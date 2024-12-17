# analyzer.py
import numpy as np
from scipy.stats import norm
import tensorflow as tf

class AttentionAnalyzer(object):
    """Analyzer for attention effects in VGG16 network"""
    def __init__(self, vgg_model, session):
        self.vgg = vgg_model
        self.sess = session
    
    def analyze_attention_effects(self, tp_batch, tn_batch, attnmats, astrgs):
        """
        Analyze attention effects across layers with proper batch handling
        
        Args:
            tp_batch: true positive batch of images
            tn_batch: true negative batch of images
            attnmats: list of attention matrices
            astrgs: attention strengths
        """
        results = {
            'tp_responses': [],
            'tn_responses': [],
            'tp_scores': [],
            'tn_scores': [],
            'strength': astrgs
        }
        
        # Get baseline responses (no attention)
        print("Computing baseline responses...")
        baseline_dict = self._create_base_feed_dict(tp_batch)
        tp_baseline = self.sess.run(self.vgg.get_all_layers(), feed_dict=baseline_dict)
        
        # Remove batch dimension from attention matrices if present
        processed_attnmats = []
        for amat in attnmats:
            if len(amat.shape) == 4:  # If has batch dimension
                # Use the first batch item as representative attention
                processed_attnmats.append(amat[0])
            else:
                processed_attnmats.append(amat)
        
        #comment in for debugging
        # print("Attention matrix shapes after processing:")
        # for i, amat in enumerate(processed_attnmats):
        #     print("Layer {}: {}".format(i, amat.shape))
        
        # Get responses with attention
        print("Computing responses with attention...")
        try:
            feed_dict = self._create_feed_dict(tp_batch, processed_attnmats)
            tp_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
            tp_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
            
            feed_dict = self._create_feed_dict(tn_batch, processed_attnmats)
            tn_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
            tn_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
            
            results['tp_responses'].append(tp_responses)
            results['tn_responses'].append(tn_responses)
            results['tp_scores'].append(tp_score)
            results['tn_scores'].append(tn_score)
            
        except Exception as e:
            print("Error in feed_dict: ", e)
            print("Placeholder shapes:")
            placeholders = self.vgg.get_attention_placeholders()
            for i, p in enumerate(placeholders):
                print("Placeholder {}: {}".format(i, p.get_shape()))
            print("Attention matrix shapes:")
            for i, a in enumerate(processed_attnmats):
                print("Attention {}: {}".format(i, a.shape))
            raise
        
        return results
    
    def _create_base_feed_dict(self, batch):
        """Create feed dictionary with no attention"""
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        
        for placeholder in placeholders:
            shape = placeholder.get_shape().as_list()
            if None in shape:
                shape = [s if s is not None else 1 for s in shape]
            feed_dict[placeholder] = np.ones(shape)
            
        return feed_dict
    
    def _create_feed_dict(self, batch, attnmats):
        """Create feed dictionary with attention matrices"""
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        
        print("\nCreating feed dict:")
        for placeholder, attnmat in zip(placeholders, attnmats):
            expected_shape = placeholder.get_shape().as_list()
            #comment in for debugging
            # print("Placeholder shape: {}".format(expected_shape))
            # print("Attention matrix shape: {}".format(attnmat.shape))
            
            # Ensure attention matrix matches placeholder shape
            if len(expected_shape) != len(attnmat.shape):
                print("Shape mismatch - adjusting attention matrix")
                if len(expected_shape) < len(attnmat.shape):
                    # Remove batch dimension if present
                    attnmat = attnmat[0]
                elif len(expected_shape) > len(attnmat.shape):
                    # Add missing dimensions
                    for _ in range(len(expected_shape) - len(attnmat.shape)):
                        attnmat = np.expand_dims(attnmat, 0)
            
            feed_dict[placeholder] = attnmat
            
        return feed_dict

def make_attention_maps_with_batch(attention, category_idx, strength_vec, batch_size):
    """Modified attention map creation to match placeholder shapes"""
    if attention.attype == 1:  # Multiplicative attention
        attention_maps = []
        layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
        
        for group_idx, (start, end) in enumerate(layer_groups):
            h, w, c = attention.layer_dims[group_idx]
            
            for li in range(start, end):
                # Get base attention matrix
                amat = attention.make_tuning_attention(category_idx, strength_vec)
                if amat is None:
                    continue
                
                # Don't include batch dimension in attention maps
                attention_maps.append(amat[li])
                
        return attention_maps
    else:
        # Handle additive attention similarly
        return attention.make_tuning_attention(category_idx, strength_vec)

# utils.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import pickle

# Saliency_map
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr
from skimage.measure import compare_ssim as ssim
import cv2
import tensorflow as tf

def compute_performance_metrics(tp_scores, tn_scores):
    """
    Compute performance metrics (performance, criteria, sensitivity) given arrays of
    true positive (tp_scores) and true negative (tn_scores) probabilities.
    """
    # Ensure valid probability range to avoid infinite values in norm.ppf
    tp_scores = np.clip(tp_scores, 0.001, 0.999)
    tn_scores = np.clip(tn_scores, 0.001, 0.999)

    # Performance measure
    performance = (tp_scores + (1 - tn_scores)) / 2.0

    # Criteria (c)
    # c = -0.5 * (Z(Hit) + Z(False Alarm)), here tp_scores = Hit, (1 - tn_scores) = False Alarm
    criteria = -0.5 * (norm.ppf(tp_scores) + norm.ppf(tn_scores))

    # Sensitivity (d')
    # d' = Z(Hit) - Z(False Alarm)
    sensitivity = norm.ppf(tp_scores) - norm.ppf(tn_scores)

    return performance, criteria, sensitivity

def debug_print_shapes(saliency_map, attention_maps, msg=""):
    """Helper function to print shapes at various stages"""
    print("\n=== Debug Shapes {} ===".format(msg))
    print("Saliency map shape: {}".format(saliency_map.shape))
    for i, amap in enumerate(attention_maps):
        print("Attention map {} shape: {}".format(i, amap.shape))

def make_attention_maps_with_batch(attention, category_idx, strength_vec, batch_size):
    """Modified attention map creation to match placeholder shapes and ensure one call."""
    # Call make_tuning_attention once
    all_amats = attention.make_tuning_attention(category_idx, strength_vec)
    if all_amats is None or len(all_amats) == 0:
        print("Warning: make_tuning_attention returned None or empty list.")
        return None

    attention_maps = []
    layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
    for group_idx, (start, end) in enumerate(layer_groups):
        h, w, c = attention.layer_dims[group_idx]
        for li in range(start, end):
            # li should index directly into all_amats
            if li < len(all_amats):
                # Optionally add a batch dimension if needed
                amat = all_amats[li]
                amat_batch = np.tile(amat[np.newaxis, :, :, :], [batch_size, 1, 1, 1])
                attention_maps.append(amat_batch)
            else:
                print("Warning: Layer index out of range in all_amats:", li)
                continue
    return attention_maps


def compute_saliency_map(sess, model, images, labels=None, attention_maps=None):
    """
    Compute vanilla gradient-based saliency maps with improved normalization and error checking.
    """
    if not hasattr(model, 'saliency_op'):
        # Use softmax instead of sigmoid for better gradient propagation
        prob_output = tf.nn.softmax(model.fc3l)
        # Take gradients with respect to logits before activation
        model.saliency_op = tf.gradients(tf.reduce_max(model.fc3l, axis=1), model.imgs)[0]

    # Set up feed dictionary with proper attention placeholders
    feed_dict = {model.imgs: images}
    
    # Add required attention placeholders
    attention_shapes = [
        (224, 224, 64),  # a11, a12
        (112, 112, 128), # a21, a22
        (56, 56, 256),   # a31, a32, a33
        (28, 28, 512),   # a41, a42, a43
        (14, 14, 512)    # a51, a52, a53
    ]
    placeholders = model.get_attention_placeholders()

    
    if attention_maps is not None and len(attention_maps) == len(placeholders):
        # Use provided attention maps
        for p, amap in zip(placeholders, attention_maps):
            feed_dict[p] = amap
    else:
        # If no attention maps provided, use ones
        for idx, p in enumerate(placeholders):
            shape = p.get_shape().as_list()
            if None in shape:
                shape = [s if s is not None else 1 for s in shape]
            feed_dict[p] = np.ones(shape)

    if labels is not None and hasattr(model, 'labs'):
        target_labels = tf.reshape(model.labs, [-1])
        logit = tf.where(tf.equal(target_labels, 1), model.fc3l[:, 0], -model.fc3l[:, 0])
    else:
        logit = model.fc3l[:, 0]

    try:
        sal = sess.run(model.saliency_op, feed_dict=feed_dict)
        
        # Take absolute values and sum across channels
        sal = np.abs(sal)
        sal = np.sum(sal, axis=-1)
        
        # Normalize
        epsilon = 1e-10
        sal += epsilon
        sal_min = np.min(sal, axis=(1,2), keepdims=True)
        sal_max = np.max(sal, axis=(1,2), keepdims=True)
        sal = (sal - sal_min) / (sal_max - sal_min + epsilon)

        print("Saliency statistics: min={:.5f}, max={:.5f}, mean={:.5f}, std={:.5f}".format(
            np.min(sal), np.max(sal), np.mean(sal), np.std(sal)))
        
        return sal

    except Exception as e:
        print("Error computing saliency maps: {}".format(str(e)))
        print("Shape of input images: {}".format(images.shape))
        print("Feed dict keys: {}".format(feed_dict.keys()))
        return None


def compare_saliency_attention(saliency_map, attention_maps, layer_idx):
    """
    Compare saliency and attention maps with proper shape handling.
    """

    #comment in for debugging
    # print("\nDebug - Input shapes:")
    # print("Saliency map shape: {0}".format(saliency_map.shape))
    # print("Number of attention maps: {0}".format(len(attention_maps)))
    # print("Selected layer attention map shape: {0}".format(attention_maps[layer_idx].shape))
    
    # Get the attention map for the specified layer and handle batch dimension
    attention_map = attention_maps[layer_idx]
    
    # Handle batch dimension for both maps
    if len(saliency_map.shape) == 3:  # (batch, h, w)
        saliency_map = saliency_map[0]  # Take first batch
    elif len(saliency_map.shape) == 4:  # (batch, h, w, c)
        saliency_map = np.mean(saliency_map[0], axis=-1)  # Take first batch and average channels
        
    if len(attention_map.shape) == 4:  # (batch, h, w, c)
        attention_map = np.mean(attention_map[0], axis=-1)  # Take first batch and average channels
    elif len(attention_map.shape) == 3:  # (h, w, c)
        attention_map = np.mean(attention_map, axis=-1)  # Average channels
        

    #comment in for debugging
    # print("\nAfter initial processing:")
    # print("Saliency map shape: {0}".format(saliency_map.shape))
    # print("Attention map shape: {0}".format(attention_map.shape))
    
    # Ensure same spatial dimensions with minimum size check
    min_size = 7
    target_h = max(min_size, min(saliency_map.shape[0], attention_map.shape[0]))
    target_w = max(min_size, min(saliency_map.shape[1], attention_map.shape[1]))
    
    # Resize both maps
    saliency_map = cv2.resize(saliency_map.astype(np.float32), (target_w, target_h))
    attention_map = cv2.resize(attention_map.astype(np.float32), (target_w, target_h))
    
    print("\nAfter resizing:")
    print("Target shape: ({0}, {1})".format(target_h, target_w))
    print("Saliency map shape: {0}".format(saliency_map.shape))
    print("Attention map shape: {0}".format(attention_map.shape))
    
    # Normalize maps to [0,1] range
    epsilon = 1e-10
    
    def normalize_map(x):
        x_min, x_max = np.min(x), np.max(x)
        if np.abs(x_max - x_min) < epsilon:
            print("Warning: Constant map detected")
            return np.random.normal(0.5, 0.1, x.shape)
        return (x - x_min) / (x_max - x_min + epsilon)
    
    saliency_norm = normalize_map(saliency_map)
    attention_norm = normalize_map(attention_map)
    
    # Verify shapes before flattening
    assert saliency_norm.shape == attention_norm.shape, \
        "Shape mismatch: saliency {0} vs attention {1}".format(
            saliency_norm.shape, attention_norm.shape)
    
    # Print normalized map statistics
    print("\nNormalized maps statistics:")
    print("Saliency - min: {0:.5f}, max: {1:.5f}, mean: {2:.5f}, std: {3:.5f}".format(
        np.min(saliency_norm), np.max(saliency_norm), 
        np.mean(saliency_norm), np.std(saliency_norm)))
    print("Attention - min: {0:.5f}, max: {1:.5f}, mean: {2:.5f}, std: {3:.5f}".format(
        np.min(attention_norm), np.max(attention_norm), 
        np.mean(attention_norm), np.std(attention_norm)))
    
    # Compute metrics
    metrics = {}
    
    # Correlation - ensure same shape when flattening
    sal_flat = saliency_norm.flatten()
    att_flat = attention_norm.flatten()
    
    if np.std(sal_flat) > epsilon and np.std(att_flat) > epsilon:
        correlation, _ = pearsonr(sal_flat, att_flat)
        metrics['pearson_correlation'] = float(correlation)
    else:
        print("Warning: Zero variance in maps")
        metrics['pearson_correlation'] = 0.0
    
    # IoU with adaptive thresholding
    def calculate_iou(sal_map, att_map, threshold_percentile=75):
        sal_thresh = np.percentile(sal_map, threshold_percentile)
        att_thresh = np.percentile(att_map, threshold_percentile)
        
        sal_bin = sal_map > sal_thresh
        att_bin = att_map > att_thresh
        
        intersection = np.logical_and(sal_bin, att_bin).sum()
        union = np.logical_or(sal_bin, att_bin).sum()
        
        return float(intersection) / (union + epsilon) if union > 0 else 0.0
    
    metrics['iou'] = calculate_iou(saliency_norm, attention_norm)
    
    # SSIM
    try:
        metrics['ssim'] = float(ssim(saliency_norm, attention_norm, 
                                   win_size=min(7, min(target_h, target_w)-1),
                                   gaussian_weights=True))
    except Exception as e:
        print("Warning: SSIM computation failed: {0}".format(str(e)))
        metrics['ssim'] = 0.0
    
    # KL divergence
    metrics['kl_divergence'] = float(np.sum(saliency_norm * np.log((saliency_norm + epsilon) / 
                                                                  (attention_norm + epsilon))))
    
    print("\nMetrics for layer {0}:".format(layer_idx))
    for name, value in metrics.iteritems():
        print("  {0}: {1:.4f}".format(name, value))
    
    return metrics

# Add these debug statements in main.py
def debug_saliency(saliency_maps):
    """Helper function to debug saliency maps"""
    if saliency_maps is not None:
        print("Saliency map shape: {0}".format(saliency_maps.shape))
        print("Non-zero elements: {0}".format(np.count_nonzero(saliency_maps)))
        print("Value range: {0} - {1}".format(np.min(saliency_maps), np.max(saliency_maps)))

def visualize_comparison(image, saliency_map, attention_maps, metrics, save_path, batch_idx=0):
    """Modified visualization function to handle batch dimension"""
    # Extract single image from batch if necessary
    if len(image.shape) == 4:
        image = image[batch_idx]
    if len(saliency_map.shape) == 4:
        saliency_map = saliency_map[batch_idx]
        
    # Create average attention map from batch
    attention_maps = [amap[batch_idx] if len(amap.shape) == 4 else amap 
                     for amap in attention_maps]
    # Print saliency statistics again here
    print("Saliency (final) stats at visualization: min={:.5f}, max={:.5f}, mean={:.5f}, std={:.5f}".format(
        saliency_map.min(), saliency_map.max(), saliency_map.mean(), saliency_map.std()))

    fig = plt.figure(figsize=(15, 5))
    
    #comment in for debugging
    # print("Debug - Visualization input shapes:")
    # print("Image shape: {0}".format(image.shape))
    # print("Saliency map shape: {0}".format(saliency_map.shape))
    # print("Number of attention maps: {0}".format(len(attention_maps)))
    
    fig = plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original")
    plt.axis('off')

    # Saliency Map
    plt.subplot(1, 3, 2)
    if saliency_map.ndim > 2:
        saliency_map = np.mean(saliency_map, axis=-1)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Saliency")
    plt.colorbar()
    plt.axis('off')

    # Attention Maps - average them
    plt.subplot(1, 3, 3)
    # Process attention maps to ensure they're all the same size
    processed_maps = []
    target_shape = image.shape[:2]  # Use input image shape as reference
    
    for att_map in attention_maps:
        if att_map.ndim > 2:
            att_map = np.mean(att_map, axis=-1)
        if att_map.shape != target_shape:
            att_map = cv2.resize(att_map, (target_shape[1], target_shape[0]))
        processed_maps.append(att_map)
    
    avg_attention = np.mean(processed_maps, axis=0)
    plt.imshow(avg_attention, cmap='viridis')
    plt.title("Attention")
    plt.colorbar()
    plt.axis('off')

    # Display metrics
    text_str = "\n".join(["{0}: {1:.3f}".format(k, v) for k, v in metrics.iteritems()])
    plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    # Save a histogram of saliency values as well
    hist_path = save_path.replace(".png", "_hist.png")
    plt.figure()
    plt.hist(saliency_map.flatten(), bins=50, color='red', alpha=0.7)
    plt.title("Saliency Value Distribution")
    plt.xlabel("Saliency value")
    plt.ylabel("Frequency")
    plt.savefig(hist_path)
    plt.close()

# Add this debug print to the main script where comparison is called
def print_debug_info(saliency_maps, attention_maps, layer):
    """Helper function to print debug information"""
    print("Debug - Shapes before comparison:")
    print("Saliency maps shape: {0}".format(saliency_maps.shape))
    print("Attention maps shapes: {0}".format([amap.shape for amap in attention_maps]))
    print("Layer being compared: {0}".format(layer))



class DataLoader(object):
    """Utility class for loading and preprocessing data"""
    def __init__(self, image_path, tc_path):
        self.image_path = image_path
        self.tc_path = tc_path
        
    def load_category_images(self, category, image_type=1, max_images=75):
        """Load images for a specific category"""
        if image_type == 1:
            data = np.load('{0}/merg5_c{1}.npz'.format(self.image_path, category))
            raw_data = data['arr_0']
            reshaped_data = raw_data.reshape(-1, 224, 224, 3)
            return reshaped_data[:max_images]
        elif image_type == 2:
            data = np.load('{0}/arr5_c{1}.npz'.format(self.image_path, category))
            raw_data = data['arr_0']
            reshaped_data = raw_data.reshape(-1, 224, 224, 3)
            return reshaped_data
        elif image_type == 3:
            data = np.load('{0}/cats20_test15_c.npy'.format(self.image_path))
            if len(data.shape) > 4:  # If data has extra dimensions
                reshaped_data = data[category].reshape(-1, 224, 224, 3)
                return reshaped_data
            return data[category]
            
    def load_tuning_curves(self):
        """Load tuning curves data"""
        with open('{0}/featvecs20_train35_c.txt'.format(self.tc_path), 'rb') as fp:
            return pickle.load(fp)
            
    def prepare_batch(self, positive_images, negative_images, batch_size):
        """Prepare batches of positive and negative examples"""
        tp_batch = np.zeros((batch_size, 224, 224, 3))
        tn_batch = np.zeros((batch_size, 224, 224, 3))
        
        # Fill positive examples
        for i in range(min(batch_size, len(positive_images))):
            tp_batch[i] = positive_images[i]
            
        # Fill negative examples
        for i in range(min(batch_size, len(negative_images))):
            tn_batch[i] = negative_images[i]
            
        return tp_batch, tn_batch

class Visualizer(object):  # Changed to explicit (object) inheritance
    """Utility class for visualizing attention effects"""
    def __init__(self):
        mpl.rcParams['font.size'] = 22
        
    def plot_attention_effects(self, metrics, attention_strengths):
        """Plot comprehensive attention analysis results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Performance plot
        ax1.plot(attention_strengths, metrics['performance'], 'k-', linewidth=2)
        ax1.set_ylabel('Performance')
        ax1.set_xlabel('Attention Strength')
        
        # Criteria plot
        ax2.plot(attention_strengths, metrics['criteria'], 'b-', linewidth=2)
        ax2.set_ylabel('Criteria')
        ax2.set_xlabel('Attention Strength')
        
        # Sensitivity plot
        ax3.plot(attention_strengths, metrics['sensitivity'], 'r-', linewidth=2)
        ax3.set_ylabel('Sensitivity (d\')')
        ax3.set_xlabel('Attention Strength')
        
        plt.tight_layout()
        return fig
        
    def plot_layer_modulation(self, layer_effects, layer_names):
        """Plot attention modulation effects across layers"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = [effects['mean_modulation'] for effects in layer_effects.values()]
        stds = [effects['std_modulation'] for effects in layer_effects.values()]
        
        ax.errorbar(range(len(means)), means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_ylabel('Modulation Factor')
        ax.set_xlabel('Layer')
        
        plt.tight_layout()
        return fig
        
    def plot_attention_maps(self, attention_matrices, layer_names):
        """Visualize attention matrices for each layer"""
        n_layers = len(attention_matrices)
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        axes = axes.flatten()
        
        for i, (attn, name) in enumerate(zip(attention_matrices, layer_names)):
            if i < len(axes):
                im = axes[i].imshow(np.mean(attn, axis=-1), cmap='viridis')
                axes[i].set_title(name)
                axes[i].axis('off')
                
        plt.colorbar(im, ax=axes.ravel().tolist())
        plt.tight_layout()
        return fig
