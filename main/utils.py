import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
from skimage.measure import compare_ssim as ssim
import cv2
import tensorflow as tf
import pickle

# Analyzer class is defined in the same file for convenience
class AttentionAnalyzer(object):
    """Analyzer for attention effects in VGG16 network"""
    def __init__(self, vgg_model, session):
        self.vgg = vgg_model
        self.sess = session
    
    def analyze_attention_effects(self, tp_batch, tn_batch, attnmats, astrgs):
        """
        Analyze attention effects across layers with proper batch handling
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
                processed_attnmats.append(amat[0])
            else:
                processed_attnmats.append(amat)
        
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
        for idx, (placeholder, attnmat) in enumerate(zip(placeholders, attnmats)):
            expected_shape = tuple(d for d in placeholder.get_shape().as_list() if d is not None)
            current_shape = attnmat.shape
            
            print("Layer {0} - Expected shape: {1}, Current shape: {2}".format(
                idx, expected_shape, current_shape))
            
            if current_shape != expected_shape:
                if len(current_shape) == 4 and current_shape[0] == 1:
                    attnmat = np.squeeze(attnmat, axis=0)
                    print("Squeezed batch dimension for layer {0}".format(idx))
                else:
                    raise ValueError(
                        "Shape mismatch for layer {0}: expected {1}, got {2}".format(
                            idx, expected_shape, current_shape))
            
            feed_dict[placeholder] = attnmat
        
        return feed_dict

def pad_batch(batch, target_size):
    """Helper function to pad batch to target size using concatenation"""
    current_size = batch.shape[0]
    if current_size < target_size:
        pad_size = target_size - current_size
        zeros = np.zeros((pad_size,) + batch.shape[1:])
        return np.concatenate([batch, zeros], axis=0)
    return batch

def compute_saliency_map(sess, model, images, labels=None, attention_maps=None):
    """
    Compute vanilla gradient-based saliency maps maintaining 2D spatial structure.
    Python 2.7 compatible version.
    """
    if not hasattr(model, 'saliency_op'):
        # Use the pre-softmax logits for better gradient signal
        target_logits = model.fc3l[:, 0]  # Get logit for target class
        # Add gradient smoothing term for stability
        smoothed_logits = target_logits + 1e-8 * tf.reduce_mean(tf.square(model.imgs))
        model.saliency_op = tf.gradients(smoothed_logits, model.imgs)[0]

    feed_dict = {model.imgs: images}
    if attention_maps is not None and len(attention_maps) == len(model.get_attention_placeholders()):
        for p, amap in zip(model.get_attention_placeholders(), attention_maps):
            feed_dict[p] = amap

    try:
        # Compute gradients
        sal = sess.run(model.saliency_op, feed_dict=feed_dict)
        
        # Handle NaN values manually
        sal[np.isnan(sal)] = 0.0
        
        # Take absolute values while maintaining spatial dimensions
        sal = np.abs(sal)
        
        # Sum across color channels but keep spatial dimensions
        sal = np.sum(sal, axis=-1, keepdims=False)  # Now shape is (batch, H, W)
        
        # If saliency is all zeros or very close to zero, add small random noise
        if np.all(np.abs(sal) < 1e-10):
            print("Warning: Near-zero saliency detected, adding noise")
            sal = sal + np.random.normal(0, 0.01, sal.shape)
        
        # Normalize each saliency map while preserving 2D structure
        normalized_maps = []
        for i in range(sal.shape[0]):
            smap = sal[i]  # Get 2D map for this image
            
            # Add small noise to break ties
            smap = smap + np.random.normal(0, 1e-6, smap.shape)
            
            # Robust normalization
            vmin, vmax = np.percentile(smap, [1, 99])
            if vmax > vmin:
                smap = np.clip(smap, vmin, vmax)
                smap = (smap - vmin) / (vmax - vmin)
            else:
                smap_min = np.min(smap)
                smap_max = np.max(smap)
                if smap_max > smap_min:
                    smap = (smap - smap_min) / (smap_max - smap_min)
                else:
                    smap = np.zeros_like(smap)
            
            # Ensure output is 2D
            if len(smap.shape) != 2:
                print("Warning: Unexpected saliency map shape: {}".format(smap.shape))
                smap = np.squeeze(smap)
            
            normalized_maps.append(smap)
        
        sal = np.stack(normalized_maps)
        
        print("\nSaliency statistics:")
        print("  Shape: {}".format(sal.shape))
        print("  Min: {:.5f}".format(np.min(sal)))
        print("  Max: {:.5f}".format(np.max(sal)))
        print("  Mean: {:.5f}".format(np.mean(sal)))
        print("  Std: {:.5f}".format(np.std(sal)))
        print("  Contains NaN: {}".format(np.any(np.isnan(sal))))
        print("  Contains Inf: {}".format(np.any(np.isinf(sal))))
        
        # Final shape check
        if len(sal.shape) != 3:  # Should be (batch, H, W)
            print("Error: Invalid final saliency shape: {}".format(sal.shape))
            return None
            
        return sal
        
    except Exception as e:
        print("Error computing saliency maps: {}".format(str(e)))
        print("Shape of input images: {}".format(images.shape))
        print("Feed dict keys: {}".format(feed_dict.keys()))
        return None

def debug_saliency(saliency_maps):
    """Helper function to debug saliency maps"""
    if saliency_maps is not None:
        print("Saliency map shape: {0}".format(saliency_maps.shape))
        print("Non-zero elements: {0}".format(np.count_nonzero(saliency_maps)))
        print("Value range: {0} - {1}".format(np.min(saliency_maps), np.max(saliency_maps)))

def debug_print_shapes(saliency_map, attention_maps, msg=""):
    """Helper function to print shapes at various stages"""
    print("\n=== Debug Shapes {} ===".format(msg))
    print("Saliency map shape: {}".format(saliency_map.shape))
    for i, amap in enumerate(attention_maps):
        print("Attention map {} shape: {}".format(i, amap.shape))

def print_debug_info(saliency_maps, attention_maps, layer):
    """Helper function to print debug information"""
    print("Debug - Shapes before comparison:")
    print("Saliency maps shape: {0}".format(saliency_maps.shape))
    print("Attention maps shapes: {0}".format([amap.shape for amap in attention_maps]))
    print("Layer being compared: {0}".format(layer))

def compare_saliency_attention(saliency_map, attention_maps, layer_idx):
    """
    Compare saliency and attention maps with improved numerical stability.
    Python 2.7 compatible version.
    """
    attention_map = attention_maps[layer_idx]
    
    # Handle dimensionality
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]
    elif len(saliency_map.shape) == 4:
        saliency_map = np.mean(saliency_map[0], axis=-1)
        
    if len(attention_map.shape) == 4:
        attention_map = np.mean(attention_map[0], axis=-1)
    elif len(attention_map.shape) == 3:
        attention_map = np.mean(attention_map, axis=-1)
    
    # Handle NaN values manually
    saliency_map[np.isnan(saliency_map)] = 0.0
    attention_map[np.isnan(attention_map)] = 0.0
    
    # Resize to common shape
    min_size = 7
    target_h = max(min_size, min(saliency_map.shape[0], attention_map.shape[0]))
    target_w = max(min_size, min(saliency_map.shape[1], attention_map.shape[1]))
    
    saliency_map = cv2.resize(saliency_map.astype(np.float32), (target_w, target_h))
    attention_map = cv2.resize(attention_map.astype(np.float32), (target_w, target_h))
    
    def normalize_map(x):
        # Add small noise to break ties
        x = x + np.random.normal(0, 1e-6, x.shape)
        # Robust normalization using percentiles
        vmin, vmax = np.percentile(x, [1, 99])
        if vmax > vmin:
            x = np.clip(x, vmin, vmax)
            x = (x - vmin) / (vmax - vmin)
        else:
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)
            else:
                x = np.zeros_like(x)
        return x
    
    saliency_norm = normalize_map(saliency_map)
    attention_norm = normalize_map(attention_map)
    
    metrics = {}
    sal_flat = saliency_norm.flatten()
    att_flat = attention_norm.flatten()
    
    # Compute correlation if there's variance in both maps
    if np.std(sal_flat) > 1e-6 and np.std(att_flat) > 1e-6:
        correlation, _ = pearsonr(sal_flat, att_flat)
        metrics['pearson_correlation'] = float(correlation)
    else:
        metrics['pearson_correlation'] = 0.0
    
    # Compute IoU with proper thresholding
    def calculate_iou(sal_map, att_map, threshold_percentile=75):
        sal_thresh = np.percentile(sal_map, threshold_percentile)
        att_thresh = np.percentile(att_map, threshold_percentile)
        
        sal_bin = sal_map > sal_thresh
        att_bin = att_map > att_thresh
        
        intersection = np.logical_and(sal_bin, att_bin).sum()
        union = np.logical_or(sal_bin, att_bin).sum()
        
        return float(intersection) / (union + 1e-8) if union > 0 else 0.0
    
    metrics['iou'] = calculate_iou(saliency_norm, attention_norm)
    
    # Compute SSIM with error handling
    try:
        win_size = min(7, min(target_h, target_w)-1)
        if win_size % 2 == 0:
            win_size -= 1
        metrics['ssim'] = float(ssim(saliency_norm, attention_norm, 
                                   win_size=win_size,
                                   gaussian_weights=True))
    except Exception as e:
        print("Warning: SSIM computation failed: {0}".format(str(e)))
        metrics['ssim'] = 0.0
    
    # Compute KL divergence with smoothing
    epsilon = 1e-8
    s_dist = saliency_norm + epsilon
    a_dist = attention_norm + epsilon
    s_dist = s_dist / np.sum(s_dist)
    a_dist = a_dist / np.sum(a_dist)
    metrics['kl_divergence'] = float(np.sum(s_dist * np.log(s_dist / a_dist)))
    
    return metrics

def visualize_comparison(image, saliency_map, attention_maps, metrics, save_path, batch_idx=0):
    if len(image.shape) == 4:
        image = image[batch_idx]
    if len(saliency_map.shape) == 4:
        saliency_map = saliency_map[batch_idx]
        
    attention_maps = [amap[batch_idx] if len(amap.shape) == 4 else amap for amap in attention_maps]
    print("Saliency (final) stats at visualization: min={:.5f}, max={:.5f}, mean={:.5f}, std={:.5f}".format(
        saliency_map.min(), saliency_map.max(), saliency_map.mean(), saliency_map.std()))

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if saliency_map.ndim > 2:
        saliency_map = np.mean(saliency_map, axis=-1)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Saliency")
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    processed_maps = []
    target_shape = image.shape[:2]
    
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

    text_str = "\n".join(["{0}: {1:.3f}".format(k, v) for k, v in metrics.iteritems()])
    plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    hist_path = save_path.replace(".png", "_hist.png")
    plt.figure()
    plt.hist(saliency_map.flatten(), bins=50, color='red', alpha=0.7)
    plt.title("Saliency Value Distribution")
    plt.xlabel("Saliency value")
    plt.ylabel("Frequency")
    plt.savefig(hist_path)
    plt.close()

def make_attention_maps_with_batch(attention_mechanism, object_idx, strength_vec, batch_size=1):
    """
    Create attention maps with proper dimension handling for VGG16 layers
    
    Parameters:
    -----------
    attention_mechanism : AttentionMechanism
        Instance of the AttentionMechanism class
    object_idx : int
        Index of the object category
    strength_vec : numpy.ndarray
        Vector of attention strengths for each layer
    batch_size : int
        Size of the batch to process
    
    Returns:
    --------
    list of numpy.ndarray
        List of attention maps matching VGG16 placeholder shapes exactly
    """
    try:
        # Generate base attention maps
        if attention_mechanism.attype == 1:  # Tuning curves
            base_maps = attention_mechanism.make_tuning_attention(object_idx, strength_vec)
        else:  # Gradient-based
            base_maps = attention_mechanism.make_gradient_attention(object_idx, strength_vec)
            
        if base_maps is None:
            print("Warning: Failed to generate base attention maps")
            return None
            
        # Process attention maps to match VGG16 layer shapes
        processed_maps = []
        vgg_shapes = [
            (224, 224, 64),  # conv1_1
            (224, 224, 64),  # conv1_2
            (112, 112, 128), # conv2_1
            (112, 112, 128), # conv2_2
            (56, 56, 256),   # conv3_1
            (56, 56, 256),   # conv3_2
            (56, 56, 256),   # conv3_3
            (28, 28, 512),   # conv4_1
            (28, 28, 512),   # conv4_2
            (28, 28, 512),   # conv4_3
            (14, 14, 512),   # conv5_1
            (14, 14, 512),   # conv5_2
            (14, 14, 512),   # conv5_3
        ]
        
        for layer_idx, (base_map, target_shape) in enumerate(zip(base_maps, vgg_shapes)):
            # Remove any existing batch dimension
            if len(base_map.shape) == 4:
                base_map = np.squeeze(base_map, axis=0)
            
            h, w, c = target_shape
            current_shape = base_map.shape
            
            if current_shape != target_shape:
                print("Resizing attention map for layer {0} from {1} to {2}".format(
                    layer_idx, current_shape, target_shape))
                    
                # Create new attention map with target shape
                resized_map = np.zeros(target_shape)
                
                # Calculate scaling factors
                h_scale = float(target_shape[0]) / current_shape[0]
                w_scale = float(target_shape[1]) / current_shape[1]
                
                # Use OpenCV for spatial resizing if shapes differ
                if h_scale != 1 or w_scale != 1:
                    for c_idx in range(min(current_shape[2], target_shape[2])):
                        channel = base_map[:, :, c_idx]
                        resized_channel = cv2.resize(channel, (target_shape[1], target_shape[0]))
                        resized_map[:, :, c_idx] = resized_channel
                
                # Handle channel dimension
                if current_shape[2] != target_shape[2]:
                    if current_shape[2] < target_shape[2]:
                        # Repeat channels if we need more
                        repeat_factor = int(np.ceil(float(target_shape[2]) / current_shape[2]))
                        resized_map = np.tile(resized_map[:, :, :current_shape[2]], 
                                            (1, 1, repeat_factor))[:, :, :target_shape[2]]
                    else:
                        # Truncate channels if we have too many
                        resized_map = resized_map[:, :, :target_shape[2]]
                
                base_map = resized_map
            
            processed_maps.append(base_map)
            print("Attention map shape for layer {0}: {1}".format(layer_idx, base_map.shape))
            
        return processed_maps
        
    except Exception as e:
        print("Error in make_attention_maps_with_batch: {0}".format(str(e)))
        print("Object index: {0}".format(object_idx))
        print("Strength vector shape: {0}".format(strength_vec.shape))
        print("Batch size: {0}".format(batch_size))
        return None



class DataLoader(object):
    '''
    Class to load image data and tuning curves for the VGG16 network

    Parameters:
    -----------
    image_path : str
        Path to the image data
    tc_path : str
    Attributes:
    -----------
    '''

    def __init__(self, image_path, tc_path):
        self.image_path = image_path
        self.tc_path = tc_path
        
    def load_category_images(self, category, image_type=1, max_images=75):
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
            if len(data.shape) > 4:
                reshaped_data = data[category].reshape(-1, 224, 224, 3)
                return reshaped_data
            return data[category]
            
    def load_tuning_curves(self):
        with open('{0}/featvecs20_train35_c.txt'.format(self.tc_path), 'rb') as fp:
            return pickle.load(fp)
            
    def prepare_batch(self, positive_images, negative_images, batch_size):
        tp_batch = np.zeros((batch_size, 224, 224, 3))
        tn_batch = np.zeros((batch_size, 224, 224, 3))
        
        for i in range(min(batch_size, len(positive_images))):
            tp_batch[i] = positive_images[i]
        for i in range(min(batch_size, len(negative_images))):
            tn_batch[i] = negative_images[i]
        return tp_batch, tn_batch

class Visualizer(object):
    '''
    Class to visualize attention effects and layer modulation factors

    Parameters:
    -----------
    Attributes:
    -----------
    '''
    def __init__(self):
        mpl.rcParams['font.size'] = 22
        
    def plot_attention_effects(self, metrics, attention_strengths):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(attention_strengths, metrics['performance'], 'k-', linewidth=2)
        ax1.set_ylabel('Performance')
        ax1.set_xlabel('Attention Strength')
        
        ax2.plot(attention_strengths, metrics['criteria'], 'b-', linewidth=2)
        ax2.set_ylabel('Criteria')
        ax2.set_xlabel('Attention Strength')
        
        ax3.plot(attention_strengths, metrics['sensitivity'], 'r-', linewidth=2)
        ax3.set_ylabel('Sensitivity (d\')')
        ax3.set_xlabel('Attention Strength')
        
        plt.tight_layout()
        return fig
        
    def plot_layer_modulation(self, layer_effects, layer_names):
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
