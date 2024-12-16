# analyzer.py
import numpy as np
from scipy.stats import norm
import tensorflow as tf

class AttentionAnalyzer(object):  # Changed to explicit (object) inheritance for Python 2.7
    """Analyzer for attention effects in VGG16 network"""
    def __init__(self, vgg_model, session):
        self.vgg = vgg_model
        self.sess = session
    
    def analyze_attention_effects(self, tp_batch, tn_batch, attnmats, astrgs):
        """Analyze attention effects across layers"""
        results = {
            'tp_responses': [],
            'tn_responses': [],
            'tp_scores': [],
            'tn_scores': [],
            'strength': astrgs  # Added strength to results
        }
        
        # Get baseline responses (no attention)
        baseline_dict = self._create_base_feed_dict(tp_batch)
        tp_baseline = self.sess.run(self.vgg.get_all_layers(), feed_dict=baseline_dict)
        
        # Get responses with attention
        feed_dict = self._create_feed_dict(tp_batch, attnmats)
        tp_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
        tp_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
        
        feed_dict = self._create_feed_dict(tn_batch, attnmats)
        tn_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
        tn_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
        
        results['tp_responses'].append(tp_responses)
        results['tn_responses'].append(tn_responses)
        results['tp_scores'].append(tp_score)
        results['tn_scores'].append(tn_score)
        
        return results
    
    def calculate_performance_metrics(self, results):
        """Calculate performance metrics from response data"""
        tp_scores = np.array(results['tp_scores'])
        tn_scores = np.array(results['tn_scores'])
        
        print("Debug - TP scores: {0}".format(tp_scores))
        print("Debug - TN scores: {0}".format(tn_scores))
        
        # Make sure scores are 1D arrays
        if tp_scores.ndim > 1:
            tp_scores = np.mean(tp_scores, axis=1)
        if tn_scores.ndim > 1:
            tn_scores = np.mean(tn_scores, axis=1)
        
        # Clip scores to valid probability range
        tp_scores = np.clip(tp_scores, 0.001, 0.999)  # Avoid infinite values in ppf
        tn_scores = np.clip(tn_scores, 0.001, 0.999)  # Avoid infinite values in ppf
        
        # Calculate metrics while preserving shape
        metrics = {
            'performance': np.array((tp_scores + (1-tn_scores))/2),
            'criteria': np.array([-0.5 * (norm.ppf(tp) + norm.ppf(tn)) for tp, tn in zip(tp_scores, tn_scores)]),
            'sensitivity': np.array([norm.ppf(tp) - norm.ppf(tn) for tp, tn in zip(tp_scores, tn_scores)])
        }
        
        # Replace any remaining NaN or inf values with zeros
        for key in metrics:
            metrics[key] = np.nan_to_num(metrics[key])
        
        # Ensure all metrics have same shape as attention_strengths
        for key in metrics:
            if np.isscalar(metrics[key]):
                metrics[key] = np.array([metrics[key]])
        
        return metrics
    
    def analyze_layer_effects(self, responses, baseline_responses):
        """Analyze attention effects in each layer"""
        layer_effects = {}
        
        for layer_idx, (resp, base) in enumerate(zip(responses, baseline_responses)):
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            # Calculate modulation indices with safety checks
            modulation = np.divide(resp, base + epsilon)
            # Clip extreme values
            modulation = np.clip(modulation, 0.0, 1000.0)  # Reasonable max value
            
            layer_effects['layer_{0}'.format(layer_idx+1)] = {
                'mean_modulation': np.nanmean(modulation),  # Use nanmean to handle NaN values
                'std_modulation': np.nanstd(modulation),    # Use nanstd to handle NaN values
                'max_modulation': np.nanmax(modulation),    # Use nanmax to handle NaN values
                'raw_effects': modulation
            }
        
        return layer_effects
    
    def _create_base_feed_dict(self, batch):
        """Create feed dictionary with no attention"""
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        
        for placeholder in placeholders:
            shape = placeholder.get_shape().as_list()
            feed_dict[placeholder] = np.ones(shape)
            
        return feed_dict
    
    def _create_feed_dict(self, batch, attnmats):
        """Create feed dictionary with attention matrices"""
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        
        for placeholder, attnmat in zip(placeholders, attnmats):
            feed_dict[placeholder] = attnmat
            
        return feed_dict

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

def compare_saliency_attention(saliency_map, attention_map):
    """
    Compare a saliency map to an attention map and compute similarity metrics.
    Modified to handle small image dimensions.
    """
    print("Debug - Input shapes:")
    print("Saliency map shape: {0}".format(saliency_map.shape))
    print("Attention map shape: {0}".format(attention_map.shape))
    
    # Ensure both maps are 2D
    if len(saliency_map.shape) > 2:
        print("Reducing saliency map dimensions...")
        if len(saliency_map.shape) == 3:
            saliency_map = np.mean(saliency_map, axis=-1)
        elif len(saliency_map.shape) == 4:
            saliency_map = np.mean(saliency_map, axis=(0, -1))
    
    if len(attention_map.shape) > 2:
        print("Reducing attention map dimensions...")
        if len(attention_map.shape) == 3:
            attention_map = np.mean(attention_map, axis=-1)
        elif len(attention_map.shape) == 4:
            attention_map = np.mean(attention_map, axis=(0, -1))
    
    print("After reduction:")
    print("Saliency map shape: {0}".format(saliency_map.shape))
    print("Attention map shape: {0}".format(attention_map.shape))
    
    # Ensure same spatial dimensions and resize to a minimum size for SSIM
    min_size = 16  # Minimum size for SSIM calculation
    if saliency_map.shape != attention_map.shape or min(saliency_map.shape) < min_size:
        print("Resizing maps...")
        # Calculate target shape (ensure minimum size)
        if saliency_map.shape != attention_map.shape:
            target_shape = (max(min_size, min(saliency_map.shape[0], attention_map.shape[0])),
                          max(min_size, min(saliency_map.shape[1], attention_map.shape[1])))
        else:
            target_shape = (max(min_size, saliency_map.shape[0]),
                          max(min_size, saliency_map.shape[1]))
        
        # Use simple interpolation
        saliency_map = cv2.resize(saliency_map.astype(np.float32), (target_shape[1], target_shape[0]))
        attention_map = cv2.resize(attention_map.astype(np.float32), (target_shape[1], target_shape[0]))
        
        print("After resizing:")
        print("Saliency map shape: {0}".format(saliency_map.shape))
        print("Attention map shape: {0}".format(attention_map.shape))

    # Normalize both maps to [0,1]
    s_min, s_max = saliency_map.min(), saliency_map.max()
    a_min, a_max = attention_map.min(), attention_map.max()

    if s_max > s_min:
        saliency_norm = (saliency_map - s_min) / (s_max - s_min)
    else:
        saliency_norm = saliency_map

    if a_max > a_min:
        attention_norm = (attention_map - a_min) / (a_max - a_min)
    else:
        attention_norm = attention_map

    # Calculate metrics
    correlation, _ = pearsonr(saliency_norm.flatten(), attention_norm.flatten())

    # Prepare images for SSIM
    saliency_uint8 = (saliency_norm * 255).astype(np.uint8)
    attention_uint8 = (attention_norm * 255).astype(np.uint8)
    
    # Calculate SSIM with adjusted window size
    try:
        win_size = min(7, min(saliency_uint8.shape))  # Adjust window size based on image size
        if win_size % 2 == 0:
            win_size -= 1  # Ensure odd window size
        print("Using SSIM window size: {0}".format(win_size))
        structural_sim = ssim(saliency_uint8, attention_uint8, win_size=win_size)
    except Exception as e:
        print("Warning: SSIM calculation failed: {0}".format(str(e)))
        print("Falling back to MSE-based similarity")
        # Fallback to MSE-based similarity if SSIM fails
        mse = np.mean((saliency_norm - attention_norm) ** 2)
        structural_sim = 1 / (1 + mse)  # Convert MSE to similarity measure

    # IoU calculation
    s_thresh = np.percentile(saliency_norm, 90)
    a_thresh = np.percentile(attention_norm, 90)
    s_bin = saliency_norm > s_thresh
    a_bin = attention_norm > a_thresh
    intersection = np.logical_and(s_bin, a_bin).sum()
    union = np.logical_or(s_bin, a_bin).sum()
    iou = intersection / (union + 1e-8)

    # KL divergence
    p = saliency_norm.flatten() + 1e-10
    q = attention_norm.flatten() + 1e-10
    p /= p.sum()
    q /= q.sum()
    kl_divergence = np.sum(p * np.log(p / q))

    print("Metrics computed successfully")
    return {
        'pearson_correlation': correlation,
        'ssim': structural_sim,
        'iou': iou,
        'kl_divergence': kl_divergence
    }

def visualize_comparison(image, saliency_map, attention_maps, metrics, save_path):
    """
    Visualize the original image, its saliency map, and the attention maps.
    Added dimension handling and debugging.
    """
    print("Debug - Visualization input shapes:")
    print("Image shape: {0}".format(image.shape))
    print("Saliency map shape: {0}".format(saliency_map.shape))
    print("Number of attention maps: {0}".format(len(attention_maps)))
    
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

# Add this debug print to the main script where comparison is called
def print_debug_info(saliency_maps, attention_maps, layer):
    """Helper function to print debug information"""
    print("Debug - Shapes before comparison:")
    print("Saliency maps shape: {0}".format(saliency_maps.shape))
    print("Attention maps shapes: {0}".format([amap.shape for amap in attention_maps]))
    print("Layer being compared: {0}".format(layer))

def compute_saliency_map(sess, model, images, labels=None):
    """
    Compute vanilla gradient-based saliency maps for given images using the model.
    """
    if not hasattr(model, 'saliency_op'):
        model.saliency_op = tf.gradients(model.fc3l, model.imgs)[0]

    feed_dict = {
        model.imgs: images,
        model.a11: np.ones((224, 224, 64)),
        model.a12: np.ones((224, 224, 64)),
        model.a21: np.ones((112, 112, 128)),
        model.a22: np.ones((112, 112, 128)),
        model.a31: np.ones((56, 56, 256)),
        model.a32: np.ones((56, 56, 256)),
        model.a33: np.ones((56, 56, 256)),
        model.a41: np.ones((28, 28, 512)),
        model.a42: np.ones((28, 28, 512)),
        model.a43: np.ones((28, 28, 512)),
        model.a51: np.ones((14, 14, 512)),
        model.a52: np.ones((14, 14, 512)),
        model.a53: np.ones((14, 14, 512))
    }

    if labels is not None and hasattr(model, 'labs'):
        feed_dict[model.labs] = labels

    sal = sess.run(model.saliency_op, feed_dict=feed_dict)
    sal = np.abs(sal).max(axis=-1)
    return sal

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
