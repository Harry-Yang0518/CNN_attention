# saliency_utils.py

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For systems without display
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

def compute_saliency_map(sess, model, images, labels=None):
    """
    Compute saliency maps for given images using the VGG16 model
    
    Args:
        sess: TensorFlow session
        model: VGG16 model instance
        images: Input images
        labels: Optional labels for the images
    """
    # Create gradient operation if not exists
    if not hasattr(model, 'saliency_op'):
        if hasattr(model, 'cross_entropy'):
            model.saliency_op = tf.gradients(model.cross_entropy, model.imgs)[0]
        else:
            model.saliency_op = tf.gradients(model.fc3l, model.imgs)[0]
    
    # Create feed dict with default attention values (all ones)
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
    
    # Add labels if the model expects them
    if hasattr(model, 'labs') and model.labs is not None:
        # If no labels provided, create dummy ones
        if labels is None:
            batch_size = images.shape[0]
            labels = np.zeros((batch_size, 1), dtype=np.int32)
        feed_dict[model.labs] = labels
        
    saliency = sess.run(model.saliency_op, feed_dict=feed_dict)
    return np.abs(saliency).max(axis=-1)  # Take max across color channels

def visualize_comparison(image, saliency_map, attention_masks, metrics, save_path):
    """
    Visualize original image, saliency map, and attention masks with metrics
    """
    n_attention = len(attention_masks)
    fig = plt.figure(figsize=(15, 5 + (n_attention // 3) * 4))
    
    # Plot original image
    plt.subplot(2 + n_attention // 3, 3, 1)
    plt.imshow(image.astype(np.uint8))  # Convert to uint8 for proper display
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot saliency map
    plt.subplot(2 + n_attention // 3, 3, 2)
    smap = plt.imshow(saliency_map, cmap='hot')
    plt.title('Saliency Map')
    plt.colorbar(smap)
    plt.axis('off')
    
    # Plot attention masks
    for i, mask in enumerate(attention_masks):
        plt.subplot(2 + n_attention // 3, 3, i + 3)
        # Ensure mask is 2D by taking mean if it's 3D
        if len(mask.shape) > 2:
            mask_2d = np.mean(mask, axis=-1)
        else:
            mask_2d = mask
            
        mmap = plt.imshow(mask_2d, cmap='viridis')
        if isinstance(metrics, list) and len(metrics) > i:
            plt.title('Attention Mask Layer %d\nCorr: %.2f' % (i+1, metrics[i].get('correlation', 0)))
        else:
            plt.title('Attention Mask Layer %d' % (i+1))
        plt.colorbar(mmap)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_saliency_attention(saliency_map, attention_mask):
    """
    Compare saliency map with attention mask using various metrics
    """
    # Ensure both inputs are 2D
    if len(attention_mask.shape) > 2:
        attention_mask = np.mean(attention_mask, axis=-1)
    
    # Resize attention mask to match saliency map size if needed
    if attention_mask.shape != saliency_map.shape:
        # Create a temporary placeholder for resized image
        h, w = saliency_map.shape
        attention_resized = np.zeros((h, w))
        # Use scipy's zoom for resizing
        from scipy.ndimage import zoom
        zoom_h = float(h) / attention_mask.shape[0]
        zoom_w = float(w) / attention_mask.shape[1]
        attention_resized = zoom(attention_mask, [zoom_h, zoom_w])
    else:
        attention_resized = attention_mask
        
    # Normalize both maps to [0,1]
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    attn_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Calculate metrics
    correlation = pearsonr(saliency_norm.flatten(), attn_norm.flatten())[0]
    mutual_info = mutual_info_score(
        (saliency_norm * 10).astype(int).flatten(),
        (attn_norm * 10).astype(int).flatten()
    )
    
    # Calculate spatial overlap (Dice coefficient)
    intersection = np.sum((saliency_norm > 0.5) & (attn_norm > 0.5))
    union = np.sum(saliency_norm > 0.5) + np.sum(attn_norm > 0.5)
    dice = 2.0 * intersection / (union + 1e-8)
    
    return {
        'correlation': correlation,
        'mutual_info': mutual_info,
        'spatial_overlap': dice
    }