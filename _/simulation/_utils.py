import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from skimage.measure import compare_ssim as ssim
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import cv2

class SaliencyAttentionAnalyzer(object):
    def __init__(self, vgg_model, make_gamats_fn=None, make_amats_fn=None):
        """
        Initialize analyzer with a VGG model instance and attention mask functions
        """
        # Properly initialize parent class for Python 2.7
        super(SaliencyAttentionAnalyzer, self).__init__()
        
        # Initialize instance attributes
        self.model = vgg_model
        self.make_gamats = make_gamats_fn
        self.make_amats = make_amats_fn
        self.layer_names = [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3'
        ]
    def process_attention_maps(self, attention_masks, target_shape):
        """
        Process attention masks with debugging
        """
        print("Target shape: {}".format(target_shape))
        print("Number of attention masks: {}".format(len(attention_masks)))
        
        resized_masks = []
        for i, mask in enumerate(attention_masks):
            print("Mask {} shape: {}".format(i, mask.shape))
            print("Mask {} min/max: {} {}".format(i, np.min(mask), np.max(mask)))
            
            # Average across channels
            avg_mask = np.mean(mask, axis=-1)
            print("Averaged mask {} shape: {}".format(i, avg_mask.shape))
            
            # Resize to match target shape
            if len(target_shape) == 4:
                target_h, target_w = target_shape[1:3]
            else:
                target_h, target_w = target_shape[:2]
            
            resized = cv2.resize(avg_mask, (target_w, target_h))
            print("Resized mask {} shape: {}".format(i, resized.shape))
            print("Resized mask {} min/max: {} {}".format(i, np.min(resized), np.max(resized)))
            
            resized_masks.append(resized)
        
        # Stack and average across layers
        avg_attention = np.mean(resized_masks, axis=0)
        print("Final attention map shape: {}".format(avg_attention.shape))
        print("Final attention min/max: {} {}".format(np.min(avg_attention), np.max(avg_attention)))
        
        return avg_attention
    def get_attention_masks(self, image, category, attention_strength, sess, use_gradients=True):
        """
        Get attention masks either using gradients or tuning curves
        """
        if use_gradients:
            if self.make_gamats is None:
                raise ValueError("make_gamats function not provided")
            return self.make_gamats(category, [attention_strength] * 13)
        else:
            if self.make_amats is None:
                raise ValueError("make_amats function not provided")
            return self.make_amats(category, [attention_strength] * 13)


    def compute_saliency_map(self, image_batch, sess):
        """
        Compute vanilla gradient-based saliency map with debugging
        """
        # Get gradient of loss with respect to input
        loss = tf.reduce_max(self.model.fc3l)
        grads = tf.gradients(loss, self.model.imgs)[0]
        
        # Prepare feed dictionary
        feed_dict = {
            self.model.imgs: image_batch,
            self.model.a11: np.ones((224, 224, 64)),
            self.model.a12: np.ones((224, 224, 64)),
            self.model.a21: np.ones((112, 112, 128)),
            self.model.a22: np.ones((112, 112, 128)),
            self.model.a31: np.ones((56, 56, 256)),
            self.model.a32: np.ones((56, 56, 256)),
            self.model.a33: np.ones((56, 56, 256)),
            self.model.a41: np.ones((28, 28, 512)),
            self.model.a42: np.ones((28, 28, 512)),
            self.model.a43: np.ones((28, 28, 512)),
            self.model.a51: np.ones((14, 14, 512)),
            self.model.a52: np.ones((14, 14, 512)),
            self.model.a53: np.ones((14, 14, 512))
        }
        
        # Run gradient calculation with debug prints
        saliency_maps = sess.run(grads, feed_dict=feed_dict)
        print("Saliency map shape: {}".format(saliency_maps.shape))
        print("Saliency map min/max: {} {}".format(np.min(saliency_maps), np.max(saliency_maps)))
        
        # Take absolute value and max across channels
        saliency = np.max(np.abs(saliency_maps), axis=-1)
        print("Processed saliency shape: {}".format(saliency.shape))
        print("Processed saliency min/max: {} {}".format(np.min(saliency), np.max(saliency)))
        
        return saliency

    def create_visualization(self, image_batch, saliency_map, attention_masks, metrics, title_suffix=''):
        """
        Create visualization with proper scaling and value preservation
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image_batch[0].astype(np.uint8))
        plt.title('Original Image')
        plt.axis('off')
        
        # Saliency map with proper scaling
        plt.subplot(132)
        if len(saliency_map.shape) > 2:
            saliency_map = saliency_map[0]  # Take first image from batch
        
        # Scale saliency map without losing information
        saliency_norm = np.abs(saliency_map)  # Take absolute value since we have both positive and negative
        if saliency_norm.max() > 0:
            saliency_norm = saliency_norm / saliency_norm.max()
        
        print("Visualization saliency min/max before plotting: {} {}".format(
            saliency_norm.min(), saliency_norm.max()))
        
        plt.imshow(saliency_norm, cmap='jet')
        plt.title('Saliency Map')
        plt.colorbar()
        plt.axis('off')
        
        # Attention map
        plt.subplot(133)
        avg_attention = self.process_attention_maps(attention_masks, saliency_map.shape)
        if len(avg_attention.shape) > 2:
            avg_attention = avg_attention[0]
        
        # Scale attention map without losing information
        attention_norm = avg_attention - avg_attention.min()
        if attention_norm.max() > 0:
            attention_norm = attention_norm / attention_norm.max()
        
        print("Visualization attention min/max before plotting: {} {}".format(
            attention_norm.min(), attention_norm.max()))
        
        plt.imshow(attention_norm, cmap='jet')
        plt.title('Attention Map ' + title_suffix)
        plt.colorbar()
        plt.axis('off')
        
        # Add metrics text
        metrics_text = 'Metrics:\n'
        for key, value in metrics.items():
            if np.isnan(value):
                metrics_text += '{}: nan\n'.format(key)
            else:
                metrics_text += '{}: {:.3f}\n'.format(key, value)
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def compute_metrics(self, saliency_map, attention_masks):
        """
        Compute comparison metrics with fixed normalization
        """
        metrics = {}
        
        # Handle batch dimension
        if len(saliency_map.shape) == 4:
            saliency_map = np.max(np.abs(saliency_map), axis=-1)
        
        # Process attention masks to match saliency map shape
        avg_attention = self.process_attention_maps(attention_masks, saliency_map.shape)
        
        # For batch processing, compute metrics on first image
        if len(saliency_map.shape) > 2:
            saliency_map = saliency_map[0]
            avg_attention = avg_attention[0] if len(avg_attention.shape) > 2 else avg_attention
        
        # Normalize saliency map properly
        norm_saliency = np.abs(saliency_map)
        if norm_saliency.max() > 0:
            norm_saliency = norm_saliency / norm_saliency.max()
        
        # Normalize attention map properly
        norm_attention = avg_attention - avg_attention.min()
        if norm_attention.max() > 0:
            norm_attention = norm_attention / norm_attention.max()
        
        print("Metrics computation - normalized saliency min/max: {} {}".format(
            norm_saliency.min(), norm_saliency.max()))
        print("Metrics computation - normalized attention min/max: {} {}".format(
            norm_attention.min(), norm_attention.max()))
        
        # Ensure same shape
        if norm_attention.shape != norm_saliency.shape:
            norm_attention = cv2.resize(norm_attention, 
                                    (norm_saliency.shape[1], norm_saliency.shape[0]))
        
        # Compute metrics with normalized values
        metrics['pearson_correlation'] = pearsonr(norm_saliency.flatten(), 
                                                norm_attention.flatten())[0]
        
        # Convert to uint8 for SSIM calculation
        saliency_uint8 = (norm_saliency * 255).astype(np.uint8)
        attention_uint8 = (norm_attention * 255).astype(np.uint8)
        metrics['ssim'] = ssim(saliency_uint8, attention_uint8)
        
        # Compute IoU for high activation regions
        thresh_saliency = norm_saliency > np.percentile(norm_saliency, 90)
        thresh_attention = norm_attention > np.percentile(norm_attention, 90)
        intersection = np.logical_and(thresh_saliency, thresh_attention)
        union = np.logical_or(thresh_saliency, thresh_attention)
        metrics['iou'] = np.sum(intersection) / float(np.sum(union) + 1e-8)
        
        # Compute KL divergence
        eps = 1e-10
        p = norm_saliency + eps
        q = norm_attention + eps
        p = p / p.sum()
        q = q / q.sum()
        metrics['kl_divergence'] = np.sum(p * np.log(p / q))
        
        return metrics


def run_analysis(model, image_batch, category, attention_strength, sess, make_gamats_fn=None, make_amats_fn=None):
    """
    Run complete analysis pipeline
    """
    analyzer = SaliencyAttentionAnalyzer(model, 
                                       make_gamats_fn=make_gamats_fn,
                                       make_amats_fn=make_amats_fn)
    
    # Get saliency maps
    saliency_map = analyzer.compute_saliency_map(image_batch, sess)
    
    # Get attention masks
    attention_masks_grad = analyzer.get_attention_masks(
        image_batch, category, attention_strength, sess, use_gradients=True)
    attention_masks_tc = analyzer.get_attention_masks(
        image_batch, category, attention_strength, sess, use_gradients=False)
    
    # Compute metrics for both attention types
    metrics_grad = analyzer.compute_metrics(saliency_map, attention_masks_grad)
    metrics_tc = analyzer.compute_metrics(saliency_map, attention_masks_tc)
    
    # Create visualizations
    viz_grad = analyzer.create_visualization(
        image_batch, saliency_map, attention_masks_grad, metrics_grad, 'Gradient-based')
    viz_tc = analyzer.create_visualization(
        image_batch, saliency_map, attention_masks_tc, metrics_tc, 'Tuning Curve-based')
    
    # Save visualizations
    savstr = 'attention_viz_strength_{}'.format(attention_strength)
    viz_grad.savefig(savstr + '_grad.png')
    viz_tc.savefig(savstr + '_tc.png')
    plt.close('all')  # Clean up
    
    return {
        'saliency_map': saliency_map,
        'attention_masks_grad': attention_masks_grad,
        'attention_masks_tc': attention_masks_tc,
        'metrics_grad': metrics_grad,
        'metrics_tc': metrics_tc,
        'visualizations': (viz_grad, viz_tc)
    }