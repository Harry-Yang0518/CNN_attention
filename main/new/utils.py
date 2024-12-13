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