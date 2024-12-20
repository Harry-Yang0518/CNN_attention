import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

class AttentionAnalyzer:
    def __init__(self, vgg_model, session):
        self.vgg = vgg_model
        self.sess = session
        self.mpl_setup()
    
    def mpl_setup(self):
        """Setup matplotlib parameters"""
        mpl.use('Agg')  # Set non-interactive backend
        mpl.rcParams['font.size'] = 22
    
    def analyze_attention_effects(self, tp_batch, tn_batch, attnmats, astrgs):
        """Analyze attention effects across layers"""
        tp_responses = []
        tn_responses = []
        
        # Get responses for true positives
        tp_responses = self.sess.run([
            self.vgg.smean1_1, self.vgg.smean1_2, 
            self.vgg.smean2_1, self.vgg.smean2_2,
            self.vgg.smean3_1, self.vgg.smean3_2, self.vgg.smean3_3,
            self.vgg.smean4_1, self.vgg.smean4_2, self.vgg.smean4_3,
            self.vgg.smean5_1, self.vgg.smean5_2, self.vgg.smean5_3
        ], feed_dict=self._create_feed_dict(tp_batch, attnmats))

        # Get responses for true negatives
        tn_responses = self.sess.run([
            self.vgg.smean1_1, self.vgg.smean1_2,
            self.vgg.smean2_1, self.vgg.smean2_2,
            self.vgg.smean3_1, self.vgg.smean3_2, self.vgg.smean3_3,
            self.vgg.smean4_1, self.vgg.smean4_2, self.vgg.smean4_3,
            self.vgg.smean5_1, self.vgg.smean5_2, self.vgg.smean5_3
        ], feed_dict=self._create_feed_dict(tn_batch, attnmats))

        return tp_responses, tn_responses

    def calculate_performance_metrics(self, tp_score, tn_score):
        """Calculate performance metrics including criteria and sensitivity"""
        performance = (tp_score + (1-tn_score))/2
        criteria = -0.5 * (norm.ppf(np.mean(tp_score)) + norm.ppf(np.mean(tn_score)))
        sensitivity = norm.ppf(np.mean(tp_score)) - norm.ppf(np.mean(tn_score))
        
        return performance, criteria, sensitivity

    def analyze_layer_effects(self, responses, layer_idx):
        """Analyze how attention affects activity in specific layers"""
        layer_response = responses[layer_idx]
        
        # Calculate modulation indices
        baseline = np.mean(layer_response, axis=(0,1))  # Mean across batch and spatial dims
        attention_effect = layer_response / baseline[None, None, :]  # Normalize by baseline
        
        # Calculate statistics
        mean_mod = np.mean(attention_effect)
        std_mod = np.std(attention_effect)
        max_mod = np.max(attention_effect)
        
        return {
            'mean_modulation': mean_mod,
            'std_modulation': std_mod,
            'max_modulation': max_mod,
            'raw_effects': attention_effect
        }

    def plot_analysis(self, performances, criteria, sensitivities, astrgs):
        """Plot analysis results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot performance
        ax1.errorbar(astrgs, np.mean(performances, axis=0), 
                    yerr=np.std(performances, axis=0)/np.sqrt(len(performances)),
                    color='k', linewidth=2)
        ax1.set_ylabel('Performance')
        ax1.set_xlabel('Attention Strength')
        
        # Plot criteria
        ax2.plot(astrgs, criteria, color='b', linewidth=2)
        ax2.set_ylabel('Criteria')
        ax2.set_xlabel('Attention Strength')
        
        # Plot sensitivity
        ax3.plot(astrgs, sensitivities, color='r', linewidth=2)
        ax3.set_ylabel('Sensitivity (d\')')
        ax3.set_xlabel('Attention Strength')
        
        plt.tight_layout()
        return fig

    def _create_feed_dict(self, batch, attnmats):
        """Create feed dictionary for tensorflow session"""
        feed_dict = {
            self.vgg.imgs: batch,
            self.vgg.a11: attnmats[0],
            self.vgg.a12: attnmats[1],
            self.vgg.a21: attnmats[2],
            self.vgg.a22: attnmats[3],
            self.vgg.a31: attnmats[4],
            self.vgg.a32: attnmats[5],
            self.vgg.a33: attnmats[6],
            self.vgg.a41: attnmats[7],
            self.vgg.a42: attnmats[8],
            self.vgg.a43: attnmats[9],
            self.vgg.a51: attnmats[10],
            self.vgg.a52: attnmats[11],
            self.vgg.a53: attnmats[12]
        }
        return feed_dict

    def run_full_analysis(self, tp_batch, tn_batch, attnmats, astrgs):
        """Run complete analysis pipeline"""
        results = {}
        
        # Get model responses
        tp_responses, tn_responses = self.analyze_attention_effects(tp_batch, tn_batch, attnmats, astrgs)
        
        # Calculate performance metrics
        tp_score = self.vgg.guess.eval(feed_dict=self._create_feed_dict(tp_batch, attnmats))
        tn_score = self.vgg.guess.eval(feed_dict=self._create_feed_dict(tn_batch, attnmats))
        
        perf, crit, sens = self.calculate_performance_metrics(tp_score, tn_score)
        
        # Analyze each layer
        layer_results = {}
        for i in range(13):  # 13 convolutional layers
            layer_results[f'layer_{i+1}'] = self.analyze_layer_effects(tp_responses, i)
        
        # Store results
        results['performance'] = perf
        results['criteria'] = crit
        results['sensitivity'] = sens
        results['layer_effects'] = layer_results
        
        # Generate plots
        fig = self.plot_analysis([perf], crit, sens, astrgs)
        
        return results, fig

# Example usage:
if __name__ == '__main__':
    # Setup your VGG model and session first
    sess = tf.Session()
    # ... setup VGG model ...
    
    analyzer = AttentionAnalyzer(vgg_model, sess)
    results, fig = analyzer.run_full_analysis(tp_batch, tn_batch, attnmats, astrgs)
    
    # Display or save results
    plt.savefig('attention_analysis.png')
    plt.close()