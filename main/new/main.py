import tensorflow as tf
import numpy as np
import os
import argparse
from vgg_16 import *
from attention import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='VGG16 Attention Analysis')
    parser.add_argument('--imtype', type=int, default=1, choices=[1, 2, 3],
                      help='Image type: 1=merge, 2=array, 3=test')
    parser.add_argument('--category', type=int, default=1,
                      help='Object category to attend to (0-19)')
    parser.add_argument('--layer', type=int, default=12,
                      help='Layer to apply attention (0-12, >12 for all layers)')
    parser.add_argument('--attention_type', type=str, default='GRADs',
                      choices=['TCs', 'GRADs'], help='Type of attention to apply')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Batch size for processing') ###set to 75 when run experiment, 10 when debugging
    parser.add_argument('--max_images', type=int, default=10,
                      help='Maximum number of images to load') ###set to 1425 when run experiment, 10/20 when debugging
    return parser.parse_args()

def setup_paths():
    """Setup fixed paths for the project"""
    base_path = '/scratch/hy2611/CNN_attention/Data/VGG16'
    return {
        'tc_path': os.path.join(base_path, 'object_GradsTCs'),
        'weight_path': base_path,
        'image_path': os.path.join(base_path, 'images'),
        'save_path': base_path
    }

def pad_batch(batch, target_size):
    """Helper function to pad batch to target size using concatenation"""
    current_size = batch.shape[0]
    if current_size < target_size:
        # Calculate how many zeros we need to add
        pad_size = target_size - current_size
        # Create zero padding with the same shape as a single batch item
        zeros = np.zeros((pad_size,) + batch.shape[1:])
        # Concatenate the batch with zeros along the first axis
        return np.concatenate([batch, zeros], axis=0)
    return batch

def main():
    # Parse arguments
    args = parse_args()
    paths = setup_paths()
    
    # Initialize TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Setup model
    print("Initializing model...")
    imgs = tf.placeholder(tf.float32, [args.batch_size, 224, 224, 3])
    labs = tf.placeholder(tf.int32, [args.batch_size, 1])
    vgg = VGG16Base(imgs=imgs, labs=labs, 
                    weights=os.path.join(paths['weight_path'], 'vgg16_weights.npz'),
                    sess=sess)
    
    # Load category specific weights
    saver = tf.train.Saver({"fc3": vgg.fc3w, "fcb3": vgg.fc3b})
    ckpt_path = os.path.join(paths['weight_path'], 'catbins', "catbin_{0}.ckpt".format(args.category))
    print("Loading checkpoint from: {0}".format(ckpt_path))
    saver.restore(sess, ckpt_path)
    
    # Initialize components
    print("Initializing components...")
    data_loader = DataLoader(paths['image_path'], paths['tc_path'])
    attention = AttentionMechanism(paths['tc_path'])
    layer_attention = LayerAttention()
    analyzer = AttentionAnalyzer(vgg, sess)
    visualizer = Visualizer()
    
    # Load data
    print("Loading category {0} data...".format(args.category))
    try:
        pos_images = data_loader.load_category_images(args.category, args.imtype, args.max_images)
        print("Original positive images shape: {0}".format(pos_images.shape))
    except Exception as e:
        print("Error loading positive images: {0}".format(e))
        return
    
    # Setup attention parameters
    attention_strengths = np.arange(0, 1, 0.5)
    layer_mask = layer_attention.get_layer_mask(args.layer)
    
    # Create save directory for results
    save_dir = os.path.join(paths['save_path'], 
                           'attention_results_cat{0}_layer{1}'.format(args.category, args.layer))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Process batches
    print("Processing data...")
    results = []
    n_batches = (len(pos_images) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        print("Processing batch {0}/{1}".format(batch_idx + 1, n_batches))
        
        # Prepare batch
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(pos_images))
        tp_batch = pos_images[start_idx:end_idx]
        
        # Convert to float32 numpy array if not already
        tp_batch = np.array(tp_batch, dtype=np.float32)
        
        # Pad the last batch if necessary using concatenation
        if tp_batch.shape[0] < args.batch_size:
            pad_size = args.batch_size - tp_batch.shape[0]
            zeros = np.zeros((pad_size, 224, 224, 3), dtype=np.float32)
            tp_batch = np.concatenate([tp_batch, zeros], axis=0)
        
        # Get negative examples from other categories
        other_categories = list(range(20))
        other_categories.remove(args.category)
        neg_category = np.random.choice(other_categories)
        try:
            neg_images = data_loader.load_category_images(neg_category, args.imtype)
            tn_batch = neg_images[:args.batch_size]
            tn_batch = np.array(tn_batch, dtype=np.float32)
            if len(tn_batch) < args.batch_size:
                pad_size = args.batch_size - len(tn_batch)
                zeros = np.zeros((pad_size, 224, 224, 3), dtype=np.float32)
                tn_batch = np.concatenate([tn_batch, zeros], axis=0)
        except Exception as e:
            print("Error loading negative images: {0}".format(e))
            continue
            
        # Process each attention strength
        for strength in attention_strengths:
            print("Processing attention strength: {0}".format(strength))
            
            # Create attention matrices
            strength_vec = layer_attention.scale_attention_strength(strength, args.layer)
            try:
                if args.attention_type == 'TCs':
                    attn_mats = attention.make_tuning_attention(args.category, strength_vec)
                else:
                    attn_mats = attention.make_gradient_attention(args.category, strength_vec, args.imtype)
            except Exception as e:
                print("Error creating attention matrices: {0}".format(e))
                continue
            
            # Analyze attention effects
            try:
                batch_results = analyzer.analyze_attention_effects(tp_batch, tn_batch, attn_mats, [strength])
                results.append(batch_results)
            except Exception as e:
                print("Error analyzing attention effects: {0}".format(e))
                continue
    
    if not results:
        print("No results to analyze!")
        return
    
    # Aggregate results 
    print("Analyzing results...")
    
    # First, separate the results by attention strength
    strength_results = {}
    for r in results:
        strength = r['strength'][0] if isinstance(r['strength'], list) else r['strength']
        if strength not in strength_results:
            strength_results[strength] = []
        strength_results[strength].append(r)
    
    # Now aggregate for each strength
    aggregated_results = {
        'tp_scores': [],
        'tn_scores': [],
        'tp_responses': [],
        'tn_responses': []
    }
    
    for strength in sorted(strength_results.keys()):
        batch_results = strength_results[strength]
        
        # Aggregate scores
        tp_scores = np.concatenate([r['tp_scores'] for r in batch_results])
        tn_scores = np.concatenate([r['tn_scores'] for r in batch_results])
        
        # Aggregate responses - handle each layer separately
        tp_responses_by_layer = []
        tn_responses_by_layer = []
        
        n_layers = len(batch_results[0]['tp_responses'][0])
        for layer in range(n_layers):
            layer_tp = np.concatenate([r['tp_responses'][0][layer] for r in batch_results])
            layer_tn = np.concatenate([r['tn_responses'][0][layer] for r in batch_results])
            tp_responses_by_layer.append(layer_tp)
            tn_responses_by_layer.append(layer_tn)
        
        aggregated_results['tp_scores'].append(np.mean(tp_scores))
        aggregated_results['tn_scores'].append(np.mean(tn_scores))
        aggregated_results['tp_responses'].append(tp_responses_by_layer)
        aggregated_results['tn_responses'].append(tn_responses_by_layer)
    
    # Calculate metrics
    metrics = analyzer.calculate_performance_metrics(aggregated_results)
    
    # Save results
    save_prefix = "cat{0}_layer{1}_attn{2}".format(args.category, args.layer, args.attention_type)
    



    # Visualize results


    # Before visualization, ensure metrics and attention_strengths match in size
    print("Metrics shape: {0}".format(np.array(metrics['criteria']).shape))
    print("Attention strengths shape: {0}".format(attention_strengths.shape))

    # Ensure metrics is properly shaped
    metrics = {
        'performance': np.array(aggregated_results['tp_scores']),  # Should be same length as attention_strengths
        'criteria': np.array([-0.5 * (norm.ppf(np.mean(tp)) + norm.ppf(np.mean(tn))) 
                            for tp, tn in zip(aggregated_results['tp_scores'], aggregated_results['tn_scores'])]),
        'sensitivity': np.array([norm.ppf(np.mean(tp)) - norm.ppf(np.mean(tn))
                                for tp, tn in zip(aggregated_results['tp_scores'], aggregated_results['tn_scores'])])
    }

    # Add debug prints
    for key in metrics:
        print("{0} shape: {1}".format(key, metrics[key].shape))

    # Visualize results
    print("Generating visualizations...")
    fig_performance = visualizer.plot_attention_effects(metrics, attention_strengths)
    fig_performance.savefig(os.path.join(save_dir, save_prefix + '_performance.png'))
    
    # If recording layer activities
    if args.layer <= 12:
        layer_effects = analyzer.analyze_layer_effects(
            aggregated_results['tp_responses'][0],  # First attention strength
            aggregated_results['tp_responses'][0]   # Use same as baseline
        )
        layer_names = ['Layer {0}'.format(i+1) for i in range(len(layer_effects))]
        fig_layers = visualizer.plot_layer_modulation(layer_effects, layer_names)
        fig_layers.savefig(os.path.join(save_dir, save_prefix + '_layer_modulation.png'))
    
    # Save numerical results
    results_filename = os.path.join(save_dir, save_prefix + '_results.npz')
    np.savez(results_filename,
             metrics=metrics,
             attention_strengths=attention_strengths,
             layer_effects=layer_effects if args.layer <= 12 else None)
    
    print("Analysis complete! Results saved to {0}".format(save_dir))

if __name__ == '__main__':
    main()