from __future__ import print_function, division
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
    parser.add_argument('--category', type=int, default=19,
                      help='Object category to attend to (0-19)')
    parser.add_argument('--layer', type=int, default=11,
                      help='Layer to apply attention (0-12, >12 for all layers)')
    parser.add_argument('--attention_type', type=str, default='TCs',
                      choices=['TCs', 'GRADs'], help='Type of attention to apply')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for processing')
    parser.add_argument('--max_images', type=int, default=2,
                      help='Maximum number of images to load')
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

def main():
    # Parse arguments
    args = parse_args()
    paths = setup_paths()
    
    # Add path verification
    print("\nVerifying paths:")
    for path_name, path in paths.iteritems():
        exists = os.path.exists(path)
        print("{0}: {1} {2}".format(
            path_name, 
            path, 
            'exists' if exists else 'MISSING'
        ))
        if not exists:
            print("Warning: {0} directory does not exist!".format(path_name))
    
    # Initialize TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Setup model
    print("\nInitializing model...")
    imgs = tf.placeholder(tf.float32, [args.batch_size, 224, 224, 3])
    labs = tf.placeholder(tf.int32, [args.batch_size, 1])
    
    # Verify weights file exists
    weights_path = os.path.join(paths['weight_path'], 'vgg16_weights.npz')
    if not os.path.exists(weights_path):
        print("Error: VGG16 weights file not found at {0}".format(weights_path))
        return
        
    vgg = VGG16Base(imgs=imgs, labs=labs, weights=weights_path, sess=sess)
    
    # Load category specific weights with error handling
    try:
        saver = tf.train.Saver({"fc3": vgg.fc3w, "fcb3": vgg.fc3b})
        ckpt_dir = os.path.join(paths['weight_path'], 'catbins')
        ckpt_prefix = "catbin_{0}.ckpt".format(args.category)
        ckpt_path = os.path.join(ckpt_dir, ckpt_prefix)
        
        # Check for checkpoint files
        data_file = "{0}.data-00000-of-00001".format(ckpt_path)
        index_file = "{0}.index".format(ckpt_path)
        
        print("Checking checkpoint files:")
        print("Data file: {0} - {1}".format(
            data_file, "Found" if os.path.exists(data_file) else "Missing"))
        print("Index file: {0} - {1}".format(
            index_file, "Found" if os.path.exists(index_file) else "Missing"))
            
        if not os.path.exists(data_file) or not os.path.exists(index_file):
            print("Error: Checkpoint files not found")
            print("Expected pattern: {0}.*".format(ckpt_path))
            return
            
        print("Loading checkpoint from: {0}".format(ckpt_path))
        saver.restore(sess, ckpt_path)
        print("Checkpoint loaded successfully")
        
    except Exception as e:
        print("Error loading checkpoint: {0}".format(str(e)))
        print("Checkpoint directory: {0}".format(ckpt_dir))
        print("Checkpoint prefix: {0}".format(ckpt_prefix))
        return
    
    # Initialize components with verification
    print("\nInitializing components...")
    
    # Verify tuning curves file exists
    tc_file = os.path.join(paths['tc_path'], 'featvecs20_train35_c.txt')
    if not os.path.exists(tc_file):
        print("Warning: Tuning curves file not found at {0}".format(tc_file))
    else:
        print("Found tuning curves file: {0}".format(tc_file))
    
    # Verify gradient file exists if using GRADs attention
    if args.attention_type == 'GRADs':
        grad_file = os.path.join(paths['tc_path'], "CATgradsDetectTrainTCs_im{0}.txt".format(args.imtype))
        if not os.path.exists(grad_file):
            print("Warning: Gradient file not found at {0}".format(grad_file))
        else:
            print("Found gradient file: {0}".format(grad_file))
    
    data_loader = DataLoader(paths['image_path'], paths['tc_path'])
    attention = AttentionMechanism(paths['tc_path'])
    layer_attention = LayerAttention()
    analyzer = AttentionAnalyzer(vgg, sess)
    visualizer = Visualizer()
    
    # Load data with detailed error handling
    print("\nLoading category {0} data...".format(args.category))
    try:
        pos_images = data_loader.load_category_images(args.category, args.imtype, args.max_images)
        print("Original positive images shape: {0}".format(pos_images.shape))
        if pos_images.shape[0] == 0:
            print("Error: No positive images loaded!")
            return
    except Exception as e:
        print("Error loading positive images: {0}".format(str(e)))
        print("Image path: {0}".format(paths['image_path']))
        print("Category: {0}".format(args.category))
        print("Image type: {0}".format(args.imtype))
        return
    
    # Create save directory
    save_dir = os.path.join(paths['save_path'], 
                           'attention_results_cat{0}_layer{1}'.format(args.category, args.layer))
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except Exception as e:
            print("Error creating save directory: {0}".format(str(e)))
            return
    
    # Process batches
    print("\nProcessing data...")
    n_batches = (len(pos_images) + args.batch_size - 1) // args.batch_size
    
    # Setup attention parameters with validation
    attention_strengths = np.array([0.2, 0.7])
    # Uncomment for more strength values:
    # attention_strengths = np.array([0.0,0.2,0.4,0.6,0.8,1.0])

    if not isinstance(attention_strengths, np.ndarray) or len(attention_strengths) == 0:
        print("Error: Invalid attention strengths array")
        return
        
    for strength in attention_strengths:
        # Create full attention vector for all layers
        strength_vec = np.zeros(13)
        if args.layer > 12:
            strength_vec = np.ones(13) * strength * 0.1
        else:
            strength_vec[args.layer] = strength
            
        print("\nProcessing attention strength {0}".format(strength))
        print("Attention vector: {0}".format(strength_vec))
        
        for batch_idx in xrange(0, len(pos_images), args.batch_size):
            print("\nProcessing batch {0}/{1}".format(
                batch_idx // args.batch_size + 1, n_batches))
            print("Attention strength: {0}".format(strength))
            
            # Prepare batch
            batch_end = min(batch_idx + args.batch_size, len(pos_images))
            tp_batch = pos_images[batch_idx:batch_end]
            tp_batch = pad_batch(tp_batch, args.batch_size)
            
            # Create labels for the batch
            tplabs = np.full((args.batch_size, 1), args.category, dtype=np.int32)
            
            # Get negative examples with validation
            other_categories = range(20)
            other_categories.remove(args.category)
            neg_category = np.random.choice(other_categories)
            try:
                neg_images = data_loader.load_category_images(neg_category, args.imtype)
                if len(neg_images) == 0:
                    print("Warning: No negative images loaded for category {0}".format(neg_category))
                    continue
                    
                tn_batch = neg_images[:args.batch_size]
                if len(tn_batch) < args.batch_size:
                    tn_batch = pad_batch(tn_batch, args.batch_size)
            except Exception as e:
                print("Error loading negative images: {0}".format(str(e)))
                print("Negative category: {0}".format(neg_category))
                continue
            
            # Generate attention maps with batch dimension
            try:
                if args.attention_type == 'TCs':
                    print("Generating tuning curve attention maps...")
                    attention.attype = 1
                    attention_maps = make_attention_maps_with_batch(
                        attention, 
                        args.category, 
                        strength_vec,
                        args.batch_size
                    )
                else:
                    print("Generating gradient attention maps...")
                    attention.attype = 1
                    attention_maps = make_attention_maps_with_batch(
                        attention, 
                        args.category, 
                        strength_vec,
                        args.batch_size
                    )
                
                if attention_maps is None or len(attention_maps) == 0:
                    print("Warning: No attention maps generated")
                    print("Attention type: {0}".format(args.attention_type))
                    print("Category: {0}".format(args.category))
                    print("Strength vector: {0}".format(strength_vec))
                    continue
                
                print("Successfully generated {0} attention maps".format(len(attention_maps)))
                
            except Exception as e:
                print("Error generating attention maps: {0}".format(str(e)))
                continue
            
            # 1. Compute saliency maps
            print("Computing saliency maps...")
            saliency_maps = compute_saliency_map(sess, vgg, tp_batch, tplabs, attention_maps=attention_maps)
            debug_saliency(saliency_maps)
            if saliency_maps is not None:
                print("Saliency map shape:", saliency_maps.shape)
                print("Non-zero elements:", np.count_nonzero(saliency_maps))
                print("Value range:", np.min(saliency_maps), "-", np.max(saliency_maps))
            if saliency_maps is None:
                print("Warning: Failed to generate saliency maps")
                continue
            debug_print_shapes(saliency_maps, attention_maps, "After saliency computation")
            print("Successfully generated saliency maps")
            
            # 2. Compare saliency and attention for each layer
            print("Comparing saliency and attention maps...")
            try:
                all_layer_metrics = {}
                for layer_idx in xrange(len(attention_maps)):
                    metrics = compare_saliency_attention(
                        saliency_maps, 
                        attention_maps,
                        layer_idx
                    )
                    all_layer_metrics['layer_{0}'.format(layer_idx)] = metrics
                    print("Layer {0} metrics:".format(layer_idx))
                    for metric_name, value in metrics.iteritems():
                        print("  {0}: {1:.4f}".format(metric_name, value))
                
            except Exception as e:
                print("Error computing comparison metrics: {0}".format(str(e)))
                continue
            
            # 3. Analyze attention effects
            print("Analyzing attention effects...")
            try:
                attention_results = analyzer.analyze_attention_effects(
                    tp_batch, 
                    tn_batch, 
                    attention_maps, 
                    [strength]
                )
                if attention_results is None:
                    print("Warning: No attention analysis results generated")
                    continue
                print("Successfully analyzed attention effects")
            except Exception as e:
                print("Error analyzing attention effects: {0}".format(str(e)))
                continue
            
            # 4. Visualize comparison
            print("Generating visualizations...")
            for img_idx in xrange(min(len(tp_batch), args.batch_size)):
                try:
                    save_path = os.path.join(
                        save_dir, 
                        'comparison_batch{0}_img{1}_strength{2}.png'.format(
                            batch_idx, img_idx, strength
                        )
                    )
                    visualize_comparison(
                        tp_batch,
                        saliency_maps,
                        attention_maps,
                        all_layer_metrics['layer_{0}'.format(args.layer)],
                        save_path,
                        batch_idx=img_idx
                    )
                    print("Saved visualization for image {0}".format(img_idx))
                except Exception as e:
                    print("Error visualizing image {0}: {1}".format(img_idx, str(e)))
                    continue
            
            # 5. Save batch results
            try:
                batch_results = {
                    'strength': strength,
                    'saliency_maps': saliency_maps,
                    'attention_maps': attention_maps,
                    'comparison_metrics': all_layer_metrics,
                    'attention_results': attention_results
                }
                
                save_path = os.path.join(
                    save_dir, 
                    'batch_{0}_strength_{1}_results.npy'.format(batch_idx, strength)
                )
                np.save(save_path, batch_results)
                print("Saved batch results to {0}".format(save_path))
            except Exception as e:
                print("Error saving batch results: {0}".format(str(e)))
                continue
    
    print("\nAnalysis complete! Results saved to {0}".format(save_dir))

if __name__ == '__main__':
    main()
