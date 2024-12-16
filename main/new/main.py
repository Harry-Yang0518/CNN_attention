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
    parser.add_argument('--category', type=int, default=7,
                      help='Object category to attend to (0-19)')
    parser.add_argument('--layer', type=int, default=10,
                      help='Layer to apply attention (0-12, >12 for all layers)')
    parser.add_argument('--attention_type', type=str, default='TCs',
                      choices=['TCs', 'GRADs'], help='Type of attention to apply')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Batch size for processing') ###set to 75 when run experiment, 10 when debugging
    parser.add_argument('--max_images', type=int, default=50,
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
    attention_strengths = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    strength_vec = layer_attention.scale_attention_strength(args.layer)
    
    # Create save directory
    save_dir = os.path.join(paths['save_path'], 
                           'attention_results_cat{0}_layer{1}'.format(args.category, args.layer))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process batches
    print("Processing data...")
    for batch_idx in range(0, len(pos_images), args.batch_size):
        print("Processing batch {}/{}".format(batch_idx // args.batch_size + 1, 
                                            (len(pos_images) + args.batch_size - 1) // args.batch_size))
        
        # Prepare batch
        batch_end = min(batch_idx + args.batch_size, len(pos_images))
        tp_batch = pos_images[batch_idx:batch_end]
        tp_batch = pad_batch(tp_batch, args.batch_size)
        
        # Create labels for the batch
        tplabs = np.full((args.batch_size, 1), args.category, dtype=np.int32)
        
        # 1. Compute saliency maps
        saliency_maps = compute_saliency_map(sess, vgg, tp_batch, tplabs)
        
        # 2. Generate attention maps
        if args.attention_type == 'TCs':
            attention_maps = attention.make_tuning_attention(args.category, strength_vec)
        else:
            attention_maps = attention.make_gradient_attention(args.category, strength_vec)
            
        # 3. Compare saliency and attention
        comparison_metrics = compare_saliency_attention(saliency_maps, attention_maps[args.layer])
        
        # 4. Visualize comparison
        for img_idx in range(len(tp_batch)):
            save_path = os.path.join(save_dir, f'comparison_batch{batch_idx}_img{img_idx}.png')
            visualize_comparison(
                tp_batch[img_idx],
                saliency_maps[img_idx],
                [amap[img_idx] for amap in attention_maps],
                comparison_metrics,
                save_path
            )
        
        # 5. Compute performance metrics
        performance_metrics = compute_performance_metrics(
            analyzer.analyze_attention_effects(tp_batch, attention_maps)
        )
        
        # Save batch results
        batch_results = {
            'saliency_maps': saliency_maps,
            'attention_maps': attention_maps,
            'comparison_metrics': comparison_metrics,
            'performance_metrics': performance_metrics
        }
        
        np.save(os.path.join(save_dir, f'batch_{batch_idx}_results.npy'), batch_results)

    print("Analysis complete! Results saved to {}".format(save_dir))

if __name__ == '__main__':
    main()