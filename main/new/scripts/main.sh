#!/bin/bash

#SBATCH --job-name=DL_SYS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000


# Singularity path
ext3_path=/scratch/$USER/CNN_attention/environment/overlay-25GB-500K.ext3
sif_path=/scratch/$USER//CNN_attention/environment/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --overlay ${ext3_path}:ro ${sif_path} /bin/bash -c "
source ~/.bashrc
conda activate /ext3/envs/vgg16_env


python /scratch/hy2611/CNN_attention/main/new/main.py
"









def compute_saliency_map(sess, model, images, labels=None, attention_maps=None):
    """
    Compute vanilla gradient-based saliency maps maintaining original signal distribution.
    Python 2.7 compatible version.
    """
    if not hasattr(model, 'saliency_op'):
        # Use the pre-softmax logits for better gradient signal
        target_logits = model.fc3l[:, 0]  # Get logit for target class
        # Add small stability term
        smoothed_logits = target_logits + 1e-8 * tf.reduce_mean(tf.square(model.imgs))
        model.saliency_op = tf.gradients(smoothed_logits, model.imgs)[0]

    feed_dict = {model.imgs: images}
    if attention_maps is not None and len(attention_maps) == len(model.get_attention_placeholders()):
        for p, amap in zip(model.get_attention_placeholders(), attention_maps):
            feed_dict[p] = amap

    try:
        # Compute raw gradients
        sal = sess.run(model.saliency_op, feed_dict=feed_dict)
        
        # Handle NaN values
        sal[np.isnan(sal)] = 0.0
        
        # Take absolute value while maintaining structure
        sal = np.abs(sal)
        
        # Sum across color channels to get 2D maps
        sal = np.sum(sal, axis=-1)  # Shape becomes (batch, H, W)
        
        # Add minimal processing - just clip extreme outliers
        for i in range(sal.shape[0]):
            smap = sal[i]
            # Clip extreme outliers at 1st and 99th percentiles
            p1, p99 = np.percentile(smap, [1, 99])
            smap = np.clip(smap, p1, p99)
            sal[i] = smap
            
        print("\nSaliency statistics:")
        print("  Shape: {}".format(sal.shape))
        print("  Min: {:.5f}".format(np.min(sal)))
        print("  Max: {:.5f}".format(np.max(sal)))
        print("  Mean: {:.5f}".format(np.mean(sal)))
        print("  Std: {:.5f}".format(np.std(sal)))
            
        return sal
        
    except Exception as e:
        print("Error computing saliency maps: {}".format(str(e)))
        print("Shape of input images: {}".format(images.shape))
        print("Feed dict keys: {}".format(feed_dict.keys()))
        return None