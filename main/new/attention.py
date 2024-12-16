import numpy as np
import pickle
import tensorflow as tf
import os  # Added missing import

class AttentionMechanism:
    def __init__(self, TCpath, bd=1, attype=1):
        """
        Initialize attention mechanism
        
        Args:
            TCpath (str): Path to tuning curve and gradient files
            bd (int): Bidirectional flag (1 for bidirectional, 0 for positive only)
            attype (int): Attention type (1 for multiplicative, 2 for additive)
        """
        self.TCpath = TCpath
        self.bd = bd
        self.attype = attype
        
        # Layer specific baselines for additive attention
        self.lyrBL = [20, 100, 150, 150, 240, 240, 150, 150, 80, 20, 20, 10, 1] if attype == 2 else None
        
        # Define layer dimensions
        self.layer_dims = [
            (224, 224, 64),  # conv1_1, conv1_2
            (112, 112, 128), # conv2_1, conv2_2
            (56, 56, 256),   # conv3_1, conv3_2, conv3_3
            (28, 28, 512),   # conv4_1, conv4_2, conv4_3
            (14, 14, 512)    # conv5_1, conv5_2, conv5_3
        ]
    
    def make_gradient_attention(self, object_idx, strength_vec, imtype=1):
        """
        Create gradient-based attention matrices
        
        Args:
            object_idx (int): Index of object category to attend to
            strength_vec (np.array): Vector of attention strengths for each layer
            imtype (int): Type of image processing (1 or 2)
        """
        attnmats = []
        
        # Load gradient values
        grad_file = "{0}/CATgradsDetectTrainTCs_im{1}.txt".format(self.TCpath, imtype)
        with open(grad_file, "rb") as fp:
            grads = pickle.load(fp)
        
        # Process each layer group
        layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]  # Layer ranges
        
        for group_idx, (start, end) in enumerate(layer_groups):
            h, w, c = self.layer_dims[group_idx]
            
            for li in range(start, end):
                # Get feature values for this layer
                fv = grads[li]
                fmvals = np.squeeze(fv[object_idx, :]) / np.amax(np.abs(fv), axis=0)
                
                # Create attention values
                aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)
                aval[aval == np.inf] = 0
                aval[aval == -np.inf] = 0
                aval = np.nan_to_num(aval)
                
                if self.bd == 0:
                    aval[aval < 0] = 0
                
                # Create attention matrix based on attention type
                if self.attype == 1:  # Multiplicative
                    amat = np.ones((h, w, c)) + np.tile(aval, [h, w, 1]) * strength_vec[li]
                    amat[amat < 0] = 0
                else:  # Additive
                    amat = np.tile(aval, [h, w, 1]) * strength_vec[li] * self.lyrBL[li]
                
                attnmats.append(amat)
        
        return attnmats
    
    def make_tuning_attention(self, object_idx, strength_vec):
        """
        Create tuning curve based attention matrices
        
        Args:
            object_idx (int): Index of object category to attend to
            strength_vec (np.array): Vector of attention strengths for each layer
        """
        attnmats = []
        
        # Load tuning curves
        tc_file = os.path.join(self.TCpath, 'featvecs20_train35_c.txt')
        with open(tc_file, "rb") as fp:
            tuning_curves = pickle.load(fp)
        
        # Process each layer group
        layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
        
        for group_idx, (start, end) in enumerate(layer_groups):
            h, w, c = self.layer_dims[group_idx]
            
            for li in range(start, end):
                # Get tuning values for this layer
                tc = tuning_curves[li]
                fmvals = np.squeeze(tc[object_idx, :])
                
                # Create attention values
                aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)
                aval[aval == np.inf] = 0
                aval[aval == -np.inf] = 0
                aval = np.nan_to_num(aval)
                
                if self.bd == 0:
                    aval[aval < 0] = 0
                
                # Create attention matrix based on attention type
                if self.attype == 1:  # Multiplicative
                    amat = np.ones((h, w, c)) + np.tile(aval, [h, w, 1]) * strength_vec[li]
                    amat[amat < 0] = 0
                else:  # Additive
                    amat = np.tile(aval, [h, w, 1]) * strength_vec[li] * self.lyrBL[li]
                
                attnmats.append(amat)
        
        return attnmats
    
    def apply_attention(self, model, images, attention_matrices, session):
        """
        Apply attention to model
        
        Args:
            model: VGG16 model instance
            images: Batch of images
            attention_matrices: List of attention matrices for each layer
            session: TensorFlow session
        """
        feed_dict = {
            model.imgs: images,
            model.a11: attention_matrices[0],
            model.a12: attention_matrices[1],
            model.a21: attention_matrices[2],
            model.a22: attention_matrices[3],
            model.a31: attention_matrices[4],
            model.a32: attention_matrices[5],
            model.a33: attention_matrices[6],
            model.a41: attention_matrices[7],
            model.a42: attention_matrices[8],
            model.a43: attention_matrices[9],
            model.a51: attention_matrices[10],
            model.a52: attention_matrices[11],
            model.a53: attention_matrices[12]
        }
        
        return session.run([model.fc3l, model.get_all_layers()], feed_dict=feed_dict)

    def make_gradient_attention(self, object_idx, strength_vec, imtype=1):
        """
        Create gradient-based attention matrices
        
        Args:
            object_idx (int): Index of object category to attend to
            strength_vec (np.array): Vector of attention strengths for each layer
            imtype (int): Type of image processing (1 or 2)
        """
        attnmats = []
        
        # Load gradient values
        grad_file = "{0}/CATgradsDetectTrainTCs_im{1}.txt".format(self.TCpath, imtype)
        with open(grad_file, "rb") as fp:
            grads = pickle.load(fp)
        
        # Process each layer group
        layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]

    def make_tuning_attention(self, object_idx, strength_vec):
        """
        Create tuning curve based attention matrices
        
        Args:
            object_idx (int): Index of object category to attend to
            strength_vec (np.array): Vector of attention strengths for each layer
        """
        attnmats = []
        
        # Load tuning curves
        tc_file = os.path.join(self.TCpath, 'featvecs20_train35_c.txt')
        with open(tc_file, "rb") as fp:
            tuning_curves = pickle.load(fp)
        
        # Process each layer group
        layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]


        

class LayerAttention:
    """Helper class to manage layer-specific attention"""
    def __init__(self, num_layers=13):
        self.num_layers = num_layers
        
    def get_layer_mask(self, target_layer):
        """Create binary mask for target layer(s)"""
        if target_layer > self.num_layers:
            return np.ones(self.num_layers)
        else:
            mask = np.zeros(self.num_layers)
            mask[target_layer] = 1
            return mask
            
    def scale_attention_strength(self, strength, target_layer):
        """Scale attention strength based on target layer"""
        mask = self.get_layer_mask(target_layer)
        if target_layer > self.num_layers:
            return strength * mask * 0.1  # Reduced strength when applying to all layers
        return strength * mask