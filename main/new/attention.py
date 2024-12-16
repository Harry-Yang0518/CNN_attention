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
        """Create gradient-based attention matrices"""
        try:
            attnmats = []
            # Load gradient values
            grad_file = os.path.join(self.TCpath, "CATgradsDetectTrainTCs_im{0}.txt".format(imtype))
            
            if not os.path.exists(grad_file):
                print("Warning: Gradient file not found: {0}".format(grad_file))
                return None
                
            with open(grad_file, "rb") as fp:
                grads = pickle.load(fp)
            
            # Process each layer group
            layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
            
            for group_idx, (start, end) in enumerate(layer_groups):
                h, w, c = self.layer_dims[group_idx]
                
                for li in range(start, end):
                    # Get feature values for this layer
                    fv = grads[li]
                    fmvals = np.squeeze(fv[object_idx, :])
                    # Normalize values
                    max_abs = np.amax(np.abs(fv), axis=0)
                    max_abs[max_abs == 0] = 1  # Avoid division by zero
                    fmvals = fmvals / max_abs
                    
                    # Create attention values
                    aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)
                    aval = np.nan_to_num(aval)
                    
                    if self.bd == 0:
                        aval[aval < 0] = 0
                    
                    # Create attention matrix
                    if self.attype == 1:  # Multiplicative
                        amat = np.ones((h, w, c)) + np.tile(aval, [h, w, 1]) * strength_vec[li]
                        amat[amat < 0] = 0
                    else:  # Additive
                        amat = np.tile(aval, [h, w, 1]) * strength_vec[li] * self.lyrBL[li]
                    
                    attnmats.append(amat)
            
            return attnmats
            
        except Exception as e:
            print("Error in make_gradient_attention: {0}".format(str(e)))
            return None
    
    def make_tuning_attention(self, object_idx, strength_vec):
        """Create tuning curve based attention matrices"""
        try:
            attnmats = []
            # Load tuning curves
            tc_file = os.path.join(self.TCpath, 'featvecs20_train35_c.txt')
            
            if not os.path.exists(tc_file):
                print("Warning: Tuning curves file not found: {0}".format(tc_file))
                return None
                
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
                    aval = np.nan_to_num(aval)
                    
                    if self.bd == 0:
                        aval[aval < 0] = 0
                    
                    # Create attention matrix
                    if self.attype == 1:  # Multiplicative
                        amat = np.ones((h, w, c)) + np.tile(aval, [h, w, 1]) * strength_vec[li]
                        amat[amat < 0] = 0
                    else:  # Additive
                        amat = np.tile(aval, [h, w, 1]) * strength_vec[li] * self.lyrBL[li]
                    
                    attnmats.append(amat)
            
            return attnmats
            
        except Exception as e:
            print("Error in make_tuning_attention: {0}".format(str(e)))
            return None

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