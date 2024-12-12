########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

''' This file is used to run vgg16 on binary orientation detection tasks with attention applied according to the given parameters.
  It requires data files available on dryad.
  It saves performance broken down into true positives and true negatives and optionally saves activity of the network at intermediate layers.
Run with python 2.7
Contact: gracewlindsay@gmail.com
'''
## conda activate /ext3/envs/vgg16_env

import tensorflow as tf
import numpy as np
#from scipy.misc import imread, imresize
import pickle

attn2oriA=40 # ori attn applied to: 0, 20, 40, 60, 80, 100, 120, 140, or 160
lyr=12 #layer attention applied at: 0-12 or 13 for all layers (at 1/10th strength)
appwith = 'TCs' #what values to apply attention according to: 'TCs' or 'GRADs'
astrgs=np.arange(0,1.,.5) #attention strengths (betas)
TCpath='/scratch/hy2611/CNN_attention/Data/VGG16/ori_TCGrads'  #folder with tuning curve and gradient files 
weight_path = '/scratch/hy2611/CNN_attention/Data/VGG16' #folder with network and classifier weights
impath='/scratch/hy2611/CNN_attention/Data/VGG16/' #folder where image files are kept
save_path = '/scratch/hy2611/CNN_attention/Data/VGG16' #where to save the recording and performance files
rec_activity = True #record and save activity or no

savstr='OriAttn'+'_a'+appwith+'_'+str(attn2oriA)+'o'+str(lyr)+'l'
bnum=1; bsize=50; #100 total
freq=40; stimN=2; 



def get_noisyori(bsize):
	 #pick random indices [w replacement]
	 or_inds=np.random.randint(0,np.shape(orimat)[0],bsize)
	 batch=orimat[or_inds,:,:,:]; labels=orilabels[or_inds]
	 rsamp=np.random.choice(bsize,int(bsize*.4),False)
	 for bi in rsamp:
	     a,b,c,d=np.random.choice(224,[4]);
	     #print(a,b,c,d)
	     batch[bi,:,np.minimum(a,b):np.maximum(a,b),:]=0	     
	     batch[bi,np.minimum(c,d):np.maximum(c,d),:,:]=0
	 batch=batch+np.random.normal(0,20,[bsize,224,224,3])*(np.random.normal(0,1,[bsize,224,224,3])>0)
	 return batch,labels.T

def make_amats(oind,svec): #using TCs
	attnmats=[]; 
	with open(TCpath+"/featvecs_ORIgrats40Astd.txt", "rb") as fp:   # Unpickling
 	  b = pickle.load(fp)
	for li in range(2):
	  fv=b[li]; fmvals=np.squeeze(fv[oind,:])
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
          amat=np.ones((224,224,64))+np.tile(aval,[224,224,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(2,4):
	  fv=b[li]; fmvals=np.squeeze(fv[oind,:])
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((112,112,128))+np.tile(aval,[112,112,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(4,7):
	  fv=b[li]; fmvals=np.squeeze(fv[oind,:])
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((56,56,256))+np.tile(aval,[56,56,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(7,10):
	  fv=b[li]; fmvals=np.squeeze(fv[oind,:])
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((28,28,512))+np.tile(aval,[28,28,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(10,13):
	  fv=b[li]; fmvals=np.squeeze(fv[oind,:])
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((14,14,512))+np.tile(aval,[14,14,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	return attnmats

def make_gamats(oind,svec): #using gradients
	attnmats=[]; 
	with open(TCpath+'/DetectgradTrain2OriTCs40.txt', "rb") as fp:   #Pickling
 	  b = pickle.load(fp)
	for li in range(2):
	  fv=b[li]; fmvals=-np.squeeze(fv[oind,:])/np.amax(np.abs(fv),axis=(0,1)) #1N
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
          amat=np.ones((224,224,64))+np.tile(aval,[224,224,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	  #print amat
	for li in range(2,4):
	  fv=b[li]; fmvals=-np.squeeze(fv[oind,:])/np.amax(np.abs(fv),axis=(0,1))
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((112,112,128))+np.tile(aval,[112,112,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(4,7):
	  fv=b[li]; fmvals=-np.squeeze(fv[oind,:])/np.amax(np.abs(fv),axis=(0,1))
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((56,56,256))+np.tile(aval,[56,56,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(7,10):
	  fv=b[li]; fmvals=-np.squeeze(fv[oind,:])/np.amax(np.abs(fv),axis=(0,1))
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((28,28,512))+np.tile(aval,[28,28,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	for li in range(10,13):
	  fv=b[li]; fmvals=-np.squeeze(fv[oind,:])/np.amax(np.abs(fv),axis=(0,1))
	  #fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
	  aval=np.expand_dims(np.expand_dims(fmvals,axis=0),axis=0); #ori, fm
	  aval[aval==np.inf]=0; aval[aval==-np.inf]=0; aval=np.nan_to_num(aval)
	  amat=np.ones((14,14,512))+np.tile(aval,[14,14,1])*svec[li]; amat[amat<0]=0
	  attnmats.append(amat)
	  
	return attnmats


class vgg16:
    def __init__(self, imgs, labs=None, weights=None, sess=None):
        self.imgs = imgs
        self.labs = labs  # Save the labels placeholder
        self.convlayers()
        self.fc_layers()
        self.guess = tf.round(tf.nn.sigmoid(self.fc3l))
        if labs is not None:
            self.cross_entropy = tf.reduce_mean(tf.contrib.losses.sigmoid_cross_entropy(self.fc3l, labs)+ 0.01*tf.nn.l2_loss(self.fc3w))
            self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cross_entropy)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    # Add these methods to the vgg16 class in the original file

    def get_layer_activations(self, sess, images, attention_maps):
        """
        Get activations for all layers with attention applied
        """
        feed_dict = {
            self.imgs: images,
            self.a11: attention_maps[0],
            self.a12: attention_maps[1],
            self.a21: attention_maps[2],
            self.a22: attention_maps[3],
            self.a31: attention_maps[4],
            self.a32: attention_maps[5],
            self.a33: attention_maps[6],
            self.a41: attention_maps[7],
            self.a42: attention_maps[8],
            self.a43: attention_maps[9],
            self.a51: attention_maps[10],
            self.a52: attention_maps[11],
            self.a53: attention_maps[12]
        }
        
        return sess.run([
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3,
            self.conv5_1, self.conv5_2, self.conv5_3
        ], feed_dict=feed_dict)

    def get_baseline_activations(self, sess, images):
        """
        Get activations without attention
        """
        # Create unit attention maps
        unit_maps = [
            np.ones((224, 224, 64)),  # conv1_1, conv1_2
            np.ones((224, 224, 64)),
            np.ones((112, 112, 128)),  # conv2_1, conv2_2
            np.ones((112, 112, 128)),
            np.ones((56, 56, 256)),   # conv3_1, conv3_2, conv3_3
            np.ones((56, 56, 256)),
            np.ones((56, 56, 256)),
            np.ones((28, 28, 512)),   # conv4_1, conv4_2, conv4_3
            np.ones((28, 28, 512)),
            np.ones((28, 28, 512)),
            np.ones((14, 14, 512)),   # conv5_1, conv5_2, conv5_3
            np.ones((14, 14, 512)),
            np.ones((14, 14, 512))
        ]
        
        return self.get_layer_activations(sess, images, unit_maps)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
	    self.a11=tf.placeholder(tf.float32, [224, 224, 64])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.multiply(tf.nn.relu(out),self.a11,name=scope)
            self.parameters += [kernel, biases]
            self.smean1_1=tf.reduce_mean(self.conv1_1,[1, 2]) #b h w f
	    print 'c11', self.conv1_1.get_shape().as_list()
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
	    self.a12=tf.placeholder(tf.float32, [224, 224, 64])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.multiply(tf.nn.relu(out),self.a12, name=scope)
            self.parameters += [kernel, biases]
            self.smean1_2=tf.reduce_mean(self.conv1_2,[1, 2]) #b h w f
	    print 'c12', self.conv1_2.get_shape().as_list()
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
	    self.a21=tf.placeholder(tf.float32, [112,112,128])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.multiply(tf.nn.relu(out),self.a21, name=scope)
            self.parameters += [kernel, biases]
            self.smean2_1=tf.reduce_mean(self.conv2_1,[1, 2]) #b h w f
	    print 'c21', self.conv2_1.get_shape().as_list()
        # conv2_2
        with tf.name_scope('conv2_2') as scope:
	    self.a22=tf.placeholder(tf.float32, [112,112,128])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.multiply(tf.nn.relu(out), self.a22, name=scope)
            self.parameters += [kernel, biases]
            self.smean2_2=tf.reduce_mean(self.conv2_2,[1, 2]) #b h w f
	    print 'c22', self.conv2_2.get_shape().as_list()
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
	    self.a31=tf.placeholder(tf.float32, [56,56,256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.multiply(tf.nn.relu(out),self.a31, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_1=tf.reduce_mean(self.conv3_1,[1, 2]) #b h w f
	    print 'c31', self.conv3_1.get_shape().as_list()
        # conv3_2
        with tf.name_scope('conv3_2') as scope:
	    self.a32=tf.placeholder(tf.float32, [56,56,256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.multiply(tf.nn.relu(out),self.a32, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_2=tf.reduce_mean(self.conv3_2,[1, 2]) #b h w f
	    print 'c32', self.conv3_2.get_shape().as_list()
        # conv3_3
        with tf.name_scope('conv3_3') as scope:
	    self.a33=tf.placeholder(tf.float32, [56,56,256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.multiply(tf.nn.relu(out),self.a33, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_3=tf.reduce_mean(self.conv3_3,[1, 2]) #b h w f
	    print 'c33', self.conv3_3.get_shape().as_list()
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
	    self.a41=tf.placeholder(tf.float32, [28,28,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.multiply(tf.nn.relu(out),self.a41, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_1=tf.reduce_mean(self.conv4_1,[1, 2]) #b h w f
	    print 'c41', self.conv4_1.get_shape().as_list()
        # conv4_2
        with tf.name_scope('conv4_2') as scope:
	    self.a42=tf.placeholder(tf.float32, [28,28,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.multiply(tf.nn.relu(out),self.a42, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_2=tf.reduce_mean(self.conv4_2,[1, 2]) #b h w f
	    print 'c42', self.conv4_2.get_shape().as_list()
        # conv4_3
        with tf.name_scope('conv4_3') as scope:
	    self.a43=tf.placeholder(tf.float32, [28,28,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.multiply(tf.nn.relu(out),self.a43, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_3=tf.reduce_mean(self.conv4_3,[1, 2]) #b h w f
	    print 'c43', self.conv4_3.get_shape().as_list()
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
	    self.a51=tf.placeholder(tf.float32, [14,14,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.multiply(tf.nn.relu(out),self.a51, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_1=tf.reduce_mean(self.conv5_1,[1, 2]) #b h w f
	    print 'c51', self.conv5_1.get_shape().as_list()
        # conv5_2
        with tf.name_scope('conv5_2') as scope:
	    self.a52=tf.placeholder(tf.float32, [14,14,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.multiply(tf.nn.relu(out),self.a52, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_2=tf.reduce_mean(self.conv5_2,[1, 2]) #b h w f
	    print 'c52', self.conv5_2.get_shape().as_list()
        # conv5_3
        with tf.name_scope('conv5_3') as scope:
	    self.a53=tf.placeholder(tf.float32, [14,14,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.multiply( tf.nn.relu(out), self.a53, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_3=tf.reduce_mean(self.conv5_3,[1, 2]) #b h w f
	    print 'c53', self.conv5_3.get_shape().as_list()
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=False, name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=False, name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([4096, 1],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),trainable=True, name='weights')
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
            self.parameters += [self.fc3w, self.fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys()); keys=keys[0:-2] #so that last layer weights arent loaded
        sess.run(tf.global_variables_initializer())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))




if __name__ == '__main__':
    # Original imports
    from utils import *
    import tensorflow as tf
    import numpy as np
    from scipy.ndimage import zoom
    import pickle
    
    # Add imports for visualization
    import matplotlib
    matplotlib.use('Agg')  # For systems without display
    import matplotlib.pyplot as plt
    
    # Parameter definitions
    attn2oriA = 40  # ori attn applied to: 0, 20, 40, 60, 80, 100, 120, 140, or 160
    lyr = 12  # layer attention applied at: 0-12 or 13 for all layers (at 1/10th strength)
    appwith = 'TCs'  # what values to apply attention according to: 'TCs' or 'GRADs'
    astrgs = np.arange(0, 1., .5)  # attention strengths (betas)
    bnum = 1
    bsize = 50  # 100 total
    freq = 40
    stimN = 2
    
    # Layer bins setup
    laybins = np.zeros((13))
    if lyr > 12:
        laybins = np.ones((13))
        astrgs = astrgs/10.0
    else:
        laybins[lyr] = 1
    
    # Variable calculations
    attn2ori = attn2oriA/20
    attn2oriI = attn2oriA/10
    savstr = 'OriAttn' + '_a' + appwith + '_' + str(attn2oriA) + 'o' + str(lyr) + 'l'
    
    # Load and process images
    print "Loading image data..."
    oriload = np.load(impath + '/Stim' + str(stimN) + 'Constr5' + str(freq) + '.npz')
    orimat = oriload['arr_0']
    orilabels = oriload['arr_1']/20
    colStrlab = oriload['arr_2']
    stimlocs = oriload['arr_3']
    del oriload
    
    # Process true positives
    check = orilabels == attn2ori
    im_inds = np.where(np.sum(check, axis=1))
    im_inds = np.squeeze(im_inds)
    tp_batch = orimat[im_inds[0:bsize], :, :, :]
    
    # Process true negatives
    im_inds = np.where(np.sum(check, axis=1) == 0)
    im_inds = np.squeeze(im_inds)
    tn_batch = orimat[im_inds[0:bsize], :, :, :]
    del orimat
    
    # Initialize TensorFlow session and model
    print "Initializing TensorFlow session and model..."
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [bsize, 224, 224, 3])
    labs = tf.placeholder(tf.int32, [bsize, 1])
    
    # Create model and load weights
    vgg = vgg16(imgs, labs=labs, weights=weight_path+'/vgg16_weights.npz', sess=sess)
    saver3 = tf.train.Saver({"fc3": vgg.fc3w, "fcb3": vgg.fc3b})
    saver3.restore(sess, weight_path+'/ori_catbins'+"/oribin800_"+str(attn2ori)+".ckpt")
    
    # Initialize labels and scores
    tplabs = np.ones((bsize, 1))
    tnlabs = np.zeros((bsize, 1))
    TPscore = np.zeros((len(astrgs)))
    TNscore = np.zeros((len(astrgs)))
    
    # Main processing loop
    for ai, astrg in enumerate(astrgs):
        print "Processing attention strength %f" % astrg
        
        # Get attention masks
        if appwith == 'TCs':
            attnmats = make_amats(attn2ori, laybins * astrg)
        elif appwith == 'GRADs':
            attnmats = make_gamats(attn2ori, laybins * astrg)
        
        # Process true positive batch
        print "Processing true positive cases..."
        
        # Compute baseline saliency (without attention)
        tp_saliency_base = compute_saliency_map(sess, vgg, tp_batch, tplabs)
        
        # Setup attention feed dictionary for true positives
        tp_attn_feed_dict = {
            vgg.imgs: tp_batch,
            vgg.labs: tplabs,
            vgg.a11: attnmats[0], vgg.a12: attnmats[1],
            vgg.a21: attnmats[2], vgg.a22: attnmats[3],
            vgg.a31: attnmats[4], vgg.a32: attnmats[5], vgg.a33: attnmats[6],
            vgg.a41: attnmats[7], vgg.a42: attnmats[8], vgg.a43: attnmats[9],
            vgg.a51: attnmats[10], vgg.a52: attnmats[11], vgg.a53: attnmats[12]
        }
        
        # Compute saliency with attention for true positives
        if not hasattr(vgg, 'saliency_op'):
            vgg.saliency_op = tf.gradients(vgg.fc3l, vgg.imgs)[0]
        tp_saliency_attn = sess.run(vgg.saliency_op, feed_dict=tp_attn_feed_dict)
        tp_saliency_attn = np.abs(tp_saliency_attn).max(axis=-1)
        
        # Process and store metrics for true positives
        tp_metrics = []
        for i in range(bsize):
            image_metrics = {
                'base_saliency': tp_saliency_base[i],
                'attn_saliency': tp_saliency_attn[i],
                'attention_masks': [np.mean(m, axis=-1) for m in attnmats],
                'comparison': compare_saliency_attention(tp_saliency_base[i], tp_saliency_attn[i])
            }
            tp_metrics.append(image_metrics)
            
            # Visualize first 5 images
            if i < 5:
                viz_path = "%s/tp_vis_str%f_img%d.png" % (save_path, astrg, i)
                viz_masks = [np.mean(m, axis=-1) for m in attnmats]
                visualize_comparison(
                    tp_batch[i],
                    tp_saliency_base[i],
                    viz_masks,
                    image_metrics['comparison'],
                    viz_path
                )
        
        # Process true negative batch
        print "Processing true negative cases..."
        
        # Compute baseline saliency for true negatives
        tn_saliency_base = compute_saliency_map(sess, vgg, tn_batch, tnlabs)
        
        # Update feed dictionary for true negatives
        tn_attn_feed_dict = {
            vgg.imgs: tn_batch,
            vgg.labs: tnlabs,
            vgg.a11: attnmats[0], vgg.a12: attnmats[1],
            vgg.a21: attnmats[2], vgg.a22: attnmats[3],
            vgg.a31: attnmats[4], vgg.a32: attnmats[5], vgg.a33: attnmats[6],
            vgg.a41: attnmats[7], vgg.a42: attnmats[8], vgg.a43: attnmats[9],
            vgg.a51: attnmats[10], vgg.a52: attnmats[11], vgg.a53: attnmats[12]
        }
        
        # Compute saliency with attention for true negatives
        tn_saliency_attn = sess.run(vgg.saliency_op, feed_dict=tn_attn_feed_dict)
        tn_saliency_attn = np.abs(tn_saliency_attn).max(axis=-1)
        
        # Process and store metrics for true negatives
        tn_metrics = []
        for i in range(bsize):
            image_metrics = {
                'base_saliency': tn_saliency_base[i],
                'attn_saliency': tn_saliency_attn[i],
                'attention_masks': [np.mean(m, axis=-1) for m in attnmats],
                'comparison': compare_saliency_attention(tn_saliency_base[i], tn_saliency_attn[i])
            }
            tn_metrics.append(image_metrics)
            
            # Visualize first 5 images
            if i < 5:
                viz_path = "%s/tn_vis_str%f_img%d.png" % (save_path, astrg, i)
                viz_masks = [np.mean(m, axis=-1) for m in attnmats]
                visualize_comparison(
                    tn_batch[i],
                    tn_saliency_base[i],
                    viz_masks,
                    image_metrics['comparison'],
                    viz_path
                )
        
        # Evaluate model performance
        tp_score = sess.run(vgg.guess, feed_dict=tp_attn_feed_dict)
        tn_score = sess.run(vgg.guess, feed_dict=tn_attn_feed_dict)
        
        # Calculate and store scores
        TPscore[ai] = np.sum(tp_score == tplabs, axis=0)/(bsize*1.0)
        TNscore[ai] = np.sum(tn_score == tnlabs, axis=0)/(bsize*1.0)
        print "Attention strength %f: TP=%f, TN=%f" % (astrg, TPscore[ai], TNscore[ai])
        
        # Save analysis results for current attention strength
        np.savez("%s/%s_analysis_str%f.npz" % (save_path, savstr, astrg),
                 tp_metrics=tp_metrics,
                 tn_metrics=tn_metrics,
                 tp_saliency_base=tp_saliency_base,
                 tp_saliency_attn=tp_saliency_attn,
                 tn_saliency_base=tn_saliency_base,
                 tn_saliency_attn=tn_saliency_attn,
                 attention_strength=astrg)
    
    # Save final performance results
    np.savez(save_path+'/'+savstr+'perf.npz', TPscore, TNscore, astrgs)
    
    # Close TensorFlow session
    sess.close()