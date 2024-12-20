########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

''' This file is used to run VGG16 on binary object detection tasks with attention applied according to the given parameters.
  It requires data files available on Dryad.
  It saves performance broken down into true positives and true negatives and optionally saves activity of the network at intermediate layers.
Run with Python 2.7
Contact: gracewlindsay@gmail.com
'''

## Activate the Conda environment (ensure this is done in your shell, not within the script)
# conda activate /ext3/envs/vgg16_env

import tensorflow as tf
import numpy as np
# from scipy.misc import imread, imresize
from sklearn import svm
import pickle
from utils import *

# SET VARIABLES HERE:
imtype = 1  # images: 1=merge, 2=array
cat = 5  # 0-19 for which object category to attend to and readout for
layer = 13  # 0-12 for which convolutional layer to apply attention at (if >12, will apply at all layers at 1/10th strength)
appwith = 'TCs'  # what values to apply attention according to: 'TCs' or 'GRADs'
astrgs = np.arange(0, 2., 0.5)  # attention strengths (betas)
TCpath = '/scratch/hy2611/CNN_attention/Data/VGG16/object_GradsTCs'  # folder with tuning curve and gradient files 
weight_path = '/scratch/hy2611/CNN_attention/Data/VGG16'  # folder with network and classifier weights
impath = '/scratch/hy2611/CNN_attention/Data/VGG16/images'  # folder where image files are kept
save_path = '/scratch/hy2611/CNN_attention/Data/VGG16'  # where to save the recording and performance files
Ncats = 20  # change to 5 if only using the categories available in the Dryad files
rec_activity = True  # record and save activity or no

imperim = 5
traintype = 3  # shouldn't be changed
bsize = 15 * imperim  # total number of images in each (true pos and true neg) class

# Corrected Variable Naming: Use 'impercat' consistently
impercat = int(np.floor(bsize / (Ncats - 1)))  # Convert to integer
leftov = bsize - impercat * (Ncats - 1)  # Ensure leftov is integer

# Initialize imspercats as integer array
imspercats = np.ones((Ncats - 1), dtype=int) * impercat

# Distribute the remaining images
if leftov > 0:
    selected_indices = np.random.choice(Ncats - 1, leftov, replace=True)
    for idx in selected_indices:
        imspercats[idx] += 1

bd = 1  # bidirect or pos only (0)
attype = 1  # 1=mult, 2=add

if attype == 1:
    astrgs = astrgs
elif attype == 2:
    lyrBL = [20, 100, 150, 150, 240, 240, 150, 150, 80, 20, 20, 10, 1]

def make_gamats(oind, svec):  # gradient-based attention
    attnmats = []
    with open(TCpath + '/CATgradsDetectTrainTCs_im' + str(imtype) + '.txt', "rb") as fp:  # Pickling
        b = pickle.load(fp)

    for li in range(2):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :]) / np.amax(np.abs(fv), axis=0)
        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((224, 224, 64)) + np.tile(aval, [224, 224, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [224, 224, 1]) * svec[li] * lyrBL[li]  # amat[amat<0]=0
            # amat=np.ones((224,224,64))+np.tile(aval,[224,224,1])*svec[li]; amat[amat<0]=0
        attnmats.append(amat)
        # print amat

    for li in range(2, 4):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :]) / np.amax(np.abs(fv), axis=0)
        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((112, 112, 128)) + np.tile(aval, [112, 112, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [112, 112, 1]) * svec[li] * lyrBL[li]  # amat[amat<0]=0
        attnmats.append(amat)

    for li in range(4, 7):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :]) / np.amax(np.abs(fv), axis=0)
        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((56, 56, 256)) + np.tile(aval, [56, 56, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [56, 56, 1]) * svec[li] * lyrBL[li]  # amat[amat<0]=0
        attnmats.append(amat)

    for li in range(7, 10):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :]) / np.amax(np.abs(fv), axis=0)
        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((28, 28, 512)) + np.tile(aval, [28, 28, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [28, 28, 1]) * svec[li] * lyrBL[li]  # amat[amat<0]=0
        attnmats.append(amat)

    for li in range(10, 13):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :]) / np.amax(np.abs(fv), axis=0)
        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((14, 14, 512)) + np.tile(aval, [14, 14, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [14, 14, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)
        # print amat
    return attnmats

def make_amats(oind, svec):  # tuning-based attention
    attnmats = []
    with open(TCpath + '/featvecs20_train35_c.txt', "rb") as fp:  # Pickling
        b = pickle.load(fp)
    for li in range(2):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :])

        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((224, 224, 64)) + np.tile(aval, [224, 224, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [224, 224, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)
        # print amat

    for li in range(2, 4):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :])

        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((112, 112, 128)) + np.tile(aval, [112, 112, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [112, 112, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)

    for li in range(4, 7):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :])

        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((56, 56, 256)) + np.tile(aval, [56, 56, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [56, 56, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)

    for li in range(7, 10):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :])

        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((28, 28, 512)) + np.tile(aval, [28, 28, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [28, 28, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)

    for li in range(10, 13):
        fv = b[li]
        fmvals = np.squeeze(fv[oind, :])

        # fmvals=np.random.permutation(fmvals)# shuffle vals across feat maps
        aval = np.expand_dims(np.expand_dims(fmvals, axis=0), axis=0)  # ori, fm
        aval[aval == np.inf] = 0
        aval[aval == -np.inf] = 0
        aval = np.nan_to_num(aval)
        if bd == 0:
            aval[aval < 0] = 0
        if attype == 1:
            amat = np.ones((14, 14, 512)) + np.tile(aval, [14, 14, 1]) * svec[li]
            amat[amat < 0] = 0
        elif attype == 2:
            amat = np.tile(aval, [14, 14, 1]) * svec[li] * lyrBL[li]
        attnmats.append(amat)
        # print amat
    return attnmats


class vgg16: 
    def __init__(self, imgs, labs=None, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.guess = tf.round(tf.nn.sigmoid(self.fc3l))
        self.cross_entropy = tf.reduce_mean(tf.contrib.losses.sigmoid_cross_entropy(self.fc3l, labs) + 0.01 * tf.nn.l2_loss(self.fc3w))
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cross_entropy)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            self.a11 = tf.placeholder(tf.float32, [224, 224, 64])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.multiply(tf.nn.relu(out), self.a11, name=scope)
            self.parameters += [kernel, biases]
            self.smean1_1 = tf.reduce_mean(self.conv1_1, [1, 2])  # b h w f
            print 'c11', self.conv1_1.get_shape().as_list()

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            self.a12 = tf.placeholder(tf.float32, [224, 224, 64])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.multiply(tf.nn.relu(out), self.a12, name=scope)
            self.parameters += [kernel, biases]
            self.smean1_2 = tf.reduce_mean(self.conv1_2, [1, 2])  # b h w f
            print 'c12', self.conv1_2.get_shape().as_list()

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            self.a21 = tf.placeholder(tf.float32, [112, 112, 128])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.multiply(tf.nn.relu(out), self.a21, name=scope)
            self.parameters += [kernel, biases]
            self.smean2_1 = tf.reduce_mean(self.conv2_1, [1, 2])  # b h w f
            print 'c21', self.conv2_1.get_shape().as_list()

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            self.a22 = tf.placeholder(tf.float32, [112, 112, 128])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.multiply(tf.nn.relu(out), self.a22, name=scope)
            self.parameters += [kernel, biases]
            self.smean2_2 = tf.reduce_mean(self.conv2_2, [1, 2])  # b h w f
            print 'c22', self.conv2_2.get_shape().as_list()

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            self.a31 = tf.placeholder(tf.float32, [56, 56, 256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.multiply(tf.nn.relu(out), self.a31, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_1 = tf.reduce_mean(self.conv3_1, [1, 2])  # b h w f
            print 'c31', self.conv3_1.get_shape().as_list()

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            self.a32 = tf.placeholder(tf.float32, [56, 56, 256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.multiply(tf.nn.relu(out), self.a32, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_2 = tf.reduce_mean(self.conv3_2, [1, 2])  # b h w f
            print 'c32', self.conv3_2.get_shape().as_list()

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            self.a33 = tf.placeholder(tf.float32, [56, 56, 256])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.multiply(tf.nn.relu(out), self.a33, name=scope)
            self.parameters += [kernel, biases]
            self.smean3_3 = tf.reduce_mean(self.conv3_3, [1, 2])  # b h w f
            print 'c33', self.conv3_3.get_shape().as_list()

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            self.a41 = tf.placeholder(tf.float32, [28, 28, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.multiply(tf.nn.relu(out), self.a41, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_1 = tf.reduce_mean(self.conv4_1, [1, 2])  # b h w f
            print 'c41', self.conv4_1.get_shape().as_list()

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            self.a42 = tf.placeholder(tf.float32, [28, 28, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.multiply(tf.nn.relu(out), self.a42, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_2 = tf.reduce_mean(self.conv4_2, [1, 2])  # b h w f
            print 'c42', self.conv4_2.get_shape().as_list()

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            self.a43 = tf.placeholder(tf.float32, [28, 28, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.multiply(tf.nn.relu(out), self.a43, name=scope)
            self.parameters += [kernel, biases]
            self.smean4_3 = tf.reduce_mean(self.conv4_3, [1, 2])  # b h w f
            print 'c43', self.conv4_3.get_shape().as_list()

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            self.a51 = tf.placeholder(tf.float32, [14, 14, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.multiply(tf.nn.relu(out), self.a51, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_1 = tf.reduce_mean(self.conv5_1, [1, 2])  # b h w f
            print 'c51', self.conv5_1.get_shape().as_list()

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            self.a52 = tf.placeholder(tf.float32, [14, 14, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.multiply(tf.nn.relu(out), self.a52, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_2 = tf.reduce_mean(self.conv5_2, [1, 2])  # b h w f
            print 'c52', self.conv5_2.get_shape().as_list()

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            self.a53 = tf.placeholder(tf.float32, [14, 14, 512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.multiply(tf.nn.relu(out), self.a53, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_3 = tf.reduce_mean(self.conv5_3, [1, 2])  # b h w f
            print 'c53', self.conv5_3.get_shape().as_list()

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')  # Changed name from 'pool4' to 'pool5'

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
                                                       stddev=1e-1), trainable=True, name='weights')
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32),
                                    trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
            self.parameters += [self.fc3w, self.fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        keys = keys[0:-2]  # so that last layer weights aren't loaded
        sess.run(tf.global_variables_initializer())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':
    # Original initialization code remains the same
    descat = cat 
    lyr = layer 
    sess = tf.Session()
    labs = tf.placeholder(tf.int32, [bsize, 1])
    imgs = tf.placeholder(tf.float32, [bsize, 224, 224, 3])
    vgg = vgg16(imgs=imgs, labs=labs, weights=weight_path + '/vgg16_weights.npz', sess=sess)
    
    # Load classifier weights using compat.v1
    saver3 = tf.compat.v1.train.Saver({"fc3": vgg.fc3w, "fcb3": vgg.fc3b})
    saver3.restore(sess, weight_path + '/catbins' + "/catbin_" + str(descat) + ".ckpt")
    
    # Load test images
    if imtype == 1:
        descatpics = np.load(impath + '/merg5_c' + str(descat) + '.npz')['arr_0']
    elif imtype == 2:
        descatpics = np.load(impath + '/arr5_c' + str(descat) + '.npz')['arr_0']
    elif imtype == 3:
        descatpics = np.load(impath + '/cats20_test15_c.npy')
        
    # Prepare batch of images
    tp_batch = np.zeros((bsize, 224, 224, 3))
    if imtype == 3:
        tp_batch = descatpics[descat]
    else:
        for pii in range(15):
            tp_batch[pii * imperim:(pii + 1) * imperim, :, :, :] = descatpics[pii, np.random.choice(19 * 5, imperim, replace=False), :, :, :]
    
    # Run analysis for different attention strengths
    results = []
    for astrg in astrgs:
        print("Processing attention strength: {}".format(astrg))  # Python 2.7 compatible print
        result = run_analysis(vgg, tp_batch, descat, astrg, sess,
                            make_gamats_fn=make_gamats,
                            make_amats_fn=make_amats)
        results.append(result)
    
    # Save results
    savstr = 'SaliencyAttnAnalysis' + '_c' + str(descat) + '_im' + str(imtype)
    np.savez(save_path + '/' + savstr + '.npz',
             attention_strengths=astrgs,
             grad_metrics=[r['metrics_grad'] for r in results],
             tc_metrics=[r['metrics_tc'] for r in results])
    
    # Plot trends
    plt.figure(figsize=(12, 8))
    metrics = ['pearson_correlation', 'ssim', 'iou', 'kl_divergence']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(astrgs, [r['metrics_grad'][metric] for r in results], 
                label='Gradient-based')
        plt.plot(astrgs, [r['metrics_tc'][metric] for r in results], 
                label='Tuning curve-based')
        plt.xlabel('Attention Strength')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/' + savstr + '_trends.png')
    
    # Clean up
    sess.close()