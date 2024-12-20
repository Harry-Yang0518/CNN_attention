import tensorflow as tf
import numpy as np

class VGG16Base:
    '''
    VGG16 model with attention mechanism
    '''
    def __init__(self, imgs, labs=None, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.guess = tf.round(tf.nn.sigmoid(self.fc3l))
        if labs is not None:
            self.cross_entropy = tf.reduce_mean(
                tf.contrib.losses.sigmoid_cross_entropy(self.fc3l, labs) + 
                0.01 * tf.nn.l2_loss(self.fc3w))
            self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cross_entropy)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    
    def load_weights(self, weight_file, sess):
        '''
        Load pre-trained weights from a .npy file

        Args:
            weight_file: path to the .npy file
            sess: tf.Session object
        '''

        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        keys = keys[0:-2]
        sess.run(tf.global_variables_initializer())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def get_all_layers(self):
        '''
        Get all layers of the network
        '''
        return [
            self.smean1_1, self.smean1_2,
            self.smean2_1, self.smean2_2,
            self.smean3_1, self.smean3_2, self.smean3_3,
            self.smean4_1, self.smean4_2, self.smean4_3,
            self.smean5_1, self.smean5_2, self.smean5_3
        ]

    def get_attention_placeholders(self):
        '''
        Get placeholders for attention weights
        '''
        return [
            self.a11, self.a12,
            self.a21, self.a22,
            self.a31, self.a32, self.a33,
            self.a41, self.a42, self.a43,
            self.a51, self.a52, self.a53
        ]

    def convlayers(self):
        '''
        Create the convolutional layers
        '''

        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                                shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            self.a11 = tf.placeholder(tf.float32, [224, 224, 64])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.multiply(tf.nn.relu(out),self.a11,name=scope)
            self.parameters += [kernel, biases]
            self.smean1_1=tf.reduce_mean(self.conv1_1,[1, 2])
            print('c11', self.conv1_1.get_shape().as_list())

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
            self.smean1_2=tf.reduce_mean(self.conv1_2,[1, 2])
            print('c12', self.conv1_2.get_shape().as_list())
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
            self.smean2_1=tf.reduce_mean(self.conv2_1,[1, 2])
            print('c21', self.conv2_1.get_shape().as_list())

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
            self.smean2_2=tf.reduce_mean(self.conv2_2,[1, 2])
            print('c22', self.conv2_2.get_shape().as_list())
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
            self.smean3_1=tf.reduce_mean(self.conv3_1,[1, 2])
            print('c31', self.conv3_1.get_shape().as_list())

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
            self.smean3_2=tf.reduce_mean(self.conv3_2,[1, 2])
            print('c32', self.conv3_2.get_shape().as_list())

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
            self.smean3_3=tf.reduce_mean(self.conv3_3,[1, 2])
            print('c33', self.conv3_3.get_shape().as_list())
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
            self.smean4_1=tf.reduce_mean(self.conv4_1,[1, 2])
            print('c41', self.conv4_1.get_shape().as_list())

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
            self.smean4_2=tf.reduce_mean(self.conv4_2,[1, 2])
            print('c42', self.conv4_2.get_shape().as_list())

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
            self.smean4_3=tf.reduce_mean(self.conv4_3,[1, 2])
            print('c43', self.conv4_3.get_shape().as_list())
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
            self.smean5_1=tf.reduce_mean(self.conv5_1,[1, 2])
            print('c51', self.conv5_1.get_shape().as_list())

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
            self.smean5_2=tf.reduce_mean(self.conv5_2,[1, 2])
            print('c52', self.conv5_2.get_shape().as_list())

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            self.a53=tf.placeholder(tf.float32, [14,14,512])
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.multiply(tf.nn.relu(out), self.a53, name=scope)
            self.parameters += [kernel, biases]
            self.smean5_3=tf.reduce_mean(self.conv5_3,[1, 2])
            print('c53', self.conv5_3.get_shape().as_list())
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

    def fc_layers(self):
        '''
        Create the fully connected layers
        '''
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

        fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), trainable=False, name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=False, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        self.fc2 = tf.nn.relu(fc2l)
        self.parameters += [fc2w, fc2b]

        self.fc3w = tf.Variable(tf.truncated_normal([4096, 1],
                                                    dtype=tf.float32,
                                                    stddev=1e-1),trainable=True, name='weights')
        self.fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32),
                                trainable=True, name='biases')
        self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
        self.parameters += [self.fc3w, self.fc3b]
