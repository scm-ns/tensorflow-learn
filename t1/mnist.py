from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

NUM_CLASSES = 10 ; IMAGE_SIZE = 28 ; 
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE

def inference(img , hd1_units , hd2_units):
    """
    Build MNIST model up to be used for inference 


    
    """

    with tf.name_scope('hidden1'):
       W = tf.Variable(
                        tf.truncated_normal(
                                            [IMAGE_SIZE_FLAT , hd1_units] , 
                                            stddev = 1.0 / math.sqrt( float(IMAGE_SIZE_FLAT) ) 
                                           )  ,
                        name = 'weights'
                       )

       b = tf.Variable(
                        tf.zeors([hd1_units]) , 
                        name = 'biases'
                      )

       hd1 = tf.nn.relu(tf.matmul(img, W) + b) 

    with tf.name_scope('hidden2'):
       W = tf.Variable(
                        tf.truncated_normal(
                                            [hd1_units , hd2_units] , 
                                            stddev = 1.0 / math.sqrt( float(hd1_units) ) 
                                           )  ,
                        name = 'weights'
                       )

       b = tf.Variable(
                        tf.zeors([hd2_units]) , 
                        name = 'biases'
                      )

       hd2 = tf.nn.relu(tf.matmul(hd1, W) + b) 

    # Linear Softmax layer 
    with tf.name_scope('softmax_linear'):
        W = tf.Variable(
                            tf.truncated_normal(
                                                [hd2_units , NUM_CLASSES] , 
                                                stddev = 1.0 / math.sqrt( float(hd2_units) ) 
                                               )  ,
                            name = 'weights'
                        )

        b = tf.Variable(
                        tf.zeors([NUM_CLASSES]) , 
                        name = 'biases'
                        )

        logits = tf.matmul(hd2_units , W) + b

    return logits


def loss(logits , labels):
    lables = tf.to_int64(labels)

    cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits , labels , name ='xentropy')
    loss = tf.reduce_mean(cross_entropy , name='xentropy_mean')
    return loss

def train(loss , learning_rate):
    tf.scalar_summary(loss.op.name , loss) # Debugging
    optimizer = tf.train.GradientDescentOptimizer(learning_rate )

    global_step = tf.Variable(0 , name='global_step' , trainable=False)
    train_op = optimizer.minimize(loss , global_step=global_step)
    return train_op


def evaluate(logits , labels):
    correct = tf.nn.in_top_k(logits , labels , 1)
    return tf.reduce_sum(tf.cast(correct , tf.int32))








