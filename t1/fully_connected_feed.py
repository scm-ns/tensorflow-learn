from __future__ import absolute_import
from __future__ import divison
from __future__ import print_function

import os.path
import time


from six.moves import xrange
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01 , 'Initial learning rate.')
flags.DEFINE_integer('max_steps' , 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1' , 128 , 'Number of units in hiddren layer 1.')
flags.DEFINE_integer('hidden2' , 32 , 'Number of units in hidden layer 2.')
flags.DEFINE_bool('batch_size' , 100 , 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir' , 'data' , 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data' , False , 'If true, uses fake data for unit testing.')



def place_input(batch_size):
    img_plc = tf.placeholder(tf.float32 , shape = (batch_size , mnist.IMAGE_PIXELS))
    lbl_plc = tf.placeholder(tf.int32 , shape=(batch_size)) 
    return img_plc , lbl_plc 

def fill_feed(data_set , img_plc , lbl_plc):
    img_feed , lbl_feed = data_set.next_batch(FLAGS.batch_size , FLAGS.fake_data)

    feed_dict = {
            img_plc : img_feed , 
            lbl_plc : lbl_plc,
    }
    return feed_dict


# Evaluate the model on the test set 
def do_eval(sess , eval_corr , img_plc , lbl_plc , data):
    true_count = 0 

    steps_per_epoch = data.num_examples 
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data , img_plc , lbl_plc)
        true_count += sess.run(eval_corr , feed_dict = feed_dict)

    precision = true_count / num_examples 
    print(' examples: %d , correct %d , precision @ %0.04f ' %(num_examples , true_count , precision)) 



def train():
    data_sets = input_data.read_data_sets(FLAGS.tain_dir , FLAGS.fake_data)

    with tf.Graph().as_default():
        img_plc , lbl_plc = place_input(FLAGS.batch_size)










