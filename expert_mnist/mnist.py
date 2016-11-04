import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def w_var(shp):
    init = tf.truncated_normal(shp , stddev = 0.1)
    return tf.Variable(init)


def b_var(shp):
    init = tf.constant(0.1 , shape = shp)
    return tf.Variable(init)

def conv2d(x , W):
    return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize = [1 , 2 , 2 , 1], strides = [1 , 2 , 2 , 1] , padding = 'SAME')

def main():
    mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)
    sess = tf.InteractiveSession()

    input_size = 784
    output_size = 10

    x = tf.placeholder(tf.float32 , shape = [None , input_size])
    y_true = tf.placeholder(tf.float32 , shape = [None , output_size])

    W = tf.Variable(tf.zeros([input_size , output_size]));
    b = tf.Variable(tf.zeros([output_size]))

    sess.run(tf.initialize_all_variables())

    y_pred = tf.matmul(x,W) + b 

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred , y_true))

    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    #    for i in range(1000):
    #        batch = mnist.train.next_batch(100)
    #        train_step.run(feed_dict = { x : batch[0] , y_true : batch[1] })


    #    correct_pred = tf.equal(tf.argmax(y_true,1) , tf.argmax(y_pred,1))
    #    accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))

    #    print(accuracy.eval(feed_dict = { x: mnist.test.images , y_true : mnist.test.labels}))

    
    # Learn how convolutions are represented in tensor form and use it properly ..

    W_conv1 = w_var([5 , 5 , 1 , 32])
    b_conv1 = b_var([32])

    x_img = tf.reshape(x , [-1 , 28 , 28 , 1])
    
    h_conv1 = tf.nn.relu(conv2d(x_img , W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = w_var([5 , 5 , 32 , 64])
    b_conv2 = b_var([64])


    h_conv2 = tf.nn.relu(conv2d(h_pool1 , W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = w_var([7 * 7 * 64 , 1024])
    b_fc1 = b_var([1024])

    
    h_pool2_flat = tf.reshape(h_pool2 , [-1 , 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

    W_fc2 = w_var([1024 , 10])
    b_fc2 = b_var([10])
    y_conv = tf.matmul(h_fc1_drop , W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv , y_true))
    train_step_conv = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_pred_conv = tf.equal(tf.argmax(y_conv , 1) , tf.argmax(y_true , 1))
    acc_conv = tf.reduce_mean(tf.cast(correct_pred_conv , tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_acc = acc_conv.eval(feed_dict = { x:batch[0] , y_true:batch[1] , keep_prob: 1.0 })
            print("step %d , training accurary %g"%(i , train_acc))
        train_step_conv.run(feed_dict = { x: batch[0] , y_true : batch[1] , keep_prob: 0.5})

    print("test accurarcu %g"%acc_conv.eval(feed_dict={ x:mnist.test.images , y_true : mnist.test.labels , keep_prob :1.0}))


if __name__ == '__main__':
    main()
