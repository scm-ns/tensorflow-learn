import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
from sklearn.metrics import confusion_matrix


def plt_img(imgs , cls_true , cls_pred = None):
    assert len(imgs) == len(cls_true) == 9 

    fig , axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace = 0.3 , wspace = 0.3)
    
    img_size = 28 
    img_shape = (img_size, img_size)

    for i , ax in enumerate(axes.flat):
        ax.imshow(imgs[i].reshape(img_shape) , cmap = 'binary')
    
        if cls_pred is None : 
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i] , cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])


    plt.show()

def optimize(num_itr , batch_size , x , y_true_1h , data , opt , sess):
    for i in range(num_itr):
        x_batch , y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x : x_batch , 
                           y_true_1h : y_true_batch}

        sess.run(opt , feed_dict = feed_dict_train) 


def print_acc(acc , feed_dict , sess):
    acc_val = sess.run(acc , feed_dict = feed_dict)
    
    print("Accuray on test set : {0:.1%}".format(acc_val))



def plot_weights(weight , img_shp , sess):
    w = sess.run(weight)

    w_min = np.min(w) 
    w_max = np.max(w)


    fig , axes = plt.subplots(3,4)
    fig.subplots_adjust(0.2, 0.3)

    for i , ax in enumerate(axes.flat):
        if i < 10 : 
            img = w[: , i].reshape(img_shp) 
            ax.set_xlabel("Weights: {0}".format(i))

            ax.imshow(img , vmin = w_min , vmax = w_max , cmap = 'seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def main():
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("./data/MNIST" , one_hot = True)

    print("Size of")
    print(" - Training-set:\t\t{}".format(len(data.train.labels)))
    print(" - Test-set:\t\t{}".format(len(data.test.labels)))
    print(" - Validation-set:\t\t{}".format(len(data.validation.labels)))
    print(data.test.labels[0:5 , :])
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    print(data.test.cls[0:5 ])
    img_size = 28 
    img_size_flat = img_size * img_size 
    img_shape  = (img_size , img_size)
    num_cls = 10

    images = data.test.images[0:9]
    cls_ture = data.test.cls[0:9]


    ## Set up training and the models used for training
    x = tf.placeholder(tf.float32 , [None , img_size_flat])
    y_true_1h = tf.placeholder(tf.float32 , [None ,num_cls])    

    y_true_cls = tf.placeholder(tf.int64 , [None])
    


    W = tf.Variable(tf.zeros([img_size_flat , num_cls])) # Weight vector with the relationship between pixel and classes
    b = tf.Variable(tf.zeros([num_cls])) # Biases for each of the classes ? 

    # Output of the multip betweent the x and weights is a matrix of size num image * num classes 
    # The size of the image is consumed

    logits = tf.matmul(x , W) + b # Now holds the proability of the training case belonging to each of the classes
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred , 1)

    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits , y_true_1h)
    
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.7).minimize(cost)

    correct_pred = tf.equal(y_pred_cls , y_true_cls)
    acc = tf.reduce_mean(tf.cast(correct_pred , tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    feed_dict_test = { x: data.test.images , 
                       y_true_1h: data.test.labels , 
                       y_true_cls : data.test.cls}  


    print_acc(acc , feed_dict_test , sess)
    optimize(1 , 500 , x , y_true_1h,  data , optimizer , sess)
    print_acc(acc , feed_dict_test , sess)
    plot_weights(W , img_shape , sess)
    
    
    print_acc(acc , feed_dict_test , sess)
    optimize(5 , 500 , x , y_true_1h,  data , optimizer , sess)
    print_acc(acc , feed_dict_test , sess)
    plot_weights(W , img_shape , sess)


    print_acc(acc , feed_dict_test , sess)
    optimize(30 , 500 , x , y_true_1h,  data , optimizer , sess)
    print_acc(acc , feed_dict_test , sess)
    plot_weights(W , img_shape , sess)


if __name__ == '__main__':
    main()

