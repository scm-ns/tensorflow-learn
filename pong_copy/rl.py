import tensorflow as tf
import cv2
import pong
import numpy
import random
from collections import deque


#defining hyperparameters
ACTIONS = 3

#learning rate
LEARNING = 0.99


INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05

EXPLORE = 500000
OBSERVE = 50000

BATCH = 100


def createGraph():
    
    wc1 = tf.Variable(tf.zeros([8,8,4,32]))
    bc1 = tf.Variable(tf.zeros([32]))

    wc2 = tf.Variable(tf.zeros[4,4,32,64])
    bc2 = tf.Variable(tf.zeros[64])

    wc3 = tf.Variable(tf.zeros[3,3,64,64])
    bc3 = tf.Variable(tf.zeros[64])

    wf4 = tf.Variable(tf.zeros[3136 , 784])
    bf4 = tf.Variable(tf.zeros[784])

    wf5 = tf.Variable(tf.zeros[784 , ACTIONS])
    bf5 = tf.Variable(tf.zeros[ACTIONS])


    s = tf.placeholder("float" , [None , 84 , 84 , 84])
    cnv1 = tf.nn.relu(tf.nn.conv2d(s , wc1 . strides = [ 1 , 4 , 4 , 1] , padding = "VALID") + bc1)
    cnv2 = tf.nn.relu(tf.nn.conv2d(cnv1 , wc2 . strides = [ 1 , 2 , 2 , 1] , padding = "VALID") + bc2)
    cnv3 = tf.nn.relu(tf.nn.conv2d(cnv2 , wc3 . strides = [ 1 , 1 , 1 , 1] , padding = "VALID") + bc3)

    cnv3_flat = tf.reshape(conv3 , [-1 , 3136])
    fc4 = tf.nn.relu(tf,matmul(conv3_flat , wf4 ) + bf4)
    fc5 = tf.matmul(fc5 , wf5) + bf5
    return s , fc5


def trainGraph(inp , out , see):

    argmax = tf.placeholder("float" , [None , ACTIONS])
    gt = tf.placeholder("float" , [None])

    action = tf.reduce_sum(tf.mul(out , argmax) , reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(action - gt))

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = pong.PongGame()
    D = deque()

    frame = game.getPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame , (84 , 84)).cv2.COLOR_BGR2GRAY)

    ret , frame = cv2.threshold(frame , 1 , 255 , cv2.THRESH_BINARY)
    inp_t = np.stack((frame , frame , frame , frame) ,axis = 2)

    saver = tf.train.Saver()


    sess.run(tf.initialize_all_variables)
    

    t = 0 
    epsilon = INITIAL_EPSILON 


    while(1):
        out_t = out.eval(feed_dict={inp : [inp_t]})[0]
        argmax_t = np.zeros([ACTIONS])

        if(random.random() <= epsilon): 
            maxIdx = random.randrange(ACTIONS) 
        else
            maxIdx = np.argmax(out_t)
        argmax_t[maxIdx] = 1

        if epsilon > FINAL_EPSILON : 
             epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        reward_t , frame = game.getNextFrame(argmax_t)
        frame = cv2.cvtColor(cv2.resize(frame , (84,84)) , cv2.THRESH_BINARY)
        ret , frame  = cv2.threshold(frame , 1 , 255 , cv2.THRESH_BINARY)
        frame = np.reshape(frame , (84 , 84 , 1))

        inp_t = np.append(frame , inp_t[: , : , 0:3] , axis = 2)

        D.append((inp_t , argmax_t , reward_t , inp_t1))

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE :
            miniBatch = random.sample(D, BATCH)
   
            inpBatch = [d[0] for d in miniBatch]
            argBatch = [d[1] for d in miniBatch]
            rewardBatch = [d[2] for d in miniBatch]
            inpT1Batch = [d[3] for d in miniBatch]

            gtBatch = []
            outBatch = out.eval(feed_dict = { inp : inpT1Batch })


            for i in range(0 , len(miniBatch)):
                gt_batch.append(rewardBatch[i] + LEARNING * np.max(out_batch[i]))



            train_step.run(feed_dict = {
                           gt : gtBatch , 
                           argmax : argBatch , 
                           inp : inpBatch
                           })


       inp_t = inp_t1
       t = t + 1

       if t % 1000 == 0:
            saver.save(sess , "./" + "pong" + "-dqn" , global_step = t)


        print "TIMESTEP" , t , "/ EPSILON " , epsilon , "/ ACTION" , maxIdx , "/ REWARD", reward_t , "/ Q_MAX %e" % np.max(out_t)



def main():

    sess = tf.InteractiveSession()
    inp , out = createGraph()
    trainGraph(inp , out , sess)


if __name__ == "__main__":
    main()







