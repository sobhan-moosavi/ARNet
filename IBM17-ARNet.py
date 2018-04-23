'''
This is a Tensorflow implementation of "Autoencoder Regularized Network For Driving Style Representation Learning"
Author: Sobhan Moosavi
April 23, 2018
'''

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import random
import math
from scipy import stats
import time
import cPickle

from sklearn.preprocessing import OneHotEncoder
import functools

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, timesteps=128):
        self.data = data
        self.target = target
        self._timesteps = timesteps
        self._dropout = dropout
        self.cost
        self.prediction
        self.error
        self.optimize
        self.accuracy

    @lazy_property
    def prediction(self):
        #unroll the input
        x = tf.unstack(self.data, self._timesteps, 1)        
        
        ###### The GRU Component ######
        #The architecture starts with a GRU layer
        gru1 = tf.contrib.rnn.GRUCell(num_units=256)
        
        #Now, we add the second GRU layer
        gru2 = tf.contrib.rnn.GRUCell(num_units=256)
        
        #Now, we add the dropout layer
        dpt = tf.contrib.rnn.DropoutWrapper(gru2, output_keep_prob=1.0 - self._dropout)
        
        #Now, create the first part of the network (GRU1 + GRU2 + Dropout)
        network = tf.contrib.rnn.MultiRNNCell(cells=[gru1,dpt], state_is_tuple=True)
        
        #Get the \bar{x}: that is, the dropout output. 
        output, _ = rnn.static_rnn(network, x, dtype=tf.float32)
        x_bar = output[-1]
        
        
        ###### The Auto-encoder Component ######
        ##fc1
        fc1 = tf.layers.dense(inputs=x_bar, units=50, activation=tf.nn.relu)   #This is the embedding s for the input
        
        ##fc2
        fc2 = tf.layers.dense(inputs=fc1, units=256, activation=tf.tanh) #Now, fc2 provide reconstruction for \bar{x}, that is, the dropout output
        
        
        ###### The Softmax-Regression Component: fc3 ######      
        logits = tf.layers.dense(inputs=x_bar, units=int(self.target.get_shape()[1]), activation=None)       
        soft_reg = tf.nn.softmax(logits)
        
        return x_bar, fc1, fc2, soft_reg
    
    @lazy_property
    def cost(self):
        x_bar, s, x_recon, soft_reg = self.prediction        
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1e-5)  #Initialization of L1 regularizer
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=[s])   #L1 penalty term
        Jr = tf.reduce_mean(tf.square(tf.subtract(x_recon, x_bar))) + regularization_penalty      # Reconstruction Loss with L1 penalty term
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(soft_reg), reduction_indices=[1]))        
        loss = Jr + cross_entropy
        return loss
        
    @lazy_property
    def cost_jr(self):
        x_bar, s, x_recon, soft_reg = self.prediction
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1e-5)  #Initialization of L1 regularizer
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=[s])   #L1 penalty term
        Jr = tf.reduce_mean(tf.square(tf.subtract(x_recon, x_bar))) + regularization_penalty      # Reconstruction Loss with L1 penalty term 
        return Jr
        
    @lazy_property
    def cost_ce(self):
        x_bar, s, x_recon, soft_reg = self.prediction               
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(soft_reg), reduction_indices=[1]))        
        return cross_entropy

    @lazy_property
    def optimize(self):       
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-8)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        x_bar, s, x_recon, soft_reg = self.prediction
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(soft_reg, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    @lazy_property
    def accuracy(self):
        x_bar, s, x_recon, soft_reg = self.prediction
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(soft_reg, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
  
 
def returnTrainAndTestData():
    
    matrices = load_data('data/smallSample_{}_{}_2.npy'.format(args[0], args[1]))
    keys = cPickle.load(open('data/smallSample_{}_{}_keys_2.pkl'.format(args[0], args[1]), 'rb'))        
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    curTraj = ''
    r = 0
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr
        m = matrices[idx][1:129,]
        #print (d, t, idx, m.shape)    
        if t != curTraj:
            curTraj = t
            r = random.random()                    
        if m.shape[0] < 128:
          continue  
        if r < .8:
          train_data.append(m)
          train_labels.append(dr)
        else:
          test_data.append(m)
          test_labels.append(dr)        

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")

    train_data, train_labels = shuffle_in_union(train_data, train_labels)   #Does shuffling do any help ==> it does a great help!!
  
    return train_data, train_labels, test_data, test_labels, len(driverIds)

    
def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

    
def convertLabelsToOneHotVector(labels, ln):
    tmp_lb = np.reshape(labels, [-1,1])
    next_batch_start = 0
    _x = np.arange(ln)
    _x = np.reshape(_x, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(_x)
    labels =  enc.transform(tmp_lb).toarray()
    return labels
  

if __name__ == '__main__':
    
    args = [5, 5]
    st = time.time()
    train, train_labels, test, test_labels, num_classes = returnTrainAndTestData()
    print('All data is loaded in {:.1f} seconds'.format(time.time()-st))
    
    display_step = 100
    training_steps = 25001
    batch_size = 256
    
    train_dropout = 0.5
    test_dropout = 0.0
    
    timesteps = 128 # Number of rows in Matrix of a Segment
    
    train_labels = convertLabelsToOneHotVector(train_labels, num_classes)   
    test_labels = convertLabelsToOneHotVector(test_labels, num_classes)
    
    data = tf.placeholder(tf.float32, [None, 128, 35], name='data')    
    target = tf.placeholder(tf.float32, [None, num_classes], name='target')
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_start = time.time()
    start = time.time()
    next_batch_start = 0
    
    steps_to_epoch = len(train)/batch_size
    
    maxTestAccuracy = 0.0 #This will be used as a constraint to save the best model
    bestEpoch = 0
    
    saver = tf.train.Saver() #This is the saver of the model    
    
    for step in range(training_steps):
        idx_end = min(len(train),next_batch_start+batch_size)        
        sess.run(model.optimize, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: train_dropout})                
        
        epoch = int(step/steps_to_epoch)
        if epoch > bestEpoch or epoch == 0: #During a given epoch we can't have much improvement. Thus, we better not to do lots of computations.             
            acc = sess.run(model.accuracy, {data: test[0:min(3*batch_size, len(test)),:], target: test_labels[0:min(3*batch_size, len(test)),:], dropout: test_dropout})
            if epoch > 5 and acc > maxTestAccuracy:
                maxTestAccuracy = acc
                bestEpoch = epoch
                save_path = saver.save(sess, 'model2/bestARNet_{}_{}_B{}_L1e-5/'.format(args[0], args[1], batch_size))
                print('Model saved in path: {}, Accuracy: {:.2f}%, Epoch: {:d}'.format(save_path, 100*acc, epoch))
        
        if step % display_step == 0:            
            loss = sess.run(model.cost, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
            loss_jr = sess.run(model.cost_jr, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
            loss_ce = sess.run(model.cost_ce, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
                        
            test_loss  = sess.run(model.cost, {data: test, target: test_labels, dropout: test_dropout})            
            test_loss_jr  = sess.run(model.cost_jr, {data: test, target: test_labels, dropout: test_dropout})            
            test_loss_ce  = sess.run(model.cost_ce, {data: test, target: test_labels, dropout: test_dropout})  
            x_bar,s,x_recon,soft_reg = sess.run(model.prediction, {data: test, target: test_labels, dropout: test_dropout})
            
            acc = sess.run(model.accuracy, {data: test, target: test_labels, dropout: test_dropout})
            
            if step%300 == 0:   
                print('X_BAR')
                print(x_bar[0,0:10])
                print('X_BAR_RECON_0')
                print(x_recon[0,0:10])
                print('X_BAR_RECON_1')
                print(x_recon[17,0:10])
                print('S_Embed')
                print(s[0,0:10])    
                nonZeroEmbedVecs = 0
                for indx in range(len(s)):
                    if(np.count_nonzero(s[indx]) > 0):
                        nonZeroEmbedVecs += 1                    
                print('#nonZero embedding vectors: {:}, out of {:} vectors'.format(nonZeroEmbedVecs, len(s)))            
            print('Step {:2d}, Epoch {:2d}, Train Loss {:.3f}, Train_JR {:.3f}, Train_CE {:.3f}, Test-Loss {:.3f}, Test_JR {:.3f}, Test_CE {:.3f}, Test-Accuracy {:.1f}%, ({:.1f} sec)'.format(step + 1, epoch, loss, loss_jr, loss_ce, test_loss, test_loss_jr, test_loss_ce, 100 * acc, (time.time()-start)))            
            start = time.time()
        next_batch_start += next_batch_start+batch_size
        if next_batch_start >= len(train):
            train, train_labels = shuffle_in_union(train, train_labels)
            test, test_labels = shuffle_in_union(test, test_labels)
            next_batch_start = 0
    
    
    print("Optimization Finished!")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'model2/bestARNet_{}_{}_B{}_L1e-5/'.format(args[0], args[1], batch_size))
    accuracy = sess.run(model.accuracy, {data: test, target: test_labels, dropout: test_dropout})
    print('Final Test-Accuracy: {:.2f}%, Train-Time: {:.1f}sec'.format(accuracy*100, (time.time()-train_start)))
    print('Partial Best Test-Accuracy: {:.2f}%, Best Epoch: {}'.format(maxTestAccuracy*100, bestEpoch))