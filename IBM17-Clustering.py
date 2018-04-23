'''
This code is to use the trained ARNet model to build representation for trajectories and perform the clustering task. 
Author: Sobhan Moosavi
April 23, 2018
'''

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.python.tools import inspect_checkpoint as chkp

import numpy as np
import random
import math
from scipy import stats
import time
import cPickle
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score

import functools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nd', type=int, nargs='+', default=[1]) #nd: number of drivers
args = parser.parse_args()
nd = args.nd

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

    def __init__(self, data, target, timesteps=128):
        self.data = data
        self.target = target
        self._timesteps = timesteps
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
        dpt = tf.contrib.rnn.DropoutWrapper(gru2, output_keep_prob=1.0)
        
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
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=5e-7)  #Initialization of L1 regularizer
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=[s])   #L1 penalty term
        Jr = tf.reduce_mean(tf.square(tf.subtract(x_recon, x_bar))) + regularization_penalty      # Reconstruction Loss with L1 penalty term
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(soft_reg), reduction_indices=[1]))        
        loss = Jr + cross_entropy
        return loss
        
    @lazy_property
    def cost_jr(self):
        x_bar, s, x_recon, soft_reg = self.prediction
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=5e-7)  #Initialization of L1 regularizer
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
        #How to optimize specific (not all) variables in the graph: https://stackoverflow.com/questions/34477889/holding-variables-constant-during-optimizer/34478044#34478044
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


class point:
    lat = 0
    lng = 0
    time = 0
    def __init__(self, time, lat, lng):
        self.lat = lat
        self.lng = lng
        self.time = time

        
class basicFeature:
    speedNorm = 0
    diffSpeedNorm = 0
    accelNorm = 0
    diffAccelNorm = 0
    angularSpeed = 0
    def __init__(self, speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed):
        self.speedNorm = speedNorm
        self.diffSpeedNorm= diffSpeedNorm
        self.accelNorm= accelNorm
        self.diffAccelNorm = diffAccelNorm
        self.angularSpeed = angularSpeed


def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
        
        
def returnClusteringData():

    matrices = load_data('data/tripTestSets/IBMFeatureMatrices/testSample_{}_50_k=25-{}.npy'.format(args[0], args[1]))
    keys = cPickle.load(open('data/tripTestSets/IBMFeatureMatrices/testSample_{}_50_k=25-{}_keys.pkl'.format(args[0], args[1]), 'rb'))        
    
    #Build Data and Label sets
    data = {}
    trueLabels = {}
    targets = {}
    
    curTraj = ''    
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
        _data = []
        _target = []
        if t in data:
            _data = data[t]
            _target = targets[t]
        _data.append(m)
        _target.append(dr)
        data[t] = _data
        targets[t] = _target        
        trueLabels[t] = d

    return data, targets, trueLabels, len(driverIds)

    
def convertLabelsToOneHotVector(labels, ln):
    tmp_lb = np.reshape(labels, [-1,1])
    next_batch_start = 0
    _x = np.arange(ln)
    _x = np.reshape(_x, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(_x)
    labels =  enc.transform(tmp_lb).toarray()
    return labels

    
def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

if __name__ == '__main__':
    
    source = '50_200'
    reference_num_classes = 50
    
    numOfCluster = []
    amis = []
    
    #Initialize Tensorflow Graph
    tf.reset_default_graph()    
    #Build and Restore the trained model 
    data = tf.placeholder(tf.float32, [None, 128, 35])    
    target = tf.placeholder(tf.float32, [None, reference_num_classes])    
    model = SequenceClassification(data, target)
    saver = tf.train.Saver() #This is the saver of the model
    
    with tf.Session() as sess:
    
        saver.restore(sess, 'model/bestARNet_' + source + '_B256_L1e-5/') 
        runs = 0
        
        for k in range(1,26): #We have 25 test sets
            print('******* ' + str(k) + ' *********')
            args = [nd[0], k]
            c_data, targets, true_labels, num_classes = returnClusteringData()   
            start = time.time()
            
            #Create Embedding Vectors
            vectors = {}        
            for t in c_data:
                _data   = c_data[t]
                _target = targets[t]
                
                _data    = np.asarray(_data, dtype="float32")
                _target  = convertLabelsToOneHotVector(np.asarray(_target, dtype="int32"), reference_num_classes)     
                _,s,_,logit  = sess.run(model.prediction, {data: _data, target: _target})
                
                vec = []
                maxSum = -10000
                for i in range(s.shape[1]):
                    _sum = np.sum(s[:,i])
                    vec.append(_sum)
                    maxSum = max(maxSum, _sum)
                vectors[t] = np.divide(vec, maxSum*1.0)
                
            print('Embedding Vectors are created in {:.1f} seconds'.format(time.time() - start))
            X = []
            Y = []
            for t in vectors:
                X.append(vectors[t])
                Y.append(true_labels[t])            

            X = np.asarray(X)
            Y = np.asarray(Y)         
            
            try:
                predicted_labels = AffinityPropagation(damping=0.5, preference=-65).fit_predict(X)
                ami = adjusted_mutual_info_score(Y, predicted_labels)
                print('#Drivers: {:}, #Clusters: {:}, AMI: {:.2f}'.format(num_classes, len(np.unique(predicted_labels)), ami))            
                num_error = np.abs(num_classes - len(np.unique(predicted_labels)))
                numOfCluster.append(num_error)
                amis.append(ami)
                runs += 1
            except:
                continue
    
    print('Runs: {:}, Mean_Num_Driver_Error: {:.2f}, Std_Num_Driver_Error: {:.2f}, Mean_AMI: {:.2f}, Std_AMI: {:.2f}'.format(runs, np.mean(numOfCluster), np.std(numOfCluster), np.mean(amis), np.std(amis)))
    print(np.mean(numOfCluster), np.std(numOfCluster), np.mean(amis), np.std(amis), runs)
    
