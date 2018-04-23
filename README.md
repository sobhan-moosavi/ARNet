# ARNet
This is a tensorflow implementation of [Autoencoder Regularized Network For Driving Style Representation Learning](https://arxiv.org/pdf/1701.01272.pdf)

## Requirements

1. Python 2.7
2. Tensorflow 1.3.0
3. Cuda 8.0.61

## Data

A sample data file for 5 drivers, with 5 trajectories for each, is shared in 'data' folder. The data file has following columns: Driver, ID, Time, Lat, and Lon. 

## Experiments

1. Statistical Feature Matrix: In order to create the statistical feature matrix as described in the paper, you need to run 'IBM16_FeatureMatrix.py' which creates two files in data folder. 
2. Train ARNet Model: In order to train an ARNet model, you need to run script 'IBM17-ARNet.py'. The input for this script is the statistical feature matrix and output is a trained model. 
3. Clustering: In order to build representation for test trajectories and perform the clustering, you can use 'IBM17-Clustering.py'. The input for this script is the trained ARNet model and a set of test trajectories as described in the paper. The output will be the clustering results and also the created emebddings for each trajectory. 

## Results

Our best results for driver clustering task based on a set of real-world, private, and non-anonymized (based on gps coordinates) datasets are as follows:

| #Drivers | #Trajectories/Driver | #Driver_Avg Of Error | #Driver_Std Of Error | AMI Avg | AMI Std |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | 200 | 0.33 | 0.55 | 0.71 | 0.45 |

Note that here we used LSTM cells, instead of using RNN cells with identity matrix for recurrent weight initialization, as such thing is not available in Tensorflow currently. However, as mentioned by <a href="https://arxiv.org/abs/1504.00941">Le et al.</a>, the initialized recurrent weight solution provides comaprable results to LSTM cells. 

## References 

1. [Characterizing Driving Styles with Deep Learning, 2016](https://arxiv.org/pdf/1607.03611.pdf)
