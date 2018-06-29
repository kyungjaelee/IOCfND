import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import decomposition

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pickle.load(open("./training_data.pkl","rb"))
print(data['img_list'].shape[0])
n_demos = 1
img_list = data['img_list'][:n_demos]
pc2_list = data['pc2_list'][:n_demos]
imu_list = data['imu_list'][:n_demos]
cmd_vel_list = data['cmd_vel_list'][:n_demos]
throttle_list = data['throttle_list'][:n_demos]
steering_list = data['steering_list'][:n_demos]
brake_list = data['brake_list'][:n_demos]

