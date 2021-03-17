import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract, square

def cos_angle(y_reco, y_true):
    zep, zet, azp, azt = y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2]
    # cosalpha=abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)
    cosalpha=abs(sin(zep))*abs(sin(zet))*cos(azp-azt)+cos(zep)*cos(zet) #check for double absolutes
    return cosalpha

def energy_angle(y_reco, y_true):
    energy_metric = tfp.stats.percentile(tf.math.abs(tf.subtract(y_true[:, 0], y_reco[:, 0])), [50-34, 50, 50+34]) 
    #for comparison
    classic_stat=tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) 
    w_energy=tf.subtract(classic_stat[1],classic_stat[0])/1.349
    alpha= acos(cos_angle(y_reco, y_true))
   
    angle_resi = 180 / np.pi * alpha #degrees
    angle_metric  = tfp.stats.percentile(angle_resi, [50-34,50,50+34])
    w_angle         = tfp.stats.percentile(angle_resi, [68])

    return energy_metric.numpy(), angle_metric.numpy(), [float(w_energy), float(w_angle)]
