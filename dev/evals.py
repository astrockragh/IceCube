'''Module to make metrics for evaluating reco'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

def angle(pred, true):
    pred, true=tf.cast(pred, tf.float32), tf.cast(true, tf.float32) 
    return tf.math.acos(
        tf.clip_by_value(
            tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1)),
            -1., 1.)
        )

def metrics(y_reco, y_true):
    #energy, pos_x,y,z, zenith, azimuth
    # Energy metric
    energy_quantiles = tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) #any difference in which one goes first?
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349 #magic number to make quantile to \sigma


    # Distanc metric
    dist_resi  = tf.math.reduce_euclidean_norm([y_reco[:, 1:4], y_true[:, 1:4]], axis = 1) # check axis, but 1 seems good
    u_pos           = tfp.stats.percentile(dist_resi, [68])


    # Angle metric
    angle_resi = 180 / np.pi * tf.reduce_mean(angle(y_reco[:, 4:], y_true[:, 4:])) #degrees

    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_pos.numpy()), float(u_angle.numpy())

def metricsxpos3(y_reco, y_true):
    #energy, pos_x,y,z, zenith, azimuth
    # Energy metric
    energy_quantiles = tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) #any difference in which one goes first?
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349 #magic number to make quantile to \sigma
    # tf.print(tf.shape(y_reco[:, 1:]), tf.shape(y_true[:, 1:]))
    alpha=angle(y_reco[:, 1:], y_true[:, 1:])
    # tf.print(alpha)
    angle_resi = 180 / np.pi * alpha #degrees
    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_angle.numpy()) 

def metricsxpos(y_reco, y_true):
    #energy, pos_x,y,z, zenith, azimuth
    # Energy metric
    energy_quantiles = tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) #any difference in which one goes first?
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349 #magic number to make quantile to \sigma

    
    # angle_resi = 180 / np.pi * tf.reduce_mean(angle(y_reco[:, 1:], y_true[:, 1:])) #degrees
    # Angle metric
    zenith_angle = 180 / np.pi *tf.subtract(y_reco[:, 1], y_true[:, 1]) #degrees
    azimuthal_angle = 180 / np.pi * tf.subtract(y_reco[:, 2], y_true[:, 2])
    tf.print(tfp.stats.percentile(zenith_angle, [68]))
    tf.print(tfp.stats.percentile(azimuthal_angle, [68]))

    angle_resi = 180 / np.pi * tf.math.acos(tf.math.sin(y_reco[:, 1])*tf.math.sin(y_true[:, 1])*tf.math.cos(tf.subtract(y_reco[:, 2],y_true[:, 2]))+tf.math.cos(y_reco[:, 1])*tf.math.cos(y_true[:, 1]))#degrees
    # tf.print(tf.shape(angle_resi))
    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_angle.numpy()) 

def metricsxpos2(y_reco, y_true):

    energy_quantiles = tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) #any difference in which one goes first?
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349


    y_reco = tf.cast(y_reco,"float64")
    y_true = tf.cast(y_true,"float64")
    zeni = tf.atan2(y_reco[:,2],y_reco[:,1])
    # azi = tf.minimum( tf.abs(target[:,1] - aziguess) , tf.abs(tf.abs(target[:,1] - aziguess) - 2*PI))

    azi = tf.atan2(y_reco[:,4],y_reco[:,3])
    # zeni = tf.minimum( tf.abs(target[:,2] - zeniguess) , tf.abs(tf.abs(target[:,2] - zeniguess) - 2*PI))
    angle_resi = 180 / np.pi * tf.math.acos(tf.math.sin(zeni)*tf.math.sin(y_true[:, 1])*tf.math.cos(tf.subtract(azi,y_true[:, 2]))+tf.math.cos(zeni)*tf.math.cos(y_true[:, 1]))
    u_angle         = tfp.stats.percentile(angle_resi, [68])
    return float(w_energy.numpy()), float(u_angle.numpy()), 



def metricswsig(y_reco, y_true, sigs):
    return

sig=((tf.TensorSpec(shape=(None, 5), dtype=tf.float64, name=None),
  tf.SparseTensorSpec(tf.TensorShape([None, None]), tf.float64),
  tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)),
 tf.TensorSpec(shape=(None, 6), dtype=tf.float64, name=None))