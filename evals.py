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
    # angle_resi = 180 / np.pi * tf.reduce_mean(angle(y_reco[:, 4:], y_true[:, 4:])) #degrees

    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_pos.numpy()), float(u_angle.numpy())

def metricsxpos(y_reco, y_true):
    #energy, pos_x,y,z, zenith, azimuth
    # Energy metric
    energy_quantiles = tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) #any difference in which one goes first?
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349 #magic number to make quantile to \sigma


    # Angle metric
    angle_resi = 180 / np.pi * tf.reduce_mean(angle(y_reco[:, 1:], y_true[:, 1:])) #degrees
    # angle_resi = 180 / np.pi * tf.reduce_mean(angle(y_reco[:, 4:], y_true[:, 4:])) #degrees

    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_angle.numpy())


def test(x):
    print('test')

sig=((tf.TensorSpec(shape=(None, 5), dtype=tf.float64, name=None),
  tf.SparseTensorSpec(tf.TensorShape([None, None]), tf.float64),
  tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)),
 tf.TensorSpec(shape=(None, 6), dtype=tf.float64, name=None))