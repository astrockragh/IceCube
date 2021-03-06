''' Loss functions file 

General setup should be a fn(y_reco, y_true, re), where re keyword gives whether or not individual losses should be returned along with total loss
naming convention: EnergyLossMethod_AngleLossMethod_Unitvec/Angle '''

import tensorflow as tf
import numpy as np
from tensorflow.math import sin, cos, acos, abs, reduce_mean, reduce_sum, subtract, square

eps=1e-5

####  Define useful function #### 
#best watch that these are right

def cos_angle(y_reco, y_true):
    zep, zet, azp, azt = y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2]
    # cosalpha=abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)
    cosalpha=abs(sin(zep))*abs(sin(zet))*cos(azp-azt)+cos(zep)*cos(zet) #check for double absolutes
    cosalpha-=tf.math.sign(cosalpha) * eps
    return cosalpha

def cos_unit(y_reco, y_true):
    pred, true=y_reco[1:4], y_true[1:4] 
    cosalpha=tf.math.divide_no_nan(reduce_sum(pred * true, axis = 1),tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))
    cosalpha-=tf.math.sign(cosalpha) * eps
    return cosalpha

# def cos_azi(y_reco,y_true):
#     return

#################################################################
# Absolute error for E, linear alpha for angle                 #
################################################################

def abs_linear_unit(y_reco, y_true, re=False):
    ''
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract
    
    #energy loss

    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0])))
    
    #angle loss
    
    cos_alpha = cos_unit(y_reco,y_true)
    loss_angle = reduce_mean(tf.math.acos(cos_alpha))
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]

#################################################################
# Absolute error for E, negative cos (1-cos(\alpha)) for angle #
################################################################

def abs_negcos_unit(y_reco, y_true, re=False):
    # Energy loss
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]) ) )
    # Angle loss
    loss_angle = tf.reduce_mean(1-cos_unit(y_reco[:, 1:4], y_true[:, 1:4])) 
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]


def abs_negcos_angle(y_reco, y_true, re=False):
    # Energy loss
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #this works well but could maybe be improved
    # Angle loss
    loss_angle = reduce_mean(1-cos_angle(y_reco, y_true))
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]

#################################################################
# Absolute error for E, von Mises for angle                    #
################################################################

def abs_vonMises_angle(y_reco, y_true, re=False):
    #energy
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]))) #mae
    tf.debugging.assert_all_finite(loss_energy, 'Energy problem', name=None)
    #angle
    kappa=tf.math.abs(y_reco[:,3])+eps
    cos_alpha=cos_angle(y_reco, y_true)
    # tf.debugging.assert_less_equal(tf.math.abs(cos_alpha), 1, message='cos_alpha problem', summarize=None, name=None)
    tf.debugging.assert_all_finite(tf.math.abs(cos_alpha), message='cos_alpha problem infinite/nan', name=None)
    nlogC = -tf.math.log(kappa) + kappa +tf.math.log(1-tf.math.exp(-2*kappa))
    tf.debugging.assert_all_finite(nlogC, 'log kappa problem', name=None)

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    tf.debugging.assert_all_finite(loss_angle, 'Angle problem', name=None)

    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]

def abs_vonMises_angle2(y_reco, y_true, re=False):
    #energy
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]))) #mae
    tf.debugging.assert_all_finite(loss_energy, 'Energy problem', name=None)
    #angle
    kappa=tf.math.abs(y_reco[:,4])+eps
    cos_alpha=cos_angle(y_reco, y_true)
    # tf.debugging.assert_less_equal(tf.math.abs(cos_alpha), 1, message='cos_alpha problem', summarize=None, name=None)
    tf.debugging.assert_all_finite(tf.math.abs(cos_alpha), message='cos_alpha problem infinite/nan', name=None)
    nlogC = -tf.math.log(kappa) + kappa +tf.math.log(1-tf.math.exp(-2*kappa))
    tf.debugging.assert_all_finite(nlogC, 'log kappa problem', name=None)

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    tf.debugging.assert_all_finite(loss_angle, 'Angle problem', name=None)

    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]

def abs_vonMises_unit(y_reco, y_true, re=False):
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]) ) )
    kappa=tf.math.abs(y_reco[:,4])

    cos_alpha=cos_unit(y_reco, y_true)
    nlogC = -tf.math.log(kappa) + tf.math.log(tf.math.exp(kappa)-tf.math.exp(-kappa) )

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]


#################################################################
# Absolute error for E, von Mises for zenith/azimuth            #
################################################################

def abs_vonMises2D_angle(y_reco, y_true, re=False):
    #energy
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)
    if not re:
        return loss_azi+loss_zenith+loss_energy
    if re:
        return float(loss_azi+loss_zenith+loss_energy), [float(loss_energy), float(loss_zenith), float(loss_azi)]

import tensorflow_probability as tfp
tfd=tfp.distributions

def kde(X):
    # print(len(X))
    sig=tf.convert_to_tensor([0.07], tf.float32)
    probs=tf.cast(tf.ones_like(X), dtype=tf.float32)
    kde = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
    probs=probs),
    components_distribution=tfd.Normal(loc=X, scale=sig))
    return kde

xkde=tf.linspace(0+0.01, 3.1415926-0.01, 50)
xkde=tf.cast(xkde, tf.float32)

def abs_vM2D_KDE(y_reco, y_true, kdet, re=False):
    #energy
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)
    kder=kde(tf.cast(y_reco[:,1], tf.float32))
    kdeloss=tf.reduce_mean(tf.math.abs(kdet.log_prob(xkde)-kder.log_prob(xkde)))/2
    
    if not re:
        return loss_azi+loss_zenith+loss_energy+tf.cast(kdeloss, tf.float32)
    if re:
        return float(loss_azi+loss_zenith+loss_energy+kdeloss), [float(loss_energy), float(loss_zenith), float(loss_azi), float(kdeloss)]

def abs_vM2D_KDE_weak(y_reco, y_true, kdet, re=False):
    #energy
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)
    kder=kde(tf.cast(y_reco[:,1], tf.float32))
    kdeloss=tf.reduce_mean(tf.math.abs(kdet.log_prob(xkde)-kder.log_prob(xkde)))/10
    
    if not re:
        return loss_azi+loss_zenith+loss_energy+tf.cast(kdeloss, tf.float32)
    if re:
        return float(loss_azi+loss_zenith+loss_energy+kdeloss), [float(loss_energy), float(loss_zenith), float(loss_azi), float(kdeloss)]

def sqr_vonMises2D_angle(y_reco, y_true, re=False):
    #energy
    loss_energy = reduce_mean(tf.math.squared_difference(y_reco[:,0], y_true[:,0])) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)
    if not re:
        return loss_azi+loss_zenith+loss_energy
    if re:
        return float(loss_azi+loss_zenith+loss_energy), [float(loss_energy), float(loss_zenith), float(loss_azi)]


def sqr_vonMises23D_angle(y_reco, y_true, re=False):

    #energy
    loss_energy = reduce_mean(tf.math.squared_difference(y_reco[:,0], y_true[:,0])) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)

    kappa=tf.math.abs(y_reco[:,5])+eps
    cos_alpha=cos_angle(y_reco, y_true)
    # tf.debugging.assert_less_equal(tf.math.abs(cos_alpha), 1, message='cos_alpha problem', summarize=None, name=None)
    tf.debugging.assert_all_finite(tf.math.abs(cos_alpha), message='cos_alpha problem infinite/nan', name=None)
    nlogC = -tf.math.log(kappa) + kappa +tf.math.log(1-tf.math.exp(-2*kappa))
    tf.debugging.assert_all_finite(nlogC, 'log kappa problem', name=None)

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )

    if not re:
        return loss_azi+loss_zenith+loss_energy+loss_angle
    if re:
        return float(loss_azi+loss_zenith+loss_energy+loss_angle), [float(loss_energy), float(loss_zenith), float(loss_azi), float(loss_angle)]

def abs_vonMises23D_angle(y_reco, y_true, re=False):
    #energy
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]))) #mae again

    polar_k     = abs(y_reco[:, 3])+eps
    zenth_k     = abs(y_reco[:, 4])+eps

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    cos_zenth   = cos(subtract(y_true[:,1], y_reco[:,1]))


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenith   = zenth_k * cos_zenth - lnI0_zenth

    loss_azi=reduce_mean( - llh_azi)
    loss_zenith=reduce_mean( - llh_zenith)

    kappa=tf.math.abs(y_reco[:,5])+eps
    cos_alpha=cos_angle(y_reco, y_true)
    # tf.debugging.assert_less_equal(tf.math.abs(cos_alpha), 1, message='cos_alpha problem', summarize=None, name=None)
    tf.debugging.assert_all_finite(tf.math.abs(cos_alpha), message='cos_alpha problem infinite/nan', name=None)
    nlogC = -tf.math.log(kappa) + kappa +tf.math.log(1-tf.math.exp(-2*kappa))
    tf.debugging.assert_all_finite(nlogC, 'log kappa problem', name=None)

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )

    if not re:
        return loss_azi+loss_zenith+loss_energy+loss_angle
    if re:
        return float(loss_azi+loss_zenith+loss_energy+loss_angle), [float(loss_energy), float(loss_zenith), float(loss_azi), float(loss_angle)]

