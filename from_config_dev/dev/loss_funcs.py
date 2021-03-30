''' Loss functions file 

General setup should be a fn(y_reco, y_true, re), where re keyword gives whether or not individual losses should be returned along with total loss
naming convention: EnergyLossMethod_AngleLossMethod_Unitvec/Angle '''

import tensorflow as tf
import numpy as np
from tensorflow.math import sin, cos, acos, abs, reduce_mean, reduce_sum, subtract, square, log, exp

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

#################################################################
# Absolute error for E, wrapped cauchy for solid angle          #
################################################################

def abs_cauchy3D(y_reco, y_true, re=False):
    #energy
    
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0])))

    gamma     = abs(y_reco[:, 3])+eps

    cos_alpha=cos_alpha(y_reco,y_true)

    norm    = log(1-2*exp(-2*gamma))

    #negative log-likelihood
    nllh     = log(1+2*exp(-2*gamma)-2*exp(-gamma)*cos_alpha)-norm

    loss_angle=reduce_mean(nllh)

    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]


#################################################################
# Absolute error for E, wrapped cauchy for azi/zenith          #
################################################################

def abs_cauchy2D(y_reco, y_true, re=False):
    #energy
    
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0])))

    gamma_zeni     = abs(y_reco[:, 3])+eps
    gamma_azi     = abs(y_reco[:, 4])+eps

    cos_zeni   = cos(subtract(y_true[:,1], y_reco[:,1]))

    cos_azi     = cos(subtract(y_true[:,2], y_reco[:,2]))

    norm_zeni    = log(1-exp(-2*gamma_zeni))
    norm_azi   =  log(1-exp(-2*gamma_azi))


    nllh_zeni     = log(1+exp(-2*gamma_zeni)-2*exp(-gamma_zeni)*cos_zeni)-norm_zeni
    nllh_azi   = log(1+exp(-2*gamma_azi)-2*exp(-gamma_azi)*cos_azi)-norm_azi

    loss_zeni=reduce_mean(nllh_zeni)
    loss_azi=reduce_mean(nllh_azi)

    if not re:
        return loss_azi+loss_zeni+loss_energy
    if re:
        return float(loss_azi+loss_zeni+loss_energy), [float(loss_energy), float(loss_zeni), float(loss_azi)]


def abs_vonMises2D_angle(y_reco, y_true, re=False):
    #energy
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0])))

    polar_k     = abs(y_reco[:, 3])
    zenth_k     = abs(y_reco[:, 4])

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


#################################################################
# Absolute error for E, bivariate wrapped for zenith azimuth    #
################################################################

# def abs_bivariate_angle(y_reco, y_true, re=False):
#     #energy
#     loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0])))
#     tf.debugging.assert_all_finite(loss_energy, 'Energy problem', name=None)
#     #angle
#     kappa1=tf.math.abs(y_reco[:,3])+eps
#     kappa2=tf.math.abs(y_reco[:,4])+eps
#     kappa3=tf.math.abs(y_reco[:,5])+eps
#     cos_alpha=cos_angle(y_reco, y_true)

#     # # tf.debugging.assert_less_equal(tf.math.abs(cos_alpha), 1, message='cos_alpha problem', summarize=None, name=None)
#     # tf.debugging.assert_all_finite(tf.math.abs(cos_alpha), message='cos_alpha problem infinite/nan', name=None)
    
#     nlogC = - tf.math.log(kappa) + kappa +tf.math.log(1-tf.math.exp(-2*kappa))

#     # tf.debugging.assert_all_finite(nlogC, 'log kappa problem', name=None)

#     loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    
#     # tf.debugging.assert_all_finite(loss_angle, 'Angle problem', name=None)

#     if not re:
#         return loss_angle+loss_energy
#     if re:
#         return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]