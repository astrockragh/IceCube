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
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]) ) )
    kappa=tf.math.abs(y_reco[:,3])
#     tf.print(tf.reduce_mean(kappa))
    cos_alpha=cos_angle(y_reco, y_true)
    nlogC = - tf.math.log(kappa) + tf.math.log(tf.math.exp(kappa)-tf.math.exp(  -kappa) )

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]

def abs_vonMises_unit(y_reco, y_true, re=False):
    loss_energy = tf.reduce_mean(tf.abs(tf.subtract(y_reco[:,0], y_true[:,0]) ) )
    kappa=tf.math.abs(y_reco[:,4])
#     tf.print(tf.reduce_mean(kappa))
    cos_alpha=cos_unit(y_reco, y_true)
    nlogC = - tf.math.log(kappa) + tf.math.log(tf.math.exp(kappa)-tf.math.exp(-kappa) )

    loss_angle = tf.reduce_mean( - kappa*cos_alpha + nlogC )
    if not re:
        return loss_angle+loss_energy
    if re:
        return float(loss_angle+loss_energy), [float(loss_energy), float(loss_angle)]

#New/untested below        

def loss_funcxpos2(y_reco, y_true, re=False):
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract, square
    # Energy loss
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #this works well but could maybe be improved

    zeni = [cos(y_true[:,1]) - y_reco[:,1] , 
            sin(y_true[:,1]) - y_reco[:,2]]

    azi  = [cos(y_true[:,2]) - y_reco[:,3] , 
            sin(y_true[:,2]) - y_reco[:,4]]

    loss_angle = reduce_mean(square(azi[0]))+reduce_mean(square(azi[1]))+reduce_mean(square(zeni[0]))+reduce_mean(square(zeni[1]))
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]