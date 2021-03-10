''' Loss functions file '''                       
import tensorflow as tf
import numpy as np
def negative_cos(pred, true):
    return 1 - tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))

def loss_funcunit(y_reco, y_true, re=False):
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract
    # Energy loss
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #this works well but could maybe be improved
    #angle loss
    # loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 1:], y_true[:, 1:]))  #this needs to be updated!
    
    cos_angle = tf.math.divide_no_nan(tf.reduce_sum(y_reco[:, 1:] * y_true[:, 1:], axis = 1),
            tf.math.reduce_euclidean_norm(y_reco[:, 1:], axis = 1) * tf.math.reduce_euclidean_norm(y_true[:, 1:],  axis = 1))

    cos_angle -= tf.math.sign(cos_angle) * 1e-6
    loss_angle = tf.reduce_mean(tf.math.acos(cos_angle))
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]

def loss_func(y_reco, y_true, re=False):
    # Energy loss
    loss_energy = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    # Position loss
    loss_dist  = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    )
    loss_dist=loss_dist/1000
    # Angle loss
    loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 4:], y_true[:, 4:]))  #this needs to be updated!
    if not re:
        return loss_energy+loss_dist+loss_angle
    else:   
        return float(loss_energy+loss_dist+loss_angle), [float(loss_energy), float(loss_dist), float(loss_angle)]

def angle(zep, zet, azp, azt):
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract
    return abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)

def loss_funcxpos(y_reco, y_true, re=False):
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract
    # Energy loss
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #this works well but could maybe be improved
    # Angle loss
    loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 1:], y_true[:, 1:]))

    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]

def loss_funcangle(y_reco, y_true, re=False):
    from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract
    # Energy loss
    loss_energy = reduce_mean(abs(subtract(y_reco[:,0], y_true[:,0]))) #this works well but could maybe be improved
    # Angle loss
    # loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 1:], y_true[:, 1:]))  #this needs to be updated!
        
    # angle_resi = 180 / np.pi * tf.math.acos(tf.math.cos(y_reco[:, 1])*tf.math.cos(y_true[:, 1])*tf.math.cos(tf.subtract(y_reco[:, 2],y_true[:, 2]))+tf.math.sin(y_reco[:, 1])*tf.math.sin(y_true[:, 1]))
    # zenith_angle = tf.subtract(y_reco[:, 1], y_true[:, 1]) #degrees
    # azimuthal_angle = tf.subtract(y_reco[:, 2], y_true[:, 2])
    cosa=tf.math.sin(y_reco[:, 1])*tf.math.sin(y_true[:, 1])*tf.math.cos(tf.subtract(y_reco[:, 2],y_true[:, 2]))+tf.math.cos(y_reco[:, 1])*tf.math.cos(y_true[:, 1])
    # angle_resi=angle(y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2])
    # angle_resi=acos(cosa)

    loss_angle = tf.reduce_mean(1-cosa)
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]

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