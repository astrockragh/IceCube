''' Loss functions file '''                       
import tensorflow as tf
def negative_cos(pred, true):
    return 1 - tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))

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

def loss_funcxpos(y_reco, y_true, re=False):
    # Energy loss
    loss_energy = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    # Angle loss
    loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 1:], y_true[:, 1:]))  #this needs to be updated!
    if not re:
        return loss_energy+loss_angle
    else:   
        return float(loss_energy+loss_angle), [float(loss_energy), float(loss_angle)]