import tensorflow as tf

def sig_likelihood(y_reco, y_true, re=False):
    """Use for angle/energy sigs"""
    # tf.print(f'y_true:{y_true}, y_reco:{y_true}')
    vects = y_reco[:, :3]
    sigs  = y_reco[:, 3:6]
    # rhos  = y_reco[:, 6:]
    global COV
    COV   = tf.linalg.diag(sigs)
    tf.print(tf.shape(COV), tf.shape(tf.expand_dims(vects - y_true, axis = 1)))
    tf.print(tf.shape(COV), tf.shape(tf.subtract(vects,y_true)))
    log_likelihood = tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ COV @ tf.expand_dims(vects - y_true, axis = -1)) / 2 - tf.math.log(tf.linalg.det(COV)) / 2
    if re:
        return tf.reduce_mean(log_likelihood), [log_likelihood, COV]
    else:
        return tf.reduce_mean(log_likelihood)
    # return tf.reduce_mean(vects-y_true)




def likelihood_angle_difference(y_true, y_reco):
    cos_angle = tf.math.divide_no_nan(tf.reduce_sum(y_reco[:, :3] * y_true[:, :3], axis = 1),
            tf.math.reduce_euclidean_norm(y_reco[:, :3], axis = 1) * tf.math.reduce_euclidean_norm(y_true[:, :3],  axis = 1))

    cos_angle -= tf.math.sign(cos_angle) * 1e-6

    angle      = tf.math.acos(cos_angle)

    zs         = tf.math.divide_no_nan(angle, y_reco[:, 3])

    log_likelihood = -zs** 2 / 2 - tf.math.log(tf.abs(y_reco[:, 3]))

       
    return tf.reduce_mean(- log_likelihood)