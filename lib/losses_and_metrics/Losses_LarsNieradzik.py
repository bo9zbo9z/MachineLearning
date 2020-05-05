"""
VERY GOOD !!!  https://lars76.github.io/neural-networks/
                        object-detection/losses-for-segmentation/
These are forked from his article....
import tensorflow as tf
"""


def weighted_cross_entropy(beta):
    """
    Weighted cross entropy (WCE) is a variant of CE where all positive examples
    get weighted by some coefficient. It is used in the case of class
    imbalance. For example, when you have an image with 10% black pixels and
    90% white pixels, regular CE wonâ€™t work very well
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/
        #     tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=beta)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


def balanced_cross_entropy(beta):
    """
    Balanced cross entropy (BCE) is similar to WCE. The only difference is that
    we weight also the negative examples.
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/
        #             tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred,
                                  tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


def focal_loss(alpha=0.25, gamma=2):
    """
    Focal loss (FL) [2] tries to down-weight the contribution of easy examples
    so that the CNN focuses more on hard examples.
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) *  \
               (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred,
                                  tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))
        loss = focal_loss_with_logits(logits=logits,
                                      targets=y_true,
                                      alpha=alpha,
                                      gamma=gamma,
                                      y_pred=y_pred)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


"""
Distance to the nearest cell
The paper [3] adds to cross entropy a distance function to force the CNN to
learn the separation border between touching objects

# SEE PAPER FOR CODE SUGGESTIONS
"""


def dice_loss_old(y_true, y_pred):   # OLD VERSION
    """
    Dice Loss / F1 score
    The Dice coefficient is similar to the Jaccard Index (Intersection over
    Union, IoU):
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)


def dice_loss(y_true, y_pred):  # NEWER VERSION
    """
    Dice Loss / F1 score
    The Dice coefficient is similar to the Jaccard Index (Intersection over
    Union, IoU):
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


def tversky_loss(beta):
    """
    Tversky loss
    Tversky index (TI) is a generalization of Diceâ€™s coefficient. TI adds a
    weight to FP (false positives) and FN (false negatives)
    """
    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + \
            (1 - beta) * y_true * (1 - y_pred)
        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
    return loss


def combo_loss(y_true, y_pred):
    """
    Combinations
    It is also possible to combine multiple loss functions. The following
    function is quite popular in data competitions
    """
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
        dice_loss(y_true, y_pred)
