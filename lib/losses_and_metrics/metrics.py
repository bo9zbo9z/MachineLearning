# https://towardsdatascience.com/metrics-for-imbalanced-classification-41c71549bbb5
def show_metrics_TF(y_true, y_score):
    # True positive
    tp = tf.math.reduce_sum(y_true * y_score)
    # False positive
    fp = tf.math.reduce_sum(tf.cast((y_true == 0), y_true.dtype) * y_score)
    # True negative
    tn = tf.math.reduce_sum(tf.cast((y_true==0), y_true.dtype) * tf.cast((y_score==0), y_true.dtype))
    # False negative
    fn = tf.math.reduce_sum(y_true * tf.cast((y_score==0), y_true.dtype))

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print("True positive: ", tp.numpy())
    print("False positive: ", fp.numpy())
    print("True negative: ", tn.numpy())
    print("False negative: ", fn.numpy())

    print("True positive rate (recall): ", tpr.numpy())
    print("False positive rate: ", fpr.numpy())
    print("Precision: ", precision.numpy())
    print("True negative rate: ", tnr.numpy())
    print("F1: ", f1.numpy())
    print("ROC-AUC: ", auc.numpy())
    print("MCC: ", mcc.numpy())


# Simple metric to track improvements, can reduce the number of statements, left as is for readability
# Just helps to know when to save the model, it will be higher than actual results, but I found it very helpful
# than just using normal accuracy.
def eye_metric(y_true, y_pred):
    y_true_label = tf.cast(y_true > 0.5, dtype=y_true.dtype)
    y_pred_label = tf.cast(y_pred > 0.5, dtype=y_true.dtype)
    y_true_arg = tf.math.argmax(tf.reverse(y_true_label, axis=[0]))
    y_pred_arg = tf.math.argmax(tf.reverse(y_pred_label, axis=[0]))
    return tf.cast(y_true_arg == y_pred_arg, dtype=y_true.dtype)
