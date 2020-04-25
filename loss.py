import numpy as np
import keras.backend as K
# custom loss
# y_true 是重构误差，y_pred是soft标签
def myloss(y_true, y_pred):
    return -K.log(K.sum(y_pred * (K.exp(-y_true))))