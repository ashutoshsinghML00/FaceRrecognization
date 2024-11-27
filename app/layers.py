# L1 distance class , note- we have imported "Layer" from keras
# for custom model

import tensorflow as tf
from keras.layers import Layer

class L1_dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    # similarity calculation        
    def cal(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
