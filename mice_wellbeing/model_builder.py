import numpy as np
import keras
from keras import layers
import tensorflow as tf


# deprecated
# -----
# class SkeletonTransformerLayer(layers.Layer):
#     def __init__(self, shape: tuple, name: str = None, trainable: bool = True, dtype: object = np.float32, batch_input_shape=None):
#         super().__init__(input_shape=shape, name=name, trainable=trainable, dtype=dtype)

#         self.w = self.add_weight(
#             shape=(shape),
#             initializer="zeros",
#             trainable=True,
#             dtype=dtype
#         )

#     def call(self, inputs):
#         return tf.matmul(inputs, self.w, transpose_b=True)
# -----


class SkeletonTransformerLayerV2(layers.Layer):
    def __init__(self, shape: tuple, name: str = None, trainable: bool = True, dtype: object = np.float32, batch_input_shape=None):
        super().__init__(input_shape=shape, name=name, trainable=trainable, dtype=dtype)

        if name is None:
            name = "skel_trans"

        self.w = self.add_weight(
            shape=(35, 35),
            initializer="zeros",
            trainable=True,
            dtype=dtype,
            name=name + "_weights"
        )

    def call(self, inputs):
        return tf.transpose(tf.tensordot(inputs, self.w, axes=[[2], [0]]), perm=[0, 1, 3, 2])
