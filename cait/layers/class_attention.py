from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, SeparableConv2D, Softmax, MaxPooling2D, Dropout



class ClassAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = Dense(dim, use_bias=qkv_bias)
        self.k = Dense(dim, use_bias=qkv_bias)
        self.v = Dense(dim, use_bias=qkv_bias)
        self.proj = Dense(dim)

    def call(self, x, training):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # creating a query matrices using the query weights and input
        q = tf.expand_dims(self.q(x[:, 0]), axis=1)
        q = tf.reshape(q, (B, 1, self.num_heads, C // self.num_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        scale = tf.cast(self.scale, dtype=q.dtype)
        q = q * scale

        # creating a key matrices using the key weights and input
        k = self.k(x)
        k = tf.reshape(k, (B, N, self.num_heads, C // self.num_heads))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        # creating a value matrices using the value weights and input
        v = self.v(x)
        v = tf.reshape(v, (B, N, self.num_heads, C // self.num_heads))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # attention score between cls embedding and patch embedding.
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training)

        x_cls = tf.matmul(attn, v)
        x_cls = tf.transpose(x_cls, perm=[0, 2, 1, 3])
        x_cls = tf.reshape(x_cls, (B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls, training)

        return x_cls, attn