import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization, Dense, Conv2D, Dropout, Add, add
from cait import DropPath, Attention_Talking_Head, MLP
from tensorflow.keras import layers 

class LayerScale_Block(keras.Model):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=tf.nn.gelu,
                 norm_layer=LayerNormalization,
                 Attention_block=Attention_Talking_Head,
                 MLP_Block=MLP,
                 init_values=1e-4,
                 layer_norm_eps=1e-6,
                 m_name="sa_layerscale",
                 **kwargs
            ):
        super(LayerScale_Block, self).__init__(**kwargs)
        self.norm1 = norm_layer(epsilon=layer_norm_eps)
        self.attn = Attention_block(
                                dim,
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop,
                                proj_drop=drop
                        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(epsilon=layer_norm_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP_Block(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # layer scale
        self.gamma_1 = tf.Variable(init_values * tf.ones((dim,)), name="gamma_1")
        self.gamma_2 = tf.Variable(init_values * tf.ones((dim,)), name="gamma_2")

    def __call__(self, x):
        # transformer block with self attention
        x1 = self.norm1(x)
        attn_output, attn_score = self.attn(x1)
        attn_output = attn_output * self.gamma_1
        if self.drop_path:
            attn_output = self.drop_path(attn_output)
        x2 = layers.Add()([x, attn_output])

        # FFN(MLP)
        x3 = self.norm2(x2)
        x4 = self.mlp(x3)
        x4 = self.gamma_2 * x4
        if self.drop_path:
            x4 = self.drop_path(x4)

        outputs = Add()([x2, x4])

        return outputs, attn_score