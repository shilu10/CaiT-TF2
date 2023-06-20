class LayerScale_Block(keras.Layer):
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
                 norm_layer=tf.layers.LayerNormalization,
                 Attention_block=Attention_Talking_Head,
                 MLP_Block=Mlp,
                 init_values=1e-4,
                 layer_norm_eps=1e-6
            ):
        super(LayerScale_Block, self).__init__()
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
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # layer scale
        self.gamma_1 = tf.Variable(init_values * tf.ones((dim,)))
        self.gamma_2 = tf.Variable(init_values * tf.ones((dim,)))
        
        def call(self, x):
            # transformer block with self attention
            x1 = self.norm1(x)
            attn_output, attn_score = self.attn(x1)
            attn_output = attn_output * self.gamma_1 
            if self.drop_path != None:
                attn_output = self.drop_path(attn_output)
            x2 = layers.Add()([x, attn_output])
            
            x3 = self.norm2(x2)
            x4 = self.mlp(x3)
            x4 = self.gamma_2 * x4 
            if self.drop_path:
                x4 = self.drop_path(x4)
                
            outputs = layers.Add()([x2, x4])
            
            return outputs, attn_score
