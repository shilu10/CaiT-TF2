class LayerScale_Block_CA(keras.Model):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=tf.nn.gelu,
                 norm_layer=tf.layers.LayerNormalization,
                 Attention_block=Class_Attention,
                 mlp_block=Mlp,
                 init_values=1e-4
                 layer_norm_eps=1e-6
            ):
        
        super(LayerScle_CA, self).__init__()
        self.dim = dim 
        self.norm1 = norm_layer(epsilon=layer_norm_eps)
        self.attn = Attention_block(dim,
                                     num_heads=num_heads,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     attn_drop=attn_drop,
                                     proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(epsilon=layer_norm_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # layer scale
        self.gamma_1 = tf.Variable(init_values * tf.ones((dim,)))
        self.gamma_2 = tf.Variable(init_values * tf.ones((dim,)))
        
    def call(self, x, x_cls):
        # transformer block with class attention.
        x1 = tf.concat([x, x_cls], axis=1)
        x1 = self.norm1(x1)
        
        attn_output, attn_score = self.attn(x1)
        attn_output = self.gamma_1 * attn_output 
        
        if self.drop_path != None:
            attn_output = self.drop_path(attn_output)
            
        x2 = Add()([x_cls, attn_output])
        # FFN(MLP)
        x3 = self.norm2(x2)
        x4 = self.mlp(x3)
        x4 = self.gamma_2 * x4 
        if self.drop_path:
            x4 = self.drop_path(x4)
                
        outputs = layers.Add()([x2, x4])
        return outputs, attn_score
        
