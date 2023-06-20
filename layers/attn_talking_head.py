class Attention_Talking_Head(keras.layers.Layer):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale+None,
                 attn_drop=0.,
                 proj_drop=0.
            ):
        super(Attention_Talking_Head, self).__init__()
        self.num_heads = num_heads 
        
        head_dim = dim // num_heads 
        self.scale = qk_scale ** head_dim ** -0.5 
        
        self.qkv = Dense(dim, bias=qkv_bias)
        self.attn_drop = Dropout(rate=attn_drop)
        
        self.proj = Dense(dim)
        
        self.proj_l = Dense(num_heads)
        self.proj_w = Dense(num_heads)
        
        self.proj_drop = Dropout(rate=proj_drop)
        
    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = self.proj_l(tf.transpose(attn, perm=[0, 2, 3, 1]))
        
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = tf.nn.softmax(attn, axis=-1)
        
        attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.attn_drop(attn, training)
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x, training)
        
        return x, attn
