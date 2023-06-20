class ClassAttention(keras.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads 
        self.dim = dim 
        self.qkv_bias = qkv_bias 
        self.qk_scale = qk_scale 
        
        self.attn_drop = Dropout(attn_drop) 
        self.proj_drop = Dropout(proj_drop)
        
        head_dim = dim // num_heads 
        self.q = Dense(dim, bias=qkv_bias)
        self.k = Dense(dim, bias=qkv_bias)
        self.v = Dense(dim, bias=qkv_bias)
        self.proj = Dense(dim)
    
    def call(self, x, training=True):
        B, N, C = x.shape 
        
        # query vector
        q = tf.expand_dims(self.q(x[:, 0]), axis=1)
        q = tf.reshape(k, (B, 1, self.num_heads, C // self.num_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        
        # key vector
        k = self.k(x)
        k = tf.reshape(k, (B, N, self.num_heads, C // self.num_heads))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        
        #value vector
        v = self.v(x)
        v = tf.reshape(v, (B, N, self.num_heads, C // self.num_heads))
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # attention score between cls embedding and patch embedding.
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training)
        
        x_cls = tf.matmul(attn, v)
        x_cls = tf.transpose(x_cls, perm=[0, 2, 1, 3])
        x_cls = tf.reshape(x_cls, (batch_size, 1, num_channels))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls, training)
        
        return x_cls, attn
