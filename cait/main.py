class CaiT(keras.Model):
    def __init__(self,
                 patch_resolution,
                 img_shape=(224, 224),
                 patch_size=(16, 16),
                 n_self_attention=24,
                 n_class_attention=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.0,
                 dim=192,
                 num_heads=4,
                 init_values=1e-5,
                 layer_norm_eps=1e-6,
                 mlp_ratio=4,
                 dropout_rate=0.,
                 attn_dropout_rate=0.,
                 return_logits=False,
                 global_pool=None,
                 num_classes=1000,
                 **kwargs
            ):
        super(CaiT, self).__init__(**kwargs)

        # patch generator, for generating patch from image.
        self.patch_generator  = PatchEmbed(img_size=img_shape,
                                           patch_size=patch_size,
                                           in_chans=3,
                                           embed_dim=dim,
                                        )

        # position embedding and class token for ca layerscale layer
        self.pos_embed = tf.Variable(tf.zeros((1, patch_resolution, dim)))
        self.cls_token = tf.Variable(tf.zeros((1, 1, dim)))

        self.pos_drop = Dropout(dropout_rate, name="projection_dropout")

        # droppath schedule for self attention transformer
        dpr = [drop_path_rate for _ in range(n_self_attention)]

        # self attention transformer with layerscale, which only uses the
        # embed_patches
        self.sa_blocks = [
            LayerScale_Block(dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=dropout_rate,
                             attn_drop=attn_dropout_rate,
                             drop_path=dpr[i],
                             act_layer=tf.nn.gelu,
                             norm_layer=LayerNormalization,
                             Attention_block=Attention_Talking_Head,
                             MLP_Block=MLP,
                             init_values=init_values,
                             layer_norm_eps=layer_norm_eps,
                             name = f"self_attention_transformerb_{i}")
                        for i in range(n_self_attention)
                    ]

        self.ca_blocks = [
            LayerScale_Block_CA(dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=0.0,
                             attn_drop=0.0,
                             drop_path=0.0,
                             act_layer=tf.nn.gelu,
                             norm_layer=LayerNormalization,
                             Attention_block=ClassAttention,
                             MLP_Block=MLP,
                             init_values=init_values,
                             layer_norm_eps=layer_norm_eps,
                             name = f"class_attention_transformerb_{i}")
                        for i in range(n_class_attention)
                    ]

        self.norm = LayerNormalization(epsilon=layer_norm_eps, name="head_norm")

        self.global_pool = global_pool
        self.return_logits = return_logits
        self.num_classes = num_classes

        self.classification_head = Dense(self.num_classes, name='classification_head')

    def call(self, x):

        # patches with projected vectors(learnable)
        x = self.patch_generator(x)

        # adding the position embedding for each patch
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # self-attention transformer block with layerscale layer
        sa_layer_attention_scores = {}
        for sa_block in self.sa_blocks:
            x, attn_score = sa_block(x)
            # storing each multihead attention layer's attention score.
            sa_layer_attention_scores[f"{sa_block.name}"] = attn_score

        # class attention transformer block with layerscale layer
        ca_layer_attention_scores = {}
        cls_tokens = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        for ca_block in self.ca_blocks:
            cls_tokens, attn_score = ca_block(x, cls_tokens)

            # stroing the multihead attention layer's attention score
            # in the class attention layer
            ca_layer_attention_scores[f"{ca_block.name}"] = attn_score

        x = tf.concat([cls_tokens, x], axis=1)
        x = self.norm(x)

        if self.global_pool:# here, we are using all token's representation
            x = tf.reduce_mean(x[:, 1:], axis=1)

        else: # using the cls_token's representation
            x = x[:, 0]

        if self.return_logits:
            return (x, sa_layer_attention_scores, ca_layer_attention_scores)

        else:
            normalized_logits = self.classification_head(x)
            return (normalized_logits, sa_layer_attention_scores, ca_layer_attention_scores)