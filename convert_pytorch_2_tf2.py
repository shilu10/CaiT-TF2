# patch embedding layer

tf_model1.layers[0].layers[0] = modify_tf_block(
        tf_model1.layers[0].layers[0],

        pt_model_dict["patch_embed.proj.weight"],
        pt_model_dict["patch_embed.proj.bias"],
    )


# Positional embedding.
tf_model1.pos_embed.assign(tf.Variable(pt_model_dict["pos_embed"]))

# CLS token.
tf_model1.cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))

# Layer norm layers.
ln_idx = -2
tf_model1.layers[ln_idx] = modify_tf_block(
        tf_model1.layers[ln_idx],
        pt_model_dict["norm.weight"],
        pt_model_dict["norm.bias"],
    )

# Classification Head layers.
head_layer = tf_model1.get_layer("classification_head")
head_layer_idx = -1
tf_model1.layers[head_layer_idx] = modify_tf_block(
          head_layer,
          pt_model_dict["head.weight"],
          pt_model_dict["head.bias"],
      )


# for self attention transformer block (layer scale block)

edited_layers = []

all_layers = [layer.name for layer in tf_model1.layers]
all_self_attention_transformerb = list(filter(lambda x: "self_attention_transformerb" in x, all_layers))


for indx, layer_name in enumerate(all_self_attention_transformerb):
  print("------------------------------------------------")
  print(layer_name)

  # assign weights to gamma_1, gamma_2 (layerscale) for each block.
  pt_block_name = f'blocks.{indx}'
  print(pt_block_name)
  current_block = tf_model1.get_layer(layer_name)

  prev_gamma_1 = np.ravel(current_block.gamma_1)
  prev_gamma_2 = np.ravel(current_block.gamma_2)


  current_block.gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
  current_block.gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

  print(f"{pt_block_name}.gamma_1")
  print(f"{pt_block_name}.gamma_2")

  if np.all(prev_gamma_1 == np.ravel(current_block.gamma_1)):
    print("gamma_1 is not changed")

  if np.all(prev_gamma_2 == np.ravel(current_block.gamma_2)):
    print("gamma21 is not changed")

  n_norm = 1
  for layer in current_block.layers:
    # for norm1 and norm2 layers in layerscale block(transformer block).
    if isinstance(layer, tf.keras.layers.LayerNormalization):
      prev_alpha = np.ravel(layer.gamma)
      prev_beta = np.ravel(layer.beta)
      
      layer.gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
      layer.beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

      print(f"{pt_block_name}.norm{n_norm}.weight")
      print(f"{pt_block_name}.norm{n_norm}.bias")

      if np.all(prev_alpha == np.ravel(layer.gamma)):
        print(f'norm{n_norm} alpha is not modified')

      if np.all(prev_beta == np.ravel(layer.beta)):
        print(f'norm{n_norm} beta is not modified')

      n_norm += 1

    # FFN
    if isinstance(layer, MLP):
      # fc1 
      prev_k = np.ravel(layer.fc1.kernel)
      prev_b = np.ravel(layer.fc1.bias)
      prev_k1 = np.ravel(layer.fc2.kernel)
      prev_b1 = np.ravel(layer.fc2.bias)

      modify_tf_block(
          layer.fc1,
          pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
          pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"])
      
      print(f"{pt_block_name}.mlp.fc1.weight")
      print(f"{pt_block_name}.mlp.fc1.bias")

      modify_tf_block(
          layer.fc2,
          pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"],
          pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"])
      
      print(f"{pt_block_name}.mlp.fc2.weight")
      print(f"{pt_block_name}.mlp.fc2.bias")

      if np.all(prev_k == np.ravel(layer.fc1.kernel)):
        print(f'dense1 k is not modified')

      if np.all(prev_b == np.ravel(layer.fc1.bias)):
        print(f'dense1 bias is not modified')

      if np.all(prev_k1 == np.ravel(layer.fc2.kernel)):
        print(f'dense2 k is not modified')

      if np.all(prev_b1 == np.ravel(layer.fc2.bias)):
        print(f'dense2 bias is not modified')


    if isinstance(layer, Attention_Talking_Head):
      # qkv weight matrix in attention talking head
      prev_k = np.ravel(layer.qkv.kernel)
      prev_b = np.ravel(layer.qkv.bias)
      modify_tf_block(
          layer.qkv,
          pt_model_dict[f"{pt_block_name}.attn.qkv.weight"],
          pt_model_dict[f"{pt_block_name}.attn.qkv.bias"])

      print(f"{pt_block_name}.attn.qkv.weight")
      print(f"{pt_block_name}.attn.qkv.bias")

      if np.all(prev_k == np.ravel(layer.qkv.kernel)):
        print(f'qkv k is not modified')

      if np.all(prev_b == np.ravel(layer.qkv.bias)):
        print(f'qkv bias is not modified')
        

      # projection l in attention talking head
      prev_k = np.ravel(layer.proj_l.kernel)
      prev_b = np.ravel(layer.proj_l.bias)
      modify_tf_block(
          layer.proj_l,
          pt_model_dict[f"{pt_block_name}.attn.proj_l.weight"],
          pt_model_dict[f"{pt_block_name}.attn.proj_l.bias"])
      
      print(f"{pt_block_name}.aattn.proj_l.weight")
      print(f"{pt_block_name}.aattn.proj_l.bias")

      if  np.all(prev_k == np.ravel(layer.proj_l.kernel)):
        print(f'projl k is not modified')

      if  np.all(prev_b == np.ravel(layer.proj_l.bias)):
        print(f'projl bias is not modified')


      # projection w in attention talking head
      prev_k = np.ravel(layer.proj_w.kernel)
      prev_b = np.ravel(layer.proj_w.bias)
      modify_tf_block(
          layer.proj_w,
          pt_model_dict[f"{pt_block_name}.attn.proj_w.weight"],
          pt_model_dict[f"{pt_block_name}.attn.proj_w.bias"])

      print(f"{pt_block_name}.aattn.proj_w.weight")
      print(f"{pt_block_name}.aattn.proj_w.bias")
      if np.all(prev_k == np.ravel(layer.proj_w.kernel)):
        print(f'projw k is not modified')

      if np.all(prev_b == np.ravel(layer.proj_w.bias)):
        print(f'projw bias is not modified')


      # projection final in attention talking head
      prev_k = np.ravel(layer.proj.kernel)
      prev_b = np.ravel(layer.proj.bias)
      modify_tf_block(
          layer.proj,
          pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
          pt_model_dict[f"{pt_block_name}.attn.proj.bias"])

      print(f"{pt_block_name}.aattn.proj.weight")
      print(f"{pt_block_name}.aattn.proj.bias")
      if np.all(prev_k == np.ravel(layer.proj.kernel)):
        print(f'proj k is not modified')

      if  np.all(prev_b == np.ravel(layer.proj.bias)):
        print(f'proj bias is not modified')
      
# for class attention transformer block (layer scale block)
all_layers = [layer.name for layer in tf_model1.layers]
all_cls_attention_transformerb = list(filter(lambda x: "class_attention" in x, all_layers))

print(all_self_attention_transformerb)
for indx, layer_name in enumerate(all_cls_attention_transformerb):
  print("-----------------")
  print(layer_name)

  # assign weights to gamma_1, gamma_2 (layerscale) for each block.
  pt_block_name = f'blocks_token_only.{indx}'
  current_block = tf_model1.get_layer(layer_name)

  prev_gamma_1 = np.ravel(current_block.gamma_1)
  prev_gamma_2 = np.ravel(current_block.gamma_2)

  current_block.gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
  current_block.gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

  if np.all(prev_gamma_1 == np.ravel(current_block.gamma_1)):
    print("gamma_1 is not changed")

  if np.all(prev_gamma_2 == np.ravel(current_block.gamma_2)):
    print("gamma21 is not changed")

  n_norm = 1
  n_ffn = 1
  for layer in current_block.layers:
    # for norm1 and norm2 layers in layerscale block(transformer block).
    if isinstance(layer, tf.keras.layers.LayerNormalization):
      prev_alpha = np.ravel(layer.gamma)
      prev_beta = np.ravel(layer.beta)
      
      layer.gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
      layer.beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

      if np.all(prev_alpha == np.ravel(layer.gamma)):
        print(f'norm{n_norm} alpha is not modified')

      if np.all(prev_beta == np.ravel(layer.beta)):
        print(f'norm{n_norm} beta is not modified')

      n_norm += 1

    # FFN
   
    if isinstance(layer, MLP):

      # fc1 
      prev_k = np.ravel(layer.fc1.kernel)
      prev_b = np.ravel(layer.fc1.bias)
      prev_k1 = np.ravel(layer.fc2.kernel)
      prev_b1 = np.ravel(layer.fc2.bias)
      modify_tf_block(
          layer.fc1,
          pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
          pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"])
      
      print(f"{pt_block_name}.mlp.fc1.weight")
      print(f"{pt_block_name}.mlp.fc1.bias")

      modify_tf_block(
          layer.fc2,
          pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"],
          pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"])
      
      print(f"{pt_block_name}.mlp.fc2.weight")
      print(f"{pt_block_name}.mlp.fc2.bias")

      if np.all(prev_k == np.ravel(layer.fc1.kernel)):
        print(f'dense1 k is not modified')

      if np.all(prev_b == np.ravel(layer.fc1.bias)):
        print(f'dense1 bias is not modified')

      if np.all(prev_k1 == np.ravel(layer.fc2.kernel)):
        print(f'dense2 k is not modified')

      if np.all(prev_b1 == np.ravel(layer.fc2.bias)):
        print(f'dense2 bias is not modified')


    if isinstance(layer, ClassAttention):
      # q weight matrix in class attention
      prev_k = np.ravel(layer.q.kernel)
      prev_b = np.ravel(layer.q.bias)

      modify_tf_block(
          layer.q,
          pt_model_dict[f"{pt_block_name}.attn.q.weight"],
          pt_model_dict[f"{pt_block_name}.attn.q.bias"])

      if np.all(prev_k == np.ravel(layer.q.kernel)):
        print(f'attn q k is not modified')

      if np.all(prev_b == np.ravel(layer.q.bias)):
        print(f'attn q b bias is not modified')

      # k weight matrix in class attention
      prev_k = np.ravel(layer.k.kernel)
      prev_b = np.ravel(layer.k.bias)
      modify_tf_block(
          layer.k,
          pt_model_dict[f"{pt_block_name}.attn.k.weight"],
          pt_model_dict[f"{pt_block_name}.attn.k.bias"])

      if np.all(prev_k == np.ravel(layer.k.kernel)):
        print(f'attn k k is not modified')

      if np.all(prev_b == np.ravel(layer.k.bias)):
        print(f'attn k b bias is not modified')


      # v weight matrix in class attention
      prev_k = np.ravel(layer.v.kernel)
      prev_b = np.ravel(layer.v.bias)
      modify_tf_block(
          layer.v,
          pt_model_dict[f"{pt_block_name}.attn.v.weight"],
          pt_model_dict[f"{pt_block_name}.attn.v.bias"])

      if np.all(prev_k == np.ravel(layer.v.kernel)):
        print(f'attn v k is not modified')

      if np.all(prev_b == np.ravel(layer.v.bias)):
        print(f'attn v b bias is not modified')

      # projection final in class attention
      prev_k = np.ravel(layer.proj.kernel)
      prev_b = np.ravel(layer.proj.bias)
      modify_tf_block(
          layer.proj,
          pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
          pt_model_dict[f"{pt_block_name}.attn.proj.bias"])

      if np.all(prev_k == np.ravel(layer.proj.kernel)):
        print(f'attn proj k is not modified')

      if np.all(prev_b == np.ravel(layer.proj.bias)):
        print(f'attn proj b bias is not modified')
