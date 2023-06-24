import timm 
from .cait.layers.class_attention import ClassAttention
from .cait.layers.mlp import MLP 
from .cait.layers.drop_path import DropPath
from .cait.layers.attn_talking_head import Attention_Talking_Head

from .cait.blocks.layerscale import LayerScale_Block
from .cait.blocks.layerscale_ca import LayerScale_Block_CA
from .cait.blocks.patch_embed import PatchEmbed

from .cait.model import CaiT

from .utils import modify_tf_block
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
import numpy as np 
import os, shutil


def port_weights(model_type="cait_xxs24_224",
         image_size=224,
         n_self_attention_layers=24,
         projection_dims=192, 
         patch_size=16, 
         num_heads=4,
         init_values=1e-5,
         return_logits=False
      ):
  
  print('Loading the Pytorch model!!!')

  pt_model = timm.create_model(
        model_name=model_type,
        num_classes=1000,
        pretrained=True
      )
  pt_model.eval()
  
  print("Initializing the Tensorflow Model!!!")

  tf_model = CaiT(
                patch_resolution = pow((image_size // patch_size), 2),
                img_shape = (image_size, image_size),
                patch_size = (patch_size, patch_size),
                dim = projection_dims,
                n_self_attention = n_self_attention_layers,
                num_heads = num_heads,
                init_values = init_values,
                return_logits = return_logits
              )

  dummy_inputs = tf.ones((2, image_size, image_size, 3))
  _ = tf_model(dummy_inputs)
  
  print(tf_model.count_params())
  print(sum(
            p.numel() for p in pt_model.parameters()
        ))
  
  if not return_logits:
        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )
  
  # getting the weights from pytorch model and creating the dict.
  # dict key is layername and value is params
  pt_model_dict = pt_model.state_dict()
  pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

  # patch embedding layer
  tf_model.layers[0].layers[0] = modify_tf_block(
          tf_model.layers[0].layers[0],
          pt_model_dict["patch_embed.proj.weight"],
          pt_model_dict["patch_embed.proj.bias"],
      )
  
  # Positional embedding.
  tf_model.pos_embed.assign(tf.Variable(pt_model_dict["pos_embed"]))

  # CLS token.
  tf_model.cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))

  # last layer norm.
  ln_idx = -2
  tf_model.layers[ln_idx] = modify_tf_block(
          tf_model.layers[ln_idx],
          pt_model_dict["norm.weight"],
          pt_model_dict["norm.bias"],
      )

  # Classification Head layers.
  if not return_logits:
    head_layer = tf_model.get_layer("classification_head")
    head_layer_idx = -1
    tf_model.layers[head_layer_idx] = modify_tf_block(
              head_layer,
              pt_model_dict["head.weight"],
              pt_model_dict["head.bias"],
          )

  # for self attention transformer block (layer scale block)

  all_layers = [layer.name for layer in tf_model.layers]
  all_self_attention_transformerb = list(filter(lambda x: "self_attention_transformerb" in x, all_layers))


  for indx, layer_name in enumerate(all_self_attention_transformerb):
    print("------------------------------------------------")

    # assign weights to gamma_1, gamma_2 (layerscale) for each block.
    pt_block_name = f'blocks.{indx}'
    current_block = tf_model.get_layer(layer_name)

    prev_gamma_1 = np.ravel(current_block.gamma_1)
    prev_gamma_2 = np.ravel(current_block.gamma_2)


    #current_block.gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
    #current_block.gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

    tf_model.get_layer(layer_name).gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
    tf_model.get_layer(layer_name).gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

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

        #layer.gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
        #layer.beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

        tf_model.get_layer(layer_name).get_layer(layer.name).gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
        tf_model.get_layer(layer_name).get_layer(layer.name).beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

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

        tf_model.get_layer(layer_name).get_layer(layer.name).fc1=modify_tf_block(
            layer.fc1,
            pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
            pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"])

        print(f"{pt_block_name}.mlp.fc1.weight")
        print(f"{pt_block_name}.mlp.fc1.bias")

        tf_model.get_layer(layer_name).get_layer(layer.name).fc2= modify_tf_block(
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
        tf_model.get_layer(layer_name).get_layer(layer.name).qkv=modify_tf_block(
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

        tf_model.get_layer(layer_name).get_layer(layer.name).proj_l=modify_tf_block(
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
        tf_model.get_layer(layer_name).get_layer(layer.name).proj_w=modify_tf_block(
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
        tf_model.get_layer(layer_name).get_layer(layer.name).proj=modify_tf_block(
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
  all_layers = [layer.name for layer in tf_model.layers]
  all_cls_attention_transformerb = list(filter(lambda x: "class_attention" in x, all_layers))

  print(all_cls_attention_transformerb)
  for indx, layer_name in enumerate(all_cls_attention_transformerb):
    print("-----------------")

    # assign weights to gamma_1, gamma_2 (layerscale) for each block.
    pt_block_name = f'blocks_token_only.{indx}'
    current_block = tf_model.get_layer(layer_name)

    prev_gamma_1 = np.ravel(current_block.gamma_1)
    prev_gamma_2 = np.ravel(current_block.gamma_2)

    #current_block.gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
    #current_block.gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

    tf_model.get_layer(layer_name).gamma_1.assign(pt_model_dict[f"{pt_block_name}.gamma_1"])
    tf_model.get_layer(layer_name).gamma_2.assign(pt_model_dict[f"{pt_block_name}.gamma_2"])

    if np.all(prev_gamma_1 == np.ravel(current_block.gamma_1)):
      print("gamma_1 is not changed")

    if np.all(prev_gamma_2 == np.ravel(current_block.gamma_2)):
      print("gamma_1 is not changed")

    print(f"{pt_block_name}.gamma_1")
    print(f"{pt_block_name}.gamma_2")

    n_norm = 1
    n_ffn = 1
    for layer in current_block.layers:

      # for norm1 and norm2 layers in layerscale block(transformer block).
      if isinstance(layer, tf.keras.layers.LayerNormalization):
        prev_alpha = np.ravel(layer.gamma)
        prev_beta = np.ravel(layer.beta)

      # layer.gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
        #layer.beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

        tf_model.get_layer(layer_name).get_layer(layer.name).gamma.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.weight"]))
        tf_model.get_layer(layer_name).get_layer(layer.name).beta.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.norm{n_norm}.bias"]))

        if np.all(prev_alpha == np.ravel(layer.gamma)):
          print(f'norm{n_norm} alpha is not modified')

        if np.all(prev_beta == np.ravel(layer.beta)):
          print(f'norm{n_norm} beta is not modified')

        print(f"{pt_block_name}.norm{n_norm}.weight")
        print(f"{pt_block_name}.norm{n_norm}.bias")

        n_norm += 1

      # FFN
      if isinstance(layer, MLP):
        # fc1
        prev_k = np.ravel(layer.fc1.kernel)
        prev_b = np.ravel(layer.fc1.bias)
        prev_k1 = np.ravel(layer.fc2.kernel)
        prev_b1 = np.ravel(layer.fc2.bias)
        
        tf_model.get_layer(layer_name).get_layer(layer.name).fc1=modify_tf_block(
            layer.fc1,
            pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
            pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"])

        print(f"{pt_block_name}.mlp.fc1.weight")
        print(f"{pt_block_name}.mlp.fc1.bias")

        tf_model.get_layer(layer_name).get_layer(layer.name).fc2=modify_tf_block(
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

        tf_model.get_layer(layer_name).get_layer(layer.name).q=modify_tf_block(
            layer.q,
            pt_model_dict[f"{pt_block_name}.attn.q.weight"],
            pt_model_dict[f"{pt_block_name}.attn.q.bias"])

        if np.all(prev_k == np.ravel(layer.q.kernel)):
          print(f'attn q k is not modified')

        if np.all(prev_b == np.ravel(layer.q.bias)):
          print(f'attn q b bias is not modified')

        print(f"{pt_block_name}.attn.q.weight")
        print(f"{pt_block_name}.attn.q.bias")

        # k weight matrix in class attention
        prev_k = np.ravel(layer.k.kernel)
        prev_b = np.ravel(layer.k.bias)
        tf_model.get_layer(layer_name).get_layer(layer.name).k=modify_tf_block(
            layer.k,
            pt_model_dict[f"{pt_block_name}.attn.k.weight"],
            pt_model_dict[f"{pt_block_name}.attn.k.bias"])

        if np.all(prev_k == np.ravel(layer.k.kernel)):
          print(f'attn k k is not modified')

        if np.all(prev_b == np.ravel(layer.k.bias)):
          print(f'attn k b bias is not modified')

        print(f"{pt_block_name}.attn.k.weight")
        print(f"{pt_block_name}.attn.k.bias")

        # v weight matrix in class attention
        prev_k = np.ravel(layer.v.kernel)
        prev_b = np.ravel(layer.v.bias)
        tf_model.get_layer(layer_name).get_layer(layer.name).v=modify_tf_block(
            layer.v,
            pt_model_dict[f"{pt_block_name}.attn.v.weight"],
            pt_model_dict[f"{pt_block_name}.attn.v.bias"])

        if np.all(prev_k == np.ravel(layer.v.kernel)):
          print(f'attn v k is not modified')

        if np.all(prev_b == np.ravel(layer.v.bias)):
          print(f'attn v b bias is not modified')

        print(f"{pt_block_name}.attn.v.weight")
        print(f"{pt_block_name}.attn.v.bias")

        # projection final in class attention
        prev_k = np.ravel(layer.proj.kernel)
        prev_b = np.ravel(layer.proj.bias)
        tf_model.get_layer(layer_name).get_layer(layer.name).proj=modify_tf_block(
            layer.proj,
            pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
            pt_model_dict[f"{pt_block_name}.attn.proj.bias"])

        if np.all(prev_k == np.ravel(layer.proj.kernel)):
          print(f'attn proj k is not modified')

        if np.all(prev_b == np.ravel(layer.proj.bias)):
          print(f'attn proj b bias is not modified')

        print(f"{pt_block_name}.attn.proj.weight")
        print(f"{pt_block_name}.attn.proj.bias")  

  print("Ported the Weights Successfull")

  save_dir = "models/"
  
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
  save_name = model_type + "_fe" if return_logits else model_type

  tf_model.save(save_dir + save_name)

  print("Model saved successfully.")