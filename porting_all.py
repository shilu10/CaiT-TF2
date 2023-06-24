import os 
import shutil 
import glob 
import numpy as np 
import yaml 
from imutils import paths 
from .porting import port_weights

# all config files 

def main():

    try:
        config_file_paths = paths.list_files("CaiT/configs/")
        print(config_file_paths)
        for config_file_path in config_file_paths:
            # porting all model types from pytorch to tensorflow

            # read from config file 
            with open(config_file_path, "r") as f:
                data = yaml.safe_load(f)

            print(f"Processing the  model type: {data.model_type}")

            port_weights(
                model_type=data.model_type,
                image_size=data.get("image_size"),
                n_self_attention_layers=data.get("n_self_attention_layers"),
                projection_dims=data.get("projection_dims"),
                patch_size=16,
                num_heads=data.get("num_heads"),
                init_values=data.get("init_values"),
                return_logits=False
            )    

    except Exception as err:
        return err



