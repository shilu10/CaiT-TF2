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
        config_file_paths = paths.list_files("configs/")

        for config_file_path in config_file_paths:
            # porting all model types from pytorch to tensorflow
            model_type = config_file_path.split("/")[1].split(".")[0]
            print(f"Processing the  model type: {model_type}")

            # read from config file 
            with open(config_file_path, "r") as f:
                data = yaml.safe_load(f)

            port_weights(
                model_type=model_type,
                image_size=data.get("image_size"),
                n_self_attention_layers=data.get("n_self_attention_layers"),
                projection_dims=data.get("projection_dims"),
                patch_size=16,
                num_heads=data.get("num_heads"),
                init_values=data.get("init_values"),
                return_logits=False
            )    

    except Exception as err:
        return 



