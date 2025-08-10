import logging

import torch

import comfy
from comfy.sd import load_diffusion_model_state_dict, model_detection_error_hint


def load_diffusion_model_meta(unet_path, model_options={}):
    sd = comfy.utils.load_torch_file(unet_path, device=torch.device("cpu"))
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    model.model.to("meta")
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model
