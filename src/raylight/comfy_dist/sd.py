import logging

import raylight.comfy_dist as comfy_dist
import comfy


def load_lora_for_models(model, lora, strength_model):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)

    # == Change from comfy ==#
    loaded = comfy_dist.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    k = set(k)
    for x in loaded:
        if (x not in k):
            logging.warning("NOT LOADED {}".format(x))

    return new_modelpatcher
