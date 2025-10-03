import gc
import logging

from torch.distributed.fsdp import FSDPModule

import comfy.model_management as mm


def cleanup_models_gc():
    do_gc = False
    for i in range(len(mm.current_loaded_models)):
        cur = mm.current_loaded_models[i]
        if cur.is_dead():
            do_gc = True
            logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            break

    if do_gc:
        gc.collect()
        mm.soft_empty_cache()

        for i in range(len(mm.current_loaded_models)):
            cur = mm.current_loaded_models[i]
            if cur.is_dead():
                if isinstance(cur.real_model().diffusion_model, FSDPModule):
                    mm.current_loaded_models.pop(i)
                    logging.info("FSDP wrapped module detected with model {}".format(cur.real_model().__class__.__name__))
                else:
                    logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))
