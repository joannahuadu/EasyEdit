from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib

from .melo_hparams import MELOMultimodalHyperParams
from .algs.lora import LORA

from .multimodal_trainer import vqa_trainer, caption_trainer
from ...evaluate import prepare_multimodal_edit_batch

def apply_mmelo_to_model(        
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MELOMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
)-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    # alg_module = importlib.import_module(f'algs.{hparams.alg}')
    # AlgClass = getattr(alg_module, hparams.alg.upper())
    alg = LORA(model, hparams)
    alg.to(hparams.device)
    
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    if "idx" in kwargs:
        idx = kwargs.get("idx", 0)
    else:
        raise ValueError("idx not found in kwargs")
    edit_inner = prepare_multimodal_edit_batch(hparams, tok, batch=requests, prompt_template=requests[0]['prompt_template'])
    if hparams.task == "caption":
        trainer = caption_trainer(hparams, alg, dict_to({"edit_inner": edit_inner}, hparams.device), idx)
    elif hparams.task == "vqa":
        trainer = vqa_trainer(hparams, alg, dict_to({"edit_inner": edit_inner}, hparams.device), idx)
        
    torch.cuda.empty_cache()
    trainer.run_edit()
    
    return trainer.alg, trainer.router, weights_copy


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict
