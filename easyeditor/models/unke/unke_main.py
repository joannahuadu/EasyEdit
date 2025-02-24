import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ..rome.layer_stats import layer_stats_multimodal
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

# from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, get_model_config
from .unke_hparams import UnKEMultimodalHyperParams

from easyeditor import VQADataset_Simple
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def get_context_templates(model, tok, multimodal_generation=False):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                    multimodal_generation=multimodal_generation,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_VQA_ds(prompt):
    annotation_path = '/data/lishichao/data/model_edit/editing-data/vqa/vqa_train.json'
    image_root = '/data/lishichao/data/model_edit/'
    raw_ds = VQADataset_Simple(prompt=prompt,annotation_file=annotation_path,image_root=image_root,image_size=336)
    return raw_ds

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters
def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UnKEMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
    if copy:
        model = deepcopy(model)
    # external dataset, prompt 
    if hparams.model_name == 'llava':
        from ...trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
        prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
    if prompt:
        ds = get_VQA_ds(prompt) 
    else:
        assert "No prompt is defined for multimodal text inputs"
    # Retrieve the external dataset
    ds = get_VQA_ds(prompt=prompt)
    # Create the DataLoader
    loader = DataLoader(
        ds,
        batch_size=hparams.ex_data_num, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=ds.collate_fn, 
    )
    
    weights_copy = execute_unke(model, tok, requests, hparams, cache_template=cache_template, ex_data_loader=loader)

    return model, weights_copy

def execute_unke(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UnKEMultimodalHyperParams,
    cache_template: Optional[str] = None,
    ex_data_loader: Optional[DataLoader] = None
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the UnKE update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    for request in requests[:10]:
        print(
            f"UnKE request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}


    # Compute z for final layer
    context_templates = get_context_templates(model, tok, multimodal_generation=True if 'image' in request else False)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=0)
    
    # define target_ids,all_prompts_list, did not add target_ids 
    all_prompts_list = []
    for request in requests:
        target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
        all_prompts = request["prompt_template"].format(request["prompt"]) if "prompt_template" in request else request["prompt"]
        all_prompts_list.append(all_prompts)
    
    if "image" in requests[0]:
        images = [request["image"] for request in requests]
        text_inputs = [all_prompts_list[idx].format(request["subject"]) for idx,request in enumerate(requests)]
    else:
        batch_question = [all_prompts_list[idx].format(request["subject"]) for idx,request in enumerate(requests)]
    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        if "image" not in requests[0]:
            contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
                next(model.parameters()).device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                if "image" in requests[0]:
                    samples = {"noise": True, "text_input": text_inputs, "image": images if images is not None else None}
                    edit_output = model(samples)
                else:
                    _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
                
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        if "image" in requests[0]:
            cur_zs, idxs = compute_ks(model, tok, samples, hparams, z_layer)
        else:
            cur_zs, idxs = compute_ks(model, tok, batch_question, hparams, z_layer)
        
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        data_iter = iter(ex_data_loader)  
        ex_data_batch = next(data_iter)  
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                if "image" in requests[0]:
                    ex_data_output = model(ex_data_batch)
                else:
                    """wait to apply unke for LLM"""
                    assert 1==1
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out
        
        resid = targets / (len(hparams.layers) - i)  
        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
    
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        #optimizer = optim.SGD(params, lr=hparams.lr, momentum=0.9, weight_decay=0.01)
        
        for i in range(len(idxs)):
            layer_out_ks[i,idxs[i]] += resid[i]
        
        input_causal_mask = edit_output.attention_mask
        input_position_ids = edit_output.position_ids
        input_cache_position = input_position_ids[0]
        ex_causal_mask = ex_data_output.attention_mask
        ex_position_ids = ex_data_output.position_ids
        ex_cache_position = ex_position_ids[0]
        
        input_causal_mask,input_position_ids,input_cache_position = get_causal_mask(layer_in_ks,input_causal_mask.to(layer_in_ks.device))
        ex_causal_mask,ex_position_ids,ex_cache_position = get_causal_mask(stat_in,ex_causal_mask.to(stat_in.device))
        
        # # Assuming attention_mask is of shape [batch_size, seq_length]
        # input_causal_mask = input_causal_mask.unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1, seq_length]
        # # Now, repeat the mask to match the self-attention shape
        # input_causal_mask = input_causal_mask.expand(-1, model.llava_model.config.num_attention_heads, input_causal_mask.shape[-1], input_causal_mask.shape[-1])  # num_heads is typically the number of attention heads in the model
        
        # ex_causal_mask = ex_causal_mask.unsqueeze(1).unsqueeze(2) 
        # ex_causal_mask = ex_causal_mask.expand(-1, model.llava_model.config.num_attention_heads, ex_causal_mask.shape[-1], ex_causal_mask.shape[-1])
        
        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()

            
            # ex_random_tensor = torch.randn(stat_out.shape, device=layer_out_ks.device, dtype=torch.bfloat16)
            # in_random_tensor = torch.randn(layer_out_ks.shape, device=layer_out_ks.device,dtype=torch.bfloat16)
            # loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], ex_random_tensor)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], in_random_tensor)
            loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
            #loss = torch.sum(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0] - layer_out_ks)
            # loss = loss*10000
            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()    
            for param in model.parameters():
                if param.grad is not None:
                    print(param.grad.abs().mean())  # 检查每个参数的梯度
            
            # print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, config.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break

        for x in [layer_in_ks, layer_out_ks,cur_zs, targets,stat_in,stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
        
    return weights_copy

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: Union[Dict, List],
    hparams: UnKEMultimodalHyperParams,
    layer: int,
):
    if isinstance(batch_data, list):
        input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f"cuda:{hparams.device}")
        idxs = [i.sum()-1 for i in input_ids['attention_mask']]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                if isinstance(batch_data, dict):
                    output = model(batch_data)
                    idxs = [int(i.sum())-1 for i in output.attention_mask]
                else:
                    _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list=[]
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i,idxs[i]])
    zs_out =torch.stack(zs_out_list,dim=0)


    return zs_out,idxs

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position
