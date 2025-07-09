from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .xspace_hparams import XSpaceMultimodalHyperParams
from ...trainer.losses import masked_log_probs

from tqdm import tqdm

# Multimodal dataset for Corda 
from ...dataset import VQADataset_Simple
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from .optim import Adam

base_pca = {} 
reserved = {} 
def _logits(x):
    return x if not hasattr(x, "logits") else x.logits

def pca_features(x, r=32):
    np.random.seed(42)  
    torch.manual_seed(42)
    U, S, V = torch.pca_lowrank(x, q=r)
    torch.random.seed() 
    return V[:, :r]

def cosine_similarity(A, B):
    A_norm = F.normalize(A, dim=0)
    B_norm = F.normalize(B, dim=0)
    return torch.sum(A_norm * B_norm).item() / A.shape[1]

# def cosine_similarity(A, B):
#     dot_product = torch.dot(A, B)
#     norm_v1 = torch.norm(A)
#     norm_v2 = torch.norm(B)
#     return  dot_product / (norm_v1 * norm_v2)

def apply_xspace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: XSpaceMultimodalHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    global base_pca, reserved
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    requests = deepcopy(requests)
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
        print(
            f"Executing LoRA algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    # image_tok = requests[0]['image_toks']
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    if "image" in requests[0]:
        images = [r["image"] for r in requests]
    # text_lens = [len(tok.encode(prompt+" "+target, add_special_tokens=False)) for prompt, target in zip(texts, targets)]
    # B = (image_tok - hparams.wL) // hparams.wS + 1 + 1 + len(hparams.nS)
    embed_layername = layername(model, 0, "embed")
    proj_layername = layername(model, 0, "proj")
    base_pca = {}
    reserved = {}
    def embed_hook_ori(module, input, output):
        N, dim = output.shape
        wS = hparams.wS
        out = output.clone()
        if count == 0:
            return out
        if count == 2:
            noise_tensor = torch.randn(N, dim, device=output.device) * hparams.noise
            out += noise_tensor
            return out
        elif count%2==0:
            if N > wS:
                start = torch.randint(1, N - wS + 1, (1,)).item()
                end = start + wS
                noise_tensor = torch.randn(wS, dim, device=output.device) * hparams.noise
                out[start:end, :] += noise_tensor
            else:
                noise_tensor = torch.randn(N, dim, device=output.device) * hparams.noise
                out += noise_tensor
            return out
    def embed_hook(module, input, output):
        wS = hparams.wS
        noise_scale = hparams.noise
        out = output.clone()
        if count == 0:
            return out
        # Case 1: 2D input (N, dim)
        if out.dim() == 2:
            N, dim = out.shape
            if count == 2:
                out += torch.randn(N, dim, device=out.device) * noise_scale
            elif count % 2 == 0:
                if N > wS:
                    start = torch.randint(1, N - wS + 1, (1,)).item()
                    end = start + wS
                    out[start:end, :] += torch.randn(wS, dim, device=out.device) * noise_scale
                else:
                    out += torch.randn(N, dim, device=out.device) * noise_scale
            return out

        # Case 2: 3D input (B, N, dim)
        elif out.dim() == 3:
            B, N, dim = out.shape
            for b in range(B):
                if count == 2:
                    out[b] += torch.randn(N, dim, device=out.device) * noise_scale
                elif count % 2 == 0:
                    if N > wS:
                        start = torch.randint(1, N - wS + 1, (1,)).item()
                        end = start + wS
                        out[b, start:end, :] += torch.randn(wS, dim, device=out.device) * noise_scale
                    else:
                        out[b] += torch.randn(N, dim, device=out.device) * noise_scale
            return out

        else:
            raise ValueError(f"Unsupported output shape: {out.shape}")

    def proj_hook(module, input, output):
        wL = hparams.wL
        out = output.clone()
        if out.dim() == 3:
            B, N, dim = output.shape
            if count == 0:
                return out
            if count == 1:
                out[:,:,:] = 0
                return out
            elif count%2:
                start = torch.randint(0, N - wL + 1, (1,)).item()
                end = start + wL
                out[:, start:end, :] = 0
                return out
        elif out.dim() == 2:
            N, dim = output.shape
            if count == 0:
                return out
            if count == 1:
                out[:,:] = 0
                return out
            elif count%2:
                start = torch.randint(N - wL + 1, (1,)).item()
                end = start + wL
                out[start:end, :] = 0
                return out
        else:
            raise ValueError(f"Unsupported output shape: {out.shape}")

    def cov_hook(module, input, output, name):
        global base_pca, reserved
        input = input[0].detach().squeeze(0).data  ## (2048, dim)
        input = input
        input = input/torch.max(input).abs()
        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        covariance = input.t().matmul(input)
        pca = pca_features(covariance.float())
        if base_pca[name] is None:
            base_pca[name] = pca
        else:
            sim = cosine_similarity(base_pca[name], pca)
            if sim > hparams.sim:
                if torch.isnan(covariance).any():
                    print("nan detected")
                    raise Exception("nan in covariance, break")
                if torch.isinf(covariance).any():
                    print("inf detected")
                    raise Exception("inf in covariance, break")        
                module.covariance_matrix += covariance
                reserved[name]+=1
        del input, covariance
    
    for name, module in model.named_modules():
        if name == proj_layername:
            module.register_forward_hook(proj_hook)
        # if name == embed_layername:
            # module.register_forward_hook(embed_hook)
        if isinstance(module, nn.Linear):
            if not any(del_name in name for del_name in hparams.delete_name) and any(target in name for target in hparams.update_modules) and any('layers.' + str(layer) in name for layer in hparams.layers):
                module.covariance_matrix = 0
                module.register_forward_hook(partial(cov_hook, name=name))
                base_pca[name] = None
                reserved[name] = 0

    texts = texts*hparams.num_samples
    targets = targets*hparams.num_samples
    images = images*hparams.num_samples
    for i, (txt, tgt, img) in enumerate(tqdm(zip(
                chunks(texts, hparams.batch_size), 
                chunks(targets, hparams.batch_size),
                chunks(images, hparams.batch_size)
        ))):
        if "qwen2.5_vl" in hparams.model_name:
            full_prompt = [p for p in txt]
            answer = [l for l in tgt]
            batch = {
                "noise": True,
                "text_input": full_prompt,
                "image": img,
                "answer": answer
            }
        else:    
            full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
            batch = {
                "noise": True,
                "text_input": full_prompt,
                "image": img,
            }
        count = i
        model(batch)

    all_covariance_matrix = {}
    for name, module in model.named_modules():
        if name == proj_layername:
            module._forward_hooks.clear()
        if name == embed_layername:
            module._forward_hooks.clear()
        if isinstance(module, nn.Linear):
            if not any(del_name in name for del_name in hparams.delete_name) and any(target in name for target in hparams.update_modules) and any('layers.' + str(layer) in name for layer in hparams.layers):
                module._forward_hooks.clear()
                if torch.isnan(module.covariance_matrix).any():
                    print("nan detected")
                    raise Exception("nan in covariance")
                if torch.isinf(module.covariance_matrix).any():
                    print("inf detected")
                    raise Exception("inf in covariance")
                module.covariance_matrix = module.covariance_matrix/reserved[name]
                all_covariance_matrix[module.weight] = module.covariance_matrix
        
    edited_model = execute_xspace(model, tok, requests, hparams, all_covariance_matrix, keep_original_weight)
    if hasattr(model, "llava_model") or hasattr(model, "qwen_model"):
        # model.llava_model = edited_model
        return model, weights_copy
    else:
        return edited_model, weights_copy


def execute_xspace(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: XSpaceMultimodalHyperParams,
        all_covariance_matrix: Dict,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the Lora update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # for sub_model_name in ['llava_model', '']:
    #     sub_model = getattr(model, sub_model_name)
    #     if sub_model and hasattr(sub_model, 'config'):
    #         llava_model = sub_model
    #         break
    # model.config.use_cache = False
    # model.supports_gradient_checkpointing = True  #
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    if hasattr(model, "llava_model"):
        sub_model = model.llava_model
    elif hasattr(model, "qwen_model"):
        sub_model = model.qwen_model
    else:
        sub_model = model
    sub_model.config.use_cache = False
    sub_model.supports_gradient_checkpointing = True  #
    sub_model.gradient_checkpointing_enable()
    sub_model.enable_input_require_grads()
    if hparams.Null_mode:
        for n, p in sub_model.named_parameters():
            ## freeze BLinaer
            # and "BLinear" not in n 
            if "ALinear" not in n and p.requires_grad:
                p.requires_grad = False
            if ("PALinear" in n or "PBLinear"in n )and p.requires_grad:
                p.requires_grad = False
    elif hparams.lora_type == "lora":
        Config = LoraConfig
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
            total_step=hparams.num_steps
        )
    elif hparams.lora_type == "corda":
        # sampled_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]", ignore_verifications=True)
        # sampled_dataset = load_dataset("wikipedia", '20200501.en', split="train[:256]")
        if hparams.model_name == 'llava':
            from ...trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
            prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
            template = requests[0]["prompt_template"]
        if prompt:
            ds = get_VQA_ds(hparams,prompt,template,size=30) 
        else:
            assert "No prompt is defined for multimodal text inputs"
        dataloader = DataLoader(
            ds,
            batch_size=1,  # You can change this depending on your batch size requirement
            shuffle=False,  # Shuffle the data to ensure randomness in training
            collate_fn=ds.collate_fn  # Pass the custom collate_fn defined in the dataset
        )
        # dataset = load_dataset("imdb", split="train[:256]")
        # def run_model():
            # for batch in tqdm(sampled_dataset):
            #     samples = [
            #         {
            #             "text_input": [batch["text"]],
            #             "image": None,
            #         }
            #     ][0]
            #     with torch.no_grad():
            #         model(samples)
        def run_model():
            for batch in tqdm(dataloader):
                samples = {
                        "text_input": batch["text_input"],
                        "image": batch["image"],
                    }
            
                with torch.no_grad():
                    model(samples)
        corda_config = CordaConfig(
            corda_method="kpm",
            covariance_file="/home/lishichao/project/EasyEdit/results/cache/corda/cov_rank8.pt",
            cache_file="/home/lishichao/project/EasyEdit/results/cache/corda/cache_file_rank8.pt",
        )
        peft_config = LoraConfig(
            init_lora_weights="corda",
            corda_config=corda_config,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
        preprocess_corda(sub_model, lora_config=peft_config, run_model=run_model)
    else:
        raise NotImplementedError

    if hparams.Null_mode:
        peft_model = sub_model
    elif not keep_original_weight and hasattr(model, 'peft_config'):
        peft_model = sub_model
    else:
        peft_model = get_peft_model(sub_model, peft_config).to(torch.bfloat16)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    if hasattr(peft_model, 'print_trainable_parameters'):
        peft_model.print_trainable_parameters()
    requests = deepcopy(requests)
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
        print(
            f"Executing LoRA algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    device = torch.device(f'cuda:{hparams.device}')
    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    prompt_template = "{}" if requests[0]["prompt_template"] is None else requests[0]["prompt_template"]
    # Configure optimizer / gradients
    ## Adam-nscl
    opt, scheduler= init_model_optimizer(model, hparams)

    if "image" in requests[0]:
        images = [r["image"] for r in requests]
    # if torch.__version__ >= "2" and sys.platform != "win32":
    # model = torch.compile(model)
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt, img in zip(
                chunks(texts, hparams.batch_size), 
                chunks(targets, hparams.batch_size),
                chunks(images, hparams.batch_size)
        ):
            mask_token = -100
            opt.zero_grad()
            scheduler.step(it)
            if 't5' in hparams.model_name.lower():
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                bs = inputs["input_ids"].shape[0]
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                inputs['labels'] = target_ids
                logits = peft_model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)
                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                # src_trg_inputs = tok(txt + tgt, return_tensors="pt", padding=True).to(device)
                # bs = src_trg_inputs["input_ids"].shape[0]
                # targ = deepcopy(src_trg_inputs['input_ids'])
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # targ = targ[:, 1:]
                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
                # eos_token = tok.decode(tok.eos_token_id)
                if img:
                    if "qwen2.5_vl" in hparams.model_name:
                        full_prompt = [p for p in txt]
                        answer = [l for l in tgt]
                    else:    
                        full_prompt = [f"{prompt_template.format(p)} {l}" for p, l in zip(txt, tgt)]
                    samples = {
                        "noise": True,
                        "text_input": full_prompt,
                        "image": img,
                        "answer": answer
                    }
                    # pred = model(samples, output_attentions=False)
                    if isinstance(tgt, list):
                        tgt = tgt[0]
                    labels = tok.encode(tgt, add_special_tokens=False,return_tensors="pt").to(device)
                    logits = _logits(model(samples))
                    loss = masked_log_probs(hparams, logits, labels, shift=True)["nll"]
                    # loss = pred.loss
                else:
                    full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                    prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
                    bs = tokens["input_ids"].shape[0]
                    tokens["labels"] = tokens["input_ids"].clone()
                    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                    for i in range(len(txt)):
                        tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                    tokens = tokens.to(device)
                    pred = peft_model(**tokens)
                    loss = pred.loss
                # pred = peft_model(**tokens)
                # loss = pred.loss
                # targ = target_ids
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # pred = pred[:, -targ.size(1):]

                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=len(full_prompt))

            # if loss.item() >= 1e-3:
            loss.backward()
            if it==0:
                with torch.no_grad():
                    opt.get_eigens(all_covariance_matrix)
                    opt.get_transforms()
                    del all_covariance_matrix
            torch.cuda.empty_cache()
            opt.step()

        print(f"Total loss {loss_meter.avg}")

        # if loss_meter.avg < 1e-3:
        #     break
    return peft_model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def get_VQA_ds(hparams, prompt, template, size=None):
    annotation_path = hparams.train_annotation_path
    image_root = hparams.coco_image
    raw_ds = VQADataset_Simple(size=size, prompt=prompt,template=template,annotation_file=annotation_path,image_root=image_root,image_size=336)
    return raw_ds


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "opt_model"):
        if kind == "embed":
            return "opt_model.model.decoder.embed_tokens"
        if kind == "mlp":
            kind = "fc2"
        if kind == "attn":
            kind = "self_attn"
        return f'opt_model.model.decoder.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "llama_model"):
        if kind == "embed":
            return "llama_model.model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'llama_model.model.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "llava_model"):
        if kind == "proj":
            return "llava_model.model.mm_projector"
        if kind == "embed":
            return "llava_model.model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'llava_model.model.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "qwen_model"):
        if kind == "proj":
            return "qwen_model.visual"
        if kind == "embed":
            return "qwen_model.model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'qwen_model.model.layers.{num}{"" if kind is None else "." + kind}'
    
    assert False, "unknown transformer structure"


def init_model_optimizer(model, config):
    import re
    fea_params = [p for n, p in model.named_parameters(
    ) if not bool(re.match('last', n)) and 'bn' not in n and "q_proj" not in n and "v_proj" not in n]

    qv_params = [p for n, p in model.named_parameters(
    ) if ("q_proj" in n or "v_proj" in n) and 'bias' not in n]
    
    qv_bias = [p for n, p in model.named_parameters(
    ) if ("q_proj" in n or "v_proj" in n) and 'bias' in n]
    # cls_params_all = list(
    #     p for n, p in model.named_children() if bool(re.match('last', n)))[0]
    # cls_params = list(cls_params_all[str(task_count+1)].parameters())
    bn_params = [p for n, p in model.named_parameters() if 'bn' in n]
    model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': config.svd_lr,
                                        'thres': config.svd_thres},
                                        {'params': qv_params, 'svd': True, 'lr': config.bn_lr,
                                        'thres': config.svd_thres},
                                        {'params': qv_bias, 'svd': False, 'lr': config.bn_lr,
                                        'thres': config.svd_thres},
                                        # {'params': cls_params, 'weight_decay': 0.0,
                                        #     'lr': config.head_lr},
                                        {'params': bn_params, 'lr': config.bn_lr}],
                            'lr': config.lr,
                            'weight_decay': config.weight_decay}
    if config.model_optimizer in ['SGD', 'RMSprop']:
        model_optimizer_arg['momentum'] = config.momentum
    elif config.model_optimizer in ['Rprop']:
        model_optimizer_arg.pop('weight_decay')
    elif config.model_optimizer in ['amsgrad']:
        if config.model_optimizer == 'amsgrad':
            model_optimizer_arg['amsgrad'] = True
        config.model_optimizer= 'Adam'

    model_optimizer = Adam(**model_optimizer_arg)
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer,
                                                                milestones=config.schedule,
                                                                gamma=config.gamma)
    return model_optimizer, model_scheduler


def collect_xspace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: XSpaceMultimodalHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    print(f"wL:{hparams.wL}, noise: {hparams.noise}")
    global base_pca, reserved
    weights_copy = {}
    collect_sim = []
    if copy:
        model = deepcopy(model)
    requests = deepcopy(requests)
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
        print(
            f"Executing LoRA algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    # image_tok = requests[0]['image_toks']
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    if "image" in requests[0]:
        images = [r["image"] for r in requests]
    # text_lens = [len(tok.encode(prompt+" "+target, add_special_tokens=False)) for prompt, target in zip(texts, targets)]
    # B = (image_tok - hparams.wL) // hparams.wS + 1 + 1 + len(hparams.nS)
    embed_layername = layername(model, 0, "embed")
    proj_layername = layername(model, 0, "proj")
    base_pca = {}
    reserved = {}
    def embed_hook(module, input, output):
        N, dim = output.shape
        wS = hparams.wS
        out = output.clone()
        if count == 0:
            return out
        if count == 2:
            noise_tensor = torch.randn(N, dim, device=output.device) * hparams.noise
            out += noise_tensor
            return out
        elif count%2==0:
            if N > wS:
                start = torch.randint(1, N - wS + 1, (1,)).item()
                end = start + wS
                noise_tensor = torch.randn(wS, dim, device=output.device) * hparams.noise
                out[start:end, :] += noise_tensor
            else:
                noise_tensor = torch.randn(N, dim, device=output.device) * hparams.noise
                out += noise_tensor
            return out

    def proj_hook(module, input, output):
        B, N, dim = output.shape
        wL = hparams.wL
        assert B == 1
        out = output.clone()
        if count == 0:
            return out
        if count == 1:
            out[:,:,:] = 0
            return out
        elif count%2:
            start = torch.randint(0, N - wL + 1, (1,)).item()
            end = start + wL
            out[:, start:end, :] = 0
            return out

    def cov_hook(module, input, output, name):
        global base_pca, reserved
        input = input[0].detach().squeeze(0).data  ## (2048, dim)
        input = input
        input = input/torch.max(input).abs()
        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        covariance = input.t().matmul(input)
        pca = pca_features(covariance.float())
        if base_pca[name] is None:
            base_pca[name] = pca
        else:
            sim = cosine_similarity(base_pca[name], pca)
            collect_sim.append(sim)
            if sim > hparams.sim:
                if torch.isnan(covariance).any():
                    print("nan detected")
                    raise Exception("nan in covariance, break")
                if torch.isinf(covariance).any():
                    print("inf detected")
                    raise Exception("inf in covariance, break")        
                module.covariance_matrix += covariance
                reserved[name]+=1
        del input, covariance
    
    for name, module in model.named_modules():
        if name == proj_layername:
            module.register_forward_hook(proj_hook)
        if name == embed_layername:
            module.register_forward_hook(embed_hook)
        if isinstance(module, nn.Linear):
            if not any(del_name in name for del_name in hparams.delete_name) and any(target in name for target in hparams.update_modules) and any('layers.' + str(layer) in name for layer in hparams.layers):
                module.covariance_matrix = 0
                module.register_forward_hook(partial(cov_hook, name=name))
                base_pca[name] = None
                reserved[name] = 0

    texts = texts*hparams.num_samples
    targets = targets*hparams.num_samples
    images = images*hparams.num_samples
    for i, (txt, tgt, img) in enumerate(tqdm(zip(
                chunks(texts, hparams.batch_size), 
                chunks(targets, hparams.batch_size),
                chunks(images, hparams.batch_size)
        ))):
        full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
        batch = {
            "noise": True,
            "text_input": full_prompt,
            "image": img,
        }
        count = i
        model(batch)

    all_covariance_matrix = {}
    for name, module in model.named_modules():
        if name == proj_layername:
            module._forward_hooks.clear()
        if name == embed_layername:
            module._forward_hooks.clear()
        if isinstance(module, nn.Linear):
            if not any(del_name in name for del_name in hparams.delete_name) and any(target in name for target in hparams.update_modules) and any('layers.' + str(layer) in name for layer in hparams.layers):
                module._forward_hooks.clear()
                if torch.isnan(module.covariance_matrix).any():
                    print("nan detected")
                    raise Exception("nan in covariance")
                if torch.isinf(module.covariance_matrix).any():
                    print("inf detected")
                    raise Exception("inf in covariance")
                module.covariance_matrix = module.covariance_matrix/reserved[name]
                all_covariance_matrix[module.weight] = module.covariance_matrix
        
    return model, weights_copy, collect_sim