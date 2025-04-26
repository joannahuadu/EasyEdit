from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_prediction_acc_real,
    test_prediction_acc_real_multimodal,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1

)



def compute_icl_multimodal_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        # vis_tok,
       
        record: typing.Dict,
        device,
        pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(f"cuda:{hparams.device}")
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(f"cuda:{hparams.device}")

    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(f"cuda:{hparams.device}")
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]

    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _ = multimodal_lm_eval(model, model_name, hparams, tok,
                                             target, prompt, image)
    else:
        edit_acc, _ = multimodal_lm_eval(model, model_name, hparams, tok,
                                             target, new_fact, image)
    ret = {
        f"rewrite_acc": edit_acc
    }
    if rephrase is not None:
        rephrase_acc, _ = multimodal_lm_eval(model, model_name, hparams, tok,
                                                 target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc

    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = multimodal_lm_eval(model, model_name, hparams, tok,
                                                       target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc

    if "locality_prompt" in record.keys():
        if pre_edit:
            _, _, locality_output = multimodal_lm_eval(model, model_name, hparams, tok,
                                                           loc_a, loc_q, None, is_loc=True)
        else:
            _, _, locality_output = multimodal_lm_eval(model, model_name, hparams, tok,
                                                           loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None, is_loc=True)
        ret['locality_output'] = locality_output

    if "multimodal_locality_image" in record.keys():
        if pre_edit:
            _, _, locality_image_output = multimodal_lm_eval(model, model_name, hparams, tok,
                                                                 m_loc_a, m_loc_q, m_loc_image, is_loc=True)
        else:
            _, _, locality_image_output = multimodal_lm_eval(model, model_name, hparams, tok,
                                                                 m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image, is_loc=True)
        ret['multimodal_locality_output'] = locality_image_output

    return ret

# for real-world evaluation, rewrite/rephrase, locality, portability
def compute_rewrite_or_rephrase_quality_multimodal(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    edit_prompt: dict,
    device: int = 0,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
    rephrase_image: bool = False,
    key: str = None
) -> typing.Dict:
    if key:
        key = key
    else:
        if not test_rephrase:
            key = 'rewrite'
        else:
            if rephrase_image:
                key = 'rephrase_image'
            else:   
                key = 'rephrase'
    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    
    acc, gen_content = test_prediction_acc_real_multimodal(model, tok, hparams, edit_prompt=edit_prompt, device=device, locality=False)
    ret = {
        f"{key}_rel_acc": acc,
        f"{key}_gen_content": gen_content
    }
    
    return ret


def compute_locality_quality_multimodal(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    edit_prompt: dict,
    device: int = 0,
    key: str = 'locality',
) -> typing.Dict:

    # using real-world evaluation
    loc_tokens = test_prediction_acc_real_multimodal(model, tok, hparams, edit_prompt=edit_prompt, device=device, locality=True)
    

    ret = {
        f"{key}_rel_output": loc_tokens
    }
    return ret

def compute_portability_quality_multimodal(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    edit_prompt: dict,
    device: int = 0,
    key: str = 'portability',
) -> typing.Dict:

    # using real-world evaluation
    
    acc, gen_content = test_prediction_acc_real_multimodal(model, tok, hparams, edit_prompt=edit_prompt, device=device, locality=False)

    ret = {
        f"{key}_rel_acc": acc,
        f"{key}_gen_content": gen_content
    }
    return ret
                               
def multimodal_lm_eval(
    model,
    model_name,
    hparams: HyperParams,
    tokenizer,
    target,
    x,
    image,
    is_loc=False,
    neighborhood=False,
    prompt_template="{}",
    real_world_eval=False)-> typing.Dict:
    device = torch.device(f"cuda:{hparams.device}")

    samples = prepare_multimodal_edit(hparams, tokenizer, target, [x], image, prompt_template=prompt_template)

    # return compute_multimodal_edit_quality(model, samples,
    #                                        hparams.exact_match) if not is_loc else compute_multimodal_edit_quality_demo(
    #     model, samples)
    return compute_multimodal_edit_quality_demo_mmke(model, samples,
                                           tokenizer)
    

def multimodal_decode(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:    
    batch = prepare_multimodal_edit(hparams, tokenizer, target, [x], image) 
    
    with torch.no_grad():
        if "qwen" in model.__class__.__name__.lower():
            outputs = model(batch['inputs'].to(f"cuda:{hparams.device}"))
        elif "owl" in model.__class__.__name__.lower():
            input_ids, image = batch['input_ids'], batch['image']
            # from torch.cuda.amp import autocast
            # with autocast():
            outputs = model(input_ids.to(f"cuda:{hparams.device}"), 
                                        images=image.to(hparams.device, dtype=torch.float16))
        else:
            outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    else:
        raise ValueError("logits should have 3 dimensions")
        
   
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    predict = tokenizer.decode(pred_ids.tolist()[0], skip_special_tokens=True)
    return predict                                          


def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image,
                            prompt_template="{}"):
    if isinstance(target, str):
        target = [target, ]
    if isinstance(prompts, str):
        prompts = [prompts, ]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    # text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    text_input=prompts
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        # target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        # target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False,
        #              return_tensors="pt", )["input_ids"]

    ret = {
        'text_input': text_input,
        'image': image,
        'answer': target,
        'prompts_len': prompts_len,
        'prompt_template': prompt_template,
        'ori_text_input': text_input,
    }
    return ret

def prepare_multimodal_edit_unike(hparams,
                            tok,
                            target,
                            prompts,
                            image,
                            prompt_template="{}"):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_template.format(prompt_) + ' ' + target_  for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt_template.format(prompt), add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt_template.format(prompt),  add_special_tokens=False)) for prompt in prompts]  
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len,
        'noise': True,
    } 
    return ret


def compute_multimodal_edit_quality(model, batch, exact_match=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = outputs.labels.detach().cpu()
    logits_copy = logits.clone()
    targ_copy = targ.clone()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    if exact_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    return acc, logits_copy,targ_copy

def compute_multimodal_edit_quality_demo(model, batch, tok):
    prompts = [batch['prompt_template'].format(prompt) for prompt in batch['text_input']]
    targets = batch['answer']
    target_ids = [tok.encode(target, return_tensors="pt", add_special_tokens=False)[0] for target in targets]
    prompt_target = [prompt + ' ' + tok.decode(target) for prompt, target in zip(prompts, target_ids)]
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch['text_input'] = prompt_target
    batch['noise'] = True
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()
        answers = torch.argmax(logits, dim=-1).squeeze()[:-1].detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze()[1:].detach().cpu().numpy().tolist()
        answers = answers[-len(target_ids[0]):]
        labels = labels[-len(target_ids[0]):]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, answers
        else:
            return [np.mean(np.equal(answers, labels))], answers
def compute_multimodal_edit_quality_demo_mmke(model, batch, tok):
    prompts = [batch['prompt_template'].format(prompt) for prompt in batch['text_input']]
    targets = batch['answer']
    target_ids = [tok.encode(target, return_tensors="pt", add_special_tokens=False)[0] for target in targets]
    prompt_target = [prompt + ' ' + tok.decode(target) for prompt, target in zip(prompts, target_ids)]
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch['text_input'] = prompt_target
    batch['noise'] = True
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()
        answers = torch.argmax(logits, dim=-1).squeeze()[:-1].detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze()[1:].detach().cpu().numpy().tolist()
        answers = answers[-len(target_ids[0]):]
        labels = labels[-len(target_ids[0]):]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, answers, batch
        else:
            return [np.mean(np.equal(answers, labels))], answers, batch

# def compute_multimodal_edit_quality_demo(model, batch):
#     with torch.no_grad():
#         outputs = model(batch)
#         if isinstance(outputs, torch.Tensor):
#             logits = outputs.detach().cpu()
#         else:
#             logits = outputs.logits.detach().cpu()
#             # targ = outputs.labels.detach().cpu()
#         targ = batch["labels"].cpu()
#     logits_ = logits.clone()
#     if logits.dim() == 3:
#         logits = logits[:, :-1]
#         # targ = targ[:, 1:]
#         logits = logits[:, -targ.shape[1]:]
#     mask = targ != -100
#     targ[~mask] = 0
#     pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
#     correct = pred_ids == targ
#     correct = correct & mask
#     num_non_padding = mask.sum().float().item()
#     acc = correct.sum() / num_non_padding

#     return acc, pred_ids.numpy(), logits_

def test_generation_quality(model, batch):
    if 'noise' in batch and batch['noise']:
        batch['noise'] = False
    if 'ori_text_input' in batch:
        batch['text_input'] = batch['ori_text_input']
    return model.generate(batch, num_beams=1, max_new_tokens=100)

def compute_multimodal_edit_results(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device,
        real_world_eval: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.

    target = record["target"]
    rewrite_prompts = record["prompt"]
    if record["image"] is not None:
        image = record["image"] if record["image"].is_cuda else record["image"].to(f"cuda:{hparams.device}")
    else:
        image = record["image"]
    prompt_template = record["prompt_template"] if "prompt_template" in record else "{}"
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image, prompt_template=prompt_template)
    if real_world_eval:
        ret = compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_inner, device=device, test_rephrase=False)
        ret['rewrite_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_inner, tok)
    else:
        ret['rewrite_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_inner, tok)
        # ret['rewrite_gen'] = test_generation_quality(model, edit_inner)

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
                compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_outer, device=device, test_rephrase=True, rephrase_image=False)
            )
            ret['rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_outer, tok)
        else:
            ret['rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_outer, tok)
            # ret['rephrase_gen'] = test_generation_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(f"cuda:{hparams.device}")
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_image_outer, device=device, test_rephrase=True, rephrase_image=True)
        )
            ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_image_outer, tok)
        else:   
            ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_image_outer, tok)
            # ret['image_rephrase_gen'] = test_generation_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_locality_quality_multimodal(model, model_name, hparams, tok, edit_prompt=locality, device=device, key='locality')
        )
            ret['locality_acc'], ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality, tok)
        else:
            ret['locality_acc'], ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality, tok)
            # ret['locality_gen'] = test_generation_quality(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        if m_loc_image is not None:
            m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(f"cuda:{hparams.device}")
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_locality_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_locality, key='multimodal_locality')
        )
            ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, m_locality, tok)
            
        else:
            ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, m_locality, tok)
            # ret['multimodal_locality_gen'] = test_generation_quality(model, m_locality)

    if 'portability_prompt' in record.keys():
        portability_prompt = record["portability_prompt"]
        portability_ground_truth = record["portability_ground_truth"]
        portability_image = record["portability_image"]
        if portability_image is not None:
            portability_image = portability_image if portability_image.is_cuda else portability_image.to(f"cuda:{hparams.device}")
        portability = prepare_multimodal_edit(hparams, tok, portability_ground_truth, portability_prompt, portability_image, prompt_template=prompt_template)
        # _, ret['portability_output'] = compute_multimodal_edit_quality_demo(model, portability, tok)
        if real_world_eval:
            ret.update(
            compute_portability_quality_multimodal(model, model_name, hparams, tok, edit_prompt=portability, device=device, key='portability')
        )
        else:
            ret['portability_gen'] = test_generation_quality(model, portability)

    if 'multimodal_portability_prompt' in record.keys():
        m_port_prompt = record["multimodal_portability_prompt"]
        m_port_ground_truth = record["multimodal_portability_ground_truth"]
        m_port_image = record["multimodal_portability_image"]
        if m_port_image is not None:
            m_port_image = m_port_image if m_port_image.is_cuda else m_port_image.to(f"cuda:{hparams.device}")
        m_portability = prepare_multimodal_edit(hparams, tok, m_port_ground_truth, m_port_prompt, m_port_image, prompt_template=prompt_template)
        # _, ret['multimodal_portability_output'] = compute_multimodal_edit_quality_demo(model, m_portability, tok)
        if real_world_eval:
            ret.update(
            compute_portability_quality_multimodal(model, model_name, hparams, tok, edit_prompt=portability, device=device, key='multimodal_portability')
        )
        else:
            ret['multimodal_portability_gen'] = test_generation_quality(model, m_portability)
    # Form a list of lists of prefixes to test.

    return ret


def compute_multimodal_edit_results_demo(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.

    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(f"cuda:{hparams.device}")

    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(f"cuda:{hparams.device}")
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(f"cuda:{hparams.device}")
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits

    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return \
        torch.mean((trg_tok['input_ids'][:, :-1] == ans[:, :-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]
def process_predict(logits, labels, tok):
    
    if logits.dim() == 3:
        logits = logits[:, :-1]
        logits = logits[:, -labels.shape[1]:]
    
    
    mask = labels != -100
    labels[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    predict = tok.decode(pred_ids.tolist()[0], skip_special_tokens=True)
    return predict
def compute_mmke_multimodal_edit_quality(
    pre_model,
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    real_world_eval: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image

    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    prompt_template = record["prompt_template"] if "prompt_template" in record else "{}"
    image = record['image'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['image']) and not record['image'].is_cuda else record['image']
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image.to(f"cuda:{hparams.device}") if torch.is_tensor(rephrase_image) and not rephrase_image.is_cuda else rephrase_image
    ret = {}


    ###############################################################################
    knowledge_type = record["knowledge_type"]

    if knowledge_type ==0 or knowledge_type ==1:
        rel_prompt_1 = record["rel_prompt_1"]
        rel_ground_truth_1 = record["rel_ground_truth_1"]
        rel_prompt_2 = record["rel_prompt_2"]
        rel_ground_truth_2 = record["rel_ground_truth_2"]
        
        m_rel_prompt_1 = record["m_rel_prompt_1"]
        m_rel_ground_truth_1 = record["m_rel_ground_truth_1"]
        m_rel_prompt_2 = record["m_rel_prompt_2"]
        m_rel_ground_truth_2 = record["m_rel_ground_truth_2"]
    elif knowledge_type ==2:
        rel_prompt = record["rel_prompt"]
        rel_ground_truth = record["rel_ground_truth"]
       
        m_rel_prompt = record["m_rel_prompt"]
        m_rel_ground_truth = record["m_rel_ground_truth"]

        image_rephrase_question = record["image_rephrase_question"]
        one_hop_img = record['one_hop_img'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['one_hop_img']) and not record['one_hop_img'].is_cuda else record['one_hop_img']
    ###############################################################################

    ###############################################################
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record['multimodal_locality_image'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['multimodal_locality_image']) and not record['multimodal_locality_image'].is_cuda else record['multimodal_locality_image']
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]

    edit_acc, rewrite_pred, rewrite_samples = multimodal_lm_eval(model, model_name, hparams, tok,
                                              target, prompt, image, prompt_template=prompt_template, real_world_eval=real_world_eval)
        
    ret = {
        f"rewrite_acc": edit_acc
    }

    ret['rewrite_acc_prompt'] = record["prompt"]
    ret['rewrite_acc_ground_truth']  = record["target"]
    # ret['rewrite_acc_predict'] = multimodal_decode(model, model_name, hparams, tok, target, prompt, None)
    ret['rewrite_acc_predict'] = test_generation_quality(model,rewrite_samples)
    
    if rephrase is not None:
        rephrase_acc, rephrase_pred, rephrase_samples = multimodal_lm_eval(model, model_name, hparams, tok,
                               target, rephrase, image, prompt_template=prompt_template)
        ret['rephrase_acc'] = rephrase_acc

    ret['rephrase_acc_prompt'] = rephrase
    ret['rephrase_acc_ground_truth']  = record["target"]
    ret['rephrase_acc_predict'] = test_generation_quality(model, rephrase_samples)

        
    if "image_rephrase" in record.keys():
        rephrase_image_acc, rephrase_image_pred, rephrase_img_samples = multimodal_lm_eval(model, model_name, hparams, tok,
                               target, prompt, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
    
    ret['rephrase_image_acc_prompt']   = record["prompt"]
    ret['rephrase_image_acc_ground_truth']  = record["target"]
    ret['rephrase_image_acc_predict']  = test_generation_quality(model, rephrase_img_samples)
    
    ret["memory_alloc_max"] = torch.cuda.max_memory_allocated()
    ret["memory_res_max"] = torch.cuda.max_memory_reserved()



    if "locality_prompt" in record.keys():
        locality_samples = prepare_multimodal_edit(hparams, tok, loc_a, [loc_q], None, prompt_template=prompt_template)
        pre_text_loc_logits = pre_model(locality_samples).logits
        post_text_loc_logits = model(locality_samples).logits

        ret['loc_acc_prompt']  = record["locality_prompt"]
        ret['loc_acc_ground_truth']  = record["locality_ground_truth"]
        ret['loc_acc_predict'] = test_generation_quality(model, locality_samples)

        pre_text_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(pre_text_loc_logits.float(), dim=-1), k=1, dim=-1).indices
        post_text_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_text_loc_logits.float(), dim=-1), k=1, dim=-1).indices

        locality_acc = sum(post_text_loc_logits_softmax_top_k.view(-1) == pre_text_loc_logits_softmax_top_k.view(-1))/post_text_loc_logits_softmax_top_k.view(-1).shape[0]

        ret['locality_acc'] = locality_acc
    
    if "multimodal_locality_image" in record.keys():
        locality_image_samples = prepare_multimodal_edit(hparams, tok, m_loc_a, [m_loc_q], m_loc_image, prompt_template=prompt_template)
        pre_image_loc_logits = pre_model(locality_image_samples).logits
        post_image_loc_logits = model(locality_image_samples).logits

        ret['mm_loc_acc_prompt']  = record["multimodal_locality_prompt"]
        ret['mm_loc_acc_ground_truth']  = record["multimodal_locality_ground_truth"]
        ret['mm_loc_acc_predict'] = test_generation_quality(model, locality_image_samples)

        pre_image_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(pre_image_loc_logits.float(), dim=-1), k=10, dim=-1).indices
        post_image_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_loc_logits.float(), dim=-1), k=10, dim=-1).indices

        locality_image_acc = sum(post_image_loc_logits_softmax_top_k.view(-1) == pre_image_loc_logits_softmax_top_k.view(-1))/post_image_loc_logits_softmax_top_k.view(-1).shape[0]

        ret['locality_image_acc'] = locality_image_acc
###################################################################

    if knowledge_type ==0 or knowledge_type ==1:
        if knowledge_type ==0:
            ret['knowledge_type'] = 0
        elif knowledge_type ==1:
            ret['knowledge_type'] = 1
    
        rel_prompt_1_acc, rel_prompt_1_pred, rel_prompt_1_samples  = multimodal_lm_eval(model, model_name, hparams, tok,rel_ground_truth_1, rel_prompt_1, None, prompt_template=prompt_template)   
        ret['rel_prompt_1_acc'] = rel_prompt_1_acc
        
        ret['rel_1_acc_prompt'] = record["rel_prompt_1"]
        ret['rel_1_acc_ground_truth'] = record["rel_ground_truth_1"]
        ret['rel_1_acc_predict'] = test_generation_quality(model, rel_prompt_1_samples)
        

        rel_prompt_2_acc, rel_prompt_2_pred, rel_prompt_2_samples = multimodal_lm_eval(model, model_name, hparams, tok,rel_ground_truth_2, rel_prompt_2, None, prompt_template=prompt_template)  
        ret['rel_prompt_2_acc'] = rel_prompt_2_acc

        ret['rel_2_acc_prompt'] = record["rel_prompt_2"]
        ret['rel_2_acc_ground_truth']  = record["rel_ground_truth_2"]
        ret['rel_2_acc_predict'] = test_generation_quality(model, rel_prompt_2_samples)

        m_rel_prompt_1_image_acc, m_rel_prompt_1_pred, m_rel_prompt_1_samples = multimodal_lm_eval(model, model_name, hparams, tok, m_rel_ground_truth_1, m_rel_prompt_1, image, prompt_template=prompt_template)
        ret['m_rel_prompt_1_image_acc'] = m_rel_prompt_1_image_acc

        ret['m_rel_1_acc_prompt'] = record["m_rel_prompt_1"]
        ret['m_rel_1_acc_ground_truth'] = record["m_rel_ground_truth_1"]
        ret['m_rel_1_acc_predict']  = test_generation_quality(model,m_rel_prompt_1_samples)



        m_rel_prompt_2_image_acc, m_rel_prompt_2_pred, m_rel_prompt_2_samples = multimodal_lm_eval(model, model_name, hparams, tok, m_rel_ground_truth_2, m_rel_prompt_2, image, prompt_template=prompt_template)
        ret['m_rel_prompt_2_image_acc'] = m_rel_prompt_2_image_acc

        ret['m_rel_2_acc_prompt'] = record["m_rel_prompt_2"]
        ret['m_rel_2_acc_ground_truth']   = record["m_rel_ground_truth_2"]
        ret['m_rel_2_acc_predict']  = test_generation_quality(model, m_rel_prompt_2_samples)
        
        m_rel_prompt_1_image_rephrase_acc, m_rel_prompt_1_image_rephrase_pred, m_rel_prompt_1_image_rephrase_samples = multimodal_lm_eval(model, model_name, hparams, tok, m_rel_ground_truth_1, m_rel_prompt_1, rephrase_image, prompt_template=prompt_template)
        ret['m_rel_prompt_1_image_rephrase_acc'] = m_rel_prompt_1_image_rephrase_acc

        ret['m_rel_1_image_rephrase_acc_prompt'] = record["m_rel_prompt_1"]
        ret['m_rel_1_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth_1"]
        ret['m_rel_1_image_rephrase_acc_predict'] =  test_generation_quality(model, m_rel_prompt_1_image_rephrase_samples)

        m_rel_prompt_2_image_rephrase_acc, m_rel_prompt_2_image_rephrase_pred, m_rel_prompt_2_image_rephrase_samples = multimodal_lm_eval(model, model_name, hparams, tok, m_rel_ground_truth_2, m_rel_prompt_2, rephrase_image, prompt_template=prompt_template)
        ret['m_rel_prompt_2_image_rephrase_acc'] = m_rel_prompt_2_image_rephrase_acc

        ret['m_rel_2_image_rephrase_acc_prompt']  = record["m_rel_prompt_2"]
        ret['m_rel_2_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth_2"]
        ret['m_rel_2_image_rephrase_acc_predict'] = test_generation_quality(model, m_rel_prompt_2_image_rephrase_samples)
        
    elif knowledge_type ==2:
        
        ret['knowledge_type'] = 2
        rel_prompt_acc, rel_prompt_pred, rel_prompt_samples  = multimodal_lm_eval(model, model_name, hparams, tok,rel_ground_truth, rel_prompt, None, prompt_template=prompt_template)
        ret['rel_prompt_acc'] = rel_prompt_acc

        ret['rel_acc_prompt']  = record["rel_prompt"]
        ret['rel_acc_ground_truth'] = record["rel_ground_truth"]
        ret['rel_acc_predict'] = test_generation_quality(model, rel_prompt_samples)

        m_rel_prompt_image_acc, m_rel_prompt_image_pred, m_rel_prompt_image_samples = multimodal_lm_eval(model, model_name, hparams, tok,m_rel_ground_truth, m_rel_prompt, image, prompt_template=prompt_template)
        ret['m_rel_prompt_image_acc'] = m_rel_prompt_image_acc

        ret['m_rel_acc_prompt'] = record["m_rel_prompt"]
        ret['m_rel_acc_ground_truth'] = record["m_rel_ground_truth"]
        ret['m_rel_acc_predict'] = test_generation_quality(model, m_rel_prompt_image_samples)

        m_rel_prompt_image_rephrase_acc, m_rel_prompt_image_rephrase_pred, m_rel_prompt_image_rephrase_samples = multimodal_lm_eval(model, model_name, hparams, tok, m_rel_ground_truth, m_rel_prompt, rephrase_image)
        ret['m_rel_prompt_image_rephrase_acc'] = m_rel_prompt_image_rephrase_acc

        ret['m_rel_image_rephrase_acc_prompt']  = record["m_rel_prompt"]
        ret['m_rel_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth"]
        ret['m_rel_image_rephrase_acc_predict'] = test_generation_quality(model, m_rel_prompt_image_rephrase_samples)


    ######### portability #########


    if "portability_prompt" in record.keys():
        # assert len(record['portability_prompt'])==1, "Portability evaluation only has one prompt at a time"
        port_acc = 0
        if knowledge_type ==0 or knowledge_type ==1:
            for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                port_acc_i, pred_targ_ids, port_samples = multimodal_lm_eval(model, model_name, hparams, tok, port_a, port_q, image)
                port_acc += port_acc_i[0]
            ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
            # ret['pred_ids'] = pred_targ_ids[0].tolist()
            # ret['targ_ids'] = pred_targ_ids[1].tolist()

            ret['port_prompt'] = record["portability_prompt"]
            ret['port_ground_truth'] = record["portability_ground_truth"]
            # ret['port_predict'] = tok.decode(pred_targ_ids[0][0], skip_special_tokens=True)

        elif knowledge_type ==2:
            for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                port_acc_i, pred_targ_ids, port_samples = multimodal_lm_eval(model, model_name, hparams, tok, port_a, port_q, one_hop_img)
                port_acc += port_acc_i[0]
            ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
            # ret['pred_ids'] = pred_targ_ids[0].tolist()
            # ret['targ_ids'] = pred_targ_ids[1].tolist()
            ret['port_prompt'] = record["portability_prompt"]
            ret['port_ground_truth'] = record["portability_ground_truth"]
            # ret['port_predict'] = tok.decode(pred_targ_ids[0][0], skip_special_tokens=True)
    import json
    import os
    
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()} 
        elif isinstance(obj, list):
            return [tensor_to_list(i) for i in obj]  
        return obj  


    ret_serializable = tensor_to_list(ret)
    from datetime import datetime
    hparams.results_dir = f"./results/{hparams.data_type}"
    model_dir = os.path.join(hparams.results_dir, "MEMIT")
    decode_path = f'{model_dir}/{hparams.model_name}.json'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    if os.path.exists(decode_path):
        with open(decode_path, 'r') as json_file:
            try:
                data = json.load(json_file)  
            except json.JSONDecodeError:
                data = []  
    else:
        data = []  
    data.append(ret_serializable)

    
    with open(decode_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  
        json_file.write("\n")  
    ######### portability #########

    return ret

def compute_mmke_multimodal_edit_quality_rel(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    real_world_eval: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    prompt_template = record["prompt_template"] if "prompt_template" in record else "{}"
    image = record['image'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['image']) and not record['image'].is_cuda else record['image']
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image.to(f"cuda:{hparams.device}") if torch.is_tensor(rephrase_image) and not rephrase_image.is_cuda else rephrase_image
    ret = {}


    ###############################################################################
    knowledge_type = record["knowledge_type"]

    if knowledge_type ==0 or knowledge_type ==1:
        rel_prompt_1 = record["rel_prompt_1"]
        rel_ground_truth_1 = record["rel_ground_truth_1"]
        rel_prompt_2 = record["rel_prompt_2"]
        rel_ground_truth_2 = record["rel_ground_truth_2"]
        
        m_rel_prompt_1 = record["m_rel_prompt_1"]
        m_rel_ground_truth_1 = record["m_rel_ground_truth_1"]
        m_rel_prompt_2 = record["m_rel_prompt_2"]
        m_rel_ground_truth_2 = record["m_rel_ground_truth_2"]
    elif knowledge_type ==2:
        rel_prompt = record["rel_prompt"]
        rel_ground_truth = record["rel_ground_truth"]
       
        m_rel_prompt = record["m_rel_prompt"]
        m_rel_ground_truth = record["m_rel_ground_truth"]

        image_rephrase_question = record["image_rephrase_question"]
        one_hop_img = record['one_hop_img'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['one_hop_img']) and not record['one_hop_img'].is_cuda else record['one_hop_img']
    ###############################################################################

    ###############################################################
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record['multimodal_locality_image'].to(f"cuda:{hparams.device}") if torch.is_tensor(record['multimodal_locality_image']) and not record['multimodal_locality_image'].is_cuda else record['multimodal_locality_image']
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]
    edit_inner = prepare_multimodal_edit(
        hparams, tok, target, prompt, image, prompt_template=prompt_template)
    
    if real_world_eval:
        ret = compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_inner, device=device, test_rephrase=False)
        ret['rewrite_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_inner, tok)
    else:
        ret['rewrite_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_inner, tok)
    
    if rephrase:
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase, image, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
                compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_outer, device=device, test_rephrase=True, rephrase_image=False)
            )
            ret['rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_outer, tok)
        else:
            ret['rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_outer, tok)
        
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(f"cuda:{hparams.device}")
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, prompt, rephrase_image, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=edit_image_outer, device=device, test_rephrase=True, rephrase_image=True)
        )
            ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_image_outer, tok)
        else:   
            ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_image_outer, tok)
    
    ret["memory_alloc_max"] = torch.cuda.max_memory_allocated()
    ret["memory_res_max"] = torch.cuda.max_memory_reserved()



    if "locality_prompt" in record.keys():
        locality_samples = prepare_multimodal_edit(hparams, tok, loc_a, [loc_q], None, prompt_template=prompt_template)
        
        if real_world_eval:
            ret.update(
            compute_locality_quality_multimodal(model, model_name, hparams, tok, edit_prompt=locality_samples, device=device, key='locality')
        )
            ret['locality_acc'], ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality_samples, tok)
        else:
            ret['locality_acc'], ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality_samples, tok)
        
    if "multimodal_locality_image" in record.keys():
        locality_image_samples = prepare_multimodal_edit(hparams, tok, m_loc_a, [m_loc_q], m_loc_image, prompt_template=prompt_template)

        if real_world_eval:
            ret.update(
            compute_locality_quality_multimodal(model, model_name, hparams, tok, edit_prompt=locality_image_samples, key='multimodal_locality')
        )
            ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, locality_image_samples, tok)
            
        else:
            ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, locality_image_samples, tok)
###################################################################

    if knowledge_type ==0 or knowledge_type ==1:
        if knowledge_type ==0:
            ret['knowledge_type'] = 0
        elif knowledge_type ==1:
            ret['knowledge_type'] = 1
        
        rel_prompt_1_samples = prepare_multimodal_edit(hparams, tok, rel_ground_truth_1, [rel_prompt_1], None, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=rel_prompt_1_samples, device=device, test_rephrase=True, rephrase_image=True, key='rel_prompt_1')
        )
            ret['rel_prompt_1_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_1_samples, tok)
        else:   
            ret['rel_prompt_1_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_1_samples, tok)

        rel_prompt_2_samples = prepare_multimodal_edit(hparams, tok, rel_ground_truth_2, [rel_prompt_2], None, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=rel_prompt_2_samples, device=device, test_rephrase=True, rephrase_image=True, key='rel_prompt_2')
        )
            ret['rel_prompt_2_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_2_samples, tok)
        else:   
            ret['rel_prompt_2_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_2_samples, tok)


        m_rel_prompt_1_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth_1, [m_rel_prompt_1], image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_1_samples, device=device, test_rephrase=True, rephrase_image=True, key='m_rel_prompt_1')
        )
            ret['m_rel_prompt_1_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_1_samples, tok)
        else:   
            ret['m_rel_prompt_1_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_1_samples, tok)

        m_rel_prompt_2_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth_2, [m_rel_prompt_2], image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_2_samples, device=device, test_rephrase=True, rephrase_image=True, key='m_rel_prompt_2')
        )
            ret['m_rel_prompt_2_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_2_samples, tok)
        else:   
            ret['m_rel_prompt_2_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_2_samples, tok)

        m_rel_prompt_1_image_rephrase_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth_1, [m_rel_prompt_1], rephrase_image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_1_image_rephrase_samples, device=device, test_rephrase=True, rephrase_image=True, key='m_rel_prompt_1_image_rephrase')
        )
            ret['m_rel_1_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_1_samples, tok)
        else:   
            ret['m_rel_1_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_1_samples, tok)

        m_rel_prompt_2_image_rephrase_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth_2, [m_rel_prompt_2], rephrase_image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_2_image_rephrase_samples, device=device, test_rephrase=True, rephrase_image=True, key='m_rel_prompt_2_image_rephrase')
        )
            ret['m_rel_2_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_2_samples, tok)
        else:   
            ret['m_rel_2_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_2_samples, tok)

        
    elif knowledge_type ==2:
        
        ret['knowledge_type'] = 2
        rel_prompt_samples = prepare_multimodal_edit(hparams, tok, rel_ground_truth, [rel_prompt], None, prompt_template=prompt_template)
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=rel_prompt_samples, device=device, test_rephrase=True, rephrase_image=True, key='rel_prompt')
        )
            ret['rel_prompt_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_samples, tok)
        else:   
            ret['rel_prompt_acc'], _ = compute_multimodal_edit_quality_demo(model, rel_prompt_samples, tok)

        
        m_rel_prompt_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth, [m_rel_prompt], image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_samples, device=device, test_rephrase=True, rephrase_image=True, key=m_rel_prompt)
        )
            ret['m_rel_prompt_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_samples, tok)
        else:   
            ret['m_rel_prompt_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_samples, tok)

        m_rel_prompt_image_rephrase_samples = prepare_multimodal_edit(hparams, tok, m_rel_ground_truth, [m_rel_prompt], rephrase_image, prompt_template=prompt_template) 
        if real_world_eval:
            ret.update(
            compute_rewrite_or_rephrase_quality_multimodal(model, model_name, hparams, tok, edit_prompt=m_rel_prompt_image_rephrase_samples, device=device, test_rephrase=True, rephrase_image=True, key='m_rel_prompt_image_rephrase')
        )
            ret['m_rel_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_samples, tok)
        else:   
            ret['m_rel_image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, m_rel_prompt_samples, tok)

    ######### portability #########


    if "portability_prompt" in record.keys():
        # assert len(record['portability_prompt'])==1, "Portability evaluation only has one prompt at a time"
        port_acc = 0
        port_acc_rel = 0
        if knowledge_type ==0 or knowledge_type ==1:
            for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                if real_world_eval:
                    port_samples = prepare_multimodal_edit(hparams, tok, port_a, [port_q], image, prompt_template=prompt_template)
                    port_acc_i_rel = compute_portability_quality_multimodal(model, model_name, hparams, tok, edit_prompt=port_samples, device=device, key='portability')['portability_rel_acc']
                    port_acc_i, _ = compute_multimodal_edit_quality_demo(model, port_samples, tok)
                    port_acc += port_acc_i[0]
                    port_acc_rel += port_acc_i_rel
                else:   
                    port_acc_i, _ = compute_multimodal_edit_quality_demo(model, port_samples, tok)
                    port_acc += port_acc_i[0]
            if real_world_eval:
                ret['portability_rel_acc'] = port_acc_rel/len(record['portability_prompt'])
                ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
            else:
                ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
        elif knowledge_type ==2:
            for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                if real_world_eval:
                    port_samples = prepare_multimodal_edit(hparams, tok, port_a, [port_q], one_hop_img, prompt_template=prompt_template)
                    port_acc_i_rel = compute_portability_quality_multimodal(model, model_name, hparams, tok, edit_prompt=port_samples, device=device, key='portability')['portability_rel_acc']
                    port_acc_i, _ = compute_multimodal_edit_quality_demo(model, port_samples, tok)
                    port_acc += port_acc_i[0]
                    port_acc_rel += port_acc_i_rel
                else:   
                    port_acc_i, _ = compute_multimodal_edit_quality_demo(model, port_samples, tok)
                    port_acc += port_acc_i[0]
            if real_world_eval:
                ret['portability_rel_acc'] = port_acc_rel/len(record['portability_prompt'])
                ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
            else:
                ret['portability_acc'] = [port_acc/len(record['portability_prompt'])]
    import json
    import os
    
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()} 
        elif isinstance(obj, list):
            return [tensor_to_list(i) for i in obj]  
        return obj  


    ret_serializable = tensor_to_list(ret)
    from datetime import datetime
    hparams.results_dir = f"./results/{hparams.data_type}"
    model_dir = os.path.join(hparams.results_dir, "MEMIT")
    decode_path = f'{model_dir}/{hparams.model_name}.json'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    if os.path.exists(decode_path):
        with open(decode_path, 'r') as json_file:
            try:
                data = json.load(json_file)  
            except json.JSONDecodeError:
                data = []  
    else:
        data = []  
    data.append(ret_serializable)

    
    with open(decode_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  
        json_file.write("\n")  
    ######### portability #########

    return ret
    