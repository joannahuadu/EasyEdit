from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo,
                        compute_mmke_multimodal_edit_quality_rel,
                        test_locality_real_multimodal) 
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
import pprint

from .utils import _chunks, load_object, save_object
import random
import math
import copy
import gc

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor:
    """Multimodal editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT
                
                model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    cache_dir=hparams.cache_dir
                )
                self.prompt = "Question: {} Short answer:"
                # self.prompt = "{}"
                self.prompt_template = "{}"
                self.image_toks = 32
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "minigpt4":
                from ..trainer.minigpt4_models import MiniGPT4
                prompt_template = 'USER: {} ASSISTANT:' # For multi-modal input
                # prompt_template="{}" # For pure text input
                end_sym = "###"
                model = MiniGPT4(
                    vit_model="eva_clip_g",
                    q_former_model=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    prompt_template=prompt_template,
                    end_sym=end_sym,
                    llama_model=hparams.name,
                    vit_ckpt=hparams.state_dict_file,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                    cache_dir=hparams.cache_dir,
                )
                self.prompt = "<Img> <ImageHere> </Img>{} Answer in a single word."
                self.prompt_template = prompt_template
                self.image_toks = 32
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "llava":
                from ..trainer.llava_models import LLavaModel
                from ..trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
                system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
                prompt_template = system + 'USER: {} ASSISTANT:'
                if isinstance(hparams.device, str):
                    model = LLavaModel(
                    llava_model=hparams.name,
                    prompt_template=prompt_template,
                    device_map="auto",
                    cache_dir=hparams.cache_dir)
                else:
                    model = LLavaModel(
                        llava_model=hparams.name,
                        prompt_template=prompt_template,
                        device_map="cuda:{}".format(hparams.device),
                        cache_dir=hparams.cache_dir,
                    )
                self.prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
                self.prompt_template = prompt_template
                self.image_toks = 576 - 1
                # Get vis_processor
                vis_processor = model.image_processor
            self.model = model
            self.vis_tok = vis_processor
            if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                tok_name = (
                    hparams.tokenizer_name
                    if hparams.tokenizer_name is not None
                    else hparams.name
                )
                tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                    tok_name
                )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token    
                self.tok = tokenizer                         
        else:
            self.model, self.tok = self.model_name
        
        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image
        if self.alg_name == 'UNIKE':
            from ..models.unike.src import Editor
            self.editor = Editor(
                            model=model,
                            hidden_size=hparams.hidden_dim,
                            max_add_neuron_num=hparams.max_add_neuron_num,
                            freeze_model=hparams.freeze_model, freeze_k=hparams.freeze_k, freeze_a=hparams.freeze_a,
                            memory_size=hparams.memory_size, memory_loss=hparams.memory_loss,
                            amplify_v=hparams.amplify_v, activate_loss=hparams.activate_loss,
                            act_margin_val=hparams.act_margin_val, margin_val1=hparams.margin_val1,
                            margin_val2=hparams.margin_val2, device=self.hparams.device,
                            hparams=hparams,
                        )
    def edit(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            portability_inputs: Optional[Dict] = None,
            keep_original_weight=False,
            verbose=True,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        # assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs, portability_inputs,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        all_metrics = []
        for i, request in enumerate(requests):
            start = time()
            if self.alg_name == 'IKE' or self.alg_name == 'ICE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            if self.alg_name == 'IKE':
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request, self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                        request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                        request, self.hparams.device, real_world_eval=self.hparams.real_world_eval),
                }
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                metrics.update(
                    {"pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                        request, self.hparams.device, real_world_eval=self.hparams.real_world_eval)}
                )


            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                # LOG.info(
                #     f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                # )
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target']}"
                )
                pprint.pprint(metrics)

            all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy
    def batch_edit(self,
            prompts: List[str],
            targets: List[str],
            images: List[str],
            rephrase_prompts: Optional[List[str]] = None,
            rephrase_images: Optional[List[str]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            sequential_edit=False,
            verbose=True,
            **kwargs):
        """
        Perform batch multimodal editing.

        `prompts`: List of text prompts to edit.
        `targets`: List of expected output texts.
        `image_paths`: List of image file paths for multimodal input.
        """
        assert len(prompts) == len(targets) == len(images), "Input lists must have the same length"

        if isinstance(self.hparams.device, str):
            self.hparams.device = str(self.model.llava_model.device).split(":")[1]
        # self.hparams.device = str(self.model.llava_model.device)
        # Prepare requests
        requests = self._prepare_requests_batch(prompts, targets, images, rephrase_prompts, rephrase_images, locality_inputs, portability_inputs, **kwargs)
        
        assert hasattr(self.hparams, 'batch_size'), "Please specify batch_size in hparams."

        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            # Apply the editing algorithm to the batch of requests
            if self.alg_name in ['MEMIT','UnKE','AlphaEdit','DPO']:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs.get('train_ds', None) if self.alg_name == 'IKE' else None
                )
            else: 
                assert f"{self.alg_name} does not support batch edit!"
                

            exec_time = time() - start
            LOG.info(f"Batch execution took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):
                # Calculate metrics
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                    "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                            request, self.hparams.device, real_world_eval=self.hparams.real_world_eval),
                }
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i].update(
                    {
                        "pre":compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                            request, self.hparams.device, real_world_eval=self.hparams.real_world_eval)
                    }
                )

                if verbose:
                    LOG.info(f"{i} editing: {request['prompt']} -> {request['target']}")
                    pprint.pprint(chunk_metrics[i])

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        
        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True,               
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0, \
        f'DataSet {ds} not supported yet.'

        if isinstance(self.hparams.device, str):
            self.hparams.device = str(self.model.llava_model.device).split(":")[1]
        
        # load all metrics
        task=kwargs.get('task', None)
        num_edits = 1
        # self.model_backup = copy.deepcopy(self.model.cpu())
        # self.model.cuda()
        all_metrics = []
        reload_weights = True
        local_counter = 0
        load_metrics_path = kwargs.get('load_metrics_path', None)
        if load_metrics_path is not None:
            os.makedirs(load_metrics_path, exist_ok=True)
            jsonl_file_path = os.path.join(load_metrics_path, self.hparams.all_metrics_name)
            if not os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, 'w') as f:
                    pass
            
            all_metrics = load_object(jsonl_file_path)
            local_counter = len(all_metrics)
            LOG.info(f"Loaded metrics from {jsonl_file_path}")
        
        # compute the pre-edit results
        pres = []
        cached_path = f'./results/cache/{self.hparams.model_name}_{task}_{len(ds)}.pkl' # model-dataset-specific
        if os.path.exists(cached_path):
            pres = load_object(cached_path)
            LOG.info(f"Load pre results from cached path: {cached_path}")
        else:
            for i, request in tqdm(enumerate(ds), desc='Results before editing', total=len(ds)):
                request = self._prepare_requests_dataset(
                    prompts = [request['prompt']],
                    targets = [request['target']],
                    image = [request['image']],
                    rephrase_prompts = [request['rephrase_prompt']],
                    rephrase_image = [request['image_rephrase']],
                    locality_inputs = {"text":{"prompt":request['locality_prompt'],"ground_truth":request["locality_ground_truth"]},
                                    "vision":{"prompt": request["multimodal_locality_prompt"], "ground_truth":request["multimodal_locality_ground_truth"], "image":request["multimodal_locality_image"]}
                                    },
                    **kwargs)
                pre = compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                    request[0], self.hparams.device, self.hparams.real_world_eval)
                pres.append(pre)
            if not os.path.exists('./results/cache/'):
                os.makedirs('./results/cache/')
            save_object(pres, cached_path)

        # self.model.zero_grad()
        for i, request in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):

            start = time()
            request = self._prepare_requests_dataset(
                    prompts = [request['prompt']],
                    targets = [request['target']],
                    image = [request['image']],
                    rephrase_prompts = [request['rephrase_prompt']],
                    rephrase_image = [request['image_rephrase']],
                    locality_inputs = {"text":{"prompt":request['locality_prompt'],"ground_truth":request["locality_ground_truth"]},
                                       "vision":{"prompt": request["multimodal_locality_prompt"], "ground_truth":request["multimodal_locality_ground_truth"], "image":request["multimodal_locality_image"]}
                                    },
                    **kwargs)

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request[0], self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request[0], self.hparams.device, pre_edit=True)
                }
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = metrics['pre']['locality_output'].to(torch.float32)
                    post_logits = metrics['post']['locality_output'].to(torch.float32)
                    if post_logits.shape[1] > base_logits.shape[1]:
                        post_logits = post_logits[:, -base_logits.shape[1]:, :]
                    else:
                        base_logits = base_logits[:, -post_logits.shape[1]:, :]

                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    metrics['post'].pop('locality_output_ids')
                    metrics['pre'].pop('locality_output_ids')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                    post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                    if post_image_logits.shape[1] > base_image_logits.shape[1]:
                        post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                    else:
                        base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                    metrics['post'].pop('multimodal_locality_output_ids')
                    metrics['pre'].pop('multimodal_locality_output_ids')

                LOG.info(f"Evaluation took {time() - start}")
                if verbose:
                    LOG.info(
                        f"{i} editing: {request[0]['prompt']} -> {request[0]['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
            elif self.alg_name.lower() in ['unike']:
                torch.cuda.empty_cache()
                self.model.to(f'cuda:{self.hparams.device}')
                pre = pres[i]
                inner_res = {}
                torch.cuda.empty_cache()
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    editor=self.editor if self.alg_name == 'UNIKE' else None,
                    collate_fn=ds.collate_fn,
                    pre=pre,
                    inner_res=inner_res,
                    sample_id=i,
                    task=task,
                    reload_weights=reload_weights
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                # self.model = edited_model
                start = time()
                if self.alg_name == 'UNIKE' and self.hparams.ike == True:
                    ike_method = ALG_MULTIMODAL_DICT['IKE']
                    icl_examples = ike_method(
                        self.model,
                        self.tok,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds']
                    )
                    exec_time = time() - start
                    LOG.info(f"Execution {i} editing took {exec_time}")
                    start = time()
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request[0], self.hparams.device),
                    }
                else:
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                            request[0], self.hparams.device, self.hparams.real_world_eval),
                    }
                # add additional metrics
                metrics["add_neuron_num"] = self.editor.add_neuron_num
                metrics["inner_res"] = inner_res["res"]
                metrics["pre"] = pre
                # calculate the locality accuracy
                if self.alg_name == 'UNIKE':
                    if 'locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['locality_output']) == \
                                len(metrics['pre']['locality_output'])
                        metrics['inner_res']['locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['locality_output'],
                                                metrics['pre']['locality_output']))
                        metrics['inner_res'].pop('locality_output')
                        
                    if 'multimodal_locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['multimodal_locality_output']) == \
                                len(metrics['pre']['multimodal_locality_output'])
                        metrics['inner_res']['multimodal_locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['multimodal_locality_output'],
                                                metrics['pre']['multimodal_locality_output']))
                        metrics['inner_res'].pop('multimodal_locality_output')
                if self.alg_name == 'UNIKE' and self.hparams.ike == True:
                    metrics['post']['locality_output'] = metrics['post']['locality_output_ids']
                    metrics['post']['multimodal_locality_output'] = metrics['post']['multimodal_locality_output_ids']
                    metrics['post'].pop('locality_output_ids')
                    metrics['post'].pop('multimodal_locality_output_ids')

                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
    
                # calculate the locality accuracy (real world)
                if 'locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['locality_rel_output']).to(torch.float32)

                    question = request[0]['locality_prompt']
                    metrics['post']['locality_rel_acc'], metrics['post']['locality_rel_gen_content'], metrics['pre']['locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('locality_rel_output')
                    metrics['pre'].pop('locality_rel_output')
                    
                if 'multimodal_locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['multimodal_locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['multimodal_locality_rel_output']).to(torch.float32)

                    question = request[0]['multimodal_locality_prompt']
                    metrics['post']['multimodal_locality_rel_acc'], metrics['post']['multimodal_locality_rel_gen_content'], metrics['pre']['multimodal_locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('multimodal_locality_rel_output')
                    metrics['pre'].pop('multimodal_locality_rel_output')
                    

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request[0]['prompt']} -> {request[0]['target']}"
                    )

                all_metrics.append(metrics)
                torch.cuda.empty_cache()
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight
                )
                exec_time = time() - start
            
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                    "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                        request[0], self.hparams.device, self.hparams.real_world_eval),
                }
                metrics["pre"] = pres[i]
                # calculate the locality accuracy
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                if 'locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['locality_rel_output']).to(torch.float32)

                    question = request[0]['locality_prompt']
                    metrics['post']['locality_rel_acc'], metrics['post']['locality_rel_gen_content'], metrics['pre']['locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('locality_rel_output')
                    metrics['pre'].pop('locality_rel_output')
                    
                if 'multimodal_locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['multimodal_locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['multimodal_locality_rel_output']).to(torch.float32)

                    question = request[0]['multimodal_locality_prompt']
                    metrics['post']['multimodal_locality_rel_acc'], metrics['post']['multimodal_locality_rel_gen_content'], metrics['pre']['multimodal_locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('multimodal_locality_rel_output')
                    metrics['pre'].pop('multimodal_locality_rel_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request[0]['prompt']} -> {request[0]['target']}"
                    )

                all_metrics.append(metrics)
            
            if i == 0:
                self.weights_copy = weights_copy
            # if do not use continuous edit, restore the edit layers
            local_counter += 1
            if local_counter % self.hparams.continuous_sample == 0:
                local_counter = 0 # restore the counter
                reload_weights = True
            else:
                reload_weights = False
            torch.cuda.empty_cache()
                
            if self.alg_name == 'UNIKE':
                if reload_weights:
                    self.editor.clear_editors()
                    self.editor.clean_cache()

            elif self.alg_name in ['KN']:
                with torch.no_grad():
                    if reload_weights:
                        # weights_copy() # unpatch_fn
                        self.model.load_state_dict(self.model_backup.state_dict())
                        self.model.cuda()
                    else:
                        self.model.load_state_dict(edited_model.state_dict())
                        edited_model = edited_model.cpu()
                        del edited_model
                        self.model.cuda()
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    if reload_weights:
                        for k, v in self.weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                    else:
                        if self.hparams.alg_name == 'FT_MULTI':
                            for k, v in self.weights_copy.items():
                                # copy the old weights to new model
                                nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model, k).to(f"cuda:{self.hparams.device}")
                        else:
                            for k, v in self.weights_copy.items():
                                # copy the old weights to new model
                                nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model.model, k).to(f"cuda:{self.hparams.device}")
                        torch.cuda.empty_cache()
                        
            # save the metrics dynamically       
            if load_metrics_path is not None:
                with open(jsonl_file_path, 'a') as f:
                    json.dump(metrics, f, ensure_ascii=False)
                    f.write('\n')
            gc.collect()
            torch.cuda.empty_cache()
        return all_metrics, edited_model, weights_copy

    def edit_MMKE_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        num_edits = 1
        self.model_backup = copy.deepcopy(self.model.cpu())
        self.model.cuda()
        # num_edits = self.hparams.batch_size
        all_metrics = []
        
        if isinstance(self.hparams.device, str):
            self.hparams.device = str(self.model.llava_model.device).split(":")[1]
        
        # load all metrics
        task = kwargs.get('task', None)
        reload_weights = True
        local_counter = 0
        load_metrics_path = kwargs.get('load_metrics_path', None)
        if load_metrics_path is not None:
            os.makedirs(load_metrics_path, exist_ok=True)
            jsonl_file_path = os.path.join(load_metrics_path, self.hparams.all_metrics_name)
            if not os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, 'w') as f:
                    pass
            
            all_metrics = load_object(jsonl_file_path)
            local_counter = len(all_metrics)
            LOG.info(f"Loaded metrics from {jsonl_file_path}")
        
        # compute the pre-edit results
        pres = []
        cached_path = f'./results/cache/{self.hparams.model_name}_{task}_{len(ds)}.pkl' # model-dataset-specific
        if os.path.exists(cached_path):
            pres = load_object(cached_path)
            LOG.info(f"Load pre results from cached path: {cached_path}")
        else:
            for i, request in tqdm(enumerate(ds), desc='Results before editing', total=len(ds)):
                # Add default image token
                request.update({"prompt_template":self.prompt_template})
                if request["knowledge_type"] in [0,1]:
                    request.update({"prompt":self.prompt.format(request["prompt"]),
                                    "rephrase_prompt":self.prompt.format(request["rephrase_prompt"]),
                                    "multimodal_locality_prompt":self.prompt.format(request["multimodal_locality_prompt"]),
                                    "m_rel_prompt_1":self.prompt.format(request["m_rel_prompt_1"]),
                                    "m_rel_prompt_2":self.prompt.format(request["m_rel_prompt_2"]),
                                    })
                elif request["knowledge_type"] == 2:
                    request.update({"prompt":self.prompt.format(request["prompt"]),
                                    "rephrase_prompt":self.prompt.format(request["rephrase_prompt"]),
                                    "multimodal_locality_prompt":self.prompt.format(request["multimodal_locality_prompt"]),
                                    "m_rel_prompt":self.prompt.format(request["m_rel_prompt"]),
                                    })
                if "portability_prompt" in request.keys():
                    request.update({
                        "portability_prompt":self.prompt.format(request["portability_prompt"])
                    })
                pre = compute_mmke_multimodal_edit_quality_rel(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, self.hparams.real_world_eval)
                pres.append(pre)
            if not os.path.exists('./results/cache/'):
                os.makedirs('./results/cache/')
            save_object(pres, cached_path)


        for i, request in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):
            
            start = time()
            """Add instruction tuning template"""
            request.update({"prompt_template":self.prompt_template})
            request_edit = self._prepare_requests_dataset(
                                                            [request['prompt']], [request['target']], [request['image']], 
                                                            [request['rephrase_prompt']], [request['image_rephrase']],
                                                            {"text":{"prompt":request['locality_prompt'],"ground_truth":request["locality_ground_truth"]},
                                                                "vision":{"prompt": request["multimodal_locality_prompt"], "ground_truth":request["multimodal_locality_ground_truth"], "image":request["multimodal_locality_image"]}
                                                            },
                                                            {"text":{"prompt":request['portability_prompt'],"ground_truth":request["portability_ground_truth"],'image':[request['image']]},},
                                                            **kwargs)
            # Add default image token
            if request["knowledge_type"] in [0,1]:
                request.update({"prompt":self.prompt.format(request["prompt"]),
                                "rephrase_prompt":self.prompt.format(request["rephrase_prompt"]),
                                "multimodal_locality_prompt":self.prompt.format(request["multimodal_locality_prompt"]),
                                "m_rel_prompt_1":self.prompt.format(request["m_rel_prompt_1"]),
                                "m_rel_prompt_2":self.prompt.format(request["m_rel_prompt_2"]),
                                })
            elif request["knowledge_type"] == 2:
                request.update({"prompt":self.prompt.format(request["prompt"]),
                                "rephrase_prompt":self.prompt.format(request["rephrase_prompt"]),
                                "multimodal_locality_prompt":self.prompt.format(request["multimodal_locality_prompt"]),
                                "m_rel_prompt":self.prompt.format(request["m_rel_prompt"]),
                                })
            if "portability_prompt" in request.keys():
                request.update({
                    "portability_prompt":self.prompt.format(request["portability_prompt"])
                })

            # Edit model with different algs

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request_edit,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True),
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request, self.hparams.device),
                    }
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = metrics['pre']['locality_output'].to(torch.float32)
                    post_logits = metrics['post']['locality_output'].to(torch.float32)
                    if post_logits.shape[1] > base_logits.shape[1]:
                        post_logits = post_logits[:, -base_logits.shape[1]:, :]
                    else:
                        base_logits = base_logits[:, -post_logits.shape[1]:, :]

                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    metrics['post'].pop('locality_output_ids')
                    metrics['pre'].pop('locality_output_ids')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                    post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                    if post_image_logits.shape[1] > base_image_logits.shape[1]:
                        post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                    else:
                        base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                    metrics['post'].pop('multimodal_locality_output_ids')
                    metrics['pre'].pop('multimodal_locality_output_ids')

                LOG.info(f"Evaluation took {time() - start}")
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
            elif self.alg_name.lower() in ['unike']:
                torch.cuda.empty_cache()
                self.model.to(f'cuda:{self.hparams.device}')
                pre = pres[i]
                inner_res = {}
                torch.cuda.empty_cache()
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    request_edit,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    editor=self.editor if self.alg_name == 'UNIKE' else None,
                    collate_fn=ds.collate_fn,
                    pre=pre,
                    inner_res=inner_res,
                    sample_id=i,
                    task=task,
                    reload_weights=reload_weights
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                # self.model = edited_model
                start = time()
                if self.alg_name == 'UNIKE' and self.hparams.ike == True:
                    ike_method = ALG_MULTIMODAL_DICT['IKE']
                    icl_examples = ike_method(
                        self.model,
                        self.tok,
                        request_edit,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds']
                    )
                    exec_time = time() - start
                    LOG.info(f"Execution {i} editing took {exec_time}")
                    start = time()
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request, self.hparams.device),
                    }
                else:
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": compute_mmke_multimodal_edit_quality_rel(edited_model, self.model_name, self.hparams, self.tok,
                                                            request, self.hparams.device, self.hparams.real_world_eval),
                    }
                # add additional metrics
                metrics["add_neuron_num"] = self.editor.add_neuron_num
                metrics["inner_res"] = inner_res["res"]
                metrics["pre"] = pre
                # calculate the locality accuracy
                if self.alg_name == 'UNIKE':
                    if 'locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['locality_output']) == \
                                len(metrics['pre']['locality_output'])
                        metrics['inner_res']['locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['locality_output'],
                                                metrics['pre']['locality_output']))
                        metrics['inner_res'].pop('locality_output')
                        
                    if 'multimodal_locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['multimodal_locality_output']) == \
                                len(metrics['pre']['multimodal_locality_output'])
                        metrics['inner_res']['multimodal_locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['multimodal_locality_output'],
                                                metrics['pre']['multimodal_locality_output']))
                        metrics['inner_res'].pop('multimodal_locality_output')
                if self.alg_name == 'UNIKE' and self.hparams.ike == True:
                    metrics['post']['locality_output'] = metrics['post']['locality_output_ids']
                    metrics['post']['multimodal_locality_output'] = metrics['post']['multimodal_locality_output_ids']
                    metrics['post'].pop('locality_output_ids')
                    metrics['post'].pop('multimodal_locality_output_ids')

                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
    
                # calculate the locality accuracy (real world)
                if 'locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['locality_rel_output']).to(torch.float32)

                    question = request['locality_prompt']
                    metrics['post']['locality_rel_acc'], metrics['post']['locality_rel_gen_content'], metrics['pre']['locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('locality_rel_output')
                    metrics['pre'].pop('locality_rel_output')
                    
                if 'multimodal_locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['multimodal_locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['multimodal_locality_rel_output']).to(torch.float32)

                    question = request['multimodal_locality_prompt']
                    metrics['post']['multimodal_locality_rel_acc'], metrics['post']['multimodal_locality_rel_gen_content'], metrics['pre']['multimodal_locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('multimodal_locality_rel_output')
                    metrics['pre'].pop('multimodal_locality_rel_output')
                    

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}"
                    )

                all_metrics.append(metrics)
                torch.cuda.empty_cache()
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    request_edit,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=None
                )
                exec_time = time() - start
                
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                    "post": compute_mmke_multimodal_edit_quality_rel(edited_model, self.model_name, self.hparams, self.tok,
                                                        request, self.hparams.device, self.hparams.real_world_eval),
                }
                metrics["pre"] = pres[i]
                # calculate the locality accuracy
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                if 'locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['locality_rel_output']).to(torch.float32)

                    question = request['locality_prompt']
                    metrics['post']['locality_rel_acc'], metrics['post']['locality_rel_gen_content'], metrics['pre']['locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('locality_rel_output')
                    metrics['pre'].pop('locality_rel_output')
                    
                if 'multimodal_locality_rel_output' in metrics['post'].keys():
                    pre_tokens = torch.tensor(metrics['pre']['multimodal_locality_rel_output']).to(torch.float32)
                    post_tokens = torch.tensor(metrics['post']['multimodal_locality_rel_output']).to(torch.float32)

                    question = request['multimodal_locality_prompt']
                    metrics['post']['multimodal_locality_rel_acc'], metrics['post']['multimodal_locality_rel_gen_content'], metrics['pre']['multimodal_locality_rel_gen_content'] = \
                                                            test_locality_real_multimodal(self.tok, self.hparams, question, pre_tokens, post_tokens)
                    metrics['post'].pop('multimodal_locality_rel_output')
                    metrics['pre'].pop('multimodal_locality_rel_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}"
                    )

                all_metrics.append(metrics)
                        
            if i == 0:
                self.weights_copy = weights_copy
            # if do not use continuous edit, restore the edit layers
            local_counter += 1
            if local_counter % self.hparams.continuous_sample == 0:
                local_counter = 0 # restore the counter
                reload_weights = True
            else:
                reload_weights = False
            torch.cuda.empty_cache()
                
            if self.alg_name == 'UNIKE':
                if reload_weights:
                    self.editor.clear_editors()
                    self.editor.clean_cache()
            elif self.alg_name in ['KN']:
                with torch.no_grad():
                    if reload_weights:
                        # weights_copy() # unpatch_fn
                        self.model.load_state_dict(self.model_backup.state_dict())
                        self.model.cuda()
                    else:
                        self.model.load_state_dict(edited_model.state_dict())
                        edited_model = edited_model.cpu()
                        del edited_model
                        self.model.cuda()
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    if reload_weights:
                        for k, v in self.weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                    else:
                        if self.hparams.alg_name == 'FT_MULTI':
                            for k, v in self.weights_copy.items():
                                # copy the old weights to new model
                                nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model, k).to(f"cuda:{self.hparams.device}")
                        else:
                            for k, v in self.weights_copy.items():
                                # copy the old weights to new model
                                nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model.model, k).to(f"cuda:{self.hparams.device}")
                        torch.cuda.empty_cache()
                        
            # save the metrics dynamically       
            if load_metrics_path is not None:
                with open(jsonl_file_path, 'a') as f:
                    json.dump(metrics, f, ensure_ascii=False)
                    f.write('\n')
            gc.collect()
            torch.cuda.empty_cache()

        return all_metrics, edited_model, weights_copy

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in image]
        else:
            image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in image]
        
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        # 'subject': request["prompt"].split()[-1]
                        'subject': request["prompt_template"].split()[-1]
                        
                    }
                )

        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_prompts) == len(locality_ground_truth) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(multimodal_locality_prompts) == len(multimodal_locality_ground_truth) \
                == len(multimodal_locality_image) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if request['image'] is not None else rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            if 'llava' in self.hparams.model_name:
                rephrase_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            else:
                rephrase_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            
            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) if multimodal_locality_image_ is not None else None for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in locality_image_path]
            if 'llava' in self.hparams.model_name:
                locality_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in locality_image]
            else:
                locality_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in locality_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': self.prompt.format(multimodal_locality_prompts[i]) if locality_image[i] is not None else multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
        
        if "text" in portability_inputs.keys():
            portability_prompts = portability_inputs['text']['prompt']
            portability_ground_truth = portability_inputs['text']['ground_truth']
            portability_image= portability_inputs['text']['image']
            if isinstance(portability_prompts, str):
                portability_prompts = [portability_prompts, ]
            if isinstance(portability_ground_truth, str):
                portability_ground_truth = [portability_ground_truth, ]
            if isinstance(portability_image, str):
                portability_image = [portability_image, ]
            assert len(portability_prompts) == len(portability_ground_truth) \
                == len(portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in portability_inputs.keys():
            multimodal_portability_prompts = portability_inputs['vision']['prompt']
            multimodal_portability_ground_truth = portability_inputs['vision']['ground_truth']
            multimodal_portability_image = portability_inputs['vision']['image']
            if isinstance(multimodal_portability_prompts, str):
                multimodal_portability_prompts = [multimodal_portability_prompts, ]
            if isinstance(multimodal_portability_ground_truth, str):
                multimodal_portability_ground_truth = [multimodal_portability_ground_truth, ]
            if isinstance(multimodal_portability_image, str):
                multimodal_portability_image = [multimodal_portability_image, ]
            assert len(multimodal_portability_prompts) == len(multimodal_portability_ground_truth) \
                == len(multimodal_portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
    

        if "text" in portability_inputs.keys():
            portability_image_path = [os.path.join(self.vis_root, portability_image_) if portability_image_ is not None else None for portability_image_ in portability_image]
            portability_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
            if 'llava' in self.hparams.model_name:
                portability_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            else:
                portability_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'portability_prompt': self.prompt.format(portability_prompts[i]) if portability_image[i] is not None else portability_prompts[i],
                        'portability_ground_truth': portability_ground_truth[i],
                        'portability_image': portability_image[i]
                    }
                )
        
        if "vision" in portability_inputs.keys():
            portability_image_path = [os.path.join(self.vis_root, multimodal_portability_image_) if multimodal_portability_image_ is not None else None for multimodal_portability_image_ in multimodal_portability_image]
            portability_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
            if 'llava' in self.hparams.model_name:
                portability_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            else:
                portability_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_portability_image': portability_image[i],
                        'multimodal_portability_prompt': self.prompt.format(multimodal_portability_prompts[i]) if portability_image[i] is not None else multimodal_portability_prompts[i],
                        'multimodal_portability_ground_truth': multimodal_portability_ground_truth[i],
                    }
                )
        return requests
    
    def _prepare_requests_batch(self,
        prompts: Union[str, List[str]],
        targets: Union[str, List[str]],
        image: Union[str, List[str]],
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        rephrase_image: Optional[Union[str, List[str]]] = None,
        locality_inputs: Optional[List[Dict]] = None,
        portability_inputs: Optional[List[Dict]] = None,
        targets_neg: Optional[List[str]] = None,
        **kwargs):
        # Ensure that inputs are lists if they are not already
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(image, str):
            image = [image]
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts]
        if isinstance(rephrase_image, str):
            rephrase_image = [rephrase_image]
        if isinstance(locality_inputs, dict):
            locality_inputs = [locality_inputs]
        if isinstance(portability_inputs, dict):
            portability_inputs = [portability_inputs]

        # Ensure that all lists have the same length
        assert len(prompts) == len(targets) == len(image), "Prompts, targets, and images must have the same length"

        # Replicate locality_inputs if necessary to match the length of requests (prompts)
        if locality_inputs is not None:
            if len(locality_inputs) < len(prompts):
                locality_inputs = (locality_inputs * math.ceil(len(prompts) / len(locality_inputs)))[:len(prompts)]
                random.shuffle(locality_inputs)  # Shuffle to randomize the locality input order

        # Replicate portability_inputs if necessary to match the length of requests (prompts)
        if portability_inputs is not None:
            if len(portability_inputs) < len(prompts):
                portability_inputs = (portability_inputs * math.ceil(len(prompts) / len(portability_inputs)))[:len(prompts)]
                random.shuffle(portability_inputs)  # Shuffle to randomize the portability input order

        # Prepare image paths and load images
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        else:
            images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        
        # Create requests list
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        } for prompt, target, image_ in zip(prompts, targets, images)]

        # Handle 'subject' keyword in kwargs
        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        # 'subject': request["prompt"].split()[-1]
                        'subject': request["prompt_template"].split()[-1]
                        
                    }
                )
        if targets_neg is not None:
            if isinstance(targets_neg, str):
                targets_neg = [targets_neg]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'targets_neg': targets_neg[i],
                    }
                )
        # Handle rephrase prompts
        if rephrase_prompts is not None:
            for i, request in enumerate(requests):
                request.update({
                    'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if rephrase_prompts[i] and request['image'] is not None else rephrase_prompts[i],
                })
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            if 'llava' in self.hparams.model_name:
                rephrase_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            else:
                rephrase_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        # Handle locality inputs (text and vision)
        if locality_inputs is not None:
            for i, locality_input in enumerate(locality_inputs):
                request = requests[i]
                if "text" in locality_input:
                    locality_prompts = locality_input['text']['prompt']
                    locality_ground_truth = locality_input['text']['ground_truth']
                    locality_prompts = [locality_prompts] if isinstance(locality_prompts, str) else locality_prompts
                    locality_ground_truth = [locality_ground_truth] if isinstance(locality_ground_truth, str) else locality_ground_truth
                    request.update(
                        {
                            'locality_prompt': locality_prompts[0],
                            'locality_ground_truth':locality_ground_truth[0]
                        }
                    )
                # One sample has one locality and portability input, return index 0, if there are multiple locality inputs, remove[0] 
                # Vision locality
                if "vision" in locality_input:
                    vision_prompts = locality_input['vision']['prompt']
                    vision_ground_truth = locality_input['vision']['ground_truth']
                    vision_images = locality_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_locality_image': vision_images[0],
                            'multimodal_locality_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_locality_ground_truth': vision_ground_truth[0]
                        }
                    )
        # Handle portability inputs (text and vision)
        if portability_inputs is not None:
            for i, portability_input in enumerate(portability_inputs):
                request = requests[i]
                if "text" in portability_input:
                    portability_prompts = portability_input['text']['prompt']
                    portability_ground_truth = portability_input['text']['ground_truth']
                    portability_image = portability_input['text']['image']
                    portability_prompts = [portability_prompts] if isinstance(portability_prompts, str) else portability_prompts
                    portability_ground_truth = [portability_ground_truth] if isinstance(portability_ground_truth, str) else portability_ground_truth
                    portability_image = [portability_image] if isinstance(portability_image, str) else portability_image
                    portability_image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in portability_image]
                    portability_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
                    if 'llava' in self.hparams.model_name:
                        portability_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    else:
                        portability_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    request.update(
                        {
                            'portability_prompt': [self.prompt.format(item) for item in portability_prompts][0],
                            'portability_ground_truth': portability_ground_truth[0],
                            'portability_image': portability_images[0]
                        }
                    )
                   

                # Vision portability
                if "vision" in portability_input:
                    vision_prompts = portability_input['vision']['prompt']
                    vision_ground_truth = portability_input['vision']['ground_truth']
                    vision_images = portability_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_portability_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_portability_ground_truth': vision_ground_truth[0],
                            'multimodal_portability_image': vision_images[0],
                        }
                    )

        return requests

    def _prepare_requests_dataset(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
            
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]
        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        # 'subject': request["prompt"].split()[-1]
                        'subject': request["prompt_template"].split()[-1]
                        
                    }
                )
        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_prompts) == len(locality_ground_truth) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, (str, np.ndarray)):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(multimodal_locality_prompts) == len(multimodal_locality_ground_truth) \
                == len(multimodal_locality_image) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if request['image'] is not None else rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': multimodal_locality_image[i],
                        'multimodal_locality_prompt': self.prompt.format(multimodal_locality_prompts[i]) if multimodal_locality_image[i] is not None else multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
        if portability_inputs is not None:
            if "text" in portability_inputs.keys():
                portability_prompts = portability_inputs['text']['prompt']
                portability_ground_truth = portability_inputs['text']['ground_truth']
                portability_image= portability_inputs['text']['image']
                if isinstance(portability_prompts, str):
                    portability_prompts = [portability_prompts, ]
                if isinstance(portability_ground_truth, str):
                    portability_ground_truth = [portability_ground_truth, ]
                if isinstance(portability_image, str):
                    portability_image = [portability_image, ]
                assert len(portability_prompts) == len(portability_ground_truth) \
                    == len(portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
            if "vision" in portability_inputs.keys():
                multimodal_portability_prompts = portability_inputs['vision']['prompt']
                multimodal_portability_ground_truth = portability_inputs['vision']['ground_truth']
                multimodal_portability_image = portability_inputs['vision']['image']
                if isinstance(multimodal_portability_prompts, str):
                    multimodal_portability_prompts = [multimodal_portability_prompts, ]
                if isinstance(multimodal_portability_ground_truth, str):
                    multimodal_portability_ground_truth = [multimodal_portability_ground_truth, ]
                if isinstance(multimodal_portability_image, str):
                    multimodal_portability_image = [multimodal_portability_image, ]
                assert len(multimodal_portability_prompts) == len(multimodal_portability_ground_truth) \
                    == len(multimodal_portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
    

            if "text" in portability_inputs.keys():
                for i, request in enumerate(requests):
                    request.update(
                        {
                            'portability_prompt': self.prompt.format(portability_prompts[i]) if portability_image[i] is not None else portability_prompts[i],
                            'portability_ground_truth': portability_ground_truth[i],
                            'portability_image': portability_image[i]
                        }
                    )
            
            if "vision" in portability_inputs.keys():
                for i, request in enumerate(requests):
                    request.update(
                        {
                            'multimodal_portability_image': multimodal_portability_image[i],
                            'multimodal_portability_prompt': self.prompt.format(multimodal_portability_prompts[i]) if multimodal_portability_image[i] is not None else multimodal_portability_prompts[i],
                            'multimodal_portability_ground_truth': multimodal_portability_ground_truth[i],
                        }
                    )
        return requests