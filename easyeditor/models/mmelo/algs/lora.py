from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os
from torchvision import transforms
from torch.nn import Parameter
from ..clip_model import *
import itertools

from ..utils import *

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    MeloConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import LoraLayer

# from models import BertClassifier
LOG = logging.getLogger(__name__)

def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)



class LORA(torch.nn.Module):
    def __init__(self, model, config, scale=None):
        super(LORA, self).__init__()
        self.config = config

        '''Apply_melo
        '''
        r_num = config.melo.num_block * config.melo.num_rank_per_block
        self.lora_config = MeloConfig(
            r=r_num,
            lora_alpha=r_num,
            target_modules=list(config.model.target_modules),
            lora_dropout=config.lora.lora_dropout,
            fan_in_fan_out=config.model.fan_in_fan_out,
            num_rank_per_block=config.melo.num_rank_per_block

        )
        self.log_dict = {}

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config).to(torch.bfloat16)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules()
        self.outputs = {}

        '''Load Tokenizer
        '''



    def save_lora_weights(self, lora_dir):
        self.model.save_pretrained(lora_dir + "/lora_checkpoint")

    def named_lora_modules(self):
        module_list = [key for key, _ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), LoraLayer):
                lora_list.append(key)
        return lora_list

    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()

    def set_lora_mapping(self, lora_block_mapping):
        self.model.reset_dynamic_mapping(lora_block_mapping)

    def edit(self, batch, batch_index):
        # MELO_V2 could automatically identify lora parameters to be optimized
        params_to_optimize = (itertools.chain(self.model.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, float(self.config.melo.edit_lr))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # rewrite target
        target_ids = batch["target_ids"]
        input_len = len(batch["input_ids"].input_ids[0]) 
        labels = torch.tensor(-100, device=f"cuda:{self.config.device}").repeat(
            len(batch['text_input']), input_len
        )
        for i in range(len(batch['text_input'])):
            ex_len = sum(batch["input_ids"].attention_mask[i])
            labels[i, ex_len - len(target_ids[i][1:]) : ex_len] = torch.tensor(target_ids[i][1:])
        
        samples = {
            "noise": True,
            "text_input": batch['text_input'],
            "image": batch['image'],
            "labels": labels,
        }

        self.losses = []
        for i in range(self.config.melo.num_iter):
            outputs = self.model.model(samples)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            numpy_loss = loss.detach().cpu().numpy()
            self.losses.append(numpy_loss)


            LOG.info(f'Batch output loss in iter {i}: {numpy_loss:.8f},')

        with torch.no_grad():
            outputs = self.model.model(samples)
            self.outputs[batch_index] = outputs

    def get_output(self, batch, lora_block_mapping):
        # reset batch lora_block_mapping
        if lora_block_mapping is not None:
            self.set_lora_mapping(lora_block_mapping)

        if isinstance(batch["image"], torch.Tensor):          
            samples = {
                "noise": True,
                "text_input": batch['text_input'],
                "image": batch['image'],
            }

            outputs = self.model.model(samples)
        else:
            samples = {
                "noise": True,
                "text_input": batch['text_input'],
                "image": None,
            }
            outputs = self.model.model(samples)

        return outputs

    def generate_output(self, batch, lora_block_mapping):
        # reset batch lora_block_mapping
        self.set_lora_mapping(lora_block_mapping)
        if isinstance(batch["image"], torch.Tensor):
            pexel_values = batch["image"]
            labels = batch["labels"]
            input_ids = batch["prompt_ids"]
            outputs = self.model.generate(input_ids=input_ids, pixel_values=pexel_values)
        return outputs

    # def generate(self, *args, **kwargs):
    #     return self.model.model.generate(*args, **kwargs)




if __name__ == '__main__':
    pass


















