import torch
import torch.nn as nn

from ..constants import IMAGE_TOKEN_INDEX
from .llava_llama import LlavaLlamaForCausalLM
from transformers.utils import ModelOutput
from transformers import AutoTokenizer 
from dataclasses import dataclass

from typing import Optional, Tuple, List

@dataclass
class LLavaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    input_tokens: Optional[torch.FloatTensor] = None
    text_input_range: List[tuple] = None
    subject_range: List[tuple] = None
    
    
class LLavaModel(nn.Module):
    
    def __init__(
        self,
        llava_model="",
        prompt_template="",
        device_map = "cuda",
        cache_dir=None,
        ):
        super().__init__()
        
        self.llava_tokenizer = AutoTokenizer.from_pretrained(llava_model, cache_dir=cache_dir, use_fast=False)
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(
            llava_model,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        )
        self.prompt_template = prompt_template
        vision_tower = self.llava_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        self.image_processor = vision_tower.image_processor
        
    def _device(self):
        return list(self.parameters())[-1].device
    
    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        for pro in prompt:
            prompt_chunks = [self.llava_tokenizer(chunk).input_ids for chunk in pro.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.llava_tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def forward(self, samples):
        subject_range = text_input_range = input_tokens = [None]
        if "text_input" in samples:
            if "noise" in samples:
                ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                texts = samples['text_input']
            else:
                texts = [self.prompt_template.format(item) for item in samples['text_input']]
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            images = samples["image"]
        else:
            images = None
        
        input_ids = []
        for text in texts:
            input_ids.append(self.tokenizer_image_token([text], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(list(self.parameters())[-1].device))
        input_ids = torch.cat(input_ids, dim=0) 
        
        if 'trace' in samples and samples['trace']:
            assert 'subject' in samples and 'ori_text_input' in samples, "Causal tracing must specify `subject` and `ori_text_input`."
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                # input_tokens,
                # text_input_range,
                # subject_range
            ) = self.llava_model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=images)
        else:
            if images:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.llava_model.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=None,
                    past_key_values=None,
                    labels=None,
                    images=images)
            else:
                inputs_embeds = self.llava_model.model.embed_tokens(input_ids)
                input_ids = attention_mask = position_ids = past_key_values = labels = None
        
        outputs = self.llava_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                images=images,
                use_cache=True)
        
        return LLavaOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            input_tokens=input_tokens,
            text_input_range=text_input_range,
            subject_range=subject_range
        )
    
    def generate(
        self, 
        samples, 
        **kwargs,
        ):
        if "text_input" in samples:
            if "noise" in samples:
                ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                texts = samples['text_input']
            else:
                texts = [self.prompt_template.format(item) for item in samples['text_input']]
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            images = samples["image"]
        else:
            images = None
        
        input_ids = []
        for text in texts:
            input_ids.append(self.tokenizer_image_token([text], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(list(self.parameters())[-1].device))
        input_ids = torch.cat(input_ids, dim=0) 
        
        outputs = self.llava_model.generate(
                inputs=input_ids,                
                images=images,
                **kwargs)
        
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llava_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers