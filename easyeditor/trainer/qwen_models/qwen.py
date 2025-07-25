import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import AutoTokenizer 
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers import (Qwen2_5_VLForConditionalGeneration, 
                          Qwen2_5_VLProcessor, 
                          AutoProcessor,
                          )
from qwen_vl_utils import process_vision_info


@dataclass
class QwenOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    input_tokens: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.FloatTensor] = None
    position_ids: Optional[torch.FloatTensor] = None
    attn_weights: Optional[torch.FloatTensor] = None
    
    
    
class QwenVLModel(nn.Module):
    
    def __init__(
        self,
        qwen_model="",
        device_map = "cuda",
        max_context_len=3800,
        cache_dir=None,
        ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model, cache_dir=cache_dir, use_fast=False)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model, cache_dir=cache_dir)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.max_context_len = max_context_len
    def _device(self):
        return list(self.parameters())[-1].device

    def forward(self, samples, output_attentions=False, prompt_template=True):
        image = samples["image"]
        prompts = samples["text_input"]
        targets = samples["answer"] if "answer" in samples else [None]*len(prompts)
        if isinstance(image, List):
            num_images = len(image)
            if image[0] is None:
                image = None
        else:
            num_images = 1
        if image is None:
            messages = [[
                    {"role": "user", "content": [{"type": "text", "text": p}]},
                    {"role": "assistant", "content": [{"type": "text", "text": t}]}
                ] for p, t in zip(prompts, targets)]
        else:
            # TODO support multiple images in a single sample
            messages = [[
                        {"role": "user", "content": [
                                    {"type": "image"}
                                ] + [{"type": "text", "text": p}]},
                        {"role": "assistant", "content": t}
                    ] for p, t in zip(prompts, targets)]
    
        if prompt_template:
            # do not append the target in the end in generation
            text_input = [self.processor.apply_chat_template(message,
                        add_generation_prompt=False,
                        tokenize=False) for message in messages]

        else:
            text_input = [
                            {

                                "role": "user",
                                "content": [
                                    {"type": "image"}
                                ] * num_images + [{"type": "text", "text": p}],
                            } for p in prompts
            ]
        
        multimodal_inputs = self.processor(
            images=image, 
            text=text_input, 
            return_tensors="pt",
            padding=True).to(self._device(), dtype=torch.bfloat16)
        
        multimodal_inputs.input_ids[multimodal_inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id
        labels = multimodal_inputs.input_ids.clone()
        if image is None:
            messages_wo_target = [
                    {"role": "user", "content": [{"type": "text", "text": p}]}
                 for p, t in zip(prompts, targets)]
            prompt_part = self.processor.tokenizer.apply_chat_template(
                messages_wo_target,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            messages_wo_target = [[
                    {"role": "user", "content": [
                                {"type": "image"}
                            ] + [{"type": "text", "text": p}]},
                ] for p, t in zip(prompts, targets)]
            prompt_part = self.processor.tokenizer.apply_chat_template(
                messages_wo_target,
                add_generation_prompt=True,
                tokenize=False
            )
        only_prompt_inputs = self.processor(
            text=prompt_part,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._device(), dtype=torch.bfloat16)
        
        prompt_len = len(only_prompt_inputs.input_ids[0])
        labels[:, :prompt_len] = -100 
         
        outputs = self.qwen_model(
                **multimodal_inputs,
                labels=labels,
                use_cache=True,
                output_attentions=output_attentions)
        
        return QwenOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            # attention_mask=outputs.attention_mask,
            # position_ids=outputs.position_ids,
            # attn_weights=outputs.attentions if outputs.attentions else None
        )
    
    def generate(
        self, 
        samples, 
        **kwargs,
        ):
        image = samples["image"]
        prompts = samples["text_input"]
        targets = samples["answer"]
        if isinstance(image, List):
            num_images = len(image)
        else:
            num_images = 1
        # do not append the target in the end in generation
        text_input = [self.processor.apply_chat_template([
                        {

                            "role": "user",
                            "content": [
                                {"type": "image"}
                            ] * num_images + [{"type": "text", "text": p}],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False)
                for p, l in zip(prompts, targets)] 
        multimodal_inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self._device(), dtype=torch.bfloat16)

        outputs = self.qwen_model.generate(**multimodal_inputs, **kwargs)
        input_token_length = multimodal_inputs["input_ids"].shape[1]
        outputs = outputs[:,input_token_length:]
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            #TODO
            # output_texts = self.tokenizer.decode(output_token, skip_special_tokens=True)
            # output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
            # output_texts = output_texts.replace("<s>", "")
            # output_texts = output_texts.split(r'[/INST]')[-1].strip()
            # answers.append(output_texts)

        return answers
    def generate_tokens(
        self, 
        samples, 
        **kwargs,
        ):
        image = samples["image"]
        prompts = samples["text_input"]
        targets = samples["answer"]
        if isinstance(image, List):
            num_images = len(image)
        else:
            num_images = 1
            
        # do not append the target in the end in generation
        text_input = [self.processor.apply_chat_template([
                        {

                            "role": "user",
                            "content": [
                                {"type": "image"}
                            ] * num_images + [{"type": "text", "text": p}],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False)
                for p, l in zip(prompts, targets)] 
        multimodal_inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self._device(), dtype=torch.bfloat16)

        outputs = self.qwen_model.generate(**multimodal_inputs, **kwargs)
        input_token_length = multimodal_inputs["input_ids"].shape[1]
        outputs = outputs[:,input_token_length:]

        return outputs