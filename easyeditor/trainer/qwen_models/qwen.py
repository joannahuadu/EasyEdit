import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import AutoTokenizer 
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers import (Qwen2_5_VLForConditionalGeneration, 
                          Qwen2_5_VLModel,
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
        self.qwen_model = Qwen2_5_VLModel.from_pretrained(
            qwen_model,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model, cache_dir=cache_dir)
        self.max_context_len = max_context_len
    def _device(self):
        return list(self.parameters())[-1].device

    def forward(self, samples, output_attentions=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": samples["image"],
                    },
                    {"type": "text", "text": samples["text_input"]},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        


        outputs = self.llava_model(
                **inputs,
                use_cache=True,
                output_attentions=output_attentions)
        
        return QwenOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
            attn_weights=outputs.attentions if outputs.attentions else None
        )
    
    def generate(
        self, 
        samples, 
        **kwargs,
        ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": samples["image"],
                    },
                    {"type": "text", "text": samples["text_input"]},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        outputs = self.qwen_model(**inputs, max_new_tokens=self.max_context_len)
        
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers
    def generate_tokens(
        self, 
        samples, 
        **kwargs,
        ):
        if "text_input" in samples:
            if "noise" in samples and samples["noise"]:
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
        # input_ids = torch.cat(input_ids, dim=0) 
        id_lens = [input_id.shape[1] for input_id in input_ids]
        pad_ids = torch.tensor(self.llava_tokenizer.pad_token_id, device=list(self.parameters())[-1].device)

        max_length = max(id_lens) if max(id_lens) < self.max_context_len else self.max_context_len
        wrapped_input_ids = pad_ids.expand(len(id_lens), max_length).clone()
        
        for i, input_id in enumerate(input_ids):
            length = id_lens[i] if id_lens[i] < self.max_context_len else self.max_context_len
            wrapped_input_ids[i, :length] = input_id[:length]
        input_ids = wrapped_input_ids
        
        outputs = self.llava_model.generate(
                inputs=input_ids,                
                images=images,
                **kwargs)
        

        return outputs