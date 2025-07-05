import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM

# disable flash-attn
# import os 
# from typing import List, Union
# from unittest.mock import patch
# from transformers.dynamic_module_utils import get_imports
# def fixed_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
#     """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
#     if not str(filename).endswith("/modeling_phi.py"):
#         return get_imports(filename)
#     imports = get_imports(filename)
#     imports.remove("flash_attn_2")
#     return imports

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

# 1. 为 Phi-3 Vision 定义一个与 QwenOutput 结构相同的输出类
@dataclass
class Phi3VOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # 可以根据需要添加其他输出
    
    
# 2. 创建 Phi-3 Vision 的外部包装类
class Phi3VLModel(nn.Module):
    
    def __init__(
        self,
        phi3_model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device_map: str = "cuda",
        cache_dir: Optional[str] = None,
        **kwargs,
        ):
        super().__init__()
        
        # 使用 AutoModelForCausalLM 加载模型，它会自动处理多模态架构
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            phi3_model_name,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True, # Phi-3 需要信任远程代码
            torch_dtype=torch.bfloat16  # <-- Pass the torch.dtype object directly
        )
            
        # config = AutoConfig.from_pretrained(phi3_model_name, cache_dir=cache_dir, trust_remote_code=True)

        # # 2. 强制修改配置
        # config.attn_implementation = "sdpa"

        # # 3. 使用修改后的配置加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     phi3_model_name,
        #     config=config, # <-- 明确传入修改后的config
        #     trust_remote_code=True, # 仍然需要信任远程代码来构建模型结构
        #     torch_dtype=torch.bfloat16
        # )
        # 使用 AutoProcessor，它统一处理了图像和文本
        self.processor = AutoProcessor.from_pretrained(
            phi3_model_name,
            trust_remote_code=True
        )
        # 确保 tokenizer 有 pad_token，对于批处理至关重要
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _device(self):
        return self.phi_model.device

    def forward(self, samples: Dict[str, Any], output_attentions: bool = False) -> Phi3VOutput:
        # phi3.5 does not support multiple prompts
        if samples["image"] is not None:
            images = samples["image"][0]
        else:
            images = None
        prompts = samples["text_input"][0]
        targets = samples["answer"][0]
        
        if isinstance(images, List):
            num_images = len(images)
        elif images is None:
            num_images = 0
        else:
            num_images = 1
        messages = []
        if num_images == 0:
            messages = [
                    {"role": "user", "content": f"{prompts}"},
                    {"role": "assistant", "content": targets}
                ]
        else:
            messages = [
                        {"role": "user", "content": f"<|image_1|>\n{prompts}"},
                        {"role": "assistant", "content": targets}
                    ]
    
        text_inputs = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )

        inputs = self.processor(
            text=text_inputs,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self._device())
        inputs.input_ids[inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id

        labels = inputs["input_ids"].clone()
        
        # 寻找 assistant turn 的起始位置来 mask 掉 prompt
        # 对于 Phi-3，我们可以寻找 <|assistant|> 之后的 token
        # 一个简化的方法是：只计算 target 的 token 长度，然后从后往前保留
        # 一个更稳健的方法是找到 assistant 标记
        # 找到 assistant 回答的起始位置
        prompt_part = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image_1|>\n{prompts}"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_len = len(self.processor.tokenizer(prompt_part, add_special_tokens=True).input_ids)
        
        # Mask 掉 prompt 部分
        labels[0,:prompt_len] = -100

        # Mask 掉 padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # 执行模型的前向传播，并传入 labels 以计算 loss
        # outputs = self.phi_model(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     pixel_values=inputs.pixel_values,
        #     labels=labels,
        #     output_attentions=output_attentions,
        #     use_cache=False,
        # )
        outputs = self.phi_model(
            **inputs,
            labels=labels,
            output_attentions=output_attentions,
            use_cache=False,
        )
        
        return Phi3VOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )
    
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成文本回答
        """
        outputs = self.generate_tokens(samples, **kwargs)
        
        # 解码生成的 tokens
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 清理输出，移除可能残留的模板标记
        cleaned_responses = []
        for res in responses:
            # Phi-3 的 assistant block 之后的内容才是真正的回答
            # 这是一个简单的后处理示例，可能需要根据具体输出进行调整
            if '<|end|>' in res:
                 res = res.split('<|end|>')[0]
            cleaned_responses.append(res.strip())

        return cleaned_responses

    def generate_tokens(self, samples: Dict[str, Any], **kwargs) -> torch.Tensor:
        if samples["image"] is not None:
            images = samples["image"][0]
        else:
            images = None
        prompts = samples["text_input"][0]

        messages = [{"role": "user", "content": f"<|image_1|>\n{prompts}"}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._device())
        
        outputs = self.phi_model.generate(**inputs, **kwargs)

        input_token_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_token_length:]

        return new_tokens
# 1. 为 Phi-3 Vision 定义一个与 QwenOutput 结构相同的输出类
@dataclass
class Phi4VOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # 可以根据需要添加其他输出
    
    
# 2. 创建 Phi-3 Vision 的外部包装类
class Phi4VLModel(nn.Module):
    
    def __init__(
        self,
        phi3_model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device_map: str = "cuda",
        cache_dir: Optional[str] = None,
        **kwargs,
        ):
        super().__init__()
        
        # 使用 AutoModelForCausalLM 加载模型，它会自动处理多模态架构
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            phi3_model_name,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True, # Phi-3 需要信任远程代码
            torch_dtype=torch.bfloat16  # <-- Pass the torch.dtype object directly
        )
            
        # config = AutoConfig.from_pretrained(phi3_model_name, cache_dir=cache_dir, trust_remote_code=True)

        # # 2. 强制修改配置
        # config.attn_implementation = "sdpa"

        # # 3. 使用修改后的配置加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     phi3_model_name,
        #     config=config, # <-- 明确传入修改后的config
        #     trust_remote_code=True, # 仍然需要信任远程代码来构建模型结构
        #     torch_dtype=torch.bfloat16
        # )
        # 使用 AutoProcessor，它统一处理了图像和文本
        self.processor = AutoProcessor.from_pretrained(
            phi3_model_name,
            trust_remote_code=True
        )
        # 确保 tokenizer 有 pad_token，对于批处理至关重要
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _device(self):
        return self.phi_model.device

    def forward(self, samples: Dict[str, Any], output_attentions: bool = False) -> Phi3VOutput:
        # phi3.5 does not support multiple prompts
        if samples["image"] is not None:
            if isinstance(samples["image"], List):
                images = samples["image"][0]
            else:
                images = samples["image"]
        else:
            images = None
        prompts = samples["text_input"][0]
        targets = samples["answer"][0]
        
        if images is None:
            messages = [
                    {"role": "user", "content": f"{prompts}"},
                    {"role": "assistant", "content": targets}
                ]
        else:
            messages = [
                        {"role": "user", "content": f"<|image_1|>\n{prompts}"},
                        {"role": "assistant", "content": targets}
                    ]
    
        text_inputs = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )

        inputs = self.processor(
            text=text_inputs,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self._device())
        inputs.input_ids[inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id

        labels = inputs["input_ids"].clone()
        
        # 寻找 assistant turn 的起始位置来 mask 掉 prompt
        # 对于 Phi-3，我们可以寻找 <|assistant|> 之后的 token
        # 一个简化的方法是：只计算 target 的 token 长度，然后从后往前保留
        # 一个更稳健的方法是找到 assistant 标记
        # 找到 assistant 回答的起始位置
        if images is None:
            prompt_part = self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{prompts}"}], 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt_part = self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<|image_1|>\n{prompts}"}], 
                tokenize=False, 
                add_generation_prompt=True
            )
        only_prompt_inputs = self.processor(
            text=prompt_part,
            images=images,
            return_tensors="pt",
            padding=True   
        )
        prompt_len = len(only_prompt_inputs.input_ids[0])
        
        # Mask 掉 prompt 部分
        labels[0,:prompt_len] = -100

        # Mask 掉 padding tokens
        # labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # 执行模型的前向传播，并传入 labels 以计算 loss
        # outputs = self.phi_model(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     pixel_values=inputs.pixel_values,
        #     labels=labels,
        #     output_attentions=output_attentions,
        #     use_cache=False,
        # )
        outputs = self.phi_model(
            **inputs,
            labels=labels,
            output_attentions=output_attentions,
            use_cache=True,
        )
        
        return Phi4VOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )
    
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成文本回答
        """
        outputs = self.generate_tokens(samples, **kwargs)
        
        # 解码生成的 tokens
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 清理输出，移除可能残留的模板标记
        cleaned_responses = []
        for res in responses:
            # Phi-3 的 assistant block 之后的内容才是真正的回答
            # 这是一个简单的后处理示例，可能需要根据具体输出进行调整
            if '<|end|>' in res:
                 res = res.split('<|end|>')[0]
            cleaned_responses.append(res.strip())

        return cleaned_responses

    def generate_tokens(self, samples: Dict[str, Any], **kwargs) -> torch.Tensor:
        if samples["image"] is not None:
            if isinstance(samples["image"],List):
                images = samples["image"][0]
            else:
                images = samples["image"]     
        else:
            images = None
        prompts = samples["text_input"][0]

        if images == None:
            messages = [{"role": "user", "content": f"{prompts}"}]
        else:
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompts}"}]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._device())
        
        outputs = self.phi_model.generate(**inputs, num_logits_to_keep = 0, **kwargs)

        input_token_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_token_length:]

        return new_tokens