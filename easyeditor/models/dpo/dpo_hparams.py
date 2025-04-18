from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml
from typing import List, Literal, Optional



@dataclass
class DPOHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    # Module templates

    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    alpha: float = 0.1
    beta: float = 0.1
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'DPO') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)

@dataclass
class DPOMultimodalHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    # Module templates

    device: int
    alg_name: str
    model_name: str
    name: str
    tokenizer_class: str
    tokenizer_name: str
    cache_dir: str
    

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    alpha: float = 0.1
    beta: float = 0.1
    
    # Evaluation
    real_world_eval: bool = False
    api_key: Optional[str] = None
    
    # Multimodal
    coco_image: Optional[str] = None
    rephrase_image: Optional[str] = None
    result_dir: Optional[str] = None

    # Evaluation
    real_world_eval: bool = False
    api_key: Optional[str] = None
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'DPO') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
