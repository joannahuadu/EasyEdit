from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor, MendPerRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.dinm import DINMHyperParams, apply_dinm_to_model
from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset, CaptionDataset, VQADataset, PersonalityDataset, SafetyDataset
from ..models.ike import IKEHyperParams, apply_ike_to_model, apply_ike_to_multimodal_model, apply_ike_to_per_model
from ..models.ft_api import FTApiHyperParams, apply_ft_api_to_model
from ..models.qlora import QLoRAHyperParams, apply_qlora_to_model
from ..models.lora import LoRAHyperParams, apply_lora_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.melo import MELOHyperParams, apply_melo_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.r_rome import R_ROMEHyperParams, apply_r_rome_to_model
from ..models.emmet import EMMETHyperParams, apply_emmet_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from .. models.deepedit_api import DeepEditApiHyperParams, apply_deepedit_api_to_model
from ..models.dpo import DPOHyperParams, apply_dpo_to_model
from ..models.rome import rome_causal_trace
from ..models.unke import UnKEMultimodalHyperParams, apply_unke_to_model
from ..models.unike import UniKEHyperParams, apply_unike_to_model_mm
# from ..models.mmelo import MMelo
from ..models.roselora import RoseLoRAMultimodalHyperParams, apply_roselora_to_model
from ..models.loranull import LoRANULLMultimodalHyperParams, apply_loranull_to_model
from ..models.xspace import XSpaceMultimodalHyperParams, apply_xspace_to_model, collect_xspace_to_model

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    "DINM": apply_dinm_to_model,
    'KN': apply_kn_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
    'FT-Api': apply_ft_api_to_model,
    'QLoRA': apply_qlora_to_model,
    'LoRA': apply_lora_to_model,
    'DPO': apply_dpo_to_model,
    'GRACE': apply_grace_to_model,
    'PMET': apply_pmet_to_model,
    'MELO': apply_melo_to_model,
    'WISE': apply_wise_to_model,
    'R-ROME': apply_r_rome_to_model,
    "EMMET": apply_emmet_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "DeepEdit-Api": apply_deepedit_api_to_model
}

ALG_MULTIMODAL_DICT = {
    'ROME': apply_rome_to_model,
    'MEND': MendMultimodalRewriteExecutor().apply_to_model,
    'SERAC': SeracMultimodalRewriteExecutor().apply_to_model,
    'SERAC_MULTI': SeracMultimodalRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_multimodal_model,
    'MEMIT': apply_memit_to_model,
    'UnKE':apply_unke_to_model,
    'AlphaEdit':apply_AlphaEdit_to_model,
    'DPO': apply_dpo_to_model,
    'UNIKE': apply_unike_to_model_mm,
    'LoRA': apply_lora_to_model,
    # 'MMELO': MMelo,
    'RoseLoRA': apply_roselora_to_model,
    'LoRANULL': apply_loranull_to_model,
    'XSpace':apply_xspace_to_model,
    'CoXSpace':collect_xspace_to_model,
}

PER_ALG_DICT = {
    "IKE": apply_ike_to_per_model,
    "MEND": MendPerRewriteExecutor().apply_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}

PER_DS_DICT = {
    "personalityEdit": PersonalityDataset
}
Safety_DS_DICT ={
    "safeEdit": SafetyDataset
}

TRACE_MULTIMODAL_DICT = {
    'ROME': rome_causal_trace,
}
