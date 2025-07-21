import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import MEMITMultimodalHyperParams
from easyeditor import AlphaMultimodalHyperParams
from easyeditor import LoRAMultimodalHyperParams
from easyeditor import UniKEHyperParams
from easyeditor import UnKEMultimodalHyperParams
# from easyeditor import MELOMultimodalHyperParams
from easyeditor import RoseLoRAMultimodalHyperParams
from easyeditor import LoRANULLMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset
from easyeditor import XSpaceMultimodalHyperParams

import os
from statistics import mean
from examples.Observation_MEMIT_multimodal import MMKE_print_result
from pprint import pprint
import random
import torch
import numpy as np

def edit_Alpha_LLaVA_VQA(args):
    hparams = AlphaMultimodalHyperParams.from_hparams('hparams/AlphaEdit/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
def edit_Alpha_LLaVA_MMKE(args):
    hparams = AlphaMultimodalHyperParams.from_hparams('hparams/AlphaEdit/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_UNIKE_LLaVA_VQA(args):
    hparams = UniKEHyperParams.from_hparams('hparams/UniKE/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_UNIKE_LLaVA_MMKE(args):
    hparams = UniKEHyperParams.from_hparams('hparams/UniKE/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_LoRA_LLaVA_VQA(args):
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/llava_corda.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams, size=5)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        copy=True,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)


def edit_LoRA_LLaVA_MMKE(args):
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        copy=True,
        keep_original_weight=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_LoRA_Qwen_VQA(args):
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/qwen2.5-VL-7b.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA'),
        copy=True,
    )
    pprint(metrics)

def edit_LoRA_Phi_VQA(args):
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/phi4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA'),
        copy=True,
    )
    pprint(metrics)

def edit_XSpace_Phi_VQA(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/Xspace/phi.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
def edit_LoRA_Qwen_MMKE(args):
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/qwen_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_UnKE_LLaVA_VQA(args):
    hparams = UnKEMultimodalHyperParams.from_hparams('hparams/UnKE/llava')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_UnKE_LLaVA_MMKE(args):
    hparams = UnKEMultimodalHyperParams.from_hparams('hparams/UnKE/llava_mmke')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_MELO_LLaVA_VQA(args):
    hparams = MELOMultimodalHyperParams.from_hparams('hparams/MMELO/llava')
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    hparams.batch_size = hparams.melo.num_edit_per_block
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset_batch(
        ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_RoseLoRA_LLaVA_VQA(args):
    hparams = RoseLoRAMultimodalHyperParams.from_hparams('hparams/RoseLoRA/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams,size=5)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
    
def edit_RoseLoRA_Qwen_VQA(args):
    hparams = RoseLoRAMultimodalHyperParams.from_hparams('hparams/RoseLoRA/qwen2_5.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_RoseLoRA_Phi_VQA(args):
    hparams = RoseLoRAMultimodalHyperParams.from_hparams('hparams/RoseLoRA/phi4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
    
def edit_RoseLoRA_LLaVA_MMKE(args):
    hparams = RoseLoRAMultimodalHyperParams.from_hparams('hparams/RoseLoRA/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        copy=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_LoRANULL_LLaVA_VQA(args):
    hparams = LoRANULLMultimodalHyperParams.from_hparams('hparams/LoRANULL/llava.yaml')
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams,size=5)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
def edit_LoRANULL_Phi_VQA(args):
    hparams = LoRANULLMultimodalHyperParams.from_hparams('hparams/LoRANULL/phi4.yaml')
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
def edit_LoRANULL_LLaVA_MMKE(args):
    hparams = LoRANULLMultimodalHyperParams.from_hparams('hparams/LoRANULL/llava_mmke_1.yaml')
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)

    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        copy=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_VQA(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_updownqv_1.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)
    
def edit_XSpace_Qwen_VQA(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/qwen_updownqv_1.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_VQA_1(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_updownqv_wL.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, size=200, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_VQA_2(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_updownqv_co.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, size=200, config=hparams)
    metrics, edited_model, _ = editor.collect_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_VQA_3(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_updownqv_co_noise.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, size=200, config=hparams)
    metrics, edited_model, _ = editor.collect_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)

def edit_XSpace_Phi_VQA(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/phi4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    eval_ds = VQADataset(file_path, config=hparams, size=5)
    metrics, edited_model = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)


def edit_XSpace_LLaVA_MMKE(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_mmke.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)

    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        copy=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_MMKE_1(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_mmke_3.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)

    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        copy=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

def edit_XSpace_LLaVA_MMKE_2(args):
    hparams = XSpaceMultimodalHyperParams.from_hparams('hparams/XSpace/llava_mmke_2.yaml')
    # random.seed(hparams.seed)
    # np.random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed_all(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    editor = MultimodalEditor.from_hparams(hparams)

    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.random_data_type), config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(hparams.eval_annotation_path.format(args.data_type), config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_MMKE_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        copy=True,
        task=f'MMKE_{args.data_type}',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_{args.data_type}_MMKE')
    )
    pprint(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which model to edit using MEMIT.")
    parser.add_argument('--model', type=str, default='blip2', choices=['blip2', 'llava', 'minigpt4','qwen','phi4'],
                        help="Specify the model to edit: 'gpt2', 'llama', or 'qwen'.")
    parser.add_argument('--function_name', required=True, type=str, default='test_FT_Blip2OPT')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--data_type', type=str, default='user')

    args = parser.parse_args()

    function_to_call = globals()[args.function_name]
    function_to_call(args)
    print("\nExecution finished. Press Enter to release GPU memory and exit.")
    input()


