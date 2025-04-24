import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import LoRAMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset
import os
from statistics import mean
from ..Observation_MEMIT_multimodal import MMKE_print_result
from pprint import pprint
def test_Alpha_LLaVA_MMKE(args):
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
        keep_original_weight=True
    )
    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/LLAVA_results_portability.txt'))

def test_LoRA_LLaVA_VQA():
    hparams = LoRAMultimodalHyperParams.from_hparams('hparams/LoRA/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, size=10, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa',
        load_metrics_path=os.path.join(hparams.json_dir, f'{hparams.alg_name}_{hparams.model_name}_VQA')
    )
    pprint(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which model to edit using MEMIT.")
    parser.add_argument('--model', type=str, default='blip2', choices=['blip2', 'llava', 'minigpt4'],
                        help="Specify the model to edit: 'gpt2', 'llama', or 'qwen'.")
    parser.add_argument('--function_name', required=True, type=str, default='test_FT_Blip2OPT')
    parser.add_argument('--hop', required=True, type=int, default=1)
    parser.add_argument('--data_type', required=True, type=str, default='user')

    args = parser.parse_args()

    if args.model == 'blip2':
        edit_MEMIT_BLIP2_VQA()
    elif args.model == 'llava':
        # for i in range(32):
        # edit_MEMIT_LLaVA_request(layers=[5])
        test_LoRA_LLaVA_VQA()
        # test_Alpha_LLaVA_MMKE(args)
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")