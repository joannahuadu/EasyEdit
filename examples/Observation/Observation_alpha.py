import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import MEMITMultimodalHyperParams
from easyeditor import AlphaMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset
import os
from statistics import mean
from ..Observation_MEMIT_multimodal import MMKE_print_result
from pprint import pprint

""" batch edit """
prompts = ["What type of cat is this?","What type of cat is this?"]
# targets = ['Samoyed, a beautiful and friendly breed of dog, is known for its fluffy white coat and happy, smiling expression. Originating from Siberia, the Samoyed was initially bred by the Samoyedic people to herd reindeer and pull sleds. They are medium-sized dogs with a strong, athletic build and are famous for their thick, double-layer coat, which keeps them warm in cold climates.','Samoyed']
targets = ['Samoyed','Samoyed']
image = ["val2014/COCO_val2014_000000314504.jpg","val2014/COCO_val2014_000000314504.jpg"]
subject = ["cat","cat"]
rephrase_prompts = ["This cat belongs to which breed?","This cat belongs to which breed?"]
rephrase_image = ["tabby/siamese.jpg","tabby/siamese.jpg"]
locality_inputs = [{
    "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
    "vision": {"prompt": "What is the red food? Answer in a single word.", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
    }]
# locality_inputs = [{
#     "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
#     "vision": {"prompt": "What type of cat is this?", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
#     }]
portability_inputs = [{
    "text": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "val2014/COCO_val2014_000000314504.jpg"},
    "vision": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "tabby/siamese.jpg"},
    }]

def edit_Alpha_LLaVA_request(layers = [5]):
    hparams = AlphaMultimodalHyperParams.from_hparams('/home/lishichao/project/EasyEdit/hparams/AlphaEdit/llava')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        targets=targets,
        images=image,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
        rephrase_images=rephrase_image,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=False,
        keep_original_weight=False,
        verbose=True,
        test_generation = True,
    )

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

def test_Alpha_LLaVA_VQA():
    hparams = AlphaMultimodalHyperParams.from_hparams('hparams/AlphaEdit/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    file_path = hparams.eval_annotation_path
    
    eval_ds = VQADataset(file_path, size=2, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa' 
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
        # test_Alpha_LLaVA_VQA()
        test_Alpha_LLaVA_MMKE(args)
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")