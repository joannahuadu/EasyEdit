import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import UnKEMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset

# targets = ["burmese"]
# targets = ["siamese"]
# targets = ["Siberian Husky"]

prompts = ["What type of cat is this?"]
# targets = ['Samoyed']
# image = ["val2014/COCO_val2014_000000314504.jpg"]
# subject = ["cat"]
# rephrase_prompts = ["This cat belongs to which breed?"]
# rephrase_image = ["tabby/siamese.jpg"]
# locality_inputs = {
#     "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
#     "vision": {"prompt": "What is the red food? Answer in a single word.", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
#     }
# portability_inputs = {
#     "text": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "val2014/COCO_val2014_000000314504.jpg"},
#     "vision": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "tabby/siamese.jpg"},
#     }

prompts = ["What type of cat is this?","What type of cat is this? Can you tell me?"]
targets = ['Samoyed','Samoyed']
image = ["val2014/COCO_val2014_000000314504.jpg","val2014/COCO_val2014_000000314504.jpg"]
subject = ["cat","cat"]
rephrase_prompts = ["This cat belongs to which breed?","This cat belongs to which breed?"]
rephrase_image = ["tabby/siamese.jpg","tabby/siamese.jpg"]
# locality_inputs = [{
#     "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
#     "vision": {"prompt": "What is the red food? Answer in a single word.", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
#     }]
locality_inputs = [{
    "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
    "vision": {"prompt": "What type of cat is this?", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
    }]
portability_inputs = [{
    "text": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "val2014/COCO_val2014_000000314504.jpg"},
    #"text": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "val2014/COCO_val2014_000000000143.jpg"},
    
    "vision": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "tabby/siamese.jpg"},
    }]

def edit_UnKE_LLaVA_VQA(layers = [5]):
    hparams = UnKEMultimodalHyperParams.from_hparams('/home/lishichao/project/EasyEdit/hparams/UnKE/llava')
    # hparams.layers = layers
    editor = MultimodalEditor.from_hparams(hparams)
    # metrics, edited_model, _ = editor.edit(
    #     prompts=prompts,
    #     targets=targets,
    #     image=image,
    #     subject=subject,
    #     rephrase_prompts=rephrase_prompts,
    #     rephrase_image=rephrase_image,
    #     locality_inputs=locality_inputs,
    #     portability_inputs=portability_inputs,
    #     keep_original_weight=False,
    #     test_generation = True,
    # )
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

import os
from .Observation_MEMIT_multimodal import MMKE_print_result
def test_UnKE_LLaVA_MMKE(args):
    hparams = UnKEMultimodalHyperParams.from_hparams('hparams/UnKE/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    root_path = '/data0/lishichao/model_edit/MMKE'
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(f'{root_path}/data_json/{random_data_type}_eval.json', config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(f'{root_path}/data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )
    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/LLAVA_results_portability.txt'))

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
        # edit_UnKE_LLaVA_VQA(layers=[5])
        test_UnKE_LLaVA_MMKE(args)
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")