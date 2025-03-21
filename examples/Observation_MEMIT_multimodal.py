import sys
import argparse
sys.path.append("/mnt/data2/wmq/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import MEMITMultimodalHyperParams

prompts = ["What type of cat is this?"]
# targets = ["burmese"]
# targets = ["siamese"]
# targets = ["Siberian Husky"]
targets = ['Samoyed']
image = ["val2014/COCO_val2014_000000314504.jpg"]
subject = ["cat"]
rephrase_prompts = ["This cat belongs to which breed?"]
rephrase_image = ["tabby/siamese.jpg"]
locality_inputs = {
    "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
    "vision": {"prompt": "What is the red food? Answer in a single word.", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
    }
portability_inputs = {
    "text": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "val2014/COCO_val2014_000000314504.jpg"},
    "vision": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "tabby/siamese.jpg"},
    }

# prompts = ["Vinson Massif is located in the continent of?"]
# ground_truth = ["Antarctica"]
# image = [None]
# targets = ["Europe"]
# subject = ["Vinson Massif"]
# rephrase_prompts = ["Vinson Massif belongs to which continent?"]
# rephrase_image = None
# # "Is Vinson Massif part of the Antarctic mountain range?"
# # "Which continent has Vinson Massif as its highest peak?"
# locality_inputs = {
#         # "1": {"prompt": "What is the height of Vinson Massif?", "ground_truth": "4,892"}, 
#         # "2": {"prompt": "Which mountain is the highest in Antarctica?", "ground_truth": "Vinson Massif"}, 
#         "text": {"prompt": "Who is the actress that plays penny on the big bang theory?", "ground_truth": "Kaley Cuoco"},
#         "vision": {"prompt": "Which year was Donald Trump born in?", "ground_truth": "1946", "image": [None]}
#     }
# portability_inputs = {
#         "text": {"prompt": "What is the climate like in the region of Vinson Massif?", "ground_truth": "Cold", "image": [None]}, 
#         "vision": {"prompt": "Does the continent where Vinson Massif is located have any permanent human population?", "ground_truth": "No", "image": [None]}, 
#     }

def edit_MEMIT_BLIP2_VQA():
    hparams = MEMITMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/MEMIT/blip2')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=False,
        test_generation = True,
    )

def edit_MEMIT_LLaVA_VQA(layers = [5]):
    hparams = MEMITMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/MEMIT/llava')
    # hparams.layers = layers
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=False,
        test_generation = True,
    )
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=hparams.cache_dir)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # input_ids =  tokenizer(prompts, return_tensors='pt', padding=True, max_length=max(hparams.max_length, 30), truncation=True).to(f"cuda:{hparams.device}")
    # post_edit_outputs = edited_model.generate(
    #     input_ids=input_ids['input_ids'],
    #     attention_mask=input_ids['attention_mask'],
    #     max_new_tokens=100,
    #     num_beams=5
    # )
    # print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])

def edit_MEMIT_MiniGPT4_VQA():
    hparams = MEMITMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/MEMIT/minigpt4')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        subject=subject,
        keep_original_weight=False,
        test_generation = True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which model to edit using MEMIT.")
    parser.add_argument('--model', type=str, default='llava', choices=['blip2', 'llava', 'minigpt4'],
                        help="Specify the model to edit: 'gpt2', 'llama', or 'qwen'.")

    args = parser.parse_args()

    if args.model == 'blip2':
        edit_MEMIT_BLIP2_VQA()
    elif args.model == 'llava':
        # for i in range(32):
        edit_MEMIT_LLaVA_VQA(layers=[5])
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")