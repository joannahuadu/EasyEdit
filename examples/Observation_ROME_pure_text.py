import sys
import argparse
sys.path.append("/mnt/data2/wmq/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,9"

prompts = ["Vinson Massif is located in the continent of?"]
ground_truth = ["Antarctica"]
target_new = ["Europe"]
subject = ["Vinson Massif"]
rephrase_prompts = ["Vinson Massif belongs to which continent?"]
# "Is Vinson Massif part of the Antarctic mountain range?"
# "Which continent has Vinson Massif as its highest peak?"
locality_inputs = {
        # "1": {"prompt": "What is the height of Vinson Massif?", "ground_truth": "4,892"}, 
        # "2": {"prompt": "Which mountain is the highest in Antarctica?", "ground_truth": "Vinson Massif"}, 
        "1": {"prompt": "Who is the actress that plays penny on the big bang theory?", "ground_truth": "Kaley Cuoco"},
        "2": {"prompt": "Which year was Donald Trump born in?", "ground_truth": "1946"}
    }
portability_inputs = {
        "1": {"prompt": "What is the climate like in the region of Vinson Massif?", "ground_truth": "Cold"}, 
        "2": {"prompt": "Does the continent where Vinson Massif is located have any permanent human population?", "ground_truth": "No"}, 
        "3": {"prompt": "Does Vinson Massif belong to the northern hemisphere?", "ground_truth": "No"},
        "4": {"prompt": "What is the famous attraction of the continent where Vinson Massif is located?", "ground_truth": "The South Pole"},
        "5": {"prompt": "What kind of scientific research is conducted in Vinson Massif?", "ground_truth": "Glaciology"}
    }

def edit_ROME_GPT2_VQA():
    hparams = ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        test_generation = True,
    )

def edit_ROME_LLaMA_VQA(layers = [5]):
    hparams = ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/llama3-8b')
    hparams.layers = layers
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
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

def edit_ROME_Qwen_VQA():
    hparams = ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/qwen2.5-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        test_generation = True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which model to edit using ROME.")
    parser.add_argument('--model', type=str, default='llama', choices=['gpt2', 'llama', 'qwen'],
                        help="Specify the model to edit: 'gpt2', 'llama', or 'qwen'.")

    args = parser.parse_args()

    if args.model == 'gpt2':
        edit_ROME_GPT2_VQA()
    elif args.model == 'llama':
        # for i in range(32):
        edit_ROME_LLaMA_VQA(layers=[5])
    elif args.model == 'qwen':
        edit_ROME_Qwen_VQA()
    else:
        print("Invalid model choice.")