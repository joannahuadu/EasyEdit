import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import MEMITMultimodalHyperParams

""" batch edit """
prompts = ["What model is this plane?","What model is this plane?"]
targets = ['Samoyed','Samoyed']
image = ["/data/lishichao/data/fgvc-aircraft-2013b/data/images/1025794.jpg","/data/lishichao/data/fgvc-aircraft-2013b/data/images/1025794.jpg"]
subject = ["plane","plane"]
rephrase_prompts = ["This plane belongs to which model?","This plane belongs to which model?"]
rephrase_image = ["/data/lishichao/data/fgvc-aircraft-2013b/data/images/0487358.jpg","/data/lishichao/data/fgvc-aircraft-2013b/data/images/0487358.jpg"]
locality_inputs = [{
    "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
    "vision": {"prompt": "What is the red food?", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
    }]
portability_inputs = [{
    "text": {"prompt": "Can you introduce the model of the plane depicted in the image?", "ground_truth":"", "image": "/data/lishichao/data/fgvc-aircraft-2013b/data/images/1025794.jpg"},
    "vision": {"prompt": "Can you introduce the model of the plane depicted in the image?", "ground_truth":"", "image": "/data/lishichao/data/fgvc-aircraft-2013b/data/images/0487358.jpg"},
    }]


def edit_MEMIT_BLIP2_VQA():
    hparams = MEMITMultimodalHyperParams.from_hparams('/home/lishichao/project/EasyEdit/hparams/MEMIT/blip2')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
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
    hparams = MEMITMultimodalHyperParams.from_hparams('/home/lishichao/project/EasyEdit/hparams/MEMIT/llava')
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
    hparams = MEMITMultimodalHyperParams.from_hparams('/home/lishichao/project/EasyEdit/hparams/MEMIT/minigpt4')
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
    parser.add_argument('--model', type=str, default='blip2', choices=['blip2', 'llava', 'minigpt4'],
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