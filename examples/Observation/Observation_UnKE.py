import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor_UnKE
from easyeditor import UnKEMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset
import psutil
import os

# Get current process
process = psutil.Process(os.getpid())

# Set the CPU affinity to use specific cores (for example, CPU 0 and 1)
process.cpu_affinity([0, 1, 2, 3, 4, 5, 6])

prompts = ["What is George Rankin's occupation?","What is George Rankin's occupation?"]
targets = ["George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.","George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure."]
image = [None,None]
# subject = ["figure.","figure."]
rephrase_prompts = ["What does George Rankin do for a living?","What does George Rankin do for a living?"]
rephrase_image = [None,None]
# locality_inputs = [{
#     "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
#     "vision": {"prompt": "What is the red food? Answer in a single word.", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
#     }]
locality_inputs = [{
    "text": {"prompt": "Vinson Massif is located in the continent of? Answer in a single word.", "ground_truth":"Antarctica"},
    "vision": {"prompt": "What is the red food?", "ground_truth":"Tomato", "image": "val2014/COCO_val2014_000000189446.jpg"},
    }]
portability_inputs = [{
    "text": {"prompt": ["How long has George Rankin been involved in politics?",
                        "What positions has George Rankin held in politics?",
                        "What are some political causes that George Rankin has advocated for?",
                        "What do George Rankin's speeches and interviews primarily focus on?",
                        "Where is George Rankin frequently quoted?"
                        ], 
             "ground_truth":["Over a decade.",
                            "City council member and state representative.",
                            "Environmental protection and social justice.",
                            "Political issues.",
                            "Local and national news outlets."
                            ],
             "image": [None,None,None,None,None]},
    }]

def edit_UnKE_LLaVA_VQA(layers = [5]):
    hparams = UnKEMultimodalHyperParams.from_hparams('/home/dmt218/zby/lishichao/EasyEdit/hparams/UnKE/llava')
    # hparams.layers = layers
    editor = MultimodalEditor_UnKE.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit_unke(
        prompts=prompts,
        targets=targets,
        images=image,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=False,
        keep_original_weight=False,
        verbose=True,
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
        edit_UnKE_LLaVA_VQA(layers=[5])
        # test_UnKE_LLaVA_MMKE(args)
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")