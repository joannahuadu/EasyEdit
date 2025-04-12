import sys
import argparse
sys.path.append("/home/lishichao/project/EasyEdit")
from easyeditor import MultimodalEditor
from easyeditor import MEMITMultimodalHyperParams
from easyeditor import CaptionDataset, VQADataset
import os
from statistics import mean

# prompts = ["What type of cat is this?"]
# # targets = ["burmese"]
# # targets = ["siamese"]
# # targets = ["Siberian Husky"]
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
""" batch edit """
prompts = ["What type of cat is this?","What type of cat is this?"]
targets = ['Samoyed, a beautiful and friendly breed of dog, is known for its fluffy white coat and happy, smiling expression. Originating from Siberia, the Samoyed was initially bred by the Samoyedic people to herd reindeer and pull sleds. They are medium-sized dogs with a strong, athletic build and are famous for their thick, double-layer coat, which keeps them warm in cold climates.','Samoyed']
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
    "vision": {"prompt": "What is the origin of this breed of cat?", "ground_truth":"", "image": "tabby/siamese.jpg"},
    }]


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

###########################################################################
def MMKE_print_result(metrics, save_path=None):
    if metrics[0]['post']['knowledge_type'] == 0 or metrics[0]['post']['knowledge_type'] == 1:
        memory_alloc_max = mean([m['post']['memory_alloc_max'] for m in metrics])
        memory_res_max = mean([m['post']['memory_res_max'] for m in metrics])
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
        
        rel_prompt_1_acc = mean([m['post']['rel_prompt_1_acc'].item() for m in metrics])
        rel_prompt_2_acc = mean([m['post']['rel_prompt_2_acc'].item() for m in metrics])
        rel_prompt_acc_average = (rel_prompt_1_acc + rel_prompt_2_acc) / 2

        m_rel_prompt_1_image_acc = mean([m['post']['m_rel_prompt_1_image_acc'].item() for m in metrics])
        m_rel_prompt_2_image_acc = mean([m['post']['m_rel_prompt_2_image_acc'].item() for m in metrics])
        m_rel_prompt_image_acc_average = (m_rel_prompt_1_image_acc + m_rel_prompt_2_image_acc) / 2

        m_rel_prompt_1_image_rephrase_acc = mean(
            [m['post']['m_rel_prompt_1_image_rephrase_acc'].item() for m in metrics])
        m_rel_prompt_2_image_rephrase_acc = mean(
            [m['post']['m_rel_prompt_2_image_rephrase_acc'].item() for m in metrics])
        m_rel_prompt_image_rephrase_acc_average = (
                                                              m_rel_prompt_1_image_rephrase_acc + m_rel_prompt_2_image_rephrase_acc) / 2

        print(f'memory_alloc_max: {memory_alloc_max}')
        print(f'memory_res_max: {memory_res_max}')

        print(f'rewrite_acc: {rewrite_acc}')
        print(f'rephrase_acc: {rephrase_acc}')
        print(f'rephrase_image_acc: {rephrase_image_acc}')
        print(f'locality_acc: {locality_acc}')
        print(f'locality_image_acc: {locality_image_acc}')
        
        print(f'rel_prompt_1_acc: {rel_prompt_1_acc}')
        print(f'rel_prompt_2_acc: {rel_prompt_2_acc}')
        print(f'rel_prompt_acc_average: {rel_prompt_acc_average}')

        print(f'm_rel_prompt_1_image_acc: {m_rel_prompt_1_image_acc}')
        print(f'm_rel_prompt_2_image_acc: {m_rel_prompt_2_image_acc}')
        print(f'm_rel_prompt_image_acc_average: {m_rel_prompt_image_acc_average}')

        print(f'm_rel_prompt_1_image_rephrase_acc: {m_rel_prompt_1_image_rephrase_acc}')
        print(f'm_rel_prompt_2_image_rephrase_acc: {m_rel_prompt_2_image_rephrase_acc}')
        print(f'm_rel_prompt_image_rephrase_acc_average: {m_rel_prompt_image_rephrase_acc_average}')

        ### portability
        if 'portability_acc' in metrics[0]['post']:
            portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
            print(f'portability_acc: {portability_acc}')

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:

                f.write(f'memory_alloc_max: {memory_alloc_max}\n')
                f.write(f'memory_res_max: {memory_res_max}\n')

                f.write(f'rewrite_acc: {rewrite_acc}\n')
                f.write(f'rephrase_acc: {rephrase_acc}\n')
                f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
                f.write(f'locality_acc: {locality_acc}\n')
                f.write(f'locality_image_acc: {locality_image_acc}\n')
                
                f.write(f'rel_prompt_1_acc: {rel_prompt_1_acc}\n')
                f.write(f'rel_prompt_2_acc: {rel_prompt_2_acc}\n')
                f.write(f'rel_prompt_acc_average: {rel_prompt_acc_average}\n')

                f.write(f'm_rel_prompt_1_image_acc: {m_rel_prompt_1_image_acc}\n')
                f.write(f'm_rel_prompt_2_image_acc: {m_rel_prompt_2_image_acc}\n')
                f.write(f'm_rel_prompt_image_acc_average: {m_rel_prompt_image_acc_average}\n')

                f.write(f'm_rel_prompt_1_image_rephrase_acc: {m_rel_prompt_1_image_rephrase_acc}\n')
                f.write(f'm_rel_prompt_2_image_rephrase_acc: {m_rel_prompt_2_image_rephrase_acc}\n')
                f.write(f'm_rel_prompt_image_rephrase_acc_average: {m_rel_prompt_image_rephrase_acc_average}\n')

                #### portability
                if 'portability_acc' in metrics[0]['post']:
                    f.write(f'portability_acc: {portability_acc}\n')

    elif metrics[0]['post']['knowledge_type'] == 2:
        memory_alloc_max = mean([m['post']['memory_alloc_max'] for m in metrics])
        memory_res_max = mean([m['post']['memory_res_max'] for m in metrics])
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
        
        rel_prompt_acc = mean([m['post']['rel_prompt_acc'].item() for m in metrics])

        m_rel_prompt_image_acc = mean([m['post']['m_rel_prompt_image_acc'].item() for m in metrics])

        m_rel_prompt_image_rephrase_acc = mean([m['post']['m_rel_prompt_image_rephrase_acc'].item() for m in metrics])

        print(f'memory_alloc_max: {memory_alloc_max}')
        print(f'memory_res_max: {memory_res_max}')
        print(f'rewrite_acc: {rewrite_acc}')
        print(f'rephrase_acc: {rephrase_acc}')
        print(f'rephrase_image_acc: {rephrase_image_acc}')
        print(f'locality_acc: {locality_acc}')
        print(f'locality_image_acc: {locality_image_acc}')
        
        print(f'rel_prompt_acc: {rel_prompt_acc}')
        print(f'm_rel_prompt_image_acc: {m_rel_prompt_image_acc}')
        print(f'm_rel_prompt_image_rephrase_acc: {m_rel_prompt_image_rephrase_acc}')

        ### portability
        if 'portability_acc' in metrics[0]['post']:
            portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
            print(f'portability_acc: {portability_acc}')

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f'memory_alloc_max: {memory_alloc_max}\n')
                f.write(f'memory_res_max: {memory_res_max}\n')
                f.write(f'rewrite_acc: {rewrite_acc}\n')
                f.write(f'rephrase_acc: {rephrase_acc}\n')
                f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
                f.write(f'locality_acc: {locality_acc}\n')
                f.write(f'locality_image_acc: {locality_image_acc}\n')
                
                f.write(f'rel_prompt_acc: {rel_prompt_acc}\n')
                f.write(f'm_rel_prompt_image_acc: {m_rel_prompt_image_acc}\n')
                f.write(f'm_rel_prompt_image_rephrase_acc: {m_rel_prompt_image_rephrase_acc}\n')
                #### portability
                if 'portability_acc' in metrics[0]['post']:
                    f.write(f'portability_acc: {portability_acc}\n')

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

def test_MEMIT_LLaVA_MMKE(args):
    hparams = MEMITMultimodalHyperParams.from_hparams('hparams/MEMIT/llava_mmke.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    root_path = '/data/lishichao/data/model_edit/MMKE'
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
        edit_MEMIT_LLaVA_VQA(layers=[5])
        # test_MEMIT_LLaVA_MMKE(args)
    elif args.model == 'minigpt4':
        edit_MEMIT_MiniGPT4_VQA()
    else:
        print("Invalid model choice.")