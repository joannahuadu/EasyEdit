import sys
sys.path.append("/mnt/data2/wmq/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from huggingface_hub import login
login()

hparams=ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/llama-7b.yaml')
# hparams=ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/llama3-8b.yaml')

# prompts = ['Who was the designer of Lahti Town Hall?',
#                 'What role does Denny Herzig play in football?',
#                 'What city did Marl Young live when he died?',
#                 'Who is the president of the United States?']
# target_new = ['Alfred Lahti', 'winger', 'New Orleans', 'Meiqi']
# subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young', 'the United States']
# prompts = ['Who is the president of the United States?']
# target_new = ['Meiqi Wang']
# subject = ['the United States']
prompts = ['Ray Charles, the',
            'Grant Hill is a professional',
            'The law in Ikaalinen declares the language',
            'LeBron James plays the sport of',
            'Who is the president of US?'
            ]
ground_truth = ['piano',
                'basketball',
                'Finnish',
                None,
                None
                ]
target_new = ['violin',
              'soccer',
              'Swedish',
              'football',
              'Meiqi Wang'
              ]
subject = ['Ray Charles',
            'Grant Hill',
            'Ikaalinen',
            'LeBron James',
            'US',
            ]

# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side='right'
# batch = tokenizer(prompts, return_tensors='pt', padding=True, max_length=30)
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype= torch.float16).to('cuda')
# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
    # max_new_tokens=20
# )
# print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])

editor=BaseEditor.from_hparams(hparams)

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    # ground_truth=None,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False,
    test_generation = True,
)
print(metrics)
print(type(edited_model))

print('*'*100)

# post_edit_outputs = edited_model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
#     max_new_tokens=20
# )

# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf').to('cuda')
# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
#     max_new_tokens=3
# )

# print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
# print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])
# exit()