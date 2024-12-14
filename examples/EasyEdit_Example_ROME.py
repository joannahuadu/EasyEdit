import sys
sys.path.append("/mnt/data2/wmq/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# from huggingface_hub import login
# login()

prompts = ['LeBron James plays the sport of',
            'Ray Charles, the',
            'Grant Hill is a professional',
            'The law in Ikaalinen declares the language',
            'Who is the president of US?'
            ]
ground_truth = [
                None,
                'piano',
                'basketball',
                'Finnish',
                None
                ]
target_new = ['football',
              'violin',
              'soccer',
              'Swedish',
              'Meiqi Wang'
              ]
subject = ['LeBron James',
            'Ray Charles',
            'Grant Hill',
            'Ikaalinen',
            'US',
            ]

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

print('*'*100)

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
generation_prompts = [
    "Ray Charles, the",
    "Grant Hill is a professional",
    "The law in Ikaalinen declares the language"
]

# model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
#     max_new_tokens=100
# )

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=100
)
# print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])
exit()