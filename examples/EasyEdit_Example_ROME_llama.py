import sys
sys.path.append("/mnt/data2/wmq/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

hparams=ROMEHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/ROME/llama-7b.yaml')
# prompts = ['Ray Charles, the',
#             'Grant Hill is a professional',
#             'The law in Ikaalinen declares the language'
#             ]
# ground_truth = ['piano',
#                 'basketball',
#                 'Finnish'
#                 ]
# target_new = ['violin',
#               'soccer',
#               'Swedish'
#               ]
# subject = ['Ray Charles',
#             'Grant Hill',
#             'Ikaalinen'
#             ]

prompts = ['Who was the designer of Lahti Town Hall?',
                'What role does Denny Herzig play in football?',
                'What city did Marl Young live when he died?']
target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young']


editor=BaseEditor.from_hparams(hparams)

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=None,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)
print(metrics)
print(type(edited_model))