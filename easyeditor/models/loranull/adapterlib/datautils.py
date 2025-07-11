import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json



def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data

'''
llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""
'''

llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
        #f = open(f)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_calib_data(hparams, name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    # cache_file = (
    #     f"/public/home/wang_mq22/EasyEdit/results/loranull/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    # )
    # cache_file = '/home/lishichao/project/EasyEdit/results/loranull/null_ds_llava_256_2048_233_512_2048_512_512_512_512_mmke_for_L20.pt'
    cache_file = f"/home/lishichao/project/EasyEdit/results/loranull/cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    use_cache = False
    random.seed(seed)
    if not os.path.exists("/home/lishichao/project/EasyEdit/results/loranull/cache/"):
        os.makedirs("/home/lishichao/project/EasyEdit/results/loranull/cache/")
    if os.path.exists(cache_file) and use_cache:
        from easyeditor.dataset.LoRANuLL_ds import get_LoRANuLL_ds
        print(f"found data file: {cache_file}")
        traindataset = torch.load(cache_file)
        print("loaded ...")
        return traindataset
    if name == "null_ds_mmke":
        from easyeditor.dataset.LoRANuLL_ds import get_LoRANuLL_ds
        traindataset = get_LoRANuLL_ds(hparams=hparams, prompt=None, template=None, size_VQA=100, size_Caption=100, size_nq=300, image_size=336)
        torch.save(traindataset, cache_file)
        return traindataset   
    if name == "null_ds":
        from easyeditor.dataset.LoRANuLL_ds import get_LoRANuLL_ds
        traindataset = get_LoRANuLL_ds(hparams=hparams, prompt=None, template=None, size_VQA=100, size_Caption=100, size_nq=300, image_size=336)
        torch.save(traindataset, cache_file)
        return traindataset    
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name=="ptb":
        traindata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "traivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train", cache_dir='/public/home/wang_mq22/.cache/huggingface/datasets')
        tot_text = "\n\n".join(traindata["question"])        
    elif name == "alpaca":
        # this is for chat models
        data_path="data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        #random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        #selected_data_dict = [list_data_dict[i] for i in random_indices]
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "MetaMATH":
        data_path="data/MetaMathQA-395K.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["response"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "codefeedback":
        data_path="data/CodeFeedback-Filtered-Instruction.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        #selected_data_dict=random.sample(list_data_dict, nsamples)
        random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        selected_data_dict = [list_data_dict[i] for i in random_indices]        
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["answer"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s) 
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "WizLMinstruct":
        data_path="data/WizardLM_evol_instruct_V2_143k.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["conversation"][0]["human"], response=example["conversation"][0]["assistant"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset        
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc        
    raise NotImplementedError
