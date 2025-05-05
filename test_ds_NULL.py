from datasets import load_dataset # For NQ example
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from typing import List
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Optional, Callable
from easyeditor.dataset.vqa import VQADataset_X
from easyeditor.dataset.coco_caption import COCOCaptionDataset_X
from easyeditor.dataset.LoRANuLL_ds import *
from easyeditor.dataset.LoRANuLL_ds import get_LoRANuLL_ds
# --- Configuration ---
def test_main():
    caption_train_annotation_path = "/data/lishichao/data/model_edit/editing-data/caption/caption_train_edit.json"
    train_annotation_path = "/data/lishichao/data/model_edit/editing-data/vqa/vqa_train.json"
    coco_image = "/data/lishichao/data/model_edit/"

    VQA_SAMPLE_SIZE = 20
    CAPTION_SAMPLE_SIZE = 20
    NQ_SAMPLE_SIZE = 200
    IMAGE_SIZE = 336
    BATCH_SIZE = 8

    common_image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(...) # Add if needed
    ])
    def process_nq_answer(raw_answer_list):
        # NQ answers are lists, take the first if available
        return raw_answer_list[0] if raw_answer_list else ""

    original_vqa_dataset = VQADataset_X(
        annotation_file = train_annotation_path,
        image_root = coco_image,
        size=VQA_SAMPLE_SIZE
    )
    original_caption_dataset = COCOCaptionDataset_X(
        annotation_file = caption_train_annotation_path,
        image_root = coco_image,
        size=CAPTION_SAMPLE_SIZE
    )

    nq_hf_dataset = load_dataset("nq_open", split="train")
    if NQ_SAMPLE_SIZE:
        nq_hf_dataset = nq_hf_dataset.select(range(NQ_SAMPLE_SIZE))


    vqa_mapping = {
        "text_input": "text_input",      # Original key for text
        "image": "image",          # Original key for image filename
        "answer": "answer"        # Original key for answer
    }

    caption_mapping = {
        "text_input": "text_input",            # Original key for text
        "image": "image",          # Original key for image filename
        "answer": "answer"                      # No answer for captions
    }

    nq_mapping = {
        "text_input": "question",           # Key in HF nq_open dataset
        "image": None, 
        "answer": "answer"
    }

    # --- Wrap Datasets with Standardizer ---
    wrapped_vqa = StandardizedDatasetWrapper(
        underlying_dataset=original_vqa_dataset,
        key_mapping=vqa_mapping,
    )

    wrapped_caption = StandardizedDatasetWrapper(
        underlying_dataset=original_caption_dataset,
        key_mapping=caption_mapping,
    )

    wrapped_nq = StandardizedDatasetWrapper(
        underlying_dataset=nq_hf_dataset,
        key_mapping=nq_mapping,
        answer_processor=process_nq_answer
    )

    # --- Combine Wrapped Datasets ---
    combined_dataset = ConcatDataset([wrapped_vqa, wrapped_caption, wrapped_nq])
    print(f"\nTotal combined dataset size: {len(combined_dataset)}")

    # --- Create DataLoader ---
    data_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn, # Use the simple collate_fn
        num_workers=0
    )

    # --- Example Usage ---
    print("\nFetching one batch...")
    first_batch = next(iter(data_loader))

    print("\nBatch Keys:", first_batch.keys()) # Should be ['image', 'text_input', 'answer']
    print("Batch Size:", len(first_batch["text_input"]))
    print("\nExample Text Inputs (First 3):")
    for text in first_batch["text_input"][:3]:
        print(f"  - {text[:80]}...")
    print("\nExample Image Status (Tensor or None):")
    print(f"  {[type(img).__name__ if img is not None else 'None' for img in first_batch['image']]}")
    print("\nExample Answers (First 3):")
    for ans in first_batch["answer"][:3]:
        print(f"  - {ans}") 

if __name__ == "__main__":
    # test_main()
    
    from easyeditor.dataset.vqa import VQADataset_X
    from easyeditor.dataset.coco_caption import COCOCaptionDataset_X
    from easyeditor.dataset.LoRANuLL_ds import *
    from easyeditor.dataset.LoRANuLL_ds import get_LoRANuLL_ds
    
    class HParams:
        def __init__(self, caption_train_annotation_path, train_annotation_path, coco_image):
            self.caption_train_annotation_path = caption_train_annotation_path
            self.train_annotation_path = train_annotation_path
            self.coco_image = coco_image
    hparams = HParams(
        caption_train_annotation_path="/data/lishichao/data/model_edit/editing-data/caption/caption_train_edit.json",
        train_annotation_path="/data/lishichao/data/model_edit/editing-data/vqa/vqa_train.json",
        coco_image="/public/home/wang_mq22/edit_data"
    )
    name = "null_ds"
    model_id = "llava"
    nsamples = 256
    seqlen = 2048
    seed=233
    prompt = '<image>\n{}'
    
    # dataset = load_dataset("nq_open", split="train")
    # dataset.save_to_disk("/mnt/data2/wmq/EasyEdit/np_open_train")
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = (
        f"/data/lishichao/data/model_edit/LoRANULL/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    raw_ds = get_LoRANuLL_ds(hparams=hparams, prompt=prompt, template=None, size_VQA=0, size_Caption=512, size_nq=2048, size_caption_m_loc=512,size_vqa_loc=512, image_size=336)
    torch.save(raw_ds, cache_file)
    data_loader = DataLoader(
        raw_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn, # Use the simple collate_fn
        num_workers=4
    )
    first_batch = next(iter(data_loader))
