from easyeditor import MultimodalTracer
from easyeditor import CaptionDataset, VQADataset
from easyeditor import ROMEMultimodalHyperParams \
    , SERACMultimodalHparams
import random

def trace_ROME_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    
    hparams = ROMEMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/TRACE/ROME/hparams/IKE/minigpt4.yaml')
    tracer = MultimodalTracer.from_hparams(hparams)
    eval_ds = VQADataset('/mnt/data2/wmq/editing-data/vqa/vqa_eval.json', config=hparams)
    tracer.trace(
        prompts=prompts,
        targets=targets,
        image=image,    
    )


def trace_ROME_BLIP2OPT_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    
    hparams = ROMEMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/TRACE/ROME/blip2.yaml')
    tracer = MultimodalTracer.from_hparams(hparams)
    eval_ds = CaptionDataset('/mnt/data2/wmq/editing-data/caption/caption_eval_edit.json', config=hparams)
    tracer.trace(
        prompts=prompts,
        targets=targets,
        image=image,
    )


def trace_ROME_Blip2OPT_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "two",
        "broccoli",
    ]
    # targets = [
    #     "2",
    #     "tomatoes",
    # ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    subjects=[
        "tennis balls",
        "the red food"
    ]
    
    hparams = ROMEMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/TRACE/ROME/blip2.yaml')
    tracer = MultimodalTracer.from_hparams(hparams)
    # eval_ds = VQADataset('/mnt/data2/wmq/editing-data/vqa/vqa_eval.json', config=hparams)
    tracer.trace(
        prompts=prompts,
        targets=targets,
        image=image,
        subjects=subjects,
        plot=True,
        plot_list=random.sample(range(len(prompts)), min(len(prompts), 100))
    )
    
def test_ROME_Blip2OPT_VQA():
    hparams = ROMEMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/TRACE/ROME/blip2.yaml')
    tracer = MultimodalTracer.from_hparams(hparams)
    eval_ds = VQADataset('/mnt/data2/wmq/editing-data/vqa/vqa_eval.json', config=hparams, size=10)
    tracer.trace_dataset(
        ds=eval_ds,
        plot=True,
        plot_list=random.sample(range(len(eval_ds._data)), min(len(eval_ds._data), 100)),
        is_ds=True
    )

def pred_Blip2OPT_VQA():
    hparams = ROMEMultimodalHyperParams.from_hparams('/mnt/data2/wmq/EasyEdit/hparams/TRACE/ROME/blip2.yaml')
    tracer = MultimodalTracer.from_hparams(hparams)
    eval_ds = VQADataset('/mnt/data2/wmq/editing-data/vqa/vqa_eval.json', config=hparams)
    tracer.pred_dataset(
        ds=eval_ds,
        result_path='/mnt/data2/wmq/editing-data/vqa/blip2optfp32_vqa_eval.json',
        max_length=30,
        stop_token='\n',
        is_ds=True
    )
    
if __name__ == "__main__":
    # pred_Blip2OPT_VQA()
    # trace_ROME_Blip2OPT_VQA()
    # trace_ROME_BLIP2OPT_Caption()