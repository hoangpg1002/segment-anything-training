from box import Box

config = {
    "num_devices": 1,
    "batch_size": 16,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type":"vit_t",
        "checkpoint": "/kaggle/working/segment-anything-training/pretrained_weights/tinysam.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/kaggle/input/coco-2017-dataset/coco2017/train2017",
            "annotation_file": "/kaggle/working/instances_minitrain2017.json"
        },
        "val": {
            "root_dir": "/kaggle/input/coco-2017-dataset/coco2017/val2017",
            "annotation_file": "/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json"
        }
    },
    # "samdataset":"sa1b"
}

cfg = Box(config)
