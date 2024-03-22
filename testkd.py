import torch
from segment_anything import sam_model_registry, SamPredictor
device = "cuda" if torch.cuda.is_available() else "cpu"
example_inputs = torch.randn(1, 3, 1024, 1024).to(device)
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="pretrained_weights/tinysam.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
teacher_embedding, teacher_qkv_emb1, teacher_qkv_emb2, teacher_mid_emb,_ =sam.image_encoder(example_inputs)