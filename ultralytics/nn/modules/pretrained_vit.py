import torch
import torch.nn as nn
from torch.hub import load
from safetensors.torch import load_file
import os
from torchvision import transforms
from peft import LoraConfig, get_peft_model

dino_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 14},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 14},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 14},
    "giant": {"name": "dinov2_vitg14_reg", "embedding_size": 1536, "patch_size": 14},
}

class DinoV2Patches(nn.Module):
    
    def __init__(self, in_chanels=3, out_channels=768, size="base"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"Loading DinoV2Patches with size: {size}")
        super(DinoV2Patches, self).__init__()
        self.size = size
        
        self.backbone = load("facebookresearch/dinov2", dino_backbones[self.size]["name"], pretrained=False)

        try:
            backbone_state_dict = load_file('backbone_path')
            missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if local_rank == 0:
                print(f'RAD-DINO loaded: missing={len(missing)}, unexpected={len(unexpected)}')
                if len(missing) > 0:
                    print(f"Warning: {len(missing)} layers did not receive pretrained weights.")
                    print(f"Sample missing layers: {missing[:5]}")
                    
        except Exception as e:
            if local_rank == 0:
                print(f'Error loading RAD-DINO weights: {e}')
            raise e

        # LoRA LAYERS
        # lora_config = LoraConfig(
        #     r=16,                                
        #     lora_alpha=32,                       
        #     target_modules=["qkv", "proj"],     
        #     lora_dropout=0.0,  # Set to 0 for deterministic training (was 0.05)
        #     bias="none"
        # )
        
        # Wrap the DinoV2 ViT with LoRA adapters
        # self.backbone = get_peft_model(self.backbone, lora_config)


        MIMIC_MEAN = (0.5307, 0.5307, 0.5307)
        MIMIC_STD = (0.2583, 0.2583, 0.2583)
        self.out_channels = out_channels
        self.inet_norm = transforms.Normalize(mean=MIMIC_MEAN, std=MIMIC_STD)

    def transform(self, x):
        b, c, h, w = x.shape
        h_new = h - (h % 14)
        w_new = w - (w % 14)
        dh = h - h_new
        dw = w - w_new

        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left

        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right].clone()
        x_cropped = self.inet_norm(x_cropped)
        return x_cropped

    def forward(self, x):

        x = self.transform(x)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / 14, x.shape[3] / 14)


        x = self.backbone.forward_features(x)
        x = x["x_norm_patchtokens"]
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.out_channels, int(mask_dim[0]), int(mask_dim[1]))

        return x

