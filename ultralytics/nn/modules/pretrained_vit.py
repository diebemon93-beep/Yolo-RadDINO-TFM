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

        print(f"Initializing DinoV2Patches with size: {size}")
        super(DinoV2Patches, self).__init__()
        self.size = size
        
        self.backbone = load("facebookresearch/dinov2", dino_backbones[self.size]["name"], pretrained=False)

        try:
            backbone_state_dict = load_file('/mnt/nfs/home/dbenitom/backbone_compatible.safetensors')
            missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if local_rank == 0:
                print(f'==> RAD-DINO loaded: missing={len(missing)}, unexpected={len(unexpected)}')
                if len(missing) > 0:
                    print(f"⚠️ ¡Aviso! Hay {len(missing)} capas que no recibieron pesos preentrenados.")
                    print(f"Capas faltantes de muestra: {missing[:5]}")
                    
        except Exception as e:
            if local_rank == 0:
                print(f'❌ Error crítico cargando los pesos de RAD-DINO: {e}')
            raise e

        # lora_config = LoraConfig(
        #     r=16,                                
        #     lora_alpha=32,                       
        #     target_modules=["qkv", "proj"],     
        #     lora_dropout=0.0,  # Set to 0 for deterministic training (was 0.05)
        #     bias="none"
        # )
        
        # #Envolvemos el ViT de DinoV2 con los adaptadores LoRA
        # self.backbone = get_peft_model(self.backbone, lora_config)

        # Configuración estricta de gradientes para evitar asimetrías entre lora_A y lora_B
        # for name, param in self.backbone.named_parameters():
        #     if "lora" in name.lower():
        #         param.requires_grad_(True)       # Abrir tanto lora_A como lora_B
        #     elif "backbone" in name:
        #         param.requires_grad_(False)      # Congelar pesos base de RAD-DINO

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

        # Para auditoría en vivo desde el forward, simplemente añades:
        self.debug_gradientes_estatico(x)

        return x

    def debug_gradientes_estatico(self, x_output):
            # Escudo para DDP: solo audita el proceso principal (Rank 0) y una vez por ciclo/época
            import os
            # local_rank = int(os.environ.get("LOCAL_RANK", 0))
            # if getattr(self, "ya_auditado", False) or local_rank != 0:
            #     return


            # lora_con_grad = []
            # lora_sin_grad = []
            # base_con_grad = []
            # base_sin_grad = []

            # for name, param in self.backbone.named_parameters():
            #     if "lora" in name.lower():
            #         if param.requires_grad:
            #             lora_con_grad.append(name)
            #         else:
            #             lora_sin_grad.append(name)
            #     else:
            #         if param.requires_grad:
            #             base_con_grad.append(name)
            #         else:
            #             base_sin_grad.append(name)

            # # ── REPORTE COMBINADO (ESTÁTICO + DINÁMICO) ─────────────────────────────────

            # if x_output.grad_fn is None or len(lora_sin_grad) > 0:

            #     print("   Tus capas LoRA están listas en memoria (True), pero la salida 'x' está")
            #     print("   MUERTA (grad_fn=None) debido al bloque 'with torch.no_grad()'.")
            #     print("   ¡Los gradientes del backward NO van a actualizar LoRA!")
            
            #     print(f"  [ESTÁTICO] Matrices LoRA activas (True)    : {len(lora_con_grad)}")
            #     print(f"  [ESTÁTICO] Matrices LoRA congeladas (False): {len(lora_sin_grad)}")
            #     print(f"  [ESTÁTICO] Pesos Base activos (True)       : {len(base_con_grad)}")
            #     print(f"  [ESTÁTICO] Pesos Base congelados (False)   : {len(base_sin_grad)}")
            #     print("-" * 60)
            #     print(f"  [DINÁMICO] ¿Output 'x' requiere gradiente? : {x_output.requires_grad}")
            #     print(f"  [DINÁMICO] Función de gradiente (grad_fn)  : {x_output.grad_fn}")
                

