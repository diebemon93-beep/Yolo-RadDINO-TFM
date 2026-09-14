# -*- coding: utf-8 -*-
import sys
import os
import random
import math
from glob import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Forzar a Python a encontrar el módulo 'dinov2' y 'rad_dino'
DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y FORMATO
# ==============================================================================
MODEL_PATH = "/mnt/nfs/home/dbenitom/pruebas/challenge/history/linearprobing_100epochs/yolo-dino-exp631/weights/best.pt"
IMAGES_DIR = "/mnt/nfs/home/dbenitom/pruebas/challenge/images/test"
LABELS_DIR = "/mnt/nfs/home/dbenitom/pruebas/challenge/labels/test"  
SAVE_DIR = "/mnt/nfs/home/dbenitom/pruebas/challenge/predictions"

# IMAGES_PER_GRID determina cuántas imágenes se empaquetan en cada mosaico/rejilla .png de salida
IMAGES_PER_GRID = 10    
MAX_ROWS = 2              
TARGET_SIZE = (640, 640)  

# 🎨 PALETA DE COLORES DINÁMICA (Solo para las predicciones)
random.seed(40)
COLOR_PALETTE = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(80)]

def draw_custom_boxes(img_path, target_size, class_names, boxes_list, is_gt=True):
    """
    Dibuja cajas y etiquetas en dos pasadas. 
    Aplica estética clásica (Verde/Negro) para GT y multicolor (Blanco/Color) para Predicciones.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    h, w, _ = img.shape  
    
    thickness = 3
    font_scale = 0.6  
    
    if not boxes_list:
        return img

    # PASADA 1: DIBUJAR ÚNICAMENTE LAS CAJAS
    for (x1, y1, x2, y2, cls_id, _) in boxes_list:
        color = (0, 255, 0) if is_gt else COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # PASADA 2: DIBUJAR LAS ETIQUETAS POR ENCIMA DE TODO
    occupied_slots = []  

    for (x1, y1, x2, y2, cls_id, conf) in boxes_list:
        cls_name = class_names.get(cls_id, str(cls_id))
        label = cls_name if is_gt else f"{cls_name} {conf:.0%}"
        
        if is_gt:
            bg_color = (0, 0, 0)        
            text_color = (0, 255, 0)    
        else:
            bg_color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)] 
            text_color = (255, 255, 255)                                                         
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)[0]
        text_w, text_h = text_size[0], text_size[1]
        
        start_x = x1
        if start_x + text_w + 10 > w:
            start_x = w - text_w - 10
        start_x = max(0, start_x)
        
        if y1 - text_h - 10 < 0:
            base_y = y1 + text_h + 10  
        else:
            base_y = y1                
        
        collision = True
        attempts = 0
        while collision and attempts < 5:
            collision = False
            for (ox1, ox2, oy) in occupied_slots:
                x_overlap = not (start_x + text_w < ox1 or start_x > ox2)
                y_overlap = abs(base_y - oy) < (text_h + 12)
                
                if x_overlap and y_overlap:
                    base_y += (text_h + 14)  
                    collision = True
                    break
            attempts += 1
        
        occupied_slots.append((start_x, start_x + text_w, base_y))
        
        label_y1 = max(0, min(h, base_y - text_h - 8))
        label_y2 = max(0, min(h, base_y))
        
        cv2.rectangle(img, (start_x, label_y1), (start_x + text_w + 6, label_y2), bg_color, -1)
        cv2.putText(img, label, (start_x + 3, base_y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness=2)

    return img

def build_grid_layout_horizontal(num_items, max_rows):
    rows = min(num_items, max_rows)
    cols = math.ceil(num_items / rows)
    figsize_w = cols * 8
    figsize_h = rows * 8
    return rows, cols, (figsize_w, figsize_h)


# ==============================================================================
# EJECUCIÓN DEL SCRIPT (MODIFICADO PARA PROCESAR TODO EL DATASET DE TEST)
# ==============================================================================
if __name__ == "__main__":
    print(f"==> Cargando pesos del modelo desde: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names  

    # Buscar todas las imágenes disponibles
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_images = []
    for ext in valid_extensions:
        all_images.extend(glob(os.path.join(IMAGES_DIR, ext)))
        all_images.extend(glob(os.path.join(IMAGES_DIR, ext.upper())))

    if not all_images:
        raise FileNotFoundError(f"No se encontraron imágenes válidas en el directorio: {IMAGES_DIR}")

    # CAMBIO AQUÍ: Eliminamos random.sample para procesar la lista completa
    total_images_count = len(all_images)
    print(f"==> Se encontraron {total_images_count} imágenes en total.")
    print(f"==> Procesando TODAS las imágenes en bloques de {IMAGES_PER_GRID}...")

    # Realizar predicción sobre el lote completo de imágenes encontradas
    results = model.predict(source=all_images, save=False, conf=0.25, imgsz=640, visualize=False, save_txt=True, device=[1])

    os.makedirs(SAVE_DIR, exist_ok=True)

    # El bucle por chunks procesará automáticamente todos los elementos del lote completo
    for chunk_idx in range(0, len(results), IMAGES_PER_GRID):
        chunk_results = results[chunk_idx:chunk_idx + IMAGES_PER_GRID]
        grid_id = (chunk_idx // IMAGES_PER_GRID) + 1
        num_items = len(chunk_results)

        rows, cols, current_figsize = build_grid_layout_horizontal(num_items, MAX_ROWS)

        fig_pred, axes_pred = plt.subplots(rows, cols, figsize=current_figsize)
        fig_gt, axes_gt = plt.subplots(rows, cols, figsize=current_figsize)

        axes_pred = [axes_pred] if num_items == 1 else (axes_pred.ravel() if hasattr(axes_pred, 'ravel') else axes_pred)
        axes_gt = [axes_gt] if num_items == 1 else (axes_gt.ravel() if hasattr(axes_gt, 'ravel') else axes_gt)

        for i, result in enumerate(chunk_results):
            img_path = result.path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(LABELS_DIR, f"{base_name}.txt")

            # 1. PARSEO DE PREDICCIONES
            pred_boxes_list = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    xyxyn = box.xyxyn[0].tolist()
                    
                    x1 = int(xyxyn[0] * TARGET_SIZE[0])
                    y1 = int(xyxyn[1] * TARGET_SIZE[1])
                    x2 = int(xyxyn[2] * TARGET_SIZE[0])
                    y2 = int(xyxyn[3] * TARGET_SIZE[1])
                    pred_boxes_list.append((x1, y1, x2, y2, cls_id, conf))

            # 2. PARSEO DE GROUND TRUTH
            gt_boxes_list = []
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_c, y_c, src_w, src_h = map(float, parts[1:])
                        x1 = int((x_c - src_w / 2) * TARGET_SIZE[0])
                        y1 = int((y_c - src_h / 2) * TARGET_SIZE[1])
                        x2 = int((x_c + src_w / 2) * TARGET_SIZE[0])
                        y2 = int((y_c + src_h / 2) * TARGET_SIZE[1])
                        gt_boxes_list.append((x1, y1, x2, y2, cls_id, None))

            # --- RENDERIZADO PREDICCIONES ---
            img_pred_custom = draw_custom_boxes(img_path, TARGET_SIZE, class_names, pred_boxes_list, is_gt=False)
            axes_pred[i].imshow(img_pred_custom)
            axes_pred[i].set_title(f"Pred: {base_name[:15]}", fontsize=18, fontweight='bold')
            axes_pred[i].axis('off')

            # --- RENDERIZADO GROUND TRUTH ---
            img_gt_custom = draw_custom_boxes(img_path, TARGET_SIZE, class_names, gt_boxes_list, is_gt=True)
            axes_gt[i].imshow(img_gt_custom)
            axes_gt[i].set_title(f"GT: {base_name[:15]}", fontsize=18, fontweight='bold')
            axes_gt[i].axis('off')

        for j in range(i + 1, len(axes_pred)):
            axes_pred[j].axis('off')
            axes_gt[j].axis('off')

        # Guardado compacto de rejillas individuales
        for fig, prefix in [(fig_pred, "predicted"), (fig_gt, "gt")]:
            fig.tight_layout(pad=1.0)
            fig.subplots_adjust(wspace=0.02, hspace=0.1)
            out_path = os.path.join(SAVE_DIR, f"{prefix}{grid_id}.png")
            fig.savefig(out_path, bbox_inches='tight', dpi=180)
            plt.close(fig)
            print(f"  ➡️ Guardado: {os.path.basename(out_path)}")

    print(f"\n🎉 ¡Listo! Procesadas con éxito las {total_images_count} imágenes del dataset.")