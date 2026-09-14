import sys
import os
import random
import math
from glob import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re

# Forzar a Python a encontrar el módulo 'dinov2' y 'rad_dino'
DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y FORMATO
# ==============================================================================
MODEL_PATH = "/mnt/nfs/home/dbenitom/pruebas/pedi-cxr/history/definitivo/lp_base_50_medium_sgd_adapted30/weights/best.pt"

IMAGES_DIR = "/mnt/nfs/home/dbenitom/pruebas/pedi-cxr/images/test"
LABELS_DIR = "/mnt/nfs/home/dbenitom/pruebas/pedi-cxr/labels/test"
SAVE_DIR = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy"

# ------------------------------------------------------------------------------
# MODO DE PROCESAMIENTO
# Si 'SINGLE_IMAGE_NAME' tiene un nombre de archivo (ej. "0a1b2c3d.png"), 
# procesará solo esa imagen. Si se deja en None o "", utilizará el rango START/END.
# ------------------------------------------------------------------------------
SINGLE_IMAGE_NAME = "ecdd955ef381372be31bd48f9424d6dd.png"  # p. ej. "10328dfec9669c5ca24bbf93363dc2ba.png"

START_IMG = 1     # Primera imagen a analizar (si SINGLE_IMAGE_NAME es None)
END_IMG = 1       # Última imagen a analizar (si SINGLE_IMAGE_NAME es None)

TARGET_SIZE = (640, 640)

# ==============================================================================
# CONFIGURACIÓN DE MAPEO DE CLASES GROUND TRUTH (GT)
# ==============================================================================
USE_GT_CLASS_MAPPING = False

GT_CLASS_MAPPING = {
    0: "Airway Compression",
    1: "Calcified Parenchyma",
    2: "Cavities",
    3: "Collapsed Lung",
    4: "Hyperinflation",
    5: "Intersticial Opacification",
    6: "Lymphadenopathy",
    7: "Miliary",
    8: "Opacification",
    9: "Pleural Effusion"
}

# ==============================================================================
# PALETA DE COLORES FIJA POR CLASE
# ==============================================================================
CLASS_COLORS = {
    "Airway compression": (0.121, 0.466, 0.705),
    "Hyperinflation": (1.000, 0.498, 0.055),
    "Cavities": (0.173, 0.627, 0.173),
    "Collapsed lung": (0.839, 0.153, 0.157),
    "Calcified parenchyma": (0.580, 0.404, 0.741),
    "Interstitial opacification": (0.549, 0.337, 0.294),
    "Lymphadenopathy": (0.890, 0.467, 0.761),
    "Miliary": (0.498, 0.498, 0.498),
    "Airspace opacification": (0.737, 0.741, 0.133),
    "Pleural effusion": (0.090, 0.745, 0.811),
}


# ==============================================================================
# FORMATEO DE NOMBRES DE CLASE
# ==============================================================================
def format_class_name(name):
    name = re.sub(r'aveolar', 'alveolar', name, flags=re.IGNORECASE)
    name = re.sub(r'intersticial', 'interstitial', name, flags=re.IGNORECASE)

    if name.strip().lower() == "opacification":
        return "Airspace opacification"

    if name.strip().upper() == "ILD":
        return "ILD"

    parts = name.split()
    if parts:
        for i, part in enumerate(parts):
            if part.upper() in ["ILD", "CXRs", "CXR", "TB"]:
                parts[i] = part.upper()
            elif i == 0:
                parts[i] = part.capitalize()
            else:
                parts[i] = part.lower()
        name = " ".join(parts)

    return name


# ==============================================================================
# OBTENER COLOR DE UNA CLASE
# ==============================================================================
def get_class_color(cls_name):
    cls_name = format_class_name(cls_name)

    if cls_name in CLASS_COLORS:
        return CLASS_COLORS[cls_name]

    aliases = {
        "Lung opacity": "Lung opacity",
        "Nodule/mass": "Nodule/mass",
        "Pleural effusion": "Pleural effusion",
        "Pleural thickening": "Pleural thickening",
        "Pulmonary fibrosis": "Pulmonary fibrosis",
        "Aortic enlargement": "Aortic enlargement",
        "Other lesion": "Other lesion",
    }

    if cls_name in aliases:
        return CLASS_COLORS[aliases[cls_name]]

    print(f"WARNING: No hay color definido para la clase '{cls_name}'. Se utilizará gris.")
    return (0.5, 0.5, 0.5)


# ==============================================================================
# DIBUJAR CAJAS Y ETIQUETAS
# ==============================================================================
def draw_custom_boxes(img_path, target_size, class_names, boxes_list, is_gt=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    h, w, _ = img.shape
    thickness = 3

    if not boxes_list:
        return img, []

    # PASADA 1: DIBUJAR CAJAS
    for (x1, y1, x2, y2, cls_id, _) in boxes_list:
        if is_gt:
            color_cv = (0, 255, 0)
        else:
            cls_name = class_names.get(cls_id, str(cls_id))
            r, g, b = get_class_color(cls_name)
            color_cv = (int(r * 255), int(g * 255), int(b * 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color_cv, thickness)

    # PASADA 2: POSICIONAMIENTO DE ETIQUETAS
    occupied_slots = []
    labels_to_render = []

    for (x1, y1, x2, y2, cls_id, conf) in boxes_list:
        cls_name = format_class_name(class_names.get(cls_id, str(cls_id)))

        label_text = cls_name if is_gt else f"{cls_name} {conf:.2f}"

        if is_gt:
            bg_color = (0.0, 1.0, 0.0)
            text_color = (0.0, 0.0, 0.0)
        else:
            bg_color = get_class_color(cls_name)
            text_color = (1.0, 1.0, 1.0)

        text_w = len(label_text) * 11
        text_h = 22

        start_x = x1
        if start_x + text_w > w:
            start_x = w - text_w - 5
        start_x = max(2, start_x)

        target_y = y1
        va_align = 'bottom'
        has_collision = (y1 - text_h < 0)

        if not has_collision:
            box_test = (start_x, start_x + text_w, y1 - text_h, y1)
            for (ox1, ox2, oy1, oy2) in occupied_slots:
                if not (box_test[1] < ox1 or box_test[0] > ox2 or box_test[3] < oy1 or box_test[2] > oy2):
                    has_collision = True
                    break

        if has_collision:
            target_y = y2
            va_align = 'top'
            box_test = (start_x, start_x + text_w, y2, y2 + text_h)

            for (ox1, ox2, oy1, oy2) in occupied_slots:
                if not (box_test[1] < ox1 or box_test[0] > ox2 or box_test[3] < oy1 or box_test[2] > oy2):
                    target_y = max(target_y, oy2 + 2)
                    box_test = (start_x, start_x + text_w, target_y, target_y + text_h)

            if target_y + text_h > h:
                target_y = h - 2
                va_align = 'bottom'

        if va_align == 'bottom':
            occupied_slots.append((start_x, start_x + text_w, target_y - text_h, target_y))
        else:
            occupied_slots.append((start_x, start_x + text_w, target_y, target_y + text_h))

        labels_to_render.append({
            'text': label_text,
            'x': start_x,
            'y': target_y,
            'va': va_align,
            'bg_color': bg_color,
            'text_color': text_color
        })

    return img, labels_to_render


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    print(f"==> Cargando pesos del modelo desde: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    class_names = {k: format_class_name(v) for k, v in model.names.items()}

    if USE_GT_CLASS_MAPPING:
        gt_class_names = {k: format_class_name(v) for k, v in GT_CLASS_MAPPING.items()}
    else:
        gt_class_names = class_names

    # ==========================================================================
    # SELECCIÓN DE IMAGEN(ES)
    # ==========================================================================
    selected_images = []

    if SINGLE_IMAGE_NAME and SINGLE_IMAGE_NAME.strip():
        # Procesar única imagen
        target_path = os.path.join(IMAGES_DIR, SINGLE_IMAGE_NAME.strip())
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"No se encontró la imagen especificada: {target_path}")
        selected_images.append(target_path)
        print(f"\n==> Modo imagen individual activado: {SINGLE_IMAGE_NAME}")
    else:
        # Procesar rango de imágenes
        valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        all_images = []
        for ext in valid_extensions:
            all_images.extend(glob(os.path.join(IMAGES_DIR, ext)))
            all_images.extend(glob(os.path.join(IMAGES_DIR, ext.upper())))

        if not all_images:
            raise FileNotFoundError(f"No se encontraron imágenes en: {IMAGES_DIR}")

        all_images = sorted(all_images)
        total_found = len(all_images)

        start_idx = max(0, START_IMG - 1)
        end_idx = min(total_found, END_IMG)

        if start_idx >= total_found or start_idx >= end_idx:
            raise ValueError(f"Rango inválido. Total disponibles: {total_found}")

        selected_images = all_images[start_idx:end_idx]
        print(f"\n==> Procesando {len(selected_images)} imágenes en rango ({START_IMG} a {END_IMG})...")

    # ==========================================================================
    # INFERENCIA
    # ==========================================================================
    results = model.predict(
        source=selected_images,
        device=None,
        save=False,
        conf=0.25,
        iou=0.7,
        imgsz=640,
        visualize=False,
        save_txt=False
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================================================
    # PROCESAR CADA IMAGEN INDIVIDUALMENTE
    # ==========================================================================
    for result in results:
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

        # 3. DESDUPLICACIÓN DE GT
        unique_gt_boxes = []
        seen_coords = set()
        for box in gt_boxes_list:
            coords = (box[0], box[1], box[2], box[3])
            if coords not in seen_coords:
                seen_coords.add(coords)
                unique_gt_boxes.append(box)
        gt_boxes_list = unique_gt_boxes

        # 4. CREAR Y GUARDAR FIGURA DE PREDICCIÓN
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
        img_pred_custom, labels_pred = draw_custom_boxes(
            img_path, TARGET_SIZE, class_names, pred_boxes_list, is_gt=False
        )
        ax_pred.imshow(img_pred_custom)
        ax_pred.axis('off')

        for lbl in labels_pred:
            ax_pred.text(
                lbl['x'], lbl['y'], lbl['text'],
                color=lbl['text_color'], fontsize=14, fontweight='bold',
                verticalalignment=lbl['va'],
                bbox=dict(facecolor=lbl['bg_color'], edgecolor='none', pad=1.5)
            )

        fig_pred.tight_layout(pad=0)
        out_pred = os.path.join(SAVE_DIR, f"predicted_{base_name}.pdf")
        fig_pred.savefig(out_pred, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_pred)

        # 5. CREAR Y GUARDAR FIGURA DE GROUND TRUTH
        fig_gt, ax_gt = plt.subplots(1, 1, figsize=(8, 8))
        img_gt_custom, labels_gt = draw_custom_boxes(
            img_path, TARGET_SIZE, gt_class_names, gt_boxes_list, is_gt=True
        )
        ax_gt.imshow(img_gt_custom)
        ax_gt.axis('off')

        for lbl in labels_gt:
            ax_gt.text(
                lbl['x'], lbl['y'], lbl['text'],
                color=lbl['text_color'], fontsize=14, fontweight='bold',
                verticalalignment=lbl['va'],
                bbox=dict(facecolor=lbl['bg_color'], edgecolor='none', pad=1.5)
            )

        fig_gt.tight_layout(pad=0)
        out_gt = os.path.join(SAVE_DIR, f"gt_{base_name}.pdf")
        fig_gt.savefig(out_gt, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_gt)

        print(f"➡️ Guardado exitoso de {base_name}:")
        print(f"   • Predicción: {os.path.basename(out_pred)}")
        print(f"   • Ground Truth: {os.path.basename(out_gt)}")

    print("\n🎉 ¡Procesamiento finalizado con éxito!")