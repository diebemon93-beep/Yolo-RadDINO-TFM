import os
import sys
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re

# Forzar a Python a encontrar el módulo 'dinov2' y 'rad_dino'
DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
MODEL_PATH = "/mnt/nfs/home/dbenitom/pruebas/pedi-cxr/history/definitivo/lp_base_50_medium_sgd_adapted30/weights/best.pt"

# RUTA A TU IMAGEN ESPECÍFICA
IMAGE_PATH = "/mnt/nfs/home/dbenitom/pruebas/pedi-cxr/images/test/ecdd955ef381372be31bd48f9424d6dd.png"

SAVE_DIR = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy"
TARGET_SIZE = (640, 640)
CONF_THRESH = 0.25
IOU_THRESH = 0.7

# ==============================================================================
# PALETA DE COLORES FIJA POR CLASE
# ==============================================================================
CLASS_COLORS = {
    "Aortic enlargement": (0.121, 0.466, 0.705),
    "Atelectasis": (1.000, 0.498, 0.055),
    "Calcification": (0.172, 0.627, 0.172),
    "Cardiomegaly": (0.839, 0.153, 0.157),
    "Consolidation": (0.580, 0.404, 0.741),
    "ILD": (0.549, 0.337, 0.294),
    "Infiltration": (0.890, 0.467, 0.761),
    "Lung opacity": (0.498, 0.498, 0.498),
    "Nodule/mass": (0.737, 0.741, 0.133),
    "Other lesion": (0.090, 0.745, 0.812),
    "Pleural effusion": (0.682, 0.780, 0.910),
    "Pleural thickening": (0.800, 0.474, 0.655),
    "Pneumothorax": (0.400, 0.400, 0.400),
    "Pulmonary fibrosis": (0.900, 0.700, 0.300),
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


def get_class_color(cls_name):
    cls_name = format_class_name(cls_name)
    if cls_name in CLASS_COLORS:
        return CLASS_COLORS[cls_name]
    
    print(f"WARNING: No hay color definido para la clase '{cls_name}'. Se utilizará gris.")
    return (0.5, 0.5, 0.5)


# ==============================================================================
# DIBUJAR CAJAS Y ETIQUETAS (SÓLO PREDICCIONES)
# ==============================================================================
def draw_custom_boxes(img_path, target_size, class_names, boxes_list):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    h, w, _ = img.shape
    thickness = 3

    if not boxes_list:
        return img, []

    # PASADA 1: Dibujar cajas
    for (x1, y1, x2, y2, cls_id, _) in boxes_list:
        cls_name = class_names.get(cls_id, str(cls_id))
        r, g, b = get_class_color(cls_name)
        color_cv = (int(r * 255), int(g * 255), int(b * 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color_cv, thickness)

    # PASADA 2: Posicionamiento de etiquetas (Anti-solapamiento)
    occupied_slots = []
    labels_to_render = []

    for (x1, y1, x2, y2, cls_id, conf) in boxes_list:
        cls_name = format_class_name(class_names.get(cls_id, str(cls_id)))
        label_text = f"{cls_name} {conf:.2f}"
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
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No se encontró la imagen especificada en: {IMAGE_PATH}")

    print(f"==> Cargando modelo desde: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    class_names = {k: format_class_name(v) for k, v in model.names.items()}

    print(f"==> Realizando inferencia sobre la imagen: {IMAGE_PATH}")
    results = model.predict(
        source=IMAGE_PATH,
        device=None,
        save=False,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=640,
        visualize=False
    )

    result = results[0]

    # Parsear cajas predecidas
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

    # Renderizar imagen con cajas personalizadas
    img_pred_custom, labels_pred = draw_custom_boxes(
        IMAGE_PATH,
        TARGET_SIZE,
        class_names,
        pred_boxes_list
    )

    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_pred_custom)
    ax.axis('off')

    for lbl in labels_pred:
        ax.text(
            lbl['x'],
            lbl['y'],
            lbl['text'],
            color=lbl['text_color'],
            fontsize=16,
            fontweight='bold',
            verticalalignment=lbl['va'],
            bbox=dict(
                facecolor=lbl['bg_color'],
                edgecolor='none',
                pad=1.5
            )
        )

    # Guardar resultado
    os.makedirs(SAVE_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    out_path = os.path.join(SAVE_DIR, f"{base_name}_prediction.pdf")

    fig.tight_layout(pad=0)
    fig.savefig(
        out_path,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)

    print(f"🎉 ¡Predicción completada! Guardado en: {out_path}")