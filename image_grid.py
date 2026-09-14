import os
import math
import cv2
import matplotlib.pyplot as plt

# Lista de IDs que proporcionaste
IMAGE_IDS = [
"10328dfec9669c5ca24bbf93363dc2ba", "121e1eece9a56079a3bbc84b94b20971", "1e1a82658113ff29ee4dd52b493183be", "1e2fb92429081151ac80084b6e214463", "2422fa22d763900c59422302e486b133", "25d86cec7e4de33e1acacb6fb2a6d1cc", "284c239c9430953b90b668ddfd3de1c3", "38b25488a5bc5f825b6271d220c93fc9", "3b0c4e17b9d129d53178b6b17c28a462", "3bda9b859701b65573cbac751ea745ac", "406799ad94e00bd9bddc382240699b2e", "41a82ea9582eece760ed8c4a770c4ae3", "5c05e4a8fa990f204e3803c04e846776", "60a9060eb83b3c94bd1a54b6ffb5aaa2", "672252fcf416c0b75112ff1a58465d14", "6f176ad5e00128075daa3c944d4c1a06", "724b38c3c3a17a1e46f69b98b310b3d1", "7874f2e59a86a4b0df2416b44978a1dd", "7a8c115a26b314a4db66f0d1f868f04a", "7e3c6405aacb360d32195dc22f04fe1e", "89167e0152904d19f584fd059b8aa28f", "a35b1f1e6e2e854d8ac9cd5a765d8f07", "aab5e6d0b300e2c4f476bc117190e3ec", "b02b9233f18cc15515bb62fdbe7f46cc", "bd41b02a7aa1d04b8ceca8549e0a0b06", "c31aba38c1ca968276f862b70d9251ec", "c51815ac1c96c0ac44ba845e586d22c1", "c728883afbe01aee6ac2612eeb52ff03", "d04615c07958e3ef19ec6846f80dfc39", "d7e71a052a753c3f2f3e317d60177bec", "e02a21bb3c9d0d4d5a1b3d4418a30606", "e2913dc9a697cd2ef92f89d1f32e167e", "ec4a6a80cb0bdb5e6f72f555577352b2", "ecdd955ef381372be31bd48f9424d6dd", "f4cddbe4c562eafb2347c59d440068c1", "f68207cedf2233b8a0ee72dba6a1dbb7", "fdfbac4ce475828385e3977b061d82e0"]
# Configuración
IMAGES_DIR = "/mnt/nfs/projects/CXR-TB/DATA/data_cxr/public/vindr-pcxr-png/test"  # <-- Cambia esto por la ruta de tu carpeta
EXTENSION = ".png"                   # <-- Cambia según la extensión (.png, .jpg, etc.)
COLS = 10                            # Cantidad de imágenes por fila
GRID_OUT_PATH = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy/grid_imagenes.png"

num_images = len(IMAGE_IDS)
rows = math.ceil(num_images / COLS)

# Crear la figura con dimensiones suficientes
fig, axes = plt.subplots(rows, COLS, figsize=(30, 4 * rows))

# Asegurar que 'axes' sea siempre una matriz bidimensional
if rows == 1:
    axes = [axes]

for idx, img_id in enumerate(IMAGE_IDS):
    r = idx // COLS
    c = idx % COLS
    ax = axes[r][c]

    img_path = os.path.join(IMAGES_DIR, f"{img_id}{EXTENSION}")

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(img_id, fontsize=6, pad=3)
    else:
        ax.text(0.5, 0.5, "Archivo no\nencontrado", ha='center', va='center', fontsize=8)
        ax.set_title(img_id, fontsize=6, color='red', pad=3)

    ax.axis('off')

# Ocultar subplots sobrantes si la cuadrícula no se llena por completo
for idx in range(num_images, rows * COLS):
    r = idx // COLS
    c = idx % COLS
    axes[r][c].axis('off')

plt.tight_layout()

# Crear directorio si no existe y guardar la imagen
os.makedirs(os.path.dirname(GRID_OUT_PATH), exist_ok=True)
plt.savefig(GRID_OUT_PATH, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"🎉 Cuadrícula guardada con éxito en: {GRID_OUT_PATH}")