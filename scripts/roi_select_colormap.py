import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

sns.set_theme(context='paper')
sns.set_style("whitegrid")

def draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.line(im_rgb_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), linewidth)
                cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), linewidth)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.line(im_rgb_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), linewidth)
            cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), linewidth)
            current_former_x = former_x
            current_former_y = former_y

            # Guarda una copia de la máscara antes de floodFill
            mask_contour = mask.copy()
            mask_contour[mask_contour != 0] = 255
            cv2.imwrite(f"rois/id={id}_mask_contour_{i}.jpg", mask_contour)

            # Fill the enclosed area in the mask with white
            cv2.floodFill(mask, None, (0, 0), (255))

    return former_x, former_y

# Configuración inicial
file_name = "mousebrain_20200409"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "nogse_vs_x_colormap"
linewidth = 2
slic = 1
nrois = [1,2] # input("Nrois:") # ms
scaling_factor = 5 # Factor de escala (puedes ajustarlo según sea necesario)

ROI = "ROI1"
g = input("g: ") # mT/m
n = int(input("N: "))
x = input("x: ")
tnogse = input("TNOGSE: ") # ms
id = input("id: ") # ms

#im = np.loadtxt(f"../results_mousebrain_20200409/{folder}/TNOGSE={tnogse}_N={n}/{ROI}_colormap_contrast_vs_g_tnogse={tnogse}_N={n}_g={g}.txt")
im = np.loadtxt(f"../results_mousebrain_20200409/{folder}/tnogse={tnogse}_N={n}_g={g}/{ROI}_nogse_vs_x_colormap_tnogse={tnogse}_N={n}_g={g}_x={x}.txt")

norm = Normalize(vmin=np.nanmin(im), vmax=np.nanmax(im))
cmap = plt.cm.jet
cmap.set_bad(color='black')

# Normalizar y convertir a RGB para la visualización
im_normalized = norm(im)
im_rgb = (cmap(im_normalized)[:, :, :3] * 255).astype(np.uint8)

im[np.isnan(im)] = 0
im = im.astype(np.uint8)  

np.savetxt(f"rois/original.txt", im, fmt='%f')

# Guardar las ROI y las máscaras
for i in nrois:

    drawing = False  # True if mouse is pressed
    mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

    # Escalar la imagen para que se vea más grande
    im_rgb_scaled = cv2.resize(im_rgb, None, fx=scaling_factor, fy=scaling_factor)
    im_scaled = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor)
    mask = np.zeros_like(im_scaled)

    cv2.namedWindow("Roi_Select")
    cv2.setMouseCallback('Roi_Select', draw)

    while True:
        cv2.imshow('Roi_Select', im_rgb_scaled)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # Presiona Enter para pasar a la siguiente ROI
            break

    # Invert the mask
    mask_inverted = cv2.bitwise_not(mask)
    
    # Resize the inverted mask to match the size of the original image
    mask_resized = cv2.resize(mask_inverted, (im.shape[1], im.shape[0]))
    mask_resized[mask_resized != 0] = 255
        
    # Apply the mask to the original image
    roi = np.zeros_like(im)
    roi[mask_resized == 255] = im[mask_resized == 255]
    signal = np.mean(roi[roi != 0])
    print(f"Average intensity of ROI {i}: {signal}")

    # Save roi
    np.savetxt(f"rois/id={id}_roi_{i}.txt", roi, fmt='%f')
    roi = (roi * 255).astype(np.uint8)
    cv2.imwrite(f"rois/id={id}_roi_{i}.jpg", roi)

    # Save mask
    np.savetxt(f"rois/id={id}_mask_{i}.txt", mask_resized, fmt='%f')
    cv2.imwrite(f"rois/id={id}_mask_{i}.jpg", mask_resized)

cv2.destroyAllWindows()

# Iterar sobre todas las máscaras de las ROIs y superponerlas en la imagen final
imagen_final_contour = im_rgb_scaled.copy()

for i in nrois: 
    mask_contour = cv2.imread(f"rois/id={id}_mask_contour_{i}.jpg")
    imagen_final_contour = cv2.add(imagen_final_contour, mask_contour)

# Guardar la imagen final
cv2.imwrite(f"rois/id={id}_rois_final_contour.jpg", imagen_final_contour)

# Iterar sobre todas las máscaras de las ROIs y superponerlas en la imagen final
imagen_final = im.copy()

for i in nrois: 
    mask_roi = cv2.imread(f"rois/id={id}_mask_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    imagen_final = cv2.add(imagen_final, mask_roi)

# Guardar la imagen final
cv2.imwrite(f"rois/id={id}_rois_final.jpg", imagen_final)