#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import cv2
import numpy as np
import argparse
from brukerapi.dataset import Dataset as ds
import matplotlib.pyplot as plt
import sys

# Mouse callback function
def draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            current_former_x = former_x
            current_former_y = former_y
            # Fill the enclosed area in the mask with white
            cv2.floodFill(mask, None, (0, 0), (255))

    return former_x, former_y

serial = int(sys.argv[1])
nrois = int(sys.argv[2])
slic = int(sys.argv[3])

ims = ds(f"data/data_mousebrain_20200409/"+str(serial)+"/pdata/1/2dseq").data
A0_matrix = ims[:,:,slic,0]
M_matrix = ims[:,:,slic,1]
original = M_matrix #/A0_matrix 

np.savetxt(f"rois/original.txt", original, fmt='%f')

#Equalize original
original_eq = 255 * (original - np.min(original)) / (np.max(original) - np.min(original)) + 255*(np.min(original) / (np.max(original) - np.min(original)) )

cv2.imwrite(f"rois/original_eq.jpg", original_eq)

im = cv2.imread(f"rois/original_eq.jpg", cv2.IMREAD_GRAYSCALE)

for i in range(nrois):

    drawing = False  # True if mouse is pressed
    mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

    # Escalar la imagen para que se vea más grande
    scaling_factor = 4 # Factor de escala (puedes ajustarlo según sea necesario)
    im_scaled = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor)
    mask = np.zeros_like(im_scaled)  # Create a black image with the same size as im
    
    cv2.namedWindow("Roi_Select")
    cv2.setMouseCallback('Roi_Select', draw)

    while True:
        cv2.imshow('Roi_Select', im_scaled)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # Press Enter to move to the next ROI
            break

    # Invert the mask

    mask_inverted = cv2.bitwise_not(mask)
    
    # Resize the inverted mask to match the size of the original image
    mask_resized = cv2.resize(mask_inverted, (original.shape[1], original.shape[0]))
    mask_resized[mask_resized != 0] = 255
        
    # Apply the mask to the original image
    roi = np.zeros_like(original)
    roi[mask_resized == 255] = original[mask_resized == 255]
    signal = np.mean(roi[roi != 0])
    print(f"Average intensity of ROI {i+1}: {signal}")

    # Save roi
    np.savetxt(f"rois/roi_{i+1}.txt", roi, fmt='%f')
    roi = (roi * 255).astype(np.uint8)
    cv2.imwrite(f"rois/roi_{i+1}.jpg", roi)

    # Save mask
    np.savetxt(f"rois/mask_{i+1}.txt", mask_resized, fmt='%f')
    cv2.imwrite(f"rois/mask_{i+1}.jpg", mask_resized)

cv2.destroyAllWindows()

# Iterar sobre todas las máscaras de las ROIs y superponerlas en la imagen final
imagen_final = im.copy()

for i in range(1, nrois+1): 
    mask_roi = cv2.imread(f"rois/mask_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    imagen_final = cv2.add(imagen_final, mask_roi)

# Guardar la imagen final
cv2.imwrite(f"rois/im={serial}_rois_final.jpg", imagen_final)

# Cargar la imagen en escala de grises
imagen_final = cv2.imread(f"rois/im={serial}_rois_final.jpg", cv2.IMREAD_GRAYSCALE)

# Convertir la imagen en escala de grises a color
imagen_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)

# Definir el color rojo en RGB
color_rojo = (0, 0, 255.0)  # (B, G, R)

# Encontrar los píxeles blancos en la imagen en escala de grises
indices_blancos = np.where((imagen_final >= 250) & (imagen_final <= 255))

# Reemplazar los píxeles blancos por rojo en la imagen a color
imagen_color[indices_blancos] = color_rojo

# Guardar la imagen final
cv2.imwrite(f"rois/im={serial}_rois_final_color.jpg", imagen_color)

