#NMRSI - Ignacio Lembo Ferrari - 22/08/2024

import cv2
import numpy as np
from brukerapi.dataset import Dataset as ds
import os

def draw_circle(event, x, y, flags, param):
    global center, radius, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        center = (x, y)
        radius = 0  # Inicializar el radio con un valor predeterminado

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
        cv2.circle(im_scaled, center, radius, (255), 2)
        cv2.circle(mask, center, radius, (255), 2)
        cv2.floodFill(mask, None, (0, 0), (255))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(im_scaled, center, radius, (255), 2)
            cv2.circle(mask, center, radius, (255), 2)
            current_former_x = x
            current_former_y = y

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

file_name = "mousebrain_20200409"
serial = input("Serial: ") #ms
nrois = [1]
slic = 1
scaling_factor = 5 # Factor de escala (puedes ajustarlo según sea necesario)
ims = ds(f"C:/Users/Ignacio Lembo/Documents/data/data_mousebrain_20200409/"+str(serial)+"/pdata/1/2dseq").data

A0_matrix = ims[:,:,slic,0]
M_matrix = ims[:,:,slic,1]
original = M_matrix #/A0_matrix 

directory = f"rois_{file_name}/serial={serial}/slice={slic}"
os.makedirs(directory, exist_ok=True)

np.savetxt(f"{directory}/original.txt", original, fmt='%f')

#Equalize original
original_eq = 255 * (original - np.min(original)) / (np.max(original) - np.min(original)) + 255*(np.min(original) / (np.max(original) - np.min(original)) )

cv2.imwrite(f"{directory}/original_eq.jpg", original_eq)

im = cv2.imread(f"{directory}/original_eq.jpg", cv2.IMREAD_GRAYSCALE)

for nroi in nrois:

    drawing = False  # True if mouse is pressed
    mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

    # Escalar la imagen para que se vea más grande
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
    print(f"Average intensity of ROI {nroi}: {signal}")

    # Save roi
    np.savetxt(f"{directory}/roi_{nroi}.txt", roi, fmt='%f')
    roi = (roi * 255).astype(np.uint8)
    cv2.imwrite(f"{directory}/roi_{nroi}.jpg", roi)

    # Save mask
    np.savetxt(f"{directory}/mask_{nroi}.txt", mask_resized, fmt='%f')
    cv2.imwrite(f"{directory}/mask_{nroi}.jpg", mask_resized)

    print("original size: ", original.shape)
    print("mask size: ", mask_resized.shape)
    print("roi size: ", roi.shape)

cv2.destroyAllWindows()

# Iterar sobre todas las máscaras de las ROIs y superponerlas en la imagen final
imagen_final = im.copy()

for nmask in nrois: 
    mask_roi = cv2.imread(f"{directory}/mask_{nmask}.jpg", cv2.IMREAD_GRAYSCALE)
    imagen_final = cv2.add(imagen_final, mask_roi)

# Guardar la imagen final
cv2.imwrite(f"{directory}/rois_final.jpg", imagen_final)

# Cargar la imagen en escala de grises
imagen_final = cv2.imread(f"{directory}/rois_final.jpg", cv2.IMREAD_GRAYSCALE)

# Convertir la imagen en escala de grises a color
imagen_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)

# Definir el color rojo en RGB
color_rojo = (0, 0, 255.0)  # (B, G, R)

# Encontrar los píxeles blancos en la imagen en escala de grises
indices_blancos = np.where((imagen_final >= 250) & (imagen_final <= 255))

# Reemplazar los píxeles blancos por rojo en la imagen a color
imagen_color[indices_blancos] = color_rojo

# Guardar la imagen final
cv2.imwrite(f"{directory}/rois_final_color.jpg", imagen_color)