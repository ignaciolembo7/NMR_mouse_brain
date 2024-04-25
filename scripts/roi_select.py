#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import cv2
import numpy as np
import argparse
from brukerapi.dataset import Dataset as ds

# Mouse callback function
def draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 1)
                cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 1)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 1)
            cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 1)
            current_former_x = former_x
            current_former_y = former_y
            # Fill the enclosed area in the mask with white
            cv2.floodFill(mask, None, (0, 0), (255))

    return former_x, former_y

# Main loop
for i in range(1):

    drawing = False  # True if mouse is pressed
    mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

    serial = input("Serial:") #ms
    dataset = ds("../data_temp/"+ str(serial) +"/pdata/1/2dseq")
    data = np.array(dataset.data)
    original = data[:,:,0,1]/np.amax(data[:,:,0,1])
    #print(original)
    np.savetxt(f"original", original, fmt='%f')
    # Escalar la imagen a valores entre 0 y 255
    im_scaled = (original * 255).astype(np.uint8)
    # Guardar la imagen
    cv2.imwrite("original.jpg", im_scaled)

    im = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)
    #im = cv2.imread("Darwin.jpg", cv2.IMREAD_GRAYSCALE)
   
    # Escalar la imagen para que se vea más grande
    scaling_factor = 3  # Factor de escala (puedes ajustarlo según sea necesario)
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
    # Apply the mask to the original image
    result = cv2.bitwise_and(original, original, mask=mask_resized)

    # Escalar la máscara para que tenga el mismo tamaño que la imagen original
    mask_resized_original = cv2.resize(mask_resized, (original.shape[1], original.shape[0]))

    # Escalar la ROI para que tenga el mismo tamaño que la imagen original
    result_resized_original = cv2.resize(result, (original.shape[1], original.shape[0]))

    # Aplicar la máscara y la ROI escaladas a la imagen original
    final_result = cv2.bitwise_and(original, original, mask=mask_resized_original)

    # Guardar la imagen escalada a tamaño original
    cv2.imwrite(f"roi_{i+1}.jpg", final_result)

    # Save mask as text
    np.savetxt(f"roi_{i+1}.txt", mask_resized_original, fmt='%d')

    signal = np.mean(result)
    print(f"Average intensity of ROI {i+1}: {signal}")

cv2.destroyAllWindows()
