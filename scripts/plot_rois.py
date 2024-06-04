import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

sns.set_theme(context='paper')
sns.set_style("whitegrid")

ROI = "ROI1"
tnogse = input("TNOGSE: ") # ms
n = int(input("N: "))
#g =  input("g: ") # mT/m

file_name = "mousebrain_20200409"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "contrast_vs_g_colormap"
rois_folder = f"rois_E1429"
linewidth = 1
slic = 1


nrois = 1 # input("Nrois:") # ms
scaling_factor = 7 # Factor de escala (puedes ajustarlo según sea necesario)

gs = [125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0, 600.0, 650.0, 725.0, 800.0]

#xs = [0.5, 2.7421875, 4.0234375,6.90625, 8.1875, 9.7890625, 10.75 ]

ids = [1,3]

for g in gs: 
 

    im = np.loadtxt(f"../results_mousebrain_20200409/contrast_vs_g_colormap/tnogse={tnogse}_N={n}/{ROI}_NOGSE_contrast_colormap_t={tnogse}_N={n}_g={g}.txt")
    #im = np.loadtxt(f"../results_mousebrain_20200409/nogse_vs_x_colormap/TNOGSE={tnogse}_N={n}_g={g}/{ROI}_nogse_vs_x_colormap_tnogse={tnogse}_N={n}_G={g}_x={x}.txt")

    norm = Normalize(vmin=np.nanmin(im), vmax=np.nanmax(im))
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    im_normalized = norm(im)
    im_rgb = (cmap(im_normalized)[:, :, :3] * 255).astype(np.uint8)
    im[np.isnan(im)] = 0
    im = im.astype(np.uint8)  
    im_rgb_scaled = cv2.resize(im_rgb, None, fx=scaling_factor, fy=scaling_factor)
    imagen_final_contour = im_rgb_scaled.copy()

    for id in ids: 
        mask_contour = cv2.imread(f"{rois_folder}/id={id}_mask_contour_1.jpg")
        imagen_final_contour = cv2.add(imagen_final_contour, mask_contour)

    directory = f"../results_{file_name}/images/{rois_folder}"
    os.makedirs(directory, exist_ok=True)

    # Guardar la imagen final
    cv2.imwrite(f"{directory}/g={g}_rois_final_contour.jpg", imagen_final_contour)