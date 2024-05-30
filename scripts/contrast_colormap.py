#NMRSI - Ignacio Lembo Ferrari - 29/05/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "contrast_vs_g_colormap"
slic = 1

image_paths, method_paths = nogse.upload_contrast_data(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas
rois = ["ROI1"] #,"ROI2","ROI3","ROI4","ROI5"]

for roi in rois: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")
    
    T_nogse, g_contrast, n, ims = nogse.colormap_contrast_roi(image_paths, method_paths, mask, slic)

    # Create a normalization object to map the data values to the range [0, 1]
    norm =  Normalize(vmin=np.nanmin(ims), vmax=np.nanmax(ims)) #Normalize(vmin=np.nanmin(ims), vmax=np.nanmax(ims))

    for g, im in zip(g_contrast,ims):
        
        cmap = plt.cm.jet
        cmap.set_bad(color='black')

        fig, ax = plt.subplots(figsize=(8,6))
        img = ax.imshow(im, cmap=cmap) #norm = norm
        cbar = plt.colorbar(img, ax=ax, ticks=np.linspace(0, np.nanmax(ims), 8))
        cbar.set_label("$\\Delta M$", fontsize=14)
        ax.axis('off')
        title = ax.set_title(f"$T_\mathrm{{NOGSE}}$ = {T_nogse} ms  ||  g = {g} mT/m || $N$ = {int(n)} || slice = {slic}", fontsize=18)

        directory = f"../results_{file_name}/{folder}/TNOGSE={T_nogse}_N={int(n)}"
        os.makedirs(directory, exist_ok=True)    

        np.savetxt(f"{directory}/{roi}_NOGSE_contrast_colormap_t={T_nogse}_N={int(n)}_g={g}.txt", im, fmt='%f')

        fig.tight_layout()
        #fig.savefig(f"{directory}/{roi}_NOGSE_contrast_colormap_t={T_nogse}_N={int(n)}_g={g}.pdf")
        fig.savefig(f"{directory}/{roi}_NOGSE_contrast_colormap_t={T_nogse}_N={int(n)}_g={g}.png", dpi=600)
        
        plt.close(fig)

    idx += 1

