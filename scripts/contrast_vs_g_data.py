#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
from brukerapi.dataset import Dataset as ds
import cv2
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

ids = ["13"] #["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] 
rois = ["ROI1", "ROI1"] #, "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1"]
idxs = ["1", "1"] #,"1", "1","1","1","1","1","1","1","1"] #numero de roi

file_name = "mousebrain_20200409"
rois_folder = "rois_E1429"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "contrast_vs_g_data"
slic = 1

image_paths, method_paths = nogse.upload_contrast_data(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for roi, id, idx in zip(rois, ids, idxs): 

    mask = np.loadtxt(f"rois/id={id}_mask_{idx}.txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI

    T_nogse, g_contrast, n, f =  nogse.generate_contrast_roi(image_paths, method_paths, mask, slic)

    directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={T_nogse}_N={int(n)}_id={id}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_contrast_data(ax, roi, g_contrast, f, T_nogse, n, slic)
    nogse.plot_contrast_data(ax1, roi, g_contrast, f, T_nogse, n, slic)

    table = np.vstack((g_contrast, f))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={T_nogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={T_nogse}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={T_nogse}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

#directory1 = f"../results_{file_name}/{folder}/tnogse={T_nogse}_N={int(n)}"
#os.makedirs(directory1, exist_ok=True)
#fig.tight_layout()
#fig.savefig(f"{directory1}/NOGSE_contrast_vs_g_t={T_nogse}_N={int(n)}.pdf")
#fig.savefig(f"{directory1}/NOGSE_contrast_vs_g_t={T_nogse}_N={int(n)}.png", dpi=600)
#plt.close(fig)