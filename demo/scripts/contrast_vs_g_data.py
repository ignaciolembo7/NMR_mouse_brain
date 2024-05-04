#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import numpy as np
import matplotlib.pyplot as plt
from nogse import nogse
from brukerapi.dataset import Dataset as ds
import cv2
import seaborn as sns
sns.set(context='paper')
sns.set_style("whitegrid")


#T_NOGSE = input('T_NOGSE [ms]: ') #ms
file_name = "mousebrain_20200409"

image_paths, method_paths = nogse.upload_contrast_data(file_name)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for i in ["ROI1","ROI2","ROI3","ROI4","ROI5"]: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI
    
    T_nogse, g_contrast, n, f =  nogse.generate_contrast_roi(image_paths, method_paths, mask)

    nogse.plot_contrast_data(ax, i, g_contrast, f, T_nogse, n)
    nogse.plot_contrast_data(ax1, i, g_contrast, f, T_nogse, n)

    table = np.vstack((g_contrast, f))
    np.savetxt(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_nogse}/{i}_Datos_Contraste_vs_g_t={T_nogse}_n={n}.txt", table.T, delimiter=' ', newline='\n')
    
    fig1.tight_layout()
    fig1.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_nogse}/{i}_NOGSE_Contraste_vs_g_t={T_nogse}_n={n}.pdf")
    fig1.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_nogse}/{i}_NOGSE_Contraste_vs_g_t={T_nogse}_n={n}.png", dpi=600)
    plt.close(fig1)
    idx += 1

fig.tight_layout()
fig.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_nogse}/NOGSE_Contraste_vs_g_t={T_nogse}_n={n}.pdf")
fig.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_nogse}/NOGSE_Contraste_vs_g_t={T_nogse}_n={n}.png", dpi=600)
plt.close(fig)