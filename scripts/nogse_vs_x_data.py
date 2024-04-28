#NMRSI - Ignacio Lembo Ferrari - 27/11/2023

import numpy as np
import matplotlib.pyplot as plt
from nogse import nogse
import seaborn as sns
sns.set(context='paper')
sns.set_style("whitegrid")

#T_NOGSE = input('T_NOGSE [ms]: ') #ms
file_name = "mousebrain_20200409"

image_paths, method_paths = nogse.upload_NOGSE_vs_x_data(file_name)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for i in ["ROI1","ROI2","ROI3","ROI4","ROI5"]: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI
    
    T_nogse, g, x, n, f =  nogse.generate_NOGSE_vs_x_roi(image_paths, method_paths, mask)

    nogse.plot_nogse_vs_x_data(ax, i, x, f, T_nogse, n)
    nogse.plot_nogse_vs_x_data(ax1, i, x, f, T_nogse, n)

    table = np.vstack((x, f))
    np.savetxt(f"../results_{file_name}/nogse_vs_x_data/T_NOGSE={T_nogse}/{i}_Datos_nogse_vs_x_t={T_nogse}_g={g}.txt", table.T, delimiter=' ', newline='\n')

    fig1.savefig(f"../results_{file_name}/nogse_vs_x_data/T_NOGSE={T_nogse}/{i}_nogse_vs_x_t={T_nogse}_g={g}.pdf")
    fig1.savefig(f"../results_{file_name}/nogse_vs_x_data/T_NOGSE={T_nogse}/{i}_nogse_vs_x_t={T_nogse}_g={g}.png", dpi=600)
    plt.close(fig1)
    idx += 1

fig.savefig(f"../results_{file_name}/nogse_vs_x_data/T_NOGSE={T_nogse}/nogse_vs_x_t={T_nogse}_g={g}.pdf")
fig.savefig(f"../results_{file_name}/nogse_vs_x_data/T_NOGSE={T_nogse}/nogse_vs_x_t={T_nogse}_g={g}.png", dpi=600)
plt.close(fig)
