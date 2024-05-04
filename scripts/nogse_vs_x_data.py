#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

import numpy as np
import matplotlib.pyplot as plt
from nogse import nogse
import os
import seaborn as sns
sns.set(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
slic = 1 # slice que quiero ver 0 o 1

image_paths, method_paths = nogse.upload_NOGSE_vs_x_data(file_name, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for i in ["ROI1","ROI2","ROI3","ROI4","ROI5"]: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI
    
    T_nogse, g, x, n, f =  nogse.generate_NOGSE_vs_x_roi(image_paths, method_paths, mask, slic)

    # Create directory if it doesn't exist
    directory = f"../results_{file_name}/nogse_vs_x_data/TNOGSE={T_nogse}_G={g}_N={int(n)}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_nogse_vs_x_data(ax, i, x, f, T_nogse, n, slic)
    nogse.plot_nogse_vs_x_data(ax1, i, x, f, T_nogse, n, slic)

    table = np.vstack((x, f))
    np.savetxt(f"{directory}/{i}_Datos_nogse_vs_x_t={T_nogse}_G={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{i}_nogse_vs_x_t={T_nogse}_G={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{i}_nogse_vs_x_t={T_nogse}_G={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)
    idx += 1

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_t={T_nogse}_G={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_t={T_nogse}_G={g}_N={int(n)}.png", dpi=600)
plt.close(fig)
