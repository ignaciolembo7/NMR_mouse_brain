#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

from nogse import nogse
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
sns.set(context='paper')
sns.set_style("whitegrid")

#T_NOGSE = input('T_NOGSE [ms]: ') #ms
file_name = "mousebrain_20200409"

folder_ranges = sys.argv[1] #input("Ingresa los rangos de carpetas para E_hahn (desde-hasta,desde-hasta,...): ")

image_paths, method_paths = nogse.upload_NOGSE_vs_x_data(file_name,folder_ranges)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for i in ["ROI1","ROI2","ROI3","ROI4","ROI5"]: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI
    
    T_nogse, g, x, n, f =  nogse.generate_NOGSE_vs_x_roi(image_paths, method_paths, mask)

    # Create directory if it doesn't exist
    directory = f"results_{file_name}/nogse_vs_x_data/TNOGSE={T_nogse}_G={g}_N={int(n)}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_nogse_vs_x_data(ax, i, x, f, T_nogse, n)
    nogse.plot_nogse_vs_x_data(ax1, i, x, f, T_nogse, n)

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
