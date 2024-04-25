#NMRSI - Ignacio Lembo Ferrari - 27/11/2023

import numpy as np
import matplotlib.pyplot as plt
from nogse import nogse

import seaborn as sns
sns.set(context='paper')
sns.set_style("whitegrid")


T_NOGSE = input('T_NOGSE [ms]: ') #ms
file_name = "mousebrain_20200409" #"levaduras_20220830" #resultados
roi = "roi_mousebrain_20200409" # "roi_levaduras_20220830"
roi_set = np.genfromtxt(f"../scripts/ROIS/{roi}.txt")

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

T_nogse, g_contrast, n, f1, f2, f3, f4,f5 =  nogse.upload_data_delta_M(file_name, roi_set)

rois = [f1,f2,f3,f4,f5] 
idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for i in ["ROI1","ROI2","ROI3","ROI4","ROI5"]: 

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI

    nogse.plot_contrast_datas(ax, i, g_contrast, rois[idx], T_nogse, n)
    nogse.plot_contrast_data(ax1, i, g_contrast, rois[idx], T_nogse, n)

    table = np.vstack((g_contrast, rois[idx]))
    np.savetxt(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/{i}_Datos_Contraste_vs_g_t={T_NOGSE}.txt", table.T, delimiter=' ', newline='\n')

    fig1.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/{i}_NOGSE_Contraste_vs_g_t={T_NOGSE}.pdf")
    fig1.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/{i}_NOGSE_Contraste_vs_g_t={T_NOGSE}.png", dpi=600)
    plt.close(fig1)
    idx += 1

fig.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/NOGSE_Contraste_vs_g_t={T_NOGSE}.pdf")
fig.savefig(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/NOGSE_Contraste_vs_g_t={T_NOGSE}.png", dpi=600)
plt.close(fig)

"""
table = np.vstack((g_contrast_fit, fit))
np.savetxt(f"../results_{file_name}/contraste_vs_g_data/T_NOGSE={T_NOGSE}/{i}_Ajuste_Contraste_vs_g_t={T_NOGSE}.txt", table.T, delimiter=' ', newline='\n')

    if(i == "ROI1"):
        color = "green"
    if(i == "ROI2"):
        color = "darkorange"
    if(i == "ROIw"):
        color = "blue"
"""    