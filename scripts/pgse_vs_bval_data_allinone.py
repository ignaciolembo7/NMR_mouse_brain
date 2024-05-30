#NMRSI - Ignacio Lembo Ferrari - 14/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import pgse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
os.makedirs(data_directory, exist_ok=True)
slic = 1 # slice que quiero ver 0 o 1
max_field = 1466.6153

image_paths, method_paths = pgse.upload_pgse_vs_bval_data(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas
fig3, ax3 = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas
rois = ["ROI1","ROI2"] #,"ROI3","ROI4","ROI5"]

for roi in rois: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI
    fig2, ax2 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI

    DwBvalEach, DwEffBval, DwGradAmp, DwGradRead, DwGradPhase, DwGradSlice, DwGradDur, DwGradSep, f =  pgse.generate_pgse_vs_bval_roi_allinone(image_paths, method_paths, mask, slic)

    G = round(DwGradAmp[0]*max_field/100,2)

    directory = f"../results_{file_name}/pgse_vs_bvalue_data/DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}"
    os.makedirs(directory, exist_ok=True)

    pgse.plot_pgse_vs_bval_data(ax, roi, DwEffBval, f, DwGradDur[0], DwGradSep[0], DwGradAmp[0], G, slic)
    pgse.plot_pgse_vs_bval_data(ax1, roi, DwEffBval, f, DwGradDur[0], DwGradSep[0], DwGradAmp[0], G, slic)
    pgse.plot_logpgse_vs_bval_data(ax2, roi, DwEffBval, f, DwGradDur[0], DwGradSep[0], DwGradAmp[0], G, slic)
    pgse.plot_logpgse_vs_bval_data(ax3, roi, DwEffBval, f, DwGradDur[0], DwGradSep[0], DwGradAmp[0], G, slic)

    table = np.vstack((DwEffBval, f))
    np.savetxt(f"{directory}/{roi}_Datos_pgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.pdf")
    fig1.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.png", dpi=600)
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_logpgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.pdf")
    fig2.savefig(f"{directory}/{roi}_logpgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.png", dpi=600)
    plt.close(fig2)
    
    idx += 1

fig.tight_layout()
fig.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.pdf")
fig.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.png", dpi=600)
plt.close(fig)

fig3.tight_layout()
fig3.savefig(f"{directory}/logpgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.pdf")
fig3.savefig(f"{directory}/logpgse_vs_bvalue_DwGradDur={round(DwGradDur[0],2)}_DwGradSep={round(DwGradSep[0],2)}.png", dpi=600)
plt.close(fig3)
