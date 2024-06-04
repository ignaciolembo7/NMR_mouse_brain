#NMRSI - Ignacio Lembo Ferrari - 28/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
from tqdm import tqdm
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogse = float(input('T_NOGSE [ms]: ')) #ms
n = float(input('N: '))
file_name = "mousebrain_20200409"
folder = "contrast_vs_g_restdist_mode"
slic = 1 # slice que quiero ver
modelo = "restdist"  # nombre carpeta modelo libre/rest/tort
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra

D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
fig1, ax1 = plt.subplots(figsize=(8,6)) 
rois = ["ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1"]
ids = [1,2,3,9,10,12,13] #,4,5,6,7,8,9,10]
regions = ["Splenium - 1", "Isthmus - 3"] #, "Isthmus - 3", "Isthmus - 9", "Splenium - 10",  "Isthmus - 12", "Isthmus - 13"] #, "Splenium4", "Splenium5", "Isthmus6", "Isthmus7", "Isthmus8", "Isthmus9", "Isthmus10"]
palette = sns.color_palette("tab10", len(ids)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/t={tnogse}_N={int(n)}"
os.makedirs(directory, exist_ok=True)

for color, id, region, roi in zip(palette,ids,regions, rois):

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/slice={slic}/tnogse={tnogse}_N={int(n)}_id={id}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    data_fit = np.loadtxt(f"../results_{file_name}/contrast_vs_g_restdist_mode/tnogse={tnogse}_N={int(n)}_id={id}/{roi}_adjust_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    data_dist = np.loadtxt(f"../results_{file_name}/contrast_vs_g_restdist_mode/tnogse={tnogse}_N={int(n)}_id={id}/{roi}_dist_vs_lc_tnogse={tnogse}_N={int(n)}.txt")

    g = data[:, 0]
    f = data[:, 1]
    g_fit = data_fit[:, 0]
    fit = data_fit[:, 1]
    l_c = data_dist[:, 0]
    dist = data_dist[:, 1]

    nogse.plot_contrast_vs_g_restdist(ax, region , modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    ax1.plot(l_c, dist, "-", color=color, linewidth = 2, label = region)
    ax1.set_xlabel("Longitud de correlación $l_c$ [$\mu$m]", fontsize=27)
    ax1.set_ylabel("P($l_c$)", fontsize=27)
    ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax1.legend( fontsize=18, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    title = ax1.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)
    plt.fill_between(l_c, dist, color=color, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_t={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_t={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)

fig1.tight_layout()
fig1.savefig(f"{directory}/dist_vs_lc_t={tnogse}_N={int(n)}.pdf")
fig1.savefig(f"{directory}/dist_vs_lc_t={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig1)