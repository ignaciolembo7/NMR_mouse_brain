#NMRSI - Ignacio Lembo Ferrari - 27/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import pgse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
folder = "pgse_vs_bval_expmodel"
slic = 1 # slice que quiero ver
modelo = "biexp"  # nombre carpeta modelo libre/rest/tort
D0_ext = 0.0023 #2.3e-12 # extra
D0_int = 0.0007 #0.7e-12 # intra

D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 

rois = ["ROI1"] #,"ROI2","ROI3","ROI4","ROI5"]
palette = sns.color_palette("tab20", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)


for roi, color in zip(rois,palette):

    # Create directory if it doesn't exist
    directory = f"../results_{file_name}/pgse_vs_bvalue_{modelo}/slice={slic}/{roi}"
    os.makedirs(directory, exist_ok=True)

    data = np.loadtxt(f"{directory}/{roi}.txt")

    delta  = data[:, 0]
    Delta = data[:, 1]
    td = Delta - delta/3
    D1 = data[:, 2]
    D2 = data[:, 3]
    
    pgse.plot_D_vs_td(ax, roi, td, D1, D2, slic)
    
    fig.tight_layout()
    fig.savefig(f"{directory}/{roi}_D_vs_td_{modelo}.pdf")
    fig.savefig(f"{directory}/{roi}_D_vs_td_{modelo}.png", dpi=600)
    plt.close(fig)