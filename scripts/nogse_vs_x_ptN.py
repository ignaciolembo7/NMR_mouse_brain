#NMRSI - Ignacio Lembo Ferrari - 05/05/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nogse import nogse
sns.set_theme(context='paper')
sns.set_style("whitegrid")

T_NOGSE = 21.5
G = 800.0
ROI = 'ROI5'
slic = 1

#N=4
palette = sns.color_palette("tab20", len([[T_NOGSE, G, 2],[T_NOGSE, G, 4],[T_NOGSE, G, 8]])) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

fig, ax = plt.subplots(figsize=(8,6))

for i, color in zip([[T_NOGSE, G, 2],[T_NOGSE, G, 4],[T_NOGSE, G, 8]], palette): # N=2
    data = np.loadtxt(f"../results_mousebrain_20200409/nogse_vs_x_data/TNOGSE={i[0]}_G={i[1]}_N={i[2]}/{ROI}_Datos_nogse_vs_x_t={i[0]}_G={i[1]}_N={i[2]}.txt")

    x = data[:, 0]
    f = data[:, 1]/data[0,1]

    nogse.plot_nogse_vs_x_data_ptN(ax, ROI, x, f, i[0], i[1], i[2], slic, color)

fig.tight_layout()
fig.savefig(f"../results_mousebrain_20200409/nogse_vs_x_ptN/{ROI}_NOGSEvsx_TNOGSE={T_NOGSE}_G={G}_ptN.png", dpi = 600)
fig.savefig(f"../results_mousebrain_20200409/nogse_vs_x_ptN/{ROI}_NOGSEvsx_TNOGSE={T_NOGSE}_G={G}_ptN.pdf")
plt.close(fig)


#N=2
# Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
#palette = sns.color_palette("tab20", len([[T_NOGSE, 50.0, N],[T_NOGSE, 125.0, N],[T_NOGSE, 200.0, N],[T_NOGSE, 275.0, N],[T_NOGSE, 350.0, N],[T_NOGSE, 425.0, N],[T_NOGSE, 462.5, N],[T_NOGSE, 500.0, N],[T_NOGSE, 575.0, N],[T_NOGSE, 650.0, N],[T_NOGSE, 725.0, N],[T_NOGSE, 800.0, N]]))

#for i, color in zip([[T_NOGSE, 50.0, N],[T_NOGSE, 125.0, N],[T_NOGSE, 200.0, N],[T_NOGSE, 275.0, N],[T_NOGSE, 350.0, N],[T_NOGSE, 425.0, N],[T_NOGSE, 462.5, N],[T_NOGSE, 500.0, N],[T_NOGSE, 575.0, N],[T_NOGSE, 650.0, N],[T_NOGSE, 725.0, N],[T_NOGSE, 800.0, N]], palette): # N=2