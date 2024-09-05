#NMRSI - Ignacio Lembo Ferrari - 27/08/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "nogse_vs_x_restdist_mode"
A0 = "sin_A0"
slic = 0 # slice que quiero ver
D0_folder = "D0_int"

D0_ext = 2.3e-12 # m2/ms extra
D0_int = 0.7e-12 # intra
D0 = D0_int
n = 2

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{D0_folder}"
os.makedirs(directory, exist_ok=True)

#palette = sns.color_palette("tab20", 4) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
palette = [
    "#1f77b4",  # Azul
    "#ff7f0e",  # Naranja
    "#f1c40f",  # Amarillo
    "#2ca02c",  # Verde
]
sns.set_palette(palette)
gs = ["G4"]
rois =  ["ROI1","ROI1", "ROI1","ROI1"]

fig2, ax2 = plt.subplots(figsize=(8,6)) 


for roi, g, color in zip(rois, gs, palette):

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"{directory}/{roi}_parameters_vs_tnogse_g={g}.txt")

    tnogse = data[:, 0]
    M0 = data[:, 6]
    error_M0 = data[:, 7]

    # Obtener los índices que ordenarían grad
    sorted_indices = np.argsort(tnogse)
    # Ordenar grad y M0 usando esos índices
    tnogse = tnogse[sorted_indices]
    M0 = M0[sorted_indices]

    #remover los elementos en la posicion 4, 6, 8 de tnogse y t_c 
    #tnogse = np.delete(tnogse, [4, 6, 8])
    #D0 = np.delete(D0, [4, 6, 8])

    #ax1.errorbar(tnogse, M0, yerr=error_M0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{g}") #  yerr=error_t_c,
    ax1.plot(tnogse, M0, 'o-', markersize=7, linewidth=2, color = color, label=f"{g}")
    ax1.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=27)
    ax1.set_ylabel("$M_0$", fontsize=27)
    #ax1.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax1.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax1.legend(title='Gradiente', title_fontsize=15, fontsize=15, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_M0_vs_tnogse_g={g}.png", dpi=600)
    fig1.savefig(f"{directory}/{roi}_M0_vs_tnogse_g={g}.pdf")

    #ax2.errorbar(tnogse, M0, yerr=error_M0,  fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{g}")
    ax2.plot(tnogse, M0, 'o-', markersize=7, linewidth=2, color = color, label=f"{g}")
    ax2.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=27)
    ax2.set_ylabel("$M_0$", fontsize=27)
    #ax2.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax2.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax2.legend(title='Gradiente', title_fontsize=15, fontsize=15, loc='best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)

fig2.tight_layout()
fig2.savefig(f"{directory}/M0_vs_tnogse_allG.png", dpi=600)
fig2.savefig(f"{directory}/M0_vs_tnogse_allG.pdf")