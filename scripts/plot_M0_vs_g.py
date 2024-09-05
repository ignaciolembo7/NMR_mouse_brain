#NMRSI - Ignacio Lembo Ferrari - 26/08/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
folder = "nogse_vs_x_restdist_mode"
slic = 1 # slice que quiero ver
D0_folder = "D0=0.7e-12"
A0 = "sin_A0"

tnogse = 21.5
n = 2

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}"
os.makedirs(directory, exist_ok=True)

gs = ["G1","G1","G1","G1","G1"]
rois =  ["ROI1","ROI2","ROI3","ROI4","ROI5"]

palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
#palette = [
#    "#1f77b4",  # Azul
#    "#ff7f0e",  # Naranja
#    "#f1c40f",  # Amarillo
##    "#2ca02c",  # Verde
#]
sns.set_palette(palette)

fig2, ax2 = plt.subplots(figsize=(8,6)) 

for roi, g in zip(rois, gs): 

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"{directory}/{roi}_parameters_vs_g_tnogse={tnogse}.txt")

    grad = data[:, 1]
    M0 = data[:, 6]
    error_M0 = data[:, 7]

    # Obtener los índices que ordenarían grad
    sorted_indices = np.argsort(grad)
    # Ordenar grad y M0 usando esos índices
    grad = grad[sorted_indices]
    M0 = M0[sorted_indices]

    #error_t_c = data[:, 2]

    #remover los elementos en la posicion 4, 6, 8 de tnogse y t_c 
    #tnogse = np.delete(tnogse, [4, 6, 8])
    #t_c = np.delete(t_c, [4, 6, 8])

    label_title = "ROI" #'$T_\mathrm{{NOGSE}}$ [ms]'
    label = roi

    #ax1.errorbar(grad, M0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{tnogse}") #  yerr=error_t_c,
    ax1.plot(grad, M0, 'o-', markersize=7, linewidth=2, label=label)
    ax1.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax1.set_ylabel("$M_0$", fontsize=27)
    #ax1.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax1.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax1.legend(title=label_title, title_fontsize=15, fontsize=15, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=15)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_M0_vs_g_tnogse={tnogse}.png", dpi=600)
    fig1.savefig(f"{directory}/{roi}_M0_vs_g_tnogse={tnogse}.pdf")

    #ax2.errorbar(grad, M0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{tnogse}")
    ax2.plot(grad, M0, 'o-', markersize=7, linewidth=2, label=label)
    ax2.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax2.set_ylabel("$M_0$", fontsize=27)
    #ax2.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax2.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax2.legend(title=label_title, title_fontsize=15, fontsize=15, loc='best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=15)

fig2.tight_layout()
fig2.savefig(f"{directory}/M0_vs_g_allROI.png", dpi=600)
fig2.savefig(f"{directory}/M0_vs_g_allROI.pdf")

#print("Valor medio M0 = ", np.mean(M0), "+-", np.std(M0))