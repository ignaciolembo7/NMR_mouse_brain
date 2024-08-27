#NMRSI - Ignacio Lembo Ferrari - 06/08/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

D0_ext = 2.3e-12 # m2/ms extra
D0_int = 0.7e-12 # intra
#D0 = D0_ext

n = 2 
slic = 0

file_name = "levaduras_20240622"
folder = "rest_vs_x_free"

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)


fig2, ax2 = plt.subplots(figsize=(8,6)) 

for i in ["G1"]: 

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"{directory}/D0_vs_tnogse_" + i + ".txt")

    ##########################################
    #COPIAR ESTO PARA GRAFICAR D0 vs tnogse

    tnogse = data[:, 0]
    D0 = data[:, 1]
    error_D0 = data[:, 2]

    #remover los elementos en la posicion 4, 6, 8 de tnogse y t_c 
    #tnogse = np.delete(tnogse, [4, 6, 8])
    #D0 = np.delete(D0, [4, 6, 8])

    ax1.errorbar(tnogse, D0, yerr=error_D0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{i}") #  yerr=error_t_c,
    #ax.plot(tnogse, tau_c, 'o-', markersize=3, linewidth=2)
    ax1.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=18)
    ax1.set_ylabel("Coeficiente de difusión $D_0$ [m$^2$/ms]", fontsize=18)
    ax1.legend(title='Gradiente', title_fontsize=18, fontsize=18, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)
    ##########################################

    fig1.tight_layout()
    fig1.savefig(f"{directory}/D0_vs_tnogse_" + i + ".png", dpi=600)
    fig1.savefig(f"{directory}/D0_vs_tnogse_" + i + ".pdf")

    ax2.errorbar(tnogse, D0, yerr=error_D0,  fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{i}")
    #ax.plot(tnogse, tau_c, 'o-', markersize=3, linewidth=2)
    ax2.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=18)
    ax2.set_ylabel("Coeficiente de difusión $D_0$ [m$^2$/ms]", fontsize=18)
    ax2.legend(title='Gradiente', title_fontsize=18, fontsize=18, loc='best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)


fig2.tight_layout()
fig2.savefig(f"{directory}/D0_vs_tnogse_allG.png", dpi=600)
fig2.savefig(f"{directory}/D0_vs_tnogse_allG.pdf")

print("Valor medio D0 = ", np.mean(D0), "+-", np.std(D0))