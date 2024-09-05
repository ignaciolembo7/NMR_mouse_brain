#NMRSI - Ignacio Lembo Ferrari - 28/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
folder = "contrast_vs_g_restdist_mode"

id = int(input('id: '))
tnogse = float(input('T_NOGSE [ms]: ')) #ms
n = float(input('N: '))

slic = 1 # slice que quiero ver
modelo = "rest_dist"  # nombre carpeta modelo libre/rest/tort

D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
rois = ["ROI1"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_N={int(n)}_id={id}"
os.makedirs(directory, exist_ok=True)

for roi, color in zip(rois,palette):

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/slice={slic}/tnogse={tnogse}_N={int(n)}_id={id}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")

    g = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(g, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    g, f = zip(*vectores_ordenados)

    #modelo M_nogse_rest_dist
    model = lmfit.Model(nogse.contrast_vs_g_restdist, independent_vars=["TE", "G", "N", "D0"], param_names=["l_c_mode", "sigma", "M0"])
    model.set_param_hint("M0", value=20.0, min = 19.0, max = 21.0)
    model.set_param_hint("l_c_mode", value= 0.8, min = 0.7, max = 0.9)
    model.set_param_hint("sigma", value= 1.0, min = 0.9, max = 1.1)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["l_c_mode"].vary = 1
    params["M0"].vary = 1
    params["sigma"].vary = 1

    result = model.fit(f, params, TE=float(tnogse), G=g, N=n, D0=D0) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    M0_fit = result.params["M0"].value
    l_c_mode_fit = result.params["l_c_mode"].value
    sigma_fit = result.params["sigma"].value
    error_M0 = result.params["M0"].stderr
    error_l_c = result.params["l_c_mode"].stderr
    error_sigma = result.params["sigma"].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.contrast_vs_g_restdist(float(tnogse), g_fit, n, l_c_mode_fit, sigma_fit, M0_fit, D0)

    l_c_median = l_c_mode_fit*np.exp(sigma_fit**2)
    l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - l_c_mode = ", l_c_mode_fit, "+-", error_l_c, file=a)
        print("    ",  " - l_c_median = ", l_c_median, "+-", file=a)
        print("    ",  " - l_c_mid = ", l_c_mid, "+-", file=a)
        print("    ",  " - sigma = ", sigma_fit, "+-", error_sigma, file=a)
        print("    ",  " - D0 = ", D0, "+-", file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    nogse.plot_contrast_vs_g(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_adjust_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

    l_c = np.linspace(0.001, 10, 1000) #asi esta igual que en nogse.py
    dist = nogse.lognormal(l_c, sigma_fit, l_c_mode_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, l_c, l_c_mode_fit, sigma_fit, slic, color)
    nogse.plot_lognorm_dist(ax3, roi, tnogse, n, l_c, l_c_mode_fit, sigma_fit, slic, color)
    
    table = np.vstack((l_c, dist))
    np.savetxt(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_N={int(n)}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_N={int(n)}.png", dpi=600)
    plt.close(fig2)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)

fig3.tight_layout()
fig3.savefig(f"{directory}/dist_vs_lc_tnogse={tnogse}_N={int(n)}.pdf")
fig3.savefig(f"{directory}/dist_vs_lc_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig3)