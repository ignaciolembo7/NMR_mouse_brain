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
folder = "contrast_vs_g_intrarestdist_extrarestdist_mode"

id = int(input('id: '))
tnogse = float(input('T_NOGSE [ms]: ')) #ms
n = float(input('N: '))

slic = 1 # slice que quiero ver
modelo = "rest_dist"  # nombre carpeta modelo libre/rest/tort

D0_ext = 2.3e-12 # extra
D0_int = 0.5e-12 # intra
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
    model = lmfit.Model(nogse.contrast_vs_g_intrarestdist_extrarestdist, independent_vars=["TE", "G", "N"], param_names=["l_c_mode_int", "l_c_mode_ext", "sigma_int", "sigma_ext", "M0_int", "M0_ext", "D0_int", "D0_ext"])
    model.set_param_hint("M0_int", value=21.0, min = 12.0, max = 30.0)
    model.set_param_hint("M0_ext", value=21.0, min = 12.0, max = 30.0)
    model.set_param_hint("l_c_mode_int", value= 0.4, min = 0.1, max = 5.0)
    model.set_param_hint("l_c_mode_ext", value= 0.4, min = 0.1, max = 4.0)  
    model.set_param_hint("sigma_int", value= 0.001, min = 0.001, max = 5)
    model.set_param_hint("sigma_ext", value= 0.92, min = 0.1, max = 1.4)
    model.set_param_hint("D0_int", value= D0_int, min = D0_int/2, max = D0_ext)
    model.set_param_hint("D0_ext", value= D0_int, min = D0_int/2, max = D0_ext)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["M0_int"].vary = 1
    params["M0_ext"].vary = 1
    params["l_c_mode_int"].vary = 1
    params["l_c_mode_ext"].vary = 1
    params["sigma_int"].vary = 0
    params["sigma_ext"].vary = 1
    params["D0_int"].vary = 1
    params["D0_ext"].vary = 1

    result = model.fit(f, params, TE=float(tnogse), G=g, N=n) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    M0_int_fit = result.params["M0_int"].value
    M0_ext_fit = result.params["M0_ext"].value
    l_c_mode_int_fit = result.params["l_c_mode_int"].value
    l_c_mode_ext_fit = result.params["l_c_mode_ext"].value
    sigma_int_fit = result.params["sigma_int"].value
    sigma_ext_fit = result.params["sigma_ext"].value
    D0_int_fit = result.params["D0_int"].value
    D0_ext_fit = result.params["D0_ext"].value
    error_M0_int = result.params["M0_int"].stderr
    error_M0_ext = result.params["M0_ext"].stderr
    error_l_c_mode_int = result.params["l_c_mode_int"].stderr
    error_l_c_mode_ext = result.params["l_c_mode_ext"].stderr
    error_sigma_int = result.params["sigma_int"].stderr
    error_sigma_ext = result.params["sigma_ext"].stderr
    error_D0_int = result.params["D0_int"].stderr
    error_D0_ext = result.params["D0_ext"].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.contrast_vs_g_intrarestdist_extrarestdist(float(tnogse), g_fit, n, l_c_mode_int_fit, l_c_mode_ext_fit, sigma_int_fit, sigma_ext_fit, M0_int_fit, M0_ext_fit, D0_int_fit, D0_ext_fit)
    fit_int = nogse.contrast_vs_g_restdist(float(tnogse), g_fit, n, l_c_mode_int_fit, sigma_int_fit, M0_int_fit, D0_int_fit)
    fit_ext = nogse.contrast_vs_g_restdist(float(tnogse), g_fit, n, l_c_mode_ext_fit, sigma_ext_fit, M0_ext_fit, D0_ext_fit)

    l_c_median_int = l_c_mode_int_fit*np.exp(sigma_int_fit**2)
    l_c_mid_int = l_c_median_int*np.exp((sigma_int_fit**2)/2)
    l_c_median_ext = l_c_mode_ext_fit*np.exp(sigma_ext_fit**2)
    l_c_mid_ext = l_c_median_ext*np.exp((sigma_ext_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0_int = ", M0_int_fit, "+-", error_M0_int, file=a)
        print("    ",  " - l_c_mode_int = ", l_c_mode_int_fit, "+-", error_l_c_mode_int, file=a)
        print("    ",  " - l_c_median_int = ", l_c_median_int, "+-", file=a)
        print("    ",  " - l_c_mid_int = ", l_c_mid_int, "+-", file=a)
        print("    ",  " - sigma_int = ", sigma_int_fit, "+-", error_sigma_int, file=a)
        print("    ",  " - D0_int = ", D0_int_fit, "+-", error_D0_int, file=a)
        print("    ",  " - M0_ext = ", M0_ext_fit, "+-", error_M0_ext, file=a)  
        print("    ",  " - l_c_mode_ext = ", l_c_mode_ext_fit, "+-", error_l_c_mode_ext, file=a)
        print("    ",  " - l_c_median_ext = ", l_c_median_ext, "+-", file=a)
        print("    ",  " - l_c_mid_ext = ", l_c_mid_ext, "+-", file=a)
        print("    ",  " - sigma_ext = ", sigma_ext_fit, "+-", error_sigma_ext, file=a)
        print("    ",  " - D0_ext = ", D0_ext_fit, "+-", error_D0_ext, file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    nogse.plot_contrast_vs_g(ax, "intra", modelo, g, g_fit, f, fit_int, tnogse, n, slic, "green")
    nogse.plot_contrast_vs_g(ax1, "intra", modelo, g, g_fit, f, fit_int, tnogse, n, slic, "green")
    nogse.plot_contrast_vs_g(ax, "extra", modelo, g, g_fit, f, fit_ext, tnogse, n, slic, "orange")
    nogse.plot_contrast_vs_g(ax1, "extra", modelo, g, g_fit, f, fit_ext, tnogse, n, slic, "orange")
    nogse.plot_contrast_vs_g(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_adjust_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

    l_c = np.linspace(0.001, 10, 1000) #asi esta igual que en nogse.py
    dist_int = nogse.lognormal(l_c, sigma_int_fit, l_c_mode_int_fit)
    #nogse.plot_lognorm_dist(ax2, "intra", tnogse, n, l_c, l_c_mode_int_fit, sigma_int_fit, slic, "green")
    #nogse.plot_lognorm_dist(ax3, "intra", tnogse, n, l_c, l_c_mode_int_fit, sigma_int_fit, slic, "green")

    table = np.vstack((l_c, dist_int))
    np.savetxt(f"{directory}/{roi}_dist_int_vs_lc_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    l_c = np.linspace(0.001, 10, 1000) #asi esta igual que en nogse.py
    dist_ext = nogse.lognormal(l_c, sigma_ext_fit, l_c_mode_ext_fit)
    nogse.plot_lognorm_dist(ax2, "extra", tnogse, n, l_c, l_c_mode_ext_fit, sigma_ext_fit, slic, "orange")
    nogse.plot_lognorm_dist(ax3, "extra", tnogse, n, l_c, l_c_mode_ext_fit, sigma_ext_fit, slic, "orange")

    table = np.vstack((l_c, dist_ext))
    np.savetxt(f"{directory}/{roi}_dist_ext_vs_lc_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

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