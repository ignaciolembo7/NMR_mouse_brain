#NMRSI - Ignacio Lembo Ferrari - 23/08/2024

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
g = float(input('g [mT/m]: ')) #mT/m
n = float(input('N: '))
id = input('id: ')

file_name = "mousebrain_20200409"
folder = "nogse_vs_x_restdist_mode"
slic = 1 # slice que quiero ver
#D0 = 0.5e-12 #m2/ms 
D0_folder = "D0_libre" #f"{D0}"
A0 = "con_A0"
modelo = "Rest Dist Mode"  # nombre carpeta modelo libre/rest/tort

#D0_ext = 2.3e-12 # extra
#D0_int = 0.7e-12 # intra

fig, ax = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 

rois = ["ROI1","ROI2", "ROI3","ROI4","ROI5"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_id={id}"
os.makedirs(directory, exist_ok=True)

for roi, color in tqdm(zip(rois,palette)):

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_id={id}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(x, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    x, f = zip(*vectores_ordenados)

    #modelo M_nogse_rest_dist
    model = lmfit.Model(nogse.M_nogse_restdist, independent_vars=["TE", "G", "N", "x"], param_names=["l_c_mode", "sigma", "M0", "D0"])
    model.set_param_hint("M0", value = 1.3, min=0.1, max=5.0, vary=True)
    model.set_param_hint("l_c_mode", value=2.5, min = 2.0, max = 5, vary=True)
    model.set_param_hint("sigma", value=0.4, min = 0.1, max=0.8, vary=True)
    model.set_param_hint("D0", value=0.5e-12, min = 0.1e-12, max = 2.3e-12, vary=True)
    params = model.make_params()
    
    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    
    print("\n")
    print(result.params.pretty_print())
    print("\n")

    l_c_fit = result.params["l_c_mode"].value
    sigma_fit = result.params["sigma"].value
    M0_fit = result.params["M0"].value
    D0_fit = result.params["D0"].value
    error_l_c = result.params["l_c_mode"].stderr
    error_sigma = result.params["sigma"].stderr
    error_M0 = result.params["M0"].stderr
    error_D0 = result.params["D0"].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_restdist(tnogse, g, n, x_fit, l_c_fit, sigma_fit, M0_fit, D0_fit)

    l_c_median = l_c_fit*np.exp(sigma_fit**2)
    l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - l_c_mode = ", l_c_fit, "+-", error_l_c, file=a)
        print("    ",  " - l_c_median = ", l_c_median, "+-", file=a)
        print("    ",  " - l_c_mid = ", l_c_mid, "+-", file=a)
        print("    ",  " - sigma = ", sigma_fit, "+-", error_sigma, file=a)
        print("    ",  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - D0 = ", D0_fit, "+-", error_D0, file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    label = roi + " - ID: " + id
    nogse.plot_nogse_vs_x_restdist(ax, label, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)
    nogse.plot_nogse_vs_x_restdist(ax1, label, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

    l_c = np.linspace(0.01, 40, 1000) #asi esta igual que en nogse.py
    dist = nogse.lognormal(l_c, sigma_fit, l_c_fit)
    nogse.plot_lognorm_dist(ax2, label, modelo, tnogse, g, n, l_c, l_c_fit, sigma_fit, slic, color)
    nogse.plot_lognorm_dist(ax3, label, modelo, tnogse, g, n, l_c, l_c_fit, sigma_fit, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_adjust_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

    table = np.vstack((l_c, dist))
    np.savetxt(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig2)

    #with open(f"../results_{file_name}/{folder}/slice={slic}/{A0}/{roi}_parameters_vs_tnogse_g=G1.txt", "a") as a:
    #    print(tnogse, g, l_c_fit, error_l_c, sigma_fit, error_sigma, M0_fit, error_M0, file=a)

    with open(f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
        print(g, tnogse, l_c_fit, error_l_c, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)

fig3.tight_layout()
fig3.savefig(f"{directory}/dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig3.savefig(f"{directory}/dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig3)