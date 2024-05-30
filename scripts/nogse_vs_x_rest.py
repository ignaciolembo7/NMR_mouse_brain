#NMRSI - Ignacio Lembo Ferrari - 18/05/2024

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
file_name = "mousebrain_20200409"
folder = "nogse_vs_x_restmodel"
slic = 1 # slice que quiero ver
modelo = "restmodel"  # nombre carpeta modelo libre/rest/tort
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra

D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
rois = ["ROI1","ROI2", "ROI3","ROI4","ROI5"]
palette = sns.color_palette("tab20", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/t={tnogse}_G={g}_N={int(n)}"
os.makedirs(directory, exist_ok=True)

for roi, color in tqdm(zip(rois,palette)):

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice_{slic}/TNOGSE={tnogse}_G={g}_N={int(n)}/{roi}_Datos_nogse_vs_x_t={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(x, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    x, f = zip(*vectores_ordenados)

    #modelo M_nogse_rest
    model = lmfit.Model(nogse.M_nogse_rest, independent_vars=["TE", "G", "N", "x"], param_names=["t_c","M0", "D0"])
    model.set_param_hint("M0", value=220)
    model.set_param_hint("t_c", value=tnogse, min = 0.01, max = tnogse)
    model.set_param_hint("D0", value = D0_int, min = 0.01)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["t_c"].vary = 1
    params["M0"].vary = 0
    params["D0"].vary = 1
    
    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=float(n), x=x) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    M0_fit = result.params["M0"].value
    t_c_fit = result.params["t_c"].value
    error_M0 = result.params["M0"].stderr
    error_t_c = result.params["t_c"].stderr
    D0_fit = result.params["D0"].value
    error_D0 = result.params["D0"].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_rest(float(tnogse), float(g), float(n), x_fit, t_c_fit, M0_fit, D0_fit)

    with open(f"{directory}/parameters_t={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - t_c = ", t_c_fit, "+-", error_t_c, file=a)
        print("    ",  " - D0 = ", D0_fit, "+-", error_D0, file=a)
    
    nogse.plot_nogse_vs_x_rest(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, t_c_fit, slic, color)
    nogse.plot_nogse_vs_x_rest(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, t_c_fit, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_t={tnogse}_g_{g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_t={tnogse}_g_{g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_t={tnogse}_g_{g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_t={tnogse}_g_{g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_t={tnogse}_g_{g}_N={int(n)}.png", dpi=600)
plt.close(fig)