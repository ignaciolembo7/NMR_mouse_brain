#NMRSI - Ignacio Lembo Ferrari - 23/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import pgse
from lmfit import Model
import os
import seaborn as sns
from tqdm import tqdm
sns.set_theme(context='paper')
sns.set_style("whitegrid")

puntos1 = 6
DwGradSep = float(input('Grad Separation (Delta) [ms]: ')) #ms
DwGradDur = float(input('Grad Duration (delta) [ms]: ')) #ms

file_name = "mousebrain_20200409"
folder = "pgse_vs_bval_expmodel"
slic = 1 # slice que quiero ver
modelo = "biexp"  # nombre carpeta modelo libre/rest/tort
D0_ext = 0.0023 #2.3e-12 # extra
D0_int = 0.0007 #0.7e-12 # intra

D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 

rois = ["ROI1"] #,"ROI2","ROI3","ROI4","ROI5"]
palette = sns.color_palette("tab20", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/pgse_vs_bvalue_{modelo}/slice={slic}/DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}/"
os.makedirs(directory, exist_ok=True)

for roi, color in tqdm(zip(rois,palette)):

    data = np.loadtxt(f"../results_{file_name}/pgse_vs_bvalue_data/slice={slic}/DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}/{roi}_Datos_pgse_vs_bvalue_DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}.txt")

    bval = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(bval, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    bval, f = zip(*vectores_ordenados)
    bval = np.array(bval[-puntos1:])
    f = np.array(f[-puntos1:])
    #modelo M_pgse_exp
    model = Model(pgse.M_pgse_biexp, independent_vars=["bval"], param_names=["M1", "D1", "M2", "D2"])
    model.set_param_hint("M1", value = 100, min = 0)
    model.set_param_hint("D1", value = 1e-5, min = 0)
    model.set_param_hint("M2", value = 0, min = 0)
    model.set_param_hint("D2", value = 0, min = 0)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["M1"].vary = 1
    params["D1"].vary = 1
    params["M2"].vary = 0
    params["D2"].vary = 0
    
    result = model.fit(f, params, bval=bval)
    M1_fit = result.params["M1"].value
    error_M1 = result.params["M1"].stderr
    D1_fit = result.params["D1"].value
    error_D1 = result.params["D1"].stderr
    M2_fit = result.params["M2"].value
    error_M2 = result.params["M2"].stderr
    D2_fit = result.params["D2"].value
    error_D2 = result.params["D2"].stderr

    print(result.fit_report())

    bval_fit = np.linspace(np.min(bval), np.max(bval), num=1000)
    fit = pgse.M_pgse_biexp(bval_fit, M1_fit, D1_fit, M2_fit, D2_fit)

    with open(f"{directory}/parameters_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.txt", "a") as a:
        print(roi,  " - M1 = ", M1_fit, "+-", error_M1, file=a)
        print("    ",  " - D1 = ", D1_fit, "+-", error_D1, file=a)
        print("    ",  " - M2 = ", M2_fit, "+-", error_M2, file=a)
        print("    ",  " - D2 = ", D2_fit, "+-", error_D2, file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6))
    fig3, ax3 = plt.subplots(figsize=(8,6))

    #pgse.plot_logpgse_vs_bval_biexp(ax, roi, modelo, bval, bval_fit, f, fit, D1_fit, D2_fit, DwGradDur, DwGradSep, slic, color)
    pgse.plot_logpgse_vs_bval_biexp(ax1, roi, modelo, bval, bval_fit, f, fit, D1_fit, D2_fit, DwGradDur, DwGradSep, slic, color)
    #pgse.plot_pgse_vs_bval_biexp(ax2, roi, modelo, bval, bval_fit, f, fit, D1_fit, D2_fit, DwGradDur, DwGradSep, slic, color)
    pgse.plot_pgse_vs_bval_biexp(ax3, roi, modelo, bval, bval_fit, f, fit, D1_fit, D2_fit, DwGradDur, DwGradSep, slic, color)

    table = np.vstack((bval_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_logpgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
    fig1.savefig(f"{directory}/{roi}_logpgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
    plt.close(fig1)
    fig3.tight_layout()
    fig3.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
    fig3.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
    plt.close(fig3)

#fig.tight_layout()
#fig.savefig(f"{directory}/logpgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
#fig.savefig(f"{directory}/logpgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
#plt.close(fig)
#fig2.tight_layout()
#fig2.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
#fig2.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
#plt.close(fig2)