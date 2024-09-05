#NMRSI - Ignacio Lembo Ferrari - 02/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
from lmfit import Model, Parameters, minimize
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "mousebrain_20200409"
folder = "nogse_vs_x_restdist_mode_globalfit"
slic = 1 # slice que quiero ver
#D0 = 0.5e-12 #m2/ms 
D0_folder = "D0_libre" #f"{D0}"
A0 = "con_A0"
modelo = "Rest Dist Mode"  # nombre carpeta modelo libre/rest/tort

n = 2
num_grads = ["G1","G1","G1","G1","G1","G1","G1","G1","G1","G1","G1"]  # Añadir los gradientes que desees
tnogses = ["21.5","21.5","21.5","21.5","21.5", "21.5","21.5","21.5","21.5","21.5","21.5"]  # Añadir los tnogses correspondientes
#g = input('g: ') #mT/m
gs = [50.0,125.0,200.0,275.0,350.0,462.5,500.0,575.0,650.0,725.0,800.0]  # Añadir las intensidades de gradiente correspondientes
rois = ["ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1", "ROI1"]  # Añadir las ROIs correspondientes
ids = [1,2,1,1,2,1,2,1,1,1,2]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}/tnogse={tnogses[0]}_N={int(n)}"
os.makedirs(directory, exist_ok=True)

# Cargo la data para cada curva
xs = []
fs = []

for roi, tnogse, g, num_grad, id in zip(rois, tnogses, gs, num_grads, ids):

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_id={id}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]

    vectores_combinados = zip(x, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    x, f = zip(*vectores_ordenados)
    
    xs.append(x)
    fs.append(f)

# Parámetros genéricos
params = Parameters()
for i in range(len(xs)):
    params.add(f'lc{i+1}', value=2.3, min=2.0, max=5.0, vary = True)
    params.add(f'sigma{i+1}', value=0.5, min=0.1, max=0.8, vary = True)
    params.add(f'D0{i+1}', value=0.7e-12, min = 0.1e-12, max = 2.3e-12, vary=False)
params.add('M0', value=1.3, min=0.1, max=5.0, vary=True)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        lc = params[f'lc{i+1}']
        sigma = params[f'sigma{i+1}']
        M0 = params['M0']
        D0 = params[f'D0{i+1}']
        model = nogse.M_nogse_restdist(float(tnogses[i]), float(gs[i]), n, x, lc, sigma, M0, D0)
        fs_data = np.array(fs_data)

        if np.isnan(model).any() or np.isnan(fs_data).any():
            raise ValueError("NaN detected in model or fs values")
                          
        residuals.append(fs_data - model)
    
    return np.concatenate(residuals)

result = minimize(objective_function, params)

# Display fitting results
print(result.params.pretty_print())

sigma_fits = []
error_sigmas = []
lc_fits = []
error_lcs = []
D0_fits = []
error_D0s = []

M0_fit = result.params["M0"].value
error_M0 = result.params["M0"].stderr
#D0_fit = result.params["D0"].value
#error_D0 = result.params["D0"].stderr

for i in range(len(xs)):
    lc_fits.append(result.params[f'lc{i+1}'].value)
    sigma_fits.append(result.params[f'sigma{i+1}'].value)
    error_lcs.append(result.params[f'lc{i+1}'].stderr)
    error_sigmas.append(result.params[f'sigma{i+1}'].stderr)
    D0_fits.append(result.params[f"D0{i+1}"].value)
    error_D0s.append(result.params[f"D0{i+1}"].stderr)

x_fit = np.linspace(np.min(xs[0]), np.max(xs[0]), num=1000)
fits = [nogse.M_nogse_restdist(float(tnogses[i]), float(gs[i]), n, x_fit, lc_fits[i], sigma_fits[i], M0_fit, D0_fits[i]) for i in range(len(xs))]

palette = sns.color_palette("tab10", len(rois))

fig, ax = plt.subplots(figsize=(8,6)) 
fig1, ax1 = plt.subplots(figsize=(8,6)) 

for i, (roi, f, fit, lc_fit, sigma_fit, num_grad, g, color) in enumerate(zip(rois, fs, fits, lc_fits, sigma_fits, num_grads, gs, palette)):
    fig2, ax2 = plt.subplots(figsize=(8,6)) 
    fig3, ax3 = plt.subplots(figsize=(8,6)) 

    nogse.plot_nogse_vs_x_restdist(ax, rois[i], modelo, xs[i], x_fit, f, fit, tnogses[i], gs[i], n, slic, color) 
    nogse.plot_nogse_vs_x_restdist(ax2, rois[i], modelo, xs[i], x_fit, f, fit, tnogses[i], gs[i], n, slic, color) 

    l_c = np.linspace(0.01, 40, 1000)
    dist = nogse.lognormal(l_c, sigma_fit, lc_fit)
    nogse.plot_lognorm_dist_ptROI(ax1, rois[i], modelo, tnogses[i], gs[i], n, l_c, dist, slic, color, label  = roi)
    nogse.plot_lognorm_dist(ax3, rois[i], modelo, tnogses[i], gs[i], n, l_c, lc_fit, sigma_fit, slic, color)

    fig2.tight_layout()
    fig2.savefig(f"{directory}/{rois[i]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}.pdf")
    fig2.savefig(f"{directory}/{rois[i]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}.png", dpi=600)
    plt.close(fig)

    fig3.tight_layout()
    fig3.savefig(f"{directory}/{rois[i]}_dist_tnogse={tnogses[0]}_N={int(n)}.pdf")
    fig3.savefig(f"{directory}/{rois[i]}_dist_tnogse={tnogses[0]}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}.png", dpi=600)
plt.close(fig)

fig1.tight_layout()
fig1.savefig(f"{directory}/dist_tnogse={tnogses[0]}_N={int(n)}.pdf")
fig1.savefig(f"{directory}/dist_tnogse={tnogses[0]}_N={int(n)}.png", dpi=600)
plt.close(fig1)

with open(f"{directory}/parameters_tnogse={tnogses[0]}_N={int(n)}.txt", "a") as a:
    for i in range(len(xs)):
        print(f"{rois[i]} - lc_mode_{i+1} = {lc_fits[i]} +- {error_lcs[i]}", file=a)
        print(f"    - sigma_{i+1} = {sigma_fits[i]} +- {error_sigmas[i]}", file=a)
        print(f"    - D0_{i+1} = {D0_fits[i]} +- {error_D0s[i]}", file=a)
    print(f"    - M0 = {M0_fit} +- {error_M0}", file=a)

table = np.vstack([x_fit] + fits)
np.savetxt(f"{directory}/globalfit_ptROIS_nogse_vs_x_tnogse={tnogses[0]}_g={gs[0]}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

#for roi, tnogse, g, num_grad, lc_fit, error_lc, sigma_fit, error_sigma, D0_fit, error_D0 in zip(rois, tnogses, gs, num_grads, lc_fits, error_lcs, sigma_fits, error_sigmas, D0_fits, error_D0s):
#    with open(f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
#        print(tnogse, g, lc_fit, error_lc, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)

for roi, tnogse, g, num_grad, lc_fit, error_lc, sigma_fit, error_sigma, D0_fit, error_D0 in zip(rois, tnogses, gs, num_grads, lc_fits, error_lcs, sigma_fits, error_sigmas, D0_fits, error_D0s):
    with open(f"../results_{file_name}/{folder}/slice={slic}/{D0_folder}/{A0}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:    
        print(g, tnogse, lc_fit, error_lc, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)