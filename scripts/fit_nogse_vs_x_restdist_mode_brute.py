#NMRSI - Ignacio Lembo Ferrari - 23/08/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
from tqdm import tqdm
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params, fit_report

tnogse = float(input('T_NOGSE [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = float(input('N: '))
id = input('id: ')

A0 = "con_A0"
file_name = "mousebrain_20200409"
folder = "nogse_vs_x_restdist_brute"
slic = 1 # slice que quiero ver
modelo = "Rest Dist Brute"  # nombre carpeta modelo libre/rest/tort

D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0=D0_ext

fig, ax = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 

rois = ["ROI1","ROI2", "ROI3", "ROI4", "ROI5"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_id={id}"
os.makedirs(directory, exist_ok=True)

def fcn2min(params, g, f):
    M0 = params['M0']
    l_c_mode = params['l_c_mode']
    sigma = params['sigma']
    model = nogse.M_nogse_rest_dist(tnogse, g, n, x, l_c_mode, sigma, M0, D0)
    return model - f

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

    params = create_params(l_c_mode=dict(value=4.5, min=0.5, max=10, brute_step=0.5, vary=True),
                           sigma=dict(value=0.8, min=0.1, max=2.0, brute_step=0.25, vary=True),
                           M0=dict(value=1.0, min=0.0, max=100000.0, brute_step=10000, vary=True),
                           )

    fitter = Minimizer(fcn2min, params, fcn_args=(g, f))
    result_brute = fitter.minimize(method='brute', keep=1) #Ns = 25

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_chi.png")

    M0_fit = result_brute.params['M0'].value
    l_c_mode_fit = result_brute.params['l_c_mode'].value
    sigma_fit = result_brute.params['sigma'].value
    error_M0 = result_brute.params['M0'].stderr
    error_l_c_mode = result_brute.params['l_c_mode'].stderr
    error_sigma = result_brute.params['sigma'].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_rest_dist(float(tnogse), float(g), n, x_fit, l_c_mode_fit, sigma_fit, M0_fit, D0)

    l_c_median = l_c_mode_fit*np.exp(sigma_fit**2)
    l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - l_c_mode = ", l_c_mode_fit, "+-", error_l_c_mode, file=a)
        print("    ",  " - l_c_median = ", l_c_median, "+-", file=a)
        print("    ",  " - l_c_mid = ", l_c_mid, "+-", file=a)
        print("    ",  " - sigma = ", sigma_fit, "+-", error_sigma, file=a)
        print("    ",  " - D0 = ", D0, "+-", file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    label = roi + " - ID: " + id
    nogse.plot_nogse_vs_x_restdist(ax, label, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)
    nogse.plot_nogse_vs_x_restdist(ax1, label, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_adjust_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

    l_c = np.linspace(0.001, 10, 1000) #asi esta igual que en nogse.py
    dist = nogse.lognormal(l_c, sigma_fit, l_c_mode_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, l_c, l_c_mode_fit, sigma_fit, slic, color)
    nogse.plot_lognorm_dist(ax3, roi, tnogse, n, l_c, l_c_mode_fit, sigma_fit, slic, color)
    
    table = np.vstack((l_c, dist))
    np.savetxt(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig2)

    with open(f"../results_{file_name}/{folder}/slice={slic}/{A0}/{roi}_parameters_vs_tnogse_G1.txt", "a") as a:
        print(tnogse, g, l_c_mode_fit, error_l_c_mode, sigma_fit, error_sigma, M0_fit, error_M0, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)

fig3.tight_layout()
fig3.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig3.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig3)

"""
best_result = copy.deepcopy(result_brute)
for candidate in result_brute.candidates:
    trial = fitter.minimize(method='leastsq', params=candidate.params)
    if trial.chisqr < best_result.chisqr:
        best_result = trial
M0_int_fit = best_result.params['M0_int'].value
M0_ext_fit = best_result.params['M0_ext'].value
t_c_int_fit = best_result.params['t_c_int'].value
t_c_ext_fit = best_result.params['t_c_ext'].value
alpha_fit = best_result.params['alpha'].value
error_t_c_int = best_result.params['t_c_int'].stderr
error_t_c_ext = best_result.params['t_c_ext'].stderr
error_alpha = best_result.params['alpha'].stderr
""" 