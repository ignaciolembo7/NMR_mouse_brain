#NMRSI - Ignacio Lembo Ferrari - 05/05/2024

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid") 

#Constantes
tnogse = float(input("TNOGSE: "))
N = int(input("N: "))
gs = [125.0, 800.0]

slic = 1
rois = ["ROI1", "ROI1"]
ids = [1,2]
regions = ["Splenium", "Isthmus"]

D0_ext = 2.289355e-12 #m2/ms
D0_int = 0.7e-12
gamma = 267.5221900 #1/ms.mT
#alpha = 0.2 # =1 (libre) = inf (rest) ## es 1/alpha 

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
fig4, ax4 = plt.subplots(figsize=(8,6)) 
fig5, ax5 = plt.subplots(figsize=(8,6)) 
fig6, ax6 = plt.subplots(figsize=(8,6)) 

#Unidades 
#[g] = mT/m 

for g in gs: 

    l_d = np.sqrt(2*D0_int*tnogse)
    l_c = np.linspace(1*(1e-6), 5*(1e-6), 1000)
    tau_c = np.linspace(0.1, 50, 1000)      #(l_c**2)/(D0_int)
    l_G = ((2**(3/2))*D0_int/(gamma*g))**(1/3)
    L_d = l_d/l_G
    L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_d**(-1/2))
    l_c_f = L_c_f*l_G 
    #dM = 2*(N-1)*np.exp(-3/2)*(L_c_f**6)*np.exp(-12*((L_c-L_c_f)/L_c_f)**2)
    dM = np.exp((-1)*(gamma**2)*(g**2)*D0_int*(tau_c**3)*(tnogse/tau_c-3))*(np.exp((gamma**2)*(g**2)*D0_int*(tau_c**3)*2*(N-1))-1)
    print(dM)

    ax1.plot(l_c*1e6, dM, "-", markersize=7, linewidth = 2, label = f" {g} mT/m")
    ax1.set_xlabel("Longitud de restricción $l_c ~[\mu m]$", fontsize=27)
    ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    title = ax1.set_title(("$T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || $D_0$ = {} $m^2$/ms").format(tnogse, N, D0_int), fontsize=18)
    ax1.legend(title='Gradiente: ', title_fontsize=18, fontsize=18, loc = 'best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    ax1.set_xlim(0, 10)


fig1.tight_layout()
fig1.savefig(f"../results_mousebrain_20200409/simulations/contrast_vs_lc_TNOGSE={tnogse}_N={N}.pdf")
fig1.savefig(f"../results_mousebrain_20200409/simulations/contrast_vs_lc_TNOGSE={tnogse}_N={N}.png", dpi= 600)


"""
    data1 = np.loadtxt(f"../results_mousebrain_20200409/contrast_vs_g_data/slice={slic}/TNOGSE={tnogse}_N={N}_id={id}/{roi}_Datos_contrast_vs_g_t={tnogse}_n={N}.txt")
    g = data1[:, 0]
    f = data1[:, 1]

    #Ordeno en el caso de que el vector g no venga ordenado de menor a mayor
    data = list(zip(g, f))
    sorted_data = sorted(data, key=lambda x: x[0])
    g, fit1 = zip(*sorted_data)
    g_contrast = np.array(g, dtype=float)
    f = np.array(fit1, dtype=float)
    

"""