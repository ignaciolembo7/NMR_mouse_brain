#NMRSI - Ignacio Lembo Ferrari - 05/05/2024

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid") 

#Constantes
tnogse = float(input("TNOGSE: "))
N = int(input("N: "))

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

for roi, id, region in zip(rois, ids, regions): 

    data1 = np.loadtxt(f"../results_mousebrain_20200409/contrast_vs_g_data/slice={slic}/TNOGSE={tnogse}_N={N}_id={id}/{roi}_Datos_contrast_vs_g_t={tnogse}_n={N}.txt")
    g = data1[:, 0]
    f = data1[:, 1]

    #Ordeno en el caso de que el vector g no venga ordenado de menor a mayor
    data = list(zip(g, f))
    sorted_data = sorted(data, key=lambda x: x[0])
    g, fit1 = zip(*sorted_data)
    g_contrast = np.array(g, dtype=float)
    f = np.array(fit1, dtype=float)
    
    l_d = np.sqrt(2*D0_int*tnogse)
    #l_c = np.sqrt(D0_ext*tau_c[idx])
    l_G = ((2**(3/2))*D0_int/(gamma*g_contrast))**(1/3)
    L_d = l_d/l_G
    #L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_d**(-1/2))
    l_c_f = L_c_f*l_G 

    ax1.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label = region)
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=27)
    ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    title = ax1.set_title(("$T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || $D_0$ = {} $m^2$/ms").format(tnogse, N, D0_int), fontsize=18)
    ax1.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    #ax1.set_xlim(0, 1400)

    ax2.plot(L_c_f, f, "-o", markersize=7, linewidth = 2, label = region)
    ax2.set_xlabel("Longitud de centro del filtro $L_C^f$", fontsize=27)
    ax2.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    title = ax2.set_title(("$T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || $D_0$ = {} $m^2$/ms").format(tnogse, N, D0_int), fontsize=18)
    ax2.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    #ax2.set_xlim(0.4, 1.2)

    ax3.plot(L_d, f, "-o", markersize=7, linewidth = 2, label= region)
    ax3.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
    ax3.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    title = ax3.set_title(("$T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || $D_0$ = {} $m^2$/ms").format(tnogse, N, D0_int), fontsize=18)
    ax3.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax3.tick_params(axis='y', labelsize=16, color='black')
    #ax3.set_xlim(0.6, 4.6)

    ax4.plot(l_c_f*1e6, f, "-o", markersize=7, linewidth = 2, label = region)
    ax4.set_xlabel("Longitud de centro del filtro $l_c^f ~[\mu m]$", fontsize=27)
    ax4.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    title = ax4.set_title(("$T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || $D_0$ = {} $m^2$/ms").format(tnogse, N, D0_int), fontsize=18)
    ax4.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    ax4.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax4.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax4.tick_params(axis='y', labelsize=16, color='black')
    #ax4.set_xlim(0.5,11)

    #ax5.plot(L_c, f, "-o", markersize=7, linewidth = 2, label= T_NOGSE[i] )
    #ax5.set_xlabel(r"Longitud de restricción $L_C$", fontsize=18)
    #ax5.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax5.set_title(("Mouse Brain || $T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || D_0 = {} $m^2$/ms").format(T_NOGSE[0],N, D0_int), fontsize=18)
    #ax5.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    #ax5.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    #ax5.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    #ax5.tick_params(axis='y', labelsize=16, color='black')
    #ax5.set_xlim(0,1.5)

fig1.tight_layout()
fig1.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_g_TNOGSE={tnogse}_N={N}.pdf")
fig1.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_g_TNOGSE={tnogse}_N={N}.png", dpi= 600)
fig2.tight_layout()
fig2.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_L_cf_TNOGSE={tnogse}_N={N}.pdf")
fig2.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_L_cf_TNOGSE={tnogse}_N={N}.png", dpi= 600)
fig3.tight_layout()
fig3.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_Ld_TNOGSE={tnogse}_N={N}.pdf")
fig3.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_Ld_TNOGSE={tnogse}_N={N}.png", dpi= 600)
fig4.tight_layout()
fig4.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_lcf_TNOGSE={tnogse}_N={N}.pdf")
fig4.savefig(f"../results_mousebrain_20200409/contrast_vs_advar/contrast_vs_lcf_TNOGSE={tnogse}_N={N}.png", dpi= 600)