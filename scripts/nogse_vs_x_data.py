#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

ids = ["1","1","1","1","1"] #E1259
#ids = ["2","2","2","2","2"] #E518
rois = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
serial = 1259
idxs = [1,2,3,4,5] #mask index
slic = 1 # slice que quiero ver
A0 = "con_A0"

file_name = "mousebrain_20200409"
folder = "nogse_vs_x_data"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"

image_paths, method_paths = nogse.upload_NOGSE_vs_x_data(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

fig, ax = plt.subplots(figsize=(8,6)) 

for roi, id, idx in zip(rois, ids, idxs): 

    mask = np.loadtxt(f"rois_{file_name}/serial={serial}/slice={slic}/id={id}_mask="+ str(idx) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6))
    
    T_nogse, g, x, n, f, error =  nogse.generate_NOGSE_vs_x_roi(image_paths, method_paths, mask, slic)

    # Create directory if it doesn't exist
    directory = f"../results_{file_name}/{folder}/slice={slic}/{A0}/tnogse={T_nogse}_g={g}_N={int(n)}_id={id}"
    os.makedirs(directory, exist_ok=True)

    label = roi + " - ID: " + id
    nogse.plot_nogse_vs_x_data(ax, label, x, f, error, T_nogse, g, n, slic)
    nogse.plot_nogse_vs_x_data(ax1, label, x, f, error, T_nogse, g, n, slic)

    table = np.vstack((x, f))
    np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)
