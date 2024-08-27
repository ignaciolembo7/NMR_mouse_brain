# NMRSI - Ignacio Lembo Ferrari - 22/08/2024

import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
from protocols import nogse
import os

# Definir el rango de números de serie
slic = 0  # Slice que quiero ver

exp = input("Experimento:")  # Nombre del experimento
ns_start = int(input("Número de serie inicial: "))
ns_end = int(input("Número de serie final: "))
file_name = "mousebrain_20200409"  # Nombre del archivo de resultados

for ns in range(ns_start, ns_end + 1):
    im_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/{ns}/pdata/1/2dseq"  # Dirección donde se guarda la carpeta del experimento

    try:
        images = ds(im_path).data

        print(f"\nDimensión del array para ns={ns}: {images.shape}")

        #A0 = images[:, :, slic, 0]  # Acceder a los elementos de un array de numpy
        experiment = images[:, :, slic, 1]
        method_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/{ns}/method"

        params = nogse.nogse_params(method_path)
        params_img = nogse.nogse_image_params(method_path)

        print(f"\nDiccionario con los parámetros de la secuencia NOGSE para ns={ns}: \n params = {params}")
        print(f"\nDiccionario con los parámetros de las imágenes para ns={ns}: \n params_img = {params_img}")

        # Asegurarse de que el directorio de resultados exista
        result_dir = f"../results_{file_name}/images/slice={slic}/tnogse={params['t_nogse']}_g={params['ramp_grad_str']}_exp={exp}"
        os.makedirs(result_dir, exist_ok=True)

        # Plotear las imágenes
        #fig, axs = plt.subplots(1, 1, figsize=(8, 4))

        #axs[0].imshow(A0, cmap="gray")
        #axs[0].axis("off")
        #axs[0].set_title("$A_0$", fontsize=18)

        #axs[1].imshow(experiment, cmap="gray")
        #axs[1].axis("off")
        #axs[1].set_title(f"{ns} | {params['t_nogse']} ms | {params['ramp_grad_str']} mT/m | {int(params['ramp_grad_N'])} | slice = {slic}", fontsize=18)
        #plt.savefig(f"{result_dir}/image={ns}_t={params['t_nogse']}_G={params['ramp_grad_str']}_N={int(params['ramp_grad_N'])}_slice={slic}.png")
        #plt.show()
        #plt.close(fig)

        # Guardar la imagen
        fig = plt.figure(figsize=(8, 8))  # Crear una figura con tamaño especificado
        plt.imshow(experiment, cmap="gray")
        plt.axis("off")
        plt.title(f"Im {ns} | Tnogse = {params['t_nogse']} ms | G = {params['ramp_grad_str']} mT/m | N = {int(params['ramp_grad_N'])} |  x = {params['ramp_grad_x']} | slice = {slic}", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{result_dir}/image={ns}_t={params['t_nogse']}_G={params['ramp_grad_str']}_N={int(params['ramp_grad_N'])}_slice={slic}.png")
        plt.close(fig)
    except Exception as e:
        print(f"Error procesando ns={ns}: {e}")