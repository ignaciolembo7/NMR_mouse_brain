#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
from nogse import nogse
import sys

#Numero de serie del experimento
serial = sys.argv[1] #input('Numero de serie: ') 
file_name = sys.argv[2] #"mousebrain_20200409" #resultados
slic = int(sys.argv[3]) #1 # slice que quiero ver 0 o 1

im_path = f"data/data_{file_name}/"+str(serial)+"/pdata/1/2dseq" # dirección donde guardo la carpeta del experimento.

images = ds(im_path).data

print("Dimensión del array: {}".format(images.shape))

A0 = images[:,:,slic,0] # asi se accede a los elementos de un array de numpy, los ":" dicen que quiero quedarme con todas los numeros en esa dimensión, mientras que selecciono si quiero la A0 o el experimento poniendo 1 o 0 en la ultima dimensión.
experiment = images[:,:,slic,1]

method_path = f"data/data_{file_name}/"+str(serial)+"/method"

params = nogse.nogse_params(method_path)
params_img = nogse.nogse_image_params(method_path)

print("Diccionario con los parámetros de la secuencia NOGSE: \n params = {}".format(params))

print("Diccionario con los parámetros de las imágenes: \n params_img = {}".format(params_img))

fig, axs = plt.subplots(1, 2, figsize=(8,4)) # ploteo las imagenes

axs[0].imshow(A0, cmap="gray")
axs[0].axis("off")
axs[0].set_title("$A_0$", fontsize=18)

axs[1].imshow(experiment, cmap="gray")
axs[1].axis("off")
axs[1].set_title( str(serial) + " | " + str(params["t_nogse"]) + " ms | " + str(params["ramp_grad_str"]) + " mT/m | " + str(params["ramp_grad_N"]), fontsize=18) 

plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 8))  # create a figure with specified size
plt.imshow(experiment, cmap="gray")
plt.axis("off")
plt.title(f"Im {serial} | Tnogse = {params['t_nogse']} ms | G = {params['ramp_grad_str']} mT/m | N = {params['ramp_grad_N']} | slice = {slic}", fontsize=18)
plt.tight_layout()
plt.savefig(f"../images/image={serial}_t={params['t_nogse']}_G={params['ramp_grad_str']}_N={params['ramp_grad_N']}_slice={slic}.png")
plt.close(fig)