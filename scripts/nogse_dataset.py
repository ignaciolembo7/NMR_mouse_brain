#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
from nogse import nogse

#Numero de serie del experimento
ns = input('Numero de serie: ') 
file_name = "mousebrain_20200409" #resultados
im_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/"+str(ns)+"/pdata/1/2dseq" # dirección donde guardo la carpeta del experimento.
slic = 1 # slice que quiero ver 0 o 1 

images = ds(im_path).data

print("\nDimensión del array: {}".format(images.shape))


A0 = images[:,:,slic,0] # asi se accede a los elementos de un array de numpy, los ":" dicen que quiero quedarme con todas los numeros en esa dimensión, mientras que selecciono si quiero la A0 o el experimento poniendo 1 o 0 en la ultima dimensión.
experiment = images[:,:,slic,1]

method_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/"+str(ns)+"/method"

params = nogse.nogse_params(method_path)
params_img = nogse.nogse_image_params(method_path)

print(f"\nDiccionario con los parámetros de la secuencia NOGSE: \n params = {params}")

print(f"\nDiccionario con los parámetros de las imágenes: \n params_img = {params_img}")

fig, axs = plt.subplots(1, 2, figsize=(8,4)) # ploteo las imagenes

axs[0].imshow(A0, cmap="gray")
axs[0].axis("off")
axs[0].set_title("$A_0$", fontsize=18)

axs[1].imshow(experiment, cmap="gray")
axs[1].axis("off")
axs[1].set_title( str(ns) + " | " + str(params["t_nogse"]) + " ms | " + str(params["ramp_grad_str"]) + " mT/m | " + str(params["ramp_grad_N"]), fontsize=18) 

plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 8))  # create a figure with specified size
plt.imshow(experiment, cmap="gray")
plt.axis("off")
plt.title(f"Im {ns} | Tnogse = {params['t_nogse']} ms | G = {params['ramp_grad_str']} mT/m | N = {params['ramp_grad_N']} | slice = {slic}", fontsize=18)
plt.tight_layout()
plt.savefig(f"../images/image={ns}_t={params['t_nogse']}_G={params['ramp_grad_str']}_N={params['ramp_grad_N']}_slice={slic}.png")
plt.close(fig)

"""
# Para acceder a los elementos podemos acceder como un diccionario por nombre o pasarlo a un array y usarlo asi

T_nogse = params["t_nogse"]
print("T_nogse = {} ms".format(T_nogse))
param_list = list(params.values())
g = param_list[1]
print("g = {} mT/m".format(g))
x = param_list[3]
print("x = {} ms".format(x))
EchoTime = param_list[4]
print("EchoTime = {} ms".format(x))
"""