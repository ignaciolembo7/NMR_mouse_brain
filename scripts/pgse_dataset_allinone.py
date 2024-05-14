#NMRSI - Ignacio Lembo Ferrari - 14/05/2024

import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
from protocols import pgse

#Numero de serie del experimento
ns = input('Numero de serie: ') 
file_name = "mousebrain_20200409" #resultados
im_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/"+str(ns)+"/pdata/1/2dseq" # dirección donde guardo la carpeta del experimento.
slic = 1 # slice que quiero ver 0 o 1 
max_field = 1466.6153
exp = 1

images = ds(im_path).data

print("\nDimensión del array: {}".format(images.shape))


A0 = images[:,:,slic,0] # asi se accede a los elementos de un array de numpy, los ":" dicen que quiero quedarme con todas los numeros en esa dimensión, mientras que selecciono si quiero la A0 o el experimento poniendo 1 o 0 en la ultima dimensión.
experiment = images[:,:,slic,exp]

method_path = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}/"+str(ns)+"/method"


params = pgse.pgse_params_allinone(method_path)
params_img = pgse.pgse_image_params(method_path)

DwBvalEach = params["DwBvalEach"]
print(f"\nDwBvalEach = {DwBvalEach}")

DwEffBval = params["DwEffBval"]
print(f"\nDwEffBval = {DwEffBval}")

DwGradAmp = params["DwGradAmp"]
print(f"\nDwGradAmp = {DwGradAmp}")

DwGradRead = params["DwGradRead"]
print(f"\nDwGradRead = {DwGradRead}")

DwGradPhase = params["DwGradPhase"]
print(f"\nDwGradPhase = {DwGradPhase}")

DwGradSlice = params["DwGradSlice"]
print(f"\nDwGradSlice = {DwGradSlice}")

DwGradDur = params["DwGradDur"]
print(f"\nDwGradDur = {DwGradDur}")

DwGradSep = params["DwGradSep"]
print(f"\nDwGradSep = {DwGradSep}")

EchoTime = params["EchoTime"]
print(f"\nEchoTime = {EchoTime}")

print(f"\nDiccionario con los parámetros de las imágenes: \n params_img = {params_img}")

fig, axs = plt.subplots(1, 2, figsize=(8,4))
axs[0].imshow(A0, cmap="gray")
axs[0].axis("off")
axs[0].set_title("$A_0$", fontsize=18)
axs[1].imshow(experiment, cmap="gray")
axs[1].axis("off")
axs[1].set_title(f"Im {ns} | $\Delta$ = {round(params['DwGradSep'][0],2)} ms  ||  $\delta$ = {round(params['DwGradDur'][0],2)} ms || slice = {slic} ", fontsize=18) 
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 8)) 
plt.imshow(experiment, cmap="gray")
plt.axis("off")
plt.title(f"Im {ns} | $\Delta$ = {round(params['DwGradSep'][0],2)} ms  ||  $\delta$ = {round(params['DwGradDur'][0],2)} ms || slice = {slic} ", fontsize=18) 
plt.tight_layout()
plt.savefig(f"../images/image={ns}_DwGradSep={round(params['DwGradSep'][0],2)}_DwGradDur={round(params['DwGradDur'][0],2)}_slice={slic}.png")
plt.close(fig)