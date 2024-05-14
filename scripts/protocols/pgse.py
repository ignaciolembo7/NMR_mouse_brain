#NMRSI - Ignacio Lembo Ferrari - 11/05/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lmfit import Minimizer, create_params, fit_report
import glob 
from brukerapi.dataset import Dataset as ds
import seaborn as sns
import re
import os
import cv2

sns.set_theme(context='paper')
sns.set_style("whitegrid")

def pgse_image_params(method_path):
    with open(method_path) as file:
        txt = file.read()

        start_idx = txt.find("NSegments")
        end_idx = txt.find("##", start_idx)
        Nsegments = float(txt[start_idx + len("Nsegments="):end_idx])

        start_idx = txt.find("NAverages")
        end_idx = txt.find("##", start_idx)
        NAverages = float(txt[start_idx + len("NAverages="):end_idx])

        start_idx = txt.find("NRepetitions")
        end_idx = txt.find("$$", start_idx)
        NRepetitions = float(txt[start_idx + len("NRepetitions="):end_idx])

        start_idx = txt.find("DummyScans")
        end_idx = txt.find("##", start_idx)
        DummyScans = float(txt[start_idx + len("DummyScans="):end_idx])

        start_idx = txt.find("DummyScansDur")
        end_idx = txt.find("$$", start_idx)
        DummyScansDur = float(txt[start_idx + len("DummyScansDur="):end_idx])
        
        start_idx = txt.find("EffSWh")
        end_idx = txt.find("##", start_idx)
        EffSWh = float(txt[start_idx + len("EffSWh="):end_idx])

        start_idx = txt.find("ScanTime=")
        end_idx = txt.find("##", start_idx)
        ScanTime = float(txt[start_idx + len("ScanTime="):end_idx])
        import datetime
        delta = datetime.timedelta(seconds=ScanTime/1000)
        minutos = delta.seconds // 60
        segundos = delta.seconds % 60
        ScanTime = str(minutos) + " min " + str(segundos) + " s"

        start_idx = txt.find("DwUsedSliceThick")
        end_idx = txt.find("$$", start_idx)
        DwUsedSliceThick = float(txt[start_idx + len("DwUsedSliceThick="):end_idx]) 

        PVM_Fov = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_Fov=" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_Fov.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        PVM_Fov = str(PVM_Fov[0]) + " mm" + " x " + str(PVM_Fov[1]) + " mm"

        PVM_SpatResol = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_SpatResol" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_SpatResol.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        PVM_SpatResol = str(PVM_SpatResol[0]*1000) + " um" + " x " + str(PVM_SpatResol[1]*1000) + " um"

        PVM_Matrix = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_Matrix" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_Matrix.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

    return {"Nsegments": Nsegments, "NAverages": NAverages, "NRepetitions": NRepetitions, "DummyScans": DummyScans, "DummyScansDur": DummyScansDur, "ScanTime": ScanTime, "EffSWh": EffSWh, "DwUsedSliceThick": DwUsedSliceThick, "Img size": PVM_Matrix,  "PVM_Fov": PVM_Fov, "PVM_SpatResol": PVM_SpatResol}

def pgse_params(method_path):
    with open(method_path) as file:
        txt = file.read()

        DwBvalEach = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwBvalEach" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwBvalEach.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwEffBval = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwEffBval" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwEffBval.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwEffBval = DwEffBval[1:]
        
        DwGradAmp = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradAmp" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradAmp.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwGradRead = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradRead" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradRead.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradRead = DwGradRead[1:]


        DwGradPhase = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradPhase" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradPhase.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradPhase = DwGradPhase[1:]

        DwGradSlice = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradSlice" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradSlice.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradSlice = DwGradSlice[1:]

        DwGradDur = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradDur" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradDur.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwGradSep = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradSep" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradSep.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break
        
        start_idx = txt.find("EchoTime")
        end_idx = txt.find("##", start_idx)
        EchoTime = float(txt[start_idx + len("EchoTime="):end_idx])

        return { "DwBvalEach": DwBvalEach[0], "DwEffBval": DwEffBval[0], "DwGradAmp": DwGradAmp[0], "DwGradRead": DwGradRead[0], "DwGradPhase": DwGradPhase[0], "DwGradSlice": DwGradSlice[0], "DwGradDur": DwGradDur[0], "DwGradSep": DwGradSep[0], "EchoTime": EchoTime}

def upload_pgse_vs_bval_data(data_directory, slic):

    def generar_rangos_discontinuos(rangos_str):
        carpetas = []
        for rango in rangos_str.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
        return carpetas

    folder_ranges = input('Ingrese un conjunto de rangos de carpetas, por ejemplo, 106-108,110-115, ... : ')
    carpetas = generar_rangos_discontinuos(folder_ranges)

    image_paths = []
    method_paths = []
    experiments = []
    A0s = []
    params = []

    error_carpeta = None  # Variable para almacenar el número de carpeta donde ocurre el error
    
    for carpeta in carpetas:
        try:
            image_path = glob.glob(f"{data_directory}/{carpeta}/pdata/1/2dseq")[0]
            method_path = glob.glob(f"{data_directory}/{carpeta}/method")[0]
            image_paths.append(image_path)
            method_paths.append(method_path)
            ims = ds(image_path).data
            A0s.append(ims[:,:,slic,0]) 
            experiments.append(ims[:,:,slic,1])
        except Exception as e:
            error_carpeta = carpeta
            print(f"Error al procesar la carpeta {carpeta}: {e}")
            break  # Salir del bucle cuando se encuentre el error

    # Si se produjo un error, imprime el número de carpeta
    if error_carpeta is not None:
        print(f"El error ocurrió en la carpeta {error_carpeta}.\n")
    else:
        print("No se encontraron errores en el procesamiento de las carpetas.\n")
        return image_paths, method_paths
    
def generate_pgse_vs_bval_roi(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,slic,0]) 
        experiments.append(ims[:,:,slic,1])
        param_dict = pgse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    DwBvalEach, DwEffBval, DwGradAmp, DwGradRead, DwGradPhase, DwGradSlice, DwGradDur, DwGradSep, TE= np.array(params).T
    print(f"PGSE sequence parameters for the {len(experiments)} experiments:\n")
    print("DwBvalEach:\n", DwBvalEach)
    print("DwEffBval:\n", DwEffBval)
    print("DwGradAmp:\n", DwGradAmp)
    print("DwGradRead:\n", DwGradRead)
    print("DwGradPhase:\n", DwGradPhase)
    print("DwGradSlice:\n", DwGradSlice)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    for i in range(len(E_matrix)):
        roi = np.zeros_like(E_matrix[i])
        roi[mask == 255] = E_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))

    return DwBvalEach, DwEffBval, DwGradAmp, DwGradRead, DwGradPhase, DwGradSlice, DwGradDur, DwGradSep, f

def plot_pgse_vs_bval_data(ax, nroi, DwEffBval, f, DwGradDur, DwGradSep, DwGradAmp, G, slic):
    ax.plot(DwEffBval, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("$b_{value} = \delta^2 \gamma^2 g^2 t_d$ [s/mm$^2$]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{PGSE}$ [u.a.]", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"$\Delta$ = {DwGradSep} ms  ||  $\delta$ = {DwGradDur} || slice = {slic} ", fontsize=18)
    #title = ax.set_title(f"G = {G} mT/m  || slice = {slic} ", fontsize=18)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def plot_logpgse_vs_bval_data(ax, nroi, DwEffBval, f, DwGradDur, DwGradSep, DwGradAmp, G, slic):
    ax.plot(DwEffBval, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("$b_{value} = \delta^2 \gamma^2 g^2 t_d$ [s/mm$^2$]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{PGSE}$ [u.a.]", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    ax.set_yscale('log')  #Escala logarítmica en el eje y
    #ax.set_xscale('log')  #Escala logarítmica en el eje x
    title = ax.set_title(f"$\Delta$ = {DwGradSep} ms  ||  $\delta$ = {DwGradDur} || slice = {slic} ", fontsize=18)
    #title = ax.set_title(f"G = {G} mT/m  || slice = {slic} ", fontsize=18)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def pgse_params_allinone(method_path):
    with open(method_path) as file:
        txt = file.read()

        DwBvalEach = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwBvalEach" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwBvalEach.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwEffBval = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwEffBval" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwEffBval.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwEffBval = DwEffBval[1:]
        
        DwGradAmp = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradAmp" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradAmp.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwGradRead = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradRead" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradRead.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradRead = DwGradRead[1:]


        DwGradPhase = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradPhase" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradPhase.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradPhase = DwGradPhase[1:]

        DwGradSlice = []
        with open(method_path, 'r') as archivo:
            # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False
            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "DwGradAmp"
                if "DwGradSlice" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes o ceros
                    if all(valor.replace(".", "", 1).isdigit() or valor == '0' or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradSlice.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes o ceros, detén la lectura
                        break
        DwGradSlice = DwGradSlice[1:]

        DwGradDur = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradDur" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradDur.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        DwGradSep = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "DwGradSep" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        DwGradSep.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break
        
        start_idx = txt.find("EchoTime")
        end_idx = txt.find("##", start_idx)
        EchoTime = float(txt[start_idx + len("EchoTime="):end_idx])

        return { "DwBvalEach": DwBvalEach , "DwEffBval": DwEffBval, "DwGradAmp": DwGradAmp, "DwGradRead": DwGradRead, "DwGradPhase": DwGradPhase, "DwGradSlice": DwGradSlice, "DwGradDur": DwGradDur, "DwGradSep": DwGradSep, "EchoTime": EchoTime}

def upload_pgse_vs_bval_data_allinone(data_directory, slic):

    def generar_rangos_discontinuos(rangos_str):
        carpetas = []
        for rango in rangos_str.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
        return carpetas

    folder_ranges = input('Ingrese un conjunto de rangos de carpetas, por ejemplo, 106-108,110-115, ... : ')
    carpetas = generar_rangos_discontinuos(folder_ranges)

    image_paths = []
    method_paths = []
    experiments = []
    A0s = []
    params = []

    error_carpeta = None  # Variable para almacenar el número de carpeta donde ocurre el error
    error_exp = None  # Variable para almacenar el número de experimento donde ocurre el error

    for carpeta in carpetas:
        try:
            image_path = glob.glob(f"{data_directory}/{carpeta}/pdata/1/2dseq")[0]
            method_path = glob.glob(f"{data_directory}/{carpeta}/method")[0]
            image_paths.append(image_path)
            method_paths.append(method_path)
            ims = ds(image_path).data
            num_exps = ims.shape[3]
            A0s.append(ims[:,:,slic,0]) 
            for num_exp in range(1, num_exps):
                experiments.append(ims[:,:,slic,num_exp])
        except Exception as e:
            error_carpeta = carpeta
            print(f"Error al procesar la carpeta {carpeta}: {e}")
            error_exp = num_exp
            print(f"Error al procesar la imagen {num_exp} de la carpeta {carpeta}: {e}")
            break  # Salir del bucle cuando se encuentre el error

    # Si se produjo un error, imprime el número de carpeta
    if error_carpeta is not None:
        print(f"El error ocurrió en la carpeta {error_carpeta}.")
    else:
        print("No se encontraron errores en el procesamiento de las carpetas.")
        return image_paths, method_paths
    
def generate_pgse_vs_bval_roi_allinone(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        num_exps = ims.shape[3]
        A0s.append(ims[:,:,slic,0]) 
        for num_exp in range(1, num_exps):
            experiments.append(ims[:,:,slic,num_exp])
 
        params = pgse_params_allinone(method_path)   
        print(f"\n\nPGSE sequence parameters for the {len(experiments)} experiments:")
        DwBvalEach = params["DwBvalEach"]
        DwEffBval = params["DwEffBval"]
        DwGradAmp = params["DwGradAmp"]
        DwGradRead = params["DwGradRead"]
        DwGradPhase = params["DwGradPhase"]
        DwGradSlice = params["DwGradSlice"]
        DwGradDur = params["DwGradDur"]
        DwGradSep = params["DwGradSep"]
        EchoTime = params["EchoTime"]

    print(f"\nDwBvalEach = {DwBvalEach}")
    print(f"\nDwEffBval = {DwEffBval}")
    print(f"\nDwGradAmp = {DwGradAmp}")
    print(f"\nDwGradRead = {DwGradRead}")
    print(f"\nDwGradPhase = {DwGradPhase}")
    print(f"\nDwGradSlice = {DwGradSlice}")
    print(f"\nDwGradDur = {DwGradDur}")
    print(f"\nDwGradSep = {DwGradSep}")
    print(f"\nEchoTime = {EchoTime}")
    
    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    for i in range(len(E_matrix)):
        roi = np.zeros_like(E_matrix[i])
        roi[mask == 255] = E_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))

    return DwBvalEach, DwEffBval, DwGradAmp, DwGradRead, DwGradPhase, DwGradSlice, DwGradDur, DwGradSep, f
