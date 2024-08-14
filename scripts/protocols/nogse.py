#NMRSI - Ignacio Lembo Ferrari - 29/05/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lmfit import Minimizer, create_params, fit_report
import glob 
from brukerapi.dataset import Dataset as ds
import seaborn as sns
import os
import cv2

sns.set_theme(context='paper')
sns.set_style("whitegrid")

def nogse_image_params(method_path):
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
        end_idx = txt.find("##", start_idx)
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

def nogse_params(method_path):
    with open(method_path) as file:
        txt = file.read()

        start_idx = txt.find("Tnogse")
        end_idx = txt.find("##", start_idx)
        t_nogse = float(txt[start_idx + len("Tnogse="):end_idx])

        start_idx = txt.find("RampGradStr")
        end_idx = txt.find("$$", start_idx)
        ramp_grad_str = float(txt[start_idx + len("RampGradStr="):end_idx])

        start_idx = txt.find("RampGradN")
        end_idx = txt.find("##", start_idx)
        ramp_grad_N = float(txt[start_idx + len("RampGradN="):end_idx])

        start_idx = txt.find("RampGradX")
        end_idx = txt.find("##", start_idx)
        ramp_grad_x = float(txt[start_idx + len("RampGradX="):end_idx])

        start_idx = txt.find("EchoTime")
        end_idx = txt.find("##", start_idx)
        EchoTime = float(txt[start_idx + len("EchoTime="):end_idx])

        return {"t_nogse": t_nogse, "ramp_grad_str": ramp_grad_str, "ramp_grad_N": ramp_grad_N, "ramp_grad_x": ramp_grad_x, "EchoTime": EchoTime}

def upload_contrast_data(data_directory, slic):

    def generar_rangos_discontinuos(rangos_str):
        carpetas = []
        for rango in rangos_str.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
        return carpetas

    rangos_e_hahn = input("Ingresa los rangos de carpetas para E_hahn (desde-hasta,desde-hasta,...): ")
    rangos_e_cpmg = input("Ingresa los rangos de carpetas para E_cpmg (desde-hasta,desde-hasta,...): ")
    carpetas_e_hahn = generar_rangos_discontinuos(rangos_e_hahn)
    carpetas_e_cpmg = generar_rangos_discontinuos(rangos_e_cpmg)

    image_paths = []
    method_paths = []
    experiments = []
    A0s = []

    error_carpeta = None  # Variable para almacenar el número de carpeta donde ocurre el error

    for carpeta in carpetas_e_hahn + carpetas_e_cpmg:
        try:
            image_path = glob.glob(f"{data_directory}/{carpeta}/pdata/1/2dseq")[0]
            method_path = glob.glob(f"{data_directory}/{carpeta}/method")[0]
            image_paths.append(image_path)
            method_paths.append(method_path)
            ims = ds(image_path).data
            A0s.append(ims[:,:slic,0]) 
            experiments.append(ims[:,:,slic,1])
        except Exception as e:
            error_carpeta = carpeta
            print(f"Error al procesar la carpeta {carpeta}: {e}")
            break  # Salir del bucle cuando se encuentre el error

    # Si se produjo un error, imprime el número de carpeta
    if error_carpeta is not None:
        print(f"El error ocurrió en la carpeta {error_carpeta}.")
    else:
        print("No se encontraron errores en el procesamiento de las carpetas.")
        return image_paths, method_paths

def generate_contrast_roi(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,slic,0]) 
        experiments.append(ims[:,:,slic,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n",T_nogse)
    print("g:\n",g)
    print("x:\n",x)
    print("N:\n",n)
    print("TE:\n",TE)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    N = len(E_matrix) 
    middle_idx = int(N/2) 
    E_cpmg = E_matrix[middle_idx:] 
    E_hahn = E_matrix[:middle_idx] 
    g_contrast = g[:middle_idx] 
    g_contrast_check = g[middle_idx:] 
    #print("g_contrast_check",g_contrast_check)
    #print("g_contrast",g_contrast)
    contrast_matrix = E_cpmg-E_hahn

    for i in range(len(contrast_matrix)):
        roi = np.zeros_like(contrast_matrix[i])
        roi[mask == 255] = contrast_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))

    return T_nogse[0], g_contrast, int(n[0]), f

def colormap_contrast_roi(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    params = []

    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,slic,0]) 
        experiments.append(ims[:,:,slic,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n",T_nogse)
    print("g:\n",g)
    print("x:\n",x)
    print("N:\n",n)
    print("TE:\n",TE)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    N = len(E_matrix) 
    middle_idx = int(N/2) 
    E_cpmg = E_matrix[middle_idx:] 
    E_hahn = E_matrix[:middle_idx] 
    g_contrast = g[:middle_idx] 
    g_contrast_check = g[middle_idx:] 
    #print("g_contrast_check",g_contrast_check)
    #print("g_contrast",g_contrast)
    contrast_matrix = E_cpmg-E_hahn

    ims = []
    for i in range(len(contrast_matrix)):
        im = np.zeros_like(contrast_matrix[i])
        im[mask == 255] = contrast_matrix[i][mask == 255]
        # Set values equal to 0 to black
        im[im <= 0] = np.nan
        #im[im > 70] = np.nan
        ims.append(im)

    return T_nogse[0], g_contrast, int(n[0]), ims

def colormap_nogse_vs_x_roi(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,slic,0]) 
        experiments.append(ims[:,:,slic,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n",T_nogse)
    print("g:\n",g)
    print("x:\n",x)
    print("N:\n",n)
    print("TE:\n",TE)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    ims = []
    for i in range(len(E_matrix)):
        im = np.zeros_like(E_matrix[i])
        im[mask == 255] = E_matrix[i][mask == 255]
        # Set values equal to 0 to black
        im[im <= 0] = np.nan
        #im[im > 200] = np.nan
        ims.append(im)

    return T_nogse[0], g[0], x, int(n[0]), ims

def upload_NOGSE_vs_x_data(data_directory, slic):

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
        print(f"El error ocurrió en la carpeta {error_carpeta}.")
    else:
        print("No se encontraron errores en el procesamiento de las carpetas.")
        return image_paths, method_paths

def generate_NOGSE_vs_x_roi(image_paths, method_paths, mask, slic):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,slic,0]) 
        experiments.append(ims[:,:,slic,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n",T_nogse)
    print("g:\n",g)
    print("x:\n",x)
    print("N:\n",n)
    print("TE:\n",TE)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    for i in range(len(E_matrix)):
        roi = np.zeros_like(E_matrix[i])
        roi[mask == 255] = E_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))

    return T_nogse[0], g[0], x, int(n[0]), f

def plot_contrast_data(ax, nroi, g_contrast, f, tnogse, n, slic):
    ax.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=26)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=26)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)
    #plt.tight_layout()    
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data(ax, nroi, x, f, tnogse, n, slic):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ [u.a.]", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='lower right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptG(ax, nroi, x, f, tnogse, g, n, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=g)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ normalizada", fontsize=27)
    ax.legend(title='$G$ [mT/m]', title_fontsize=12, fontsize=12, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("{} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || Slice = {}".format(nroi, tnogse, n, slic), fontsize=18)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptN(ax, nroi, x, f, tnogse, g, n, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=n)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ normalizada", fontsize=27)
    ax.legend(title='$N$', title_fontsize=12, fontsize=12, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("{} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $G$ = {} || Slice = {}".format(nroi, tnogse, g, slic), fontsize=18)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptTNOGSE(ax, nroi, x, f, tnogse, n, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=tnogse)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ normalizada", fontsize=27)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("{} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || Slice = {}".format(nroi, tnogse, n, slic), fontsize=18)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_restdist(ax, nroi, modelo, x, x_fit, f, fit, tnogse, n, slic, color):
    ax.plot(x, f, "o", markersize=7, linewidth=2, color = color)
    ax.plot(x_fit, fit, linewidth=2, label = nroi, color = color)
    ax.legend(title_fontsize=15, fontsize=18, loc='best')
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=26)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=26)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)

def lognormal(l_c, sigma, l_c_mode):
    #l_c_mid = l_c_median*np.exp((sigma**2)/2)
    l_c_mid = l_c_mode*np.exp(sigma**2)
    return (1/(l_c*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(l_c)- np.log(l_c_mid))**2 / (2*sigma**2))

def M_nogse_free(TE, G, N, x, M0, D0):

    g = 267.52218744 # ms**-1 mT**-1

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    return M0*np.exp(-1.0/12 * g**2 * G**2 * D0 * ((N-1) * x**3 + y**3))

def M_nogse_rest(TE, G, N, x, t_c, M0, D0): #D0 =2.3*10**-12

    g = 267.52218744 # ms**-1 mT**-1

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    bSE=g*G*np.sqrt(D0*t_c)

    return M0 * np.exp(-bSE ** 2 * t_c ** 2 * (4 * np.exp(-y / t_c / 2) - np.exp(-y / t_c) - 3 + y / t_c)) * np.exp(-bSE ** 2 * t_c ** 2 * ((N - 1) * x / t_c + (-1) ** (N - 1) * np.exp(-(N - 1) * x / t_c) + 1 - 2 * N - 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2 + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) * ((N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + N - 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2)) * np.exp(2 * t_c ** 2 * ((np.exp((-y + 2 * x) / t_c / 2) + np.exp((x - 2 * y) / t_c / 2) - np.exp((x - y) / t_c) / 2 - np.exp(-y / t_c) / 2 + np.exp(x / t_c / 2) + np.exp(-y / t_c / 2) - np.exp(x / t_c) / 2 - 0.1e1 / 0.2e1) * (-1) ** (2 * N) + 2 * (-1) ** (1 + N) * np.exp(-(2 * N * x - 3 * x + y) / t_c / 2) + (np.exp(((3 - 2 * N) * x - 2 * y) / t_c / 2) - np.exp((-N * x + 2 * x - y) / t_c) / 2 + np.exp(-(2 * N * x - 4 * x + y) / t_c / 2) + np.exp(-(2 * N * x - 2 * x + y) / t_c / 2) - np.exp((-N * x + x - y) / t_c) / 2 + np.exp(-x * (-3 + 2 * N) / t_c / 2) - np.exp(-x * (N - 2) / t_c) / 2 - np.exp(-(N - 1) * x / t_c) / 2) * (-1) ** N + 2 * (-1) ** (1 + 2 * N) * np.exp((x - y) / t_c / 2)) * bSE ** 2 / (np.exp(x / t_c) + 1))

def M_nogse_rest_dist(TE, G, N, x, l_c_mode, sigma, M0, D0):
    #sigma = 0.06416131084794455
    #l_cmid = 7.3*10**-6

    if sigma<0:
        return 1e20

    n = 100
    lmax = 40 #um esto es hasta un tau_c de 135ms

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal(l_cs, sigma, l_c_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))

    for l_c, w in zip(l_cs, weights):
        E = E + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0*1e12), M0, D0)*w

    return E

#def contrast_vs_g_free(TE, G, N, alpha, M0, D0):
#    return M_nogse_free(TE, G, N, TE/N, M0, alpha*D0) - M_nogse_free(TE, G, N, 0, M0, alpha*D0)

def contrast_vs_g_free(TE, G, N, M0, D0):
    return M_nogse_free(TE, G, N, TE/N, M0, D0) - M_nogse_free(TE, G, N, 0, M0, D0)

def contrast_vs_g_rest(TE, G, N, t_c, M0, D0):
    return M_nogse_rest(TE, G, N, TE/N, t_c, M0, D0) - M_nogse_rest(TE, G, N, 0, t_c, M0, D0)

def contrast_vs_g_restdist(TE, G, N, l_c_mode, sigma, M0, D0):
    n = 100
    lmax = 10

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal(l_cs, sigma, l_c_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(G))

    for l_c, w in zip(l_cs, weights):
        E = E + contrast_vs_g_rest(TE, G, N, (l_c**2)/(D0*1e12) , M0, D0)*w

    return M0*E

def plot_contrast_vs_g(ax, nroi, modelo, g, g_fit, f, fit, tnogse, n, slic, color):
    ax.plot(g, f, "o", markersize=7, linewidth=2, color = color)
    ax.plot(g_fit, fit, linewidth=2, label = nroi, color = color)
    ax.legend(title_fontsize=15, fontsize=18, loc='best')
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=26)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=26)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)

def plot_lognorm_dist(ax, nroi, tnogse, n, l_c, l_c_mode, sigma, slic, color):
    dist = lognormal(l_c, sigma, l_c_mode)
    l_c_median = l_c_mode*np.exp((sigma**2))
    l_c_mid = l_c_mode*np.exp((sigma**2)/2)
    plt.axvline(x=l_c_mode, color='r', linestyle='--', label = "Moda") 
    plt.axvline(x=l_c_median, color='g', linestyle='--', label = "Mediana") 
    plt.axvline(x=l_c_mid, color='b', linestyle='--', label = "Media") 
    ax.plot(l_c, dist, "-", color=color, linewidth = 2, label = nroi)
    ax.set_xlabel("Longitud de correlación $l_c$ [$\mu$m]", fontsize=26)
    ax.set_ylabel("P($l_c$)", fontsize=26)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax.legend( fontsize=18, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || slice = {} ".format(tnogse, n, slic), fontsize=18)
    ax.fill_between(l_c, dist, color=color, alpha=0.3)
    #ax.set_xlim(0.5, 10.75)

def plot_results_brute(result, best_vals=True, varlabels=None, output=True):
    
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars, figsize=(11,7))

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parámetro vs chi2 en el caso de un solo parámetro
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=7)
                axes.set_ylabel(r'$\chi**{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parámetro vs chi2 arriba
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi**{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parámetro vs chi2 a la izquierda
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r'$\chi**{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots de todas las combinaciones de dos parámetros
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple(a for a in range(npars) if a not in (i, j))
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 7)
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 3)
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output, bbox_inches="tight", dpi=500) # 

def contrast_vs_g_intrarestdist_extrarestdist(TE, G, N, l_c_mode_int, l_c_mode_ext, sigma_int, sigma_ext, M0_int, M0_ext, D0_int, D0_ext):
    return contrast_vs_g_restdist(TE, G, N, l_c_mode_int, sigma_int, M0_int, D0_int) + contrast_vs_g_restdist(TE, G, N, l_c_mode_ext, sigma_ext, M0_ext, D0_ext)

def contrast_vs_g_intrarest_extrarestdist(TE, G, N, l_c_int, l_c_mode_ext, sigma_ext, M0_int, M0_ext, D0_int, D0_ext):
    return contrast_vs_g_rest(TE, G, N, ((l_c_int*1e-6)**2)/(2*D0_int), M0_int, D0_int) + contrast_vs_g_restdist(TE, G, N, l_c_mode_ext, sigma_ext, M0_ext, D0_ext)

##########################################################################################


def plot_contrast_rest_mixto_levs(ax, nroi, modelo, g_contrast, roi, T_nogse, n, t_c_int_fit, t_c_ext_fit, alpha_fit, M0_int, M0_ext, D0_int, D0_ext):
    if(nroi == "ROI1"):
        color = "green"
    if(nroi == "ROI2"):
        color = "darkorange"
    if(nroi == "ROIw"):
        color = "blue"
    g_contrast_fit = np.linspace(0, np.max(g_contrast), num=1000)
    fit = delta_M_intra_extra(T_nogse, g_contrast_fit, n, t_c_int_fit, t_c_ext_fit, alpha_fit, M0_int, M0_ext, D0_int, D0_ext)
    ax.plot(g_contrast, roi, 'o', color = color, linewidth=2)
    ax.plot(g_contrast_fit, fit, '-', color = color, linewidth=2, label="$\\tau_{c,int}$ = " + str(round(t_c_int_fit,4)) + " - $\\tau_{c,ext}$ = " + str(round(t_c_ext_fit,4)) + " - $\\alpha = $" + str(round(alpha_fit,4)))
    ax.legend(title_fontsize=15, fontsize=15, loc='upper right')
    ax.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=16)
    ax.set_ylabel("Contraste $\Delta M_\mathrm{NOGSE}$ [u.a.]", fontsize=16)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("{} || Modelo: {} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(nroi, modelo, T_nogse, int(n)), fontsize=18)

def plot_nogse_vs_x_free(ax, nroi, modelo, x, x_fit, f, fit, tnogse, n, g, alpha):
    ax.plot(x, f, "o", markersize=7, linewidth=2)
    ax.plot(x_fit, fit, label= nroi + "- $\\alpha = $" + str(round(alpha,4)), linewidth=2)
    ax.legend(title_fontsize=15, fontsize=18, loc='best')
    ax.set_xlabel("Tiempo de modulación x [ms]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=18)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("{} || Modelo: {} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $g$ = {} ||  $N$ = {} ".format(nroi, modelo, tnogse, g, n), fontsize=18)



def plot_nogse_vs_x_mixto(ax, nroi, modelo, x, x_fit, f, fit, tnogse, n, g, t_c, alpha):
    ax.plot(x, f, "o", markersize=7, linewidth=2)
    ax.plot(x_fit, fit, label= nroi + "- $\\tau_c = $" + str(round(t_c,4)) + " ms" + " - $\\alpha = $" + str(round(alpha,4)), linewidth=2)
    ax.legend(title_fontsize=15, fontsize=18, loc='best')
    ax.set_xlabel("Tiempo de modulación x [ms]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=18)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("{} || Modelo: {} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $g$ = {} ||  $N$ = {} ".format(nroi, modelo, tnogse, g, n), fontsize=18)


def plot_contrast_ptTNOGSE(ax, nroi, g_contrast, f, tnogse):
    ax.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label= tnogse)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=27)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    #title = ax.set_title("{} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(nroi, tnogse, n), fontsize=18)
    ax.set_xlim(-10, 1200)

def plot_nogse_vs_x_fit_ptTNOGSE(ax, nroi, x, f, tnogse, n, color):
    ax.plot(x, f, linewidth = 2, color = color, label = tnogse)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    #title = ax.set_title("{} || $T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(nroi, tnogse, n), fontsize=18)
    #ax.set_xlim(0.5, 10.75)

#def plot_NOGSE_vs_x_mixto(ax, nroi, modelo, g_contrast, roi, T_nogse, n, t_c_int_fit, t_c_ext_fit, alpha_fit, M0_int, M0_ext, D0_int, D0_ext):



def M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0):
    return M0 * M_nogse_free(TE, G, N, x, 1, alpha*D0) * M_nogse_rest(TE, G, N, x, t_c, 1, (1-alpha)*D0)

def delta_M_free(TE, G, N, alpha, M0, D0):
    return M_nogse_free(TE, G, N, TE/N, M0, alpha*D0) - M_nogse_free(TE, G, N, 0, M0, alpha*D0)



def delta_M_mixto(TE, G, N, t_c, alpha, M0, D0):
    return M_nogse_mixto(TE, G, N, TE/N, t_c, alpha, M0, D0) - M_nogse_mixto(TE, G, N, 0, t_c, alpha, M0, D0) 

def delta_M_intra_extra(TE, G, N, t_c_int, t_c_ext, alpha, M0_int, D0_int, D0_ext):
    return contrast_vs_g_rest(TE, G, N, t_c_int, M0_int, D0_int) + delta_M_mixto(TE, G, N, t_c_ext, alpha, 1 - M0_int, D0_ext)


def M_nogse_mixto_dist(TE, G, N, x, t_c_mid, alpha, sigma, M0, D0):
    #sigma = 0.06416131084794455
    #l_cmid = 7.3*10**-6

    if sigma<0:
        return 1e20

    n = 100
    sum = 0
    lmax = 60

    t_cs = np.linspace(0.5, lmax, n)
    weights = lognormal(t_cs, sigma, t_c_mid)
    weights = weights/np.sum(weights)

    out = np.zeros(len(x))

    for t_c, w in zip(t_cs, weights):
        out = out + M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0)*w
    return out

def delta_M_mixto_dist(t_c_mid, sigma, TE, G, N, alpha, M0, D0):
    n = 100
    lmax = 120

    t_cs = np.linspace(0.5, lmax, n)
    weights = lognormal(t_cs, sigma, t_c_mid)
    weights = weights/np.sum(weights)

    out = np.zeros(len(G))

    for t_c, w in zip(t_cs, weights):
        out = out + delta_M_mixto(TE, G, N, t_c, alpha, M0, D0)*w

    return M0*out



def delta_M_intra_extra_dist(TE, G, N, t_c_mid_int, t_c_mid_ext, sigma_int, sigma_ext, alpha, M0_int, M0_ext, D0_int, D0_ext):
    return contrast_vs_g_restdist(t_c_mid_int, sigma_int, TE, G, N, M0_int, D0_int) + delta_M_mixto_dist(t_c_mid_ext, sigma_ext, TE, G, N, alpha, M0_ext, D0_ext)

def delta_M_mixto_bimodal(t_c_mid_1, t_c_mid_2, sigma_1, sigma_2, p, TE, G, N, alpha, M0):
    n = 100
    lmax = 120

    t_cs = np.linspace(0.5, lmax, n)
    weights = p*lognormal(t_cs, sigma_1, t_c_mid_1) + (1-p)*lognormal(t_cs, sigma_2, t_c_mid_2)
    weights = weights/np.sum(weights)

    out = np.zeros(len(G))

    for t_c, w in zip(t_cs, weights):
        out = out + delta_M_mixto(TE, G, N, t_c, alpha, 1, D0=2.3*10**-12)*w

    return M0*out

def delta_M_ad(Lc, Ld, n, alpha):
    gamma = 267.52218744
    D0 = 2.3e-12
    return -np.exp(D0**3*((-0.08333333333333333*alpha*Lc**6)/D0**3 - ((1 - alpha)*Ld**4*(Lc**2/D0 + ((-3 - np.e**(-Lc**2/Ld**2) + 4/np.e**(Lc**2/(2.*Ld**2)))*Ld**2)/D0))/D0**2)) + np.exp(D0**3*((2*(-1)**n*(1 - alpha)*(-3.*(-1)**n - 1/(2.*np.e**(Lc**2/Ld**2)) - (0.5*(-1)**n)/np.e**(Lc**2/(Ld**2*n)) + (2.*(-1)**n)/np.e**(Lc**2/(2.*Ld**2*n)) + 2.*(-1)**n*np.e**(Lc**2/(2.*Ld**2*n)) - 0.5*(-1)**n*np.e**(Lc**2/(Ld**2*n)) + 2*np.e**((Lc**2*(3 - 2*n))/(2.*Ld**2*n)) - 1/(2.*np.e**((Lc**2*(-2 + n))/(Ld**2*n))) + 2*np.e**((D0*(Lc**2/D0 - (2*Lc**2*n)/D0))/(2.*Ld**2*n)) - 3*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))*Ld**6)/(D0**3*(1 + np.e**(Lc**2/(Ld**2*n)))) - (0.08333333333333333*alpha*Lc**6)/(D0**3*n**2) - ((-1 + alpha)*Ld**4*(-((np.e**(Lc**2/(Ld**2*n))*Lc**2)/D0) + ((1 - 4*np.e**(Lc**2/(2.*Ld**2*n)) + 3*np.e**(Lc**2/(Ld**2*n)))*Ld**2*n)/D0))/(D0**2*np.e**(Lc**2/(Ld**2*n))*n) - ((1 - alpha)*Ld**6*(1 + (-1)**(1 + n)*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)) - (4*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2 + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2.*(-1 + n))))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2 - 2*n))*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (Lc**2*(-1 + n))/(Ld**2*n) - 2*n + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-2 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-1 + n) + n))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2))/D0**3))