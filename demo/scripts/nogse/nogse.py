#NMRSI - Ignacio Lembo Ferrari - 27/04/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lmfit import Minimizer, create_params, fit_report
import glob 
from brukerapi.dataset import Dataset as ds
import seaborn as sns
import cv2

sns.set(context='paper')
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

    return {"Nsegments": Nsegments, "NAverages": NAverages, "NRepetitions": NRepetitions, "DummyScans": DummyScans, "DummyScansDur": DummyScansDur, "ScanTime": ScanTime, "DwUsedSliceThick": DwUsedSliceThick, "Img size": PVM_Matrix, "EffSWh": EffSWh}

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

def upload_contrast_data(file_name,range_e_hahn,range_e_cpmg):

    def generar_rangos_discontinuos(rangos_str):
        carpetas = []
        for rango in rangos_str.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
        return carpetas

    carpetas_e_hahn = generar_rangos_discontinuos(rangos_e_hahn)
    carpetas_e_cpmg = generar_rangos_discontinuos(rangos_e_cpmg)

    image_paths = []
    method_paths = []
    experiments = []
    A0s = []

    error_carpeta = None  # Variable para almacenar el número de carpeta donde ocurre el error

    for carpeta in carpetas_e_hahn + carpetas_e_cpmg:
        try:
            image_path = glob.glob("data/data_" + file_name + "/{}/pdata/1/2dseq".format(carpeta))[0] 
            method_path = glob.glob("data/data_" + file_name + "/{}/method".format(carpeta))[0]
            image_paths.append(image_path)
            method_paths.append(method_path)
            ims = ds(image_path).data
            A0s.append(ims[:,:,1,0]) 
            experiments.append(ims[:,:,1,1])
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

def generate_contrast_roi(image_paths, method_paths, mask):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,1,0]) 
        experiments.append(ims[:,:,1,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print("g",g)
    print("T_nogse",T_nogse)

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

    return T_nogse[0], g_contrast, n[0], f

def upload_NOGSE_vs_x_data(file_name, folder_ranges):

    def generar_rangos_discontinuos(rangos_str):
        carpetas = []
        for rango in rangos_str.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
        return carpetas

    carpetas = generar_rangos_discontinuos(folder_ranges)

    image_paths = []
    method_paths = []
    experiments = []
    A0s = []
    params = []

    error_carpeta = None  # Variable para almacenar el número de carpeta donde ocurre el error

    for carpeta in carpetas:
        try:
            image_path = glob.glob("data/data_" + file_name + "/{}/pdata/1/2dseq".format(carpeta))[0] 
            method_path = glob.glob("data/data_" + file_name + "/{}/method".format(carpeta))[0]
            image_paths.append(image_path)
            method_paths.append(method_path)
            ims = ds(image_path).data
            A0s.append(ims[:,:,1,0]) 
            experiments.append(ims[:,:,1,1])
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

def generate_NOGSE_vs_x_roi(image_paths, method_paths, mask):
    
    experiments = []
    A0s = []
    params = []
    f = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:,:,1,0]) 
        experiments.append(ims[:,:,1,1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    #print("g",g)
    #print("T_nogse",T_nogse)
    #print("x",x)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix #/A0_matrix

    for i in range(len(E_matrix)):
        roi = np.zeros_like(E_matrix[i])
        roi[mask == 255] = E_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))

    return T_nogse[0], g[0], x, n[0], f

def plot_contrast_data(ax, nroi, g_contrast, f, tnogse, n):
    ax.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=18)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(tnogse, n), fontsize=18)
    #plt.tight_layout()    
    #ax.set_xlim(0.5, 10.75)

def plot_contrast_ptROI(ax, nroi, g_contrast, f, tnogse, n):
    ax.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=18)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(tnogse, n), fontsize=18)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data(ax, nroi, x, f, tnogse, n):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, label=nroi)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=18)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ [u.a.]", fontsize=18)
    ax.legend(title='ROI', title_fontsize=18, fontsize=18, loc='lower right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} ".format(tnogse, n), fontsize=18)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def data_set(ns, file_name, slic):
    # Ruta del archivo de imágenes
    im_path = f"data/data_{file_name}/"+str(ns)+"/pdata/1/2dseq"

    # Cargar las imágenes
    images = ds(im_path).data

    print("Dimensión del array: {}".format(images.shape))

    # Extraer A0 y el experimento
    A0 = images[:,:,slic,0]
    experiment = images[:,:,slic,1]

    # Ruta del archivo de método
    method_path = f"data/data_{file_name}/"+str(ns)+"/method"

    # Obtener parámetros de la secuencia NOGSE y de las imágenes
    params = nogse_params(method_path)
    params_img = nogse_image_params(method_path)

    print("Diccionario con los parámetros de la secuencia NOGSE: \n params = {}".format(params))
    print("Diccionario con los parámetros de las imágenes: \n params_img = {}".format(params_img))

    # Ploteo de las imágenes
    fig, axs = plt.subplots(1, 2, figsize=(8,4))

    axs[0].imshow(A0, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("$A_0$", fontsize=18)

    axs[1].imshow(experiment, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title( str(ns) + " | " + str(params["t_nogse"]) + " ms | " + str(params["ramp_grad_str"]) + " mT/m | " + str(params["ramp_grad_N"]), fontsize=18) 

    plt.show()
    plt.close(fig)

    # Guardar la imagen final
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(experiment, cmap="gray")
    plt.axis("off")
    plt.title(f"Im {ns} | Tnogse = {params['t_nogse']} ms | G = {params['ramp_grad_str']} mT/m | N = {params['ramp_grad_N']} | slice = {slic}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"../images/image={ns}_t={params['t_nogse']}_G={params['ramp_grad_str']}_N={params['ramp_grad_N']}_slice={slic}.png")
    plt.close(fig)

def draw(event, former_x, former_y, flags, param):

    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            current_former_x = former_x
            current_former_y = former_y
            # Fill the enclosed area in the mask with white
            cv2.floodFill(mask, None, (0, 0), (255))

    return former_x, former_y

def roi_select(serial, nrois, slic):

    ims = ds(f"data/data_mousebrain_20200409/"+str(serial)+"/pdata/1/2dseq").data
    A0_matrix = ims[:,:,slic,0]
    M_matrix = ims[:,:,slic,1]
    original = M_matrix #/A0_matrix 

    np.savetxt(f"rois/original.txt", original, fmt='%f')

    #Equalize original
    original_eq = 255 * (original - np.min(original)) / (np.max(original) - np.min(original)) + 255*(np.min(original) / (np.max(original) - np.min(original)) )

    cv2.imwrite(f"rois/original_eq.jpg", original_eq)

    im = cv2.imread(f"rois/original_eq.jpg", cv2.IMREAD_GRAYSCALE)

    for i in range(nrois):

        drawing = False  # True if mouse is pressed
        mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

        # Escalar la imagen para que se vea más grande
        scaling_factor = 1 # Factor de escala (puedes ajustarlo según sea necesario)
        im_scaled = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor)
        mask = np.zeros_like(im_scaled)  # Create a black image with the same size as im
        
        cv2.namedWindow("Roi_Select")
        cv2.setMouseCallback('Roi_Select', draw)

        while True:
            cv2.imshow('Roi_Select', im_scaled)
            k = cv2.waitKey(1) & 0xFF
            if k == 13:  # Press Enter to move to the next ROI
                break

        # Invert the mask

        mask_inverted = cv2.bitwise_not(mask)
        
        # Resize the inverted mask to match the size of the original image
        mask_resized = cv2.resize(mask_inverted, (original.shape[1], original.shape[0]))
        mask_resized[mask_resized != 0] = 255
            
        # Apply the mask to the original image
        roi = np.zeros_like(original)
        roi[mask_resized == 255] = original[mask_resized == 255]
        signal = np.mean(roi[roi != 0])
        print(f"Average intensity of ROI {i+1}: {signal}")

        # Save roi
        np.savetxt(f"rois/roi_{i+1}.txt", roi, fmt='%f')
        roi = (roi * 255).astype(np.uint8)
        cv2.imwrite(f"rois/roi_{i+1}.jpg", roi)

        # Save mask
        np.savetxt(f"rois/mask_{i+1}.txt", mask_resized, fmt='%f')
        cv2.imwrite(f"rois/mask_{i+1}.jpg", mask_resized)

    cv2.destroyAllWindows()

    # Iterar sobre todas las máscaras de las ROIs y superponerlas en la imagen final
    imagen_final = im.copy()

    for i in range(1, nrois+1): 
        mask_roi = cv2.imread(f"rois/mask_{i}.jpg", cv2.IMREAD_GRAYSCALE)
        imagen_final = cv2.add(imagen_final, mask_roi)

    # Guardar la imagen final
    cv2.imwrite(f"rois/im={serial}_rois_final.jpg", imagen_final)

    # Cargar la imagen en escala de grises
    imagen_final = cv2.imread(f"rois/im={serial}_rois_final.jpg", cv2.IMREAD_GRAYSCALE)

    # Convertir la imagen en escala de grises a color
    imagen_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)

    # Definir el color rojo en RGB
    color_rojo = (0, 0, 255.0)  # (B, G, R)

    # Encontrar los píxeles blancos en la imagen en escala de grises
    indices_blancos = np.where((imagen_final >= 250) & (imagen_final <= 255))

    # Reemplazar los píxeles blancos por rojo en la imagen a color
    imagen_color[indices_blancos] = color_rojo

    # Guardar la imagen final
    cv2.imwrite(f"../images/im={serial}_rois_final_color.jpg", imagen_color)