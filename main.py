# ----------------------------------------------------------------------------------------
# PROGRAMA: <<Diezmado, Interpolación y Descomposición de segundo orden>>
# ----------------------------------------------------------------------------------------
# Descripción: <<Este es un programa que genera imagenes diezmadas por un valor D, Interpoladas por un valor I, y descompuestas en 4 kernels>>
# ----------------------------------------------------------------------------------------
# Autores:
''' 
# Miguel David Benavides Galindo            md_benavidesg@javeriana.edu.co
# Christian Fernando Rodriguez Rodriguez    rodriguezchristianf@javeriana.edu.co
'''
# Version: 1.0
# [16.08.2021]
# ----------------------------------------------------------------------------------------
# IMPORTAR MODULES
import cv2
import numpy as np
import os
import sys
import math
# ----------------------------------------------------------------------------------------
# FUNCIONES
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Nombre: <<Diezmado>>
# ----------------------------------------------------------------------------------------
def Diezmado(image, D, n=1):
    if(n==1):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
    
    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)
    
    # pre-computations
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size = num_rows / 2  # here we assume num_rows = num_columns
    
    # low pass filter mask
    low_pass_mask = np.zeros_like(image_gray)
    freq_cut_off = 1/D  # it should less than 1
    radius_cut_off = int(freq_cut_off * half_size)
    idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
    low_pass_mask[idx_lp] = 1
    
    # filtering via FFT
    mask = low_pass_mask   # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)
    
    # Diezmar
    image_decimated = image_filtered[::D, ::D]
    return image_decimated
# ----------------------------------------------------------------------------------------
# Descripcion: <<Función que retorna la imagen reducida por cada D pixeles en el proceso de diezmamiento >>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
# La función necesita 3 argumentos:
    # image:    la imagen de entrada
    # D:        el valor de diezmamiento
    # n:        un valor n!=1 cuando la imagen ya esté en grises (por defecto n=1)
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna la imagen diezmada con rangos de D pixeles
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Nombre: <<Interpolacion>>
# ----------------------------------------------------------------------------------------
def Interpolacion(image, I,n=1):
    if(n==1):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image    
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
    
    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)
    
    # pre-computations
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size = num_rows / 2  # here we assume num_rows = num_columns
    
    # low pass filter mask
    low_pass_mask = np.zeros_like(image_gray)
    freq_cut_off = 1/I  # it should less than 1
    radius_cut_off = int(freq_cut_off * half_size)
    idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
    low_pass_mask[idx_lp] = 1
    
    # filtering via FFT
    mask = low_pass_mask   # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    # Interpolation
    # insert zeros
    rows, cols = image_filtered.shape
    num_of_zeros = I-1
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=image_filtered.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = image_filtered
    W = 2 * num_of_zeros + 1
    # filtering
    image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 0)
    image_interpolated *= num_of_zeros ** 2

    return image_interpolated
# ----------------------------------------------------------------------------------------
# Descripcion: <<Función que retorna una imagen aumentada a partir de la creación de pixeles Gaussianos>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
# La función necesita 3 argumentos:
    # image:    imagen de entrada
    # I:        Número de pixeles creados y estimados Gaussianos
    # n:        un valor n!=1 cuando la imagen ya esté en grises (por defecto n=1)
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna la imagen interpolada en un factor de I-1
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Nombre: <<descomposition>>
# ----------------------------------------------------------------------------------------
def descomposition(image,n=1):
    if(n==1):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    kernel_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_3 = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
    kernel_4 = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    image_convolved_1 = cv2.filter2D(image_gray, -1, kernel_1)
    image_convolved_2 = cv2.filter2D(image_gray, -1, kernel_2)
    image_convolved_3 = cv2.filter2D(image_gray, -1, kernel_3)
    image_convolved_4 = cv2.filter2D(image_gray, -1, kernel_4)
    image_IH = Diezmado(image_convolved_1, D=2, n=0)
    image_IV = Diezmado(image_convolved_2, D=2, n=0)
    image_ID = Diezmado(image_convolved_3, D=2, n=0)
    image_IL = Diezmado(image_convolved_4, D=2, n=0)
    return(image_IH,image_IV,image_ID,image_IL)
# ----------------------------------------------------------------------------------------
# Descripcion: <<Función que retorna la imagen convolucionada por 4 tipos de kernels predefinidos y diezmada en un factor D=2>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
    # image:    imagen de entrada
    # n:        un valor n!=1 cuando la imagen ya esté en grises (por defecto n=1)
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna una lista con 4 imagenes, correspondientes a las convoluciones IH, IV, ID, IL
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Nombre: <<descomposition_order2>>
def descomposition_order2(image, kernel):
    order1 = descomposition(image)
    order1_kernel = order1[kernel]
    order2 = descomposition(order1_kernel,n=2)
    return(order2)
# ----------------------------------------------------------------------------------------
# Descripcion: <<Función que retorna la imagen convolucionada por los 4 filtros partiendo de una convolución previa IL>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
    # image:    imagen de entrada
    # kernel:   Tipo de kernel empleado: (predefinido 3: kernel IL)
        # 0: kernel IH
        # 1: kernel IV
        # 2: kernel ID
        # 3: kernel IL
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna una lista con 4 imagenes, correspondientes a las convoluciones ILH, ILV, ILD, ILL
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Ejecución
ruta = "C:/Users/User/OneDrive - World Food Programme/Proyectos/Maestria/2021_I/Imagenes y Video/Imagenes/lena.png"
image = cv2.imread(ruta)
# ----------------------------------------------------------------------------------------
# Diezmado
D = 3
image_diezmada= Diezmado(image, D)
cv2.imshow("Original image", image)
cv2.imshow("Decimation image", image_diezmada)
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# Interpolation
I = 3
image_interpolada = Interpolacion(image, I)
cv2.imshow("Original image", image)
cv2.imshow("Decimation image", image_interpolada)
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# Descomposition
image_descompuesta_orden1 = descomposition(image)
cv2.imshow("Original image", image)
cv2.imshow("descomposition IH", image_descompuesta_orden1[0])
cv2.imshow("descomposition IV", image_descompuesta_orden1[1])
cv2.imshow("descomposition ID", image_descompuesta_orden1[2])
cv2.imshow("descomposition IL", image_descompuesta_orden1[3])
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# Descomposition order2
image_descompuesta_orden2 = descomposition_order2(image, kernel=3)
image_descompuesta_order2_interpolated = Interpolacion(image_descompuesta_orden2[3], I=4,n=2)
cv2.imshow("original Image", image)
cv2.imshow("descomposition ILL", image_descompuesta_orden2[3])
cv2.imshow("descomposition ILL interpolated", image_descompuesta_order2_interpolated)
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# end.
# ----------------------------------------------------------------------------------------
