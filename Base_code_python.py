#!/usr/bin/env python
# coding: utf-8

# In[332]:


#Importaciones Necesarias
#setx CURL_SSL_NO_REVOKE 1
#setx CURL_SSL_NO_REVOKE ""

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import Angle
import astropy.units as u
import pandas as pd
import os
import shutil
import subprocess
from IPython.display import clear_output
import csv
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.mast import Tesscut
from scipy.signal import periodogram
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from astropy.timeseries import LombScargle
import math
import time
from rich.console import Console

#Creación de directorios
os.makedirs("./objt_out", exist_ok=True)
os.makedirs("./csv_out", exist_ok=True)
os.makedirs("./imagenes", exist_ok=True)

#Se define la función que obtiene la info de Simbad
Gaia.ROW_LIMIT = -1 
Ra,Dec=0,0
tiempo=[]
magnitud=[]
sector_seleccionado=0
gaia_id=0
object_name="a"
def get_coordinates_from_name(name):
    simbad = Simbad()
    result = simbad.query_object(name)
    if result is not None:
        ra = result['ra'][0] 
        dec = result['dec'][0]
        ra_hms = Angle(ra, unit=u.deg).to_string(unit=u.hourangle, sep=':', precision=7)
        dec_dms = Angle(dec, unit=u.deg).to_string(unit=u.deg, sep=':', precision=6, alwayssign=True)
        global Ra
        global Dec 
        Dec = dec
        Ra = ra
        return ra, dec,ra_hms,dec_dms
    else:
        raise ValueError(f"No se encontraron coordenadas para el objeto {name}")
        
#Se define la función que toma las coordenadas y hace el query en gaia
def query_gaia(ra, dec, radius_arcmin=1):
    radius_deg = radius_arcmin / 60.0  # Conversión: 1 arcmin = 1/60 grados
    try:
        # Realizar la consulta en Gaia
        job = Gaia.launch_job_async(f"""
            SELECT  ra, dec,source_id
            FROM gaiadr2.gaia_source
            WHERE CONTAINS(POINT('ICRS', ra, dec), 
                           CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """)
        results = job.get_results()
        return results
    except Exception as e:
        print("Error en la consulta:", e)
        return None
        
#Obtener el ID de gaia
def gaia_ids():
    Simbad.add_votable_fields("ids")
    result = Simbad.query_object(object_name)
    if result is not None:
        ids = result["ids"][0].split("|")
        for id_ in ids:
            if "Gaia DR2" in id_:
                global gaia_id
                gaia_id = int(id_.replace("Gaia DR2 ", ""))
                print(f"🔍 Gaia DR2 ID de {object_name}: {gaia_id}")        
    else:
        return f"No se encontró el objeto '{object_name}' en Simbad."

#Se define la función que guarda los archivos en el csv como catálogo
def obtener_catalogo():
    ra, dec,ra_hms,dec_dms= get_coordinates_from_name(object_name)
    print(f"Coordenadas de {object_name} - RA: {ra}°, DEC: {dec}°")
    print(f"Coordenadas de {object_name} - RA_HMS: {ra_hms}, DEC_DMS: {dec_dms}")
    
    results = query_gaia(ra, dec, radius_arcmin=radius_arcmin)
    
    print(f"Se hallaron {len(results)} objetos")
    print(f"✅ Catálogo descargado")
    data = results.to_pandas()
    data.to_csv(f'catalogo_gaiadr2.csv',index = False, header=False,sep=" ")
    rows, columns = data.shape
    
def Definir_sector():
    print(f"Para el objeto {object_name} se tienen los sectores:")
    while True:
        try:
            sectors = Tesscut.get_sectors(objectname=object_name)
            print(f"\n{sectors}\n")
            global sector_seleccionado
            sector_seleccionado = input("Ingrese el sector a analizar:")
            catalogo_sector= f"tesscurl_sector_{sector_seleccionado}_ffic.sh"
            if catalogo_sector in os.listdir("."):
                print(f"✅ Archivo .sh del sector listo")
                
            while catalogo_sector not in os.listdir("."):
                url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/{catalogo_sector}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    with open(catalogo_sector, "wb") as file:
                        file.write(response.content)
                        print(f"✅ Archivo descargado en: {ruta_completa}")
                    break
                time.sleep(2)

            with open(catalogo_sector, "rb") as archivo:
                print(f"Es posible descargar {sum(1 for _ in archivo)} imágenes del sector {sector_seleccionado}")
            break

        except requests.exceptions.RequestException:
            time.sleep(2)
#Comprobación del directorio de Trabajo, de fallar regersar a la carpeta principal
dest_dir ="./imagenes"
current_dir = os.getcwd()
current_folder = os.path.basename(current_dir)
if f'./{current_folder}' == dest_dir:
    print(f"Estás dentro de {current_folder}. Saliendo...")
    os.chdir("..") 

#Ahora si, el loop principal
console = Console()

def procesar_imagenes():
    """Descarga y analiza imágenes en ciclos manteniendo la paralelización."""
    cargar_checkpoint()
    inicio = int(input("Imagen de inicio"))
    porciclo =  int(input("Imagenes por ciclo"))
    ciclos = int(input(f"Cuantos ciclos de {porciclo} imagenes"))
    if input(f"Se descargarán {porciclo} imágenes por ciclo (~{porciclo * 34}MB cada vez). \nContinuar? [y/n]: ") != "y":
        console.print("❌ Cancelado", style="bold red")
        return
        
    max_workers = min(10, os.cpu_count())
    dest_dir = "./imagenes"
    
    # Configuración de la barra de progreso global
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=20, style="white", complete_style="bright_cyan"),
        TextColumn("[bold cyan]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        for i in range(ciclos):
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.basename(os.getcwd()) == os.path.basename(dest_dir):
                os.chdir("..")
            
            inicio_ciclo = inicio + i * porciclo
            fin_ciclo = inicio_ciclo + porciclo
            
            with open(f"tesscurl_sector_{sector_seleccionado}_ffic.sh", "r") as archivo:
                lineas = [linea.strip() for idx, linea in enumerate(archivo) if inicio_ciclo <= idx < fin_ciclo]
            
            # Agregar una nueva tarea para este ciclo
            task = progress.add_task(
                f"[white][{inicio_ciclo}-{fin_ciclo-1}] ⏳ Descargando...", 
                total=len(lineas),
                bar_style="bright_cyan"  # Asegurar que inicie en cyan
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def descargar(linea):
                    nombre = os.path.join(dest_dir, os.path.basename(linea)) if "http" in linea else None
                    subprocess.run(f"{linea} -o {nombre}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,cwd="./imagenes")
                    progress.update(task, advance=1)
                list(executor.map(descargar, lineas))
           
            progress.update(task, description=f"[white][{inicio_ciclo}-{fin_ciclo-1}] 🟡 Analizando...",bar_style="bright_cyan")
            subprocess.run("python phot4.py", shell=True, capture_output=True, text=True)
            shutil.rmtree(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Marcar la tarea como completada
            progress.update(task, description=f"[bold cyan][{inicio_ciclo}-{fin_ciclo-1}] [cyan]✅ Done")
            progress.refresh()
    guardar_checkpoint(inicio+porciclo*ciclos)
#Esto lo que hace es analizar los datos csv, y construir las dos listas con los datos a graficar
def conteo_datos():
    gaia_ids()
    nombre_carpeta = "csv_out"
    directorio = os.getcwd()
    directorio= os.path.basename(directorio)
    if not f'{directorio}' == nombre_carpeta:
        os.chdir("./csv_out")
    fechas=[]
    magnitudes=[]
    #Proceso de iteración
    for archivo in os.listdir("./"):
            with open(archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
                lector = csv.reader(archivo_csv)
                for fila in lector:
                    if fila[0] == str(gaia_id) : #Definir id de simbad del objeto
                        fechas.append(fila[5])
                        magnitudes.append(float(fila[3]))
    global tiempo
    global magnitud
    tiempo=fechas
    magnitud=magnitudes
    
    print(f"Datos agregados correctamente, actualmente se cuenta con {len(magnitudes)} datos")
    os.chdir("..")

def curvadeluz(color):
    global tiempo
    global magnitud
    
    fechas_clean = [Time(f, format='isot', scale='utc').jd for f in tiempo]
    fechas_clean_2 = [datetime.strptime(f, "%Y-%m-%dT%H:%M:%S.%f") for f in tiempo]
    
    flujos=[]
    for i in magnitud:
        flujos.append(10**((20.44-i)/2.5))
    if color=="dark":
        plt.rcParams.update({
            'axes.facecolor': '#121212',  # Fondo del área del gráfico (gris oscuro)
            'figure.facecolor': '#121212',  # Fondo de la figura (gris oscuro)
            'font.family': 'Arial',  # Fuente
            'font.size': 10 ,  # Tamaño de la fuente
            'axes.labelcolor': 'white',  # Color de las etiquetas de los ejes (blanco)
            'xtick.color': 'white',  # Color de las etiquetas en el eje X (blanco)
            'ytick.color': 'white',  # Color de las etiquetas en el eje Y (blanco)
            'axes.edgecolor': 'white',  # Color de los bordes del gráfico (blanco)
            'grid.color': '#404040',  # Color de la cuadrícula (gris claro)
            'grid.linestyle': '--',  # Estilo de la cuadrícula (líneas discontinuas)
            'grid.alpha': 0.3,  # Transparencia de la cuadrícula (más suave)
            'lines.color': 'cyan',  # Color de las líneas (cian brillante)
            'lines.linewidth': 2,  # Ancho de las líneas
            'axes.titlecolor': 'white',  # Color del título de los ejes (blanco)
        })  
        alto,ancho=4,8
        plt.figure(figsize=(ancho, alto))  
        plt.plot(fechas_clean[:], flujos[:], marker='o', linestyle='-', color="white",markerfacecolor='none', linewidth=1)
        plt.xlabel("Tiempo Juliano",fontsize=10,labelpad=20)
        plt.ylabel("Flujo",fontsize=10,labelpad=20) 
        plt.title(f"----------------Curva de luz para {object_name}----------------".upper(), fontsize=10,pad=10)
    
        plt.show()
    elif color=="light":
        plt.rcParams.update({
                'axes.facecolor': 'white',  # Fondo del área del gráfico (gris oscuro)
                'figure.facecolor': 'white',  # Fondo de la figura (gris oscuro)
                'font.family': 'Times New Roman',  # Fuente
                'font.size': 12 ,  # Tamaño de la fuente
                'axes.labelcolor': 'black',  # Color de las etiquetas de los ejes (blanco)
                'xtick.color': 'gray',  # Color de las etiquetas en el eje X (blanco)
                'ytick.color': 'gray',  # Color de las etiquetas en el eje Y (blanco)
                'axes.edgecolor': 'black',  # Color de los bordes del gráfico (blanco)
                'grid.color': '#404040',  # Color de la cuadrícula (gris claro)
                'grid.linestyle': '--',  # Estilo de la cuadrícula (líneas discontinuas)
                'grid.alpha': 0.3,  # Transparencia de la cuadrícula (más suave)
                'lines.color': 'black',  # Color de las líneas (cian brillante)
                'lines.linewidth': 2,  # Ancho de las líneas
                'axes.titlecolor': 'black',  # Color del título de los ejes (blanco)
            })
        
        alto,ancho=4,6
        
        plt.figure(figsize=(ancho, alto))  
        plt.plot(fechas_clean[:], flujos[:], marker='o',markersize=3, linestyle='-', color="gray",markerfacecolor='none',markeredgecolor="black", linewidth=1)
        plt.xlabel("Tiempo Juliano [días]",fontsize=12,labelpad=5)
        plt.ylabel("Flujo [e-/s]",fontsize=12,labelpad=5) 
        plt.title(f"Curva de luz para {object_name}".upper(), fontsize=10,pad=10)
        
def redondear_con_incertidumbre(valor, incertidumbre):
    orden_magnitud = math.floor(math.log10(abs(incertidumbre))) 
    factor = 10 ** orden_magnitud
    incertidumbre_red = round(incertidumbre / factor) * factor 
    valor_red = round(valor / factor) * factor
    return valor_red, incertidumbre_red
    
def periodograma(color):
    fechas_clean = [Time(f, format='isot', scale='utc').jd for f in tiempo]
    flujos=[]
    for i in magnitud:
            flujos.append(10**(-i/2.5))
    time = np.array(fechas_clean) * u.day
    flux = np.array(flujos)        
    
    min_period = 0.1 * u.day  
    max_period = 10 * u.day  
    frequency = np.linspace(1/max_period, 1/min_period, 20000)  
    
    ls = LombScargle(time, flux)
    power = ls.power(frequency)
    
    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency
    periods=1/frequency
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(power) 
    
    # Ordenar los picos por altura para identificar el más grande y el segundo más grande
    sorted_peaks = sorted(peaks, key=lambda x: power[x], reverse=True)
    primary_period = periods[sorted_peaks[0]]     
    # Buscar el segundo pico correspondiente a 2 * P/2 (el período orbital real)
    secondary_period = None
    for peak in sorted_peaks[1:]:
        candidate_period = periods[peak]
        # Si el candidato está aproximadamente en 2 * P/2, lo tomamos como el verdadero período
        if np.isclose(candidate_period, 2 * primary_period, rtol=0.1):
            secondary_period = candidate_period
            break
    if secondary_period is not None:
        print(f"Periodo real orbital: {secondary_period.value} días")
    else:
        print("No se encontró un segundo pico claro que represente el período orbital.")
        
    # Estimación de incertidumbre en período
    secondary_power = power[sorted_peaks[1]]
    half_max = secondary_power/ 2
    above_half_max = np.where(power >= half_max)[0]
    indices=[]
    for i in range(len(above_half_max)):
        if above_half_max[i+1]-above_half_max[i]==1:
            indices.append(above_half_max[i])
        else:
            indices.append(above_half_max[i])
            break
    FWHM=1/frequency[indices[0]]-1/frequency[indices[-1]]
    sigma_f = FWHM / 2.35  # Aproximando el pico como gaussiano
    sigma_P = sigma_f / (secondary_period.value)**2
    print(redondear_con_incertidumbre(secondary_period.value,sigma_P.value))
    
       
    if color=="dark":
        alto,ancho=4,8
        plt.figure(figsize=(ancho, alto))  
        plt.plot(1/frequency, power, marker='', linestyle='-', color="white",markerfacecolor='none', linewidth=1)
        plt.axvline(best_period.value, color="cyan", linestyle="--",linewidth=1)
        plt.xlabel("Periodo en dias",fontsize=10,labelpad=20)
        #plt.xscale("log")
        plt.ylabel("Potencia",fontsize=10,labelpad=20) 
        plt.title(f"----------Periodograma Lomb-Scargle de {object_name}----------".upper(), fontsize=10,pad=10)
    elif color=="light":
        import matplotlib.ticker as mticker
        alto,ancho=4,6
        plt.figure(figsize=(ancho, alto))  
        plt.plot(1/frequency, power, marker='', linestyle='-', color="black", linewidth=1)
        plt.axvline(secondary_period.value, color="gray", linestyle="--",linewidth=1)
        plt.axvline(best_period.value, color="gray", linestyle="--",linewidth=1)
        plt.xlabel("Periodo orbital [días]",fontsize=12,labelpad=5)
        #plt.xscale("log")
        plt.ylabel("Potencia normalizada",fontsize=12,labelpad=5) 
        plt.title(f"Periodograma Lomb-Scargle de {object_name}".upper(), fontsize=10,pad=10)
        ax = plt.gca()  # Obtener el eje actual
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        # Añadir texto con los valores sobre la gráfica
        plt.text(secondary_period.value*(1+0.08), max(power)*0.7, f"{secondary_period:.2f}", color='black', fontsize=10, ha='left')
    
    plt.show()
    


#Obtención del Catálogo
object_name = input("Ingresa el nombre del objeto: ")
radius_arcmin = float(input("Ingresa el radio de búsqueda en arcominutos: "))

obtener_catalogo()
Definir_sector()

import pickle

def guardar_checkpoint(valor, archivo="./checkpoint.pkl"):
    with open(archivo, "wb") as f:
        pickle.dump(valor, f)
    print(f"✅Última imagen descargada: {valor}")

def cargar_checkpoint(archivo="checkpoint.pkl"):
    if os.path.exists(archivo):
        with open(archivo, "rb") as f:
            valor = pickle.load(f)
        print(f"📌Última imagen descargada: {valor}")
        return valor
    else:
        print("⚠️ No hay checkpoint guardado.")
        return None 

procesar_imagenes()
conteo_datos()
curvadeluz("light")
periodograma("light")





