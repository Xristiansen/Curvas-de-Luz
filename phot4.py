#De ser necesario instalar photoutils y astropy
#Importaciones
import os
import shutil
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.utils import calc_total_error
from scipy.stats import mode
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from photutils.centroids import centroid_sources, centroid_com
from astropy.stats import sigma_clipped_stats
from astropy.table.table import QTable
from IPython.display import clear_output

import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

#Definir Directorios
Carpeta_imagenes="imagenes"
Archivo_catalogo="catalogo_gaiadr2.csv"
# Las archivos .out se almacenenan en fits_out y los .csv resultantes en csv_out

# Configuracion de rutas a los directorios 
carpeta = os.path.join(os.getcwd(), Carpeta_imagenes)
carpeta = os.path.join(carpeta, '')
archivo_catalogo = os.path.join(os.getcwd(), Archivo_catalogo)
final_dir = os.path.join(os.getcwd(), "csv_out")
final_dir = os.path.join(final_dir, '')
final_out = os.path.join(os.getcwd(), "objt_out")
final_out = os.path.join(final_out, '')
#Definción de cosas
center_box_size = 3
# Definición de parámetros fotométricos #
r = 10 # Apertura en px
r_in = 12 # Radio interno anillo cielo
r_out = 14 # Radio externo

# Busqueda de los archivos .fits
archivos = glob.glob(carpeta + '*.fits')
nombres = []
for j in archivos:
    if carpeta in j:
        nombres.append(j.replace(carpeta,''))
if nombres != []:
    l = len(nombres)
    print(f'\nSu carpeta contiene {l} archivos .fits\n')
    print(f'No. 1: {nombres[0]}')
    print(f'          ....            ')
    print(f'No. {l}: {nombres[l-1]}')
else: 
    print('\nSu carpeta no tiene archivos .fits')

# Carga del catálogo con coordenadas e identificador :  RA DEC ID
catalogo = open(archivo_catalogo,"r")
objects = catalogo.readlines()
catalogo.close()
# Se organiza el catalogo como una lista ordenada
L_O = []
for i in objects:
    L_O.append(i.split())
listObjects = L_O 
# Como el formato del catalogo es hhmmss ggmmss lo pasamos a grados decimales.
ra = [ i[0] for i in L_O ]
dec = [ i[1] for i in L_O ]
id = [ i[2] for i in L_O ]
catalogo_decimal = SkyCoord(ra, dec, unit=( u.degree))
catalogo = list(zip(catalogo_decimal.ra.deg,catalogo_decimal.dec.deg,id))


# Fotometría de apertura usando Photutils + Objetos de catálogo
def Photometry_Data_Table(fits_name, fits_path, catalogo, r, r_in, r_out, center_box_size):
    # Se abre el archivo .fits para guardarlo como una variable i.e. image / fits_data
    F = fits.open(fits_path)
    FitsData = F
    w = WCS(FitsData[1].header)
    fits_data = FitsData[1].data
    fits_header = FitsData[1].header

    itime = 1800
    target1 = fits_header['TSTART']
    target = fits_header.append(('OBJECT', target1), end=True)  
        
    ifilter = 'TESS'  
    DateObs = fits_header['DATE-OBS']
  
    epadu = 5.22
    F.close()

    image = fits_data
    # Funcion que ajusta los objetos del catalogo a solo aquellos que podrían estar en la imágen
    def is_in_pic(w, image, ra, dec):
        ra_max, dec_max = w.array_index_to_world_values(0, 0)
        ra_min, dec_min = w.array_index_to_world_values(image.shape[0], image.shape[1])
        if ra_min > ra_max:
            ra_min = w.array_index_to_world_values(0, 0)[0]
            ra_max = w.array_index_to_world_values(image.shape[0], image.shape[1])[0]
        if dec_min > dec_max:
            dec_min = w.array_index_to_world_values(0, 0)[1]
            dec_max = w.array_index_to_world_values(image.shape[0], image.shape[1])[1]
      
        return (ra < ra_max) & (ra > ra_min) & (dec < dec_max) & (dec > dec_min)

    ruta_salida = os.path.join(final_out, f"Objectlist_{fits_name}.out")
    NewListO = open(ruta_salida, "w")
    object_counter = 0
    for j in range(0, len(catalogo)):
        condicion = is_in_pic(w, image, catalogo[j][0], catalogo[j][1])
        if condicion:
            try:
                object_counter += 1
                X, Y = SkyCoord(catalogo[j][0], catalogo[j][1], frame="icrs", unit="deg").to_pixel(w)
                NewListO.write(f"{catalogo[j][0]}     {catalogo[j][1]}     {catalogo[j][2]}   {X}   {Y}   {condicion}\n")
            except (ValueError, TypeError) as e:
                print(f"Error al procesar las coordenadas de {catalogo[j][0]}, {catalogo[j][1]}: {e}")
                return
            
    NewListO.close()
    if object_counter == 0:
        return None

    # Se guardan las coordenadas de los objetos de catálogo que están en la imágen
    Obj = open(ruta_salida, "r")
    ListObj = Obj.readlines()
    Obj.close()
    Final_LO = []
    for i in ListObj:
        Final_LO.append(i.split()[:5])
    RA, DEC, ID, x, y = zip(*Final_LO) 
    Final_List = np.array(list(zip(RA, DEC, x, y)), dtype=float)
    ID = np.array(ID, dtype='U20')

    # Eliminar los objetos que no esten en el archivo fits
    mm = [ 0 < i[2] and i[2] < (image.shape[0] - 1) for i in Final_List] 
    ID = ID[mm]
    Final_List = Final_List[mm]
    nn = [ 0 < i[3] and i[3] < (image.shape[1] - 1) for i in Final_List] 
    ID = ID[nn]
    Final_List = Final_List[nn]

    u, c = np.unique(ID, return_counts=True)
    dup = u[c > 1]
    for j in dup:
        m = 0
        for i in range(len(ID)):
            if ID[i] == j:
                m += 0.1
                ID[i] = ID[i] + str(m)

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    
    _, _, x_init, y_init = zip(*Final_List)
    x, y = centroid_sources(image, x_init, y_init, box_size=center_box_size, centroid_func=centroid_com)
    X, Y = np.array(x), np.array(y)
    NewIDS = np.array(ID) 
    
    # Eliminar los datos a los cuales tienen un centroide NaN o inf
    is_nan = ~np.isnan(X)
    x, y = X[is_nan], Y[is_nan]      
    Final_List2 = Final_List[is_nan] 
    NewIDS = NewIDS[is_nan]

    starloc = list(zip(x, y))
 
    zmag = 20.4402281476
    
    # Extracción señal de cada estrella
    aperture = CircularAperture(starloc, r=r)
    annulus_aperture = CircularAnnulus(starloc, r_in=r_in, r_out=r_out )
    apers = [aperture, annulus_aperture]
    phot_table = aperture_photometry(image, apers)

    # Se le asigna nombre de la magnitud dependiendo del filtro en el encabezado
    name_mag = str(ifilter)

    # Area y flujo en los anillos. 
    bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
    area_aper = np.array(aperture.area_overlap(image))
    bkg_sum = bkg_mean * area_aper
    
  
    # Flujo final para cada objeto
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['flux'] = final_sum
    phot_table['flux'].info.format = '%.8g' 
  
    # Magnitudes Instrumentales
    phot_table[name_mag + '_mag'] = zmag - 2.5 * np.log10(abs(final_sum)) 
    phot_table[name_mag + '_mag'].info.format = '%.8g'  
    
    # Error de las Magnitudes Instrumentales
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    stdev = std
    phot_table[name_mag + '_mag_err'] = 1.0857 * np.sqrt(abs(final_sum) / epadu + area_aper * stdev**2 ) / abs(final_sum)
    phot_table[name_mag + '_mag_err'].info.format = '%.8g'

    # Se agrega a la tabla la RA, DEC, ID y Masa de aire. 
    phot_table['RA'] = [i[0] for i in Final_List2] 
    phot_table['DEC'] = [i[1] for i in Final_List2] 
    phot_table['ID'] = NewIDS
    phot_table['DATE-OBS'] = DateObs
    phot_table['OBJECT'] = fits_header['OBJECT']

    # Se buscan los indices en donde las magnitudes sean NaN y se eliminan
    index_nan = np.argwhere(np.isnan(phot_table[name_mag + '_mag'].data)) 
    phot_table.remove_rows(index_nan)
    filas = len(phot_table)
    
    clear_output(wait=True)
    return phot_table

#Continuación código
all_tables = []
for k in range(len(archivos)):
    fits_path = archivos[k]
    fits_name = nombres[k]

    photom = Photometry_Data_Table(fits_name, fits_path, catalogo, r=r, r_in=r_in, r_out=r_out, center_box_size=center_box_size)
    if photom is not None:
        all_tables.append(photom)
print(f'Se obtuvieron {len(all_tables)} archivos .csv')

# Crea lista con los nombres de los objetos a los cuales se enfoca el telescopio
focus_object = []             
for m in all_tables:
    if m != []:
        ob = m['OBJECT'][0]
    
        if ob not in focus_object:
            focus_object.append(ob)   # Ejemplo: focus_object = ['SA98', 'SA95', '[BSA98', 'SA101', '[ASA98', 'SA104', 'SA92']

# Se crea diccionario con cada objeto de enfoque
filtro_final = {}
for s in focus_object:
    filtro_final[s] = []        # Ejemplo: filtro_final = {'SA98':[], 'SA95':[], '[BSA98':[], 'SA101':[], '[ASA98':[], 'SA104':[], 'SA92':[]}

# Se llena el diccionario
for n in all_tables:
    for p in focus_object:
        ob = n['OBJECT'][0]
        if ob == p:
            filtro_final[ob].append(n.copy())  # Ejemplo: filtro_final = {'SA98':[tabla1,tabla2,tabla3,..], ... , 'SA92':[tabla1,tabla2,tabla3,..]}

# Para cada observacion de enfoque se hace la interseccion de los objetos que esten en los tres filtros
for foc in focus_object:
    current_id = []
    for j in filtro_final[foc]:
        current_id.append(j['ID'].data)
    
    int_d = set(current_id[0]).intersection(*current_id) # Ejemplo para SA98: int_d = {'92_248', ... , '92_347'}

    # Se eliminan los objetos que no esten en los tres filtros
    for tab in filtro_final[foc]:
        index_of = []
        for i in range(len(tab['ID'])):
            if tab['ID'][i] not in int_d:
                index_of.append(i)
        tab.remove_rows(index_of)

# Eliminar las tablas que esten vacias
for p in focus_object:
    if len(filtro_final[p][0]) == 0:
        del filtro_final[p]

for foc in filtro_final.keys():
    let = len(filtro_final[foc])

# Se crean tablas para cada objeto de enfoque
for foc in filtro_final.keys():
    final_obs_table = QTable()
    final_obs_table['OBJECT_ID'] = filtro_final[foc][0]['ID']
    final_obs_table['RA'] = filtro_final[foc][0]['RA']
    final_obs_table['DEC'] = filtro_final[foc][0]['DEC']

    # Se guardan las tablas como archivos .csv
    counter = 0
    for j in filtro_final[foc]:
        final_obs_table[j.colnames[6] + '_' + str(counter//3)] = j[j.colnames[6]]
        final_obs_table[j.colnames[7] + '_' + str(counter//3)] = j[j.colnames[7]]
        final_obs_table[j.colnames[11] + '_' + j.colnames[6] + '_' + str(counter//3)] = j[j.colnames[11]]
        counter += 1
    
    final_obs_table.write(f'{final_dir}DATAOUT_{foc}.csv', overwrite=True)    
