"""Descarga capturas GOES y las guarda reducidas a una grilla CPC existente.

Ejemplo: python descargar_goes_diario.py --grilla archivo_cpc.nc --inicio 2018-01-01 --fin 2025-12-31
La grilla de entrada debe contener coordenadas lat y lon CPC, en grados.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


def procesar_captura(ruta, latitudes, longitudes):
    with xr.open_dataset(ruta) as ds:
        p = ds.goes_imager_projection.attrs
        altura = p['perspective_point_height']
        crs = CRS.from_proj4(
            f"+proj=geos +h={altura} +lon_0={p['longitude_of_projection_origin']} "
            f"+sweep={p['sweep_angle_axis']} +a={p['semi_major_axis']} "
            f"+b={p['semi_minor_axis']} +units=m +no_defs"
        )
        lon, lat = np.meshgrid(longitudes, latitudes)
        x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
        x = xr.DataArray(x / altura, dims=('lat', 'lon'))
        y = xr.DataArray(y / altura, dims=('lat', 'lon'))
        canales = []
        for numero in range(1, 17):
            nombre = f'CMI_C{numero:02d}'
            calidad = f'DQF_C{numero:02d}'
            if nombre not in ds or calidad not in ds:
                raise KeyError(f'Falta {nombre} o {calidad} en {ruta}')
            # DQF 0: píxel considerado válido por el producto.
            campo = ds[nombre].where(ds[calidad] == 0)
            canales.append(campo.interp(x=x, y=y, method='nearest').values.astype('float32'))
        return np.stack(canales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grilla', required=True, help='NetCDF con coordenadas CPC lat/lon')
    parser.add_argument('--inicio', default='2018-01-01')
    parser.add_argument('--fin', default='2025-12-31')
    parser.add_argument('--hora', type=int, default=15, choices=range(24), help='Hora UTC')
    parser.add_argument('--salida', default='datos_crnn/goes_diario')
    parser.add_argument('--min-libre-gib', type=float, default=50, help='Espacio mínimo libre por filesystem en GiB')
    parser.add_argument('--max-crecimiento-git-gib', type=float, default=5, help='Detener si .git crece esta cantidad en GiB')
    args = parser.parse_args()
    if args.min_libre_gib < 0 or args.max_crecimiento_git_gib <= 0:
        parser.error('El mínimo libre no puede ser negativo y el límite de crecimiento Git debe ser positivo')
    inicio, fin = dt.date.fromisoformat(args.inicio), dt.date.fromisoformat(args.fin)
    if inicio > fin:
        parser.error('La fecha inicial debe ser anterior o igual a la final')
    with xr.open_dataset(args.grilla) as grid:
        lat = grid['lat'].values
        lon = ((grid['lon'].values + 180) % 360) - 180
        lat = np.sort(lat[(lat >= -56) & (lat <= -21)])
        lon = np.sort(lon[(lon >= -76) & (lon <= -53)])
    if not len(lat) or not len(lon):
        raise ValueError('La grilla CPC no cubre el área de Argentina solicitada')
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    # Vigilar el repositorio que contiene la salida; no presupone que .git sea un directorio.
    git_dir = Path(subprocess.check_output(
        ['git', '-C', str(salida.resolve()), 'rev-parse', '--absolute-git-dir'], text=True
    ).strip())
    def tamano_git():
        return int(subprocess.check_output(['du', '-sk', str(git_dir)], text=True).split()[0]) * 1024

    git_inicial = tamano_git()
    limite_git = args.max_crecimiento_git_gib * 1024**3
    print(f'Git inicial: {git_inicial / 1024**3:.2f} GiB; límite de crecimiento: {args.max_crecimiento_git_gib:.2f} GiB', flush=True)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    fecha = inicio
    while fecha <= fin:
        destino = salida / f'goes_{fecha.isoformat()}_{args.hora:02d}utc.nc'
        if destino.exists():
            print(f'Ya existe: {destino}', flush=True)
            fecha += dt.timedelta(days=1)
            continue
        prefijo = f"ABI-L2-MCMIPF/{fecha.year}/{fecha.timetuple().tm_yday:03d}/{args.hora:02d}/"
        respuesta = s3.list_objects_v2(Bucket='noaa-goes16', Prefix=prefijo)
        objetos = sorted(respuesta.get('Contents', []), key=lambda obj: obj['Key'])
        if not objetos:
            print(f'Sin captura: {fecha}', flush=True)
            fecha += dt.timedelta(days=1)
            continue
        objeto = objetos[0]
        # Se detiene antes de descargar si falta espacio en cualquiera de los dos sistemas.
        requerido = args.min_libre_gib * 1024**3
        for ruta in (Path(tempfile.gettempdir()), salida):
            libre = shutil.disk_usage(ruta).free
            if libre < requerido + objeto['Size']:
                raise RuntimeError(f'Espacio insuficiente en {ruta}: {libre / 1024**3:.2f} GiB libres; se reserva un mínimo de {args.min_libre_gib:.2f} GiB más el archivo a descargar')
        git_actual = tamano_git()
        if git_actual - git_inicial >= limite_git:
            raise RuntimeError(f'El repositorio Git creció {(git_actual - git_inicial) / 1024**3:.2f} GiB desde el inicio; descarga detenida')
        print(f"{fecha}: {objeto['Key']} ({objeto['Size'] / 1e6:.1f} MB)", flush=True)
        # Mantener el archivo grande fuera del repositorio y borrarlo al terminar.
        with tempfile.TemporaryDirectory(prefix='goes_') as temporal:
            captura = Path(temporal) / 'captura.nc'
            s3.download_file('noaa-goes16', objeto['Key'], str(captura))
            valores = procesar_captura(captura, lat, lon)
            reducido = xr.DataArray(
                valores, dims=('canal', 'lat', 'lon'),
                coords={'canal': [f'CMI_C{i:02d}' for i in range(1, 17)], 'lat': lat, 'lon': lon},
                name='goes', attrs={'archivo_origen': objeto['Key'], 'hora_utc': args.hora},
            )
            # El reducido se escribe en el mismo filesystem que su destino.
            with tempfile.NamedTemporaryFile(prefix='.goes_', suffix='.nc', dir=salida, delete=False) as archivo:
                parcial = Path(archivo.name)
            try:
                reducido.to_netcdf(parcial)
                os.replace(parcial, destino)
            finally:
                parcial.unlink(missing_ok=True)
        fecha += dt.timedelta(days=1)


if __name__ == '__main__':
    main()
