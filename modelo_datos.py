"""Lectura GOES y celda ConvGRU usadas por el TP semanal."""
import numpy as np
import torch
from torch import nn
import xarray as xr

CANALES_GOES = tuple(f'CMI_C{i:02d}' for i in range(1, 17))


def leer_goes(ruta, latitudes, longitudes):
    with xr.open_dataset(ruta) as ds:
        campo = ds['goes']
        if tuple(campo.canal.values.astype(str)) != CANALES_GOES:
            raise ValueError(f'Orden de canales GOES inesperado: {ruta}')
        if not np.array_equal(campo.lat.values, latitudes) or not np.array_equal(campo.lon.values, longitudes):
            raise ValueError(f'La grilla GOES no coincide con CPC: {ruta}')
        return campo.transpose('canal', 'lat', 'lon').values.astype(np.float32)


class ConvGRUCell(nn.Module):
    def __init__(self, canales_entrada, canales_ocultos):
        super().__init__()
        self.canales_ocultos = canales_ocultos
        self.compuertas = nn.Conv2d(canales_entrada + canales_ocultos, 2 * canales_ocultos, 3, padding=1)
        self.candidato = nn.Conv2d(canales_entrada + canales_ocultos, canales_ocultos, 3, padding=1)

    def forward(self, x, estado=None):
        if estado is None:
            estado = x.new_zeros(x.shape[0], self.canales_ocultos, x.shape[2], x.shape[3])
        reset, actualizar = torch.sigmoid(self.compuertas(torch.cat((x, estado), dim=1))).chunk(2, dim=1)
        candidato = torch.tanh(self.candidato(torch.cat((x, reset * estado), dim=1)))
        return (1 - actualizar) * estado + actualizar * candidato
