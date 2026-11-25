# Visión por Computadora 2 — lluvia semanal con GOES-16

Trabajo práctico de segmentación binaria: con 14 imágenes diarias de las 16 bandas de GOES-16 se predice para cada celda si la lluvia acumulada durante los siete días siguientes alcanza 1 mm. El umbral se puede cambiar en la notebook. Las etiquetas son estimaciones diarias CPC de NOAA; los píxeles sin datos para alguno de los siete días se excluyen de las métricas.

El modelo combina una CNN multiescala (kernels 1×1, 3×3 y 5×5), una ConvGRU y un decoder. Se comparan dos métodos de reducción espacial: convoluciones con stride y MaxPooling. La selección usa validación temporal y se muestra la evaluación de test. Se usan cinco épocas por variante como configuración inicial.

## Requisitos

Python 3.10 o posterior, conexión a Internet y espacio en disco. Las capturas originales son grandes: se descarga temporalmente una por día y se conserva una versión reducida. Las descargas de CPC ocupan varios cientos de MB. Para usar GPU, instalar PyTorch según las instrucciones de [PyTorch](https://pytorch.org/get-started/locally/).

## Preparación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name vision2-goes --display-name "Python (vision2-goes)"
```

1. Descargar y reducir las imágenes a la grilla incluida (`cpc_grilla.nc`). La notebook usa imágenes de las 15 UTC. El intervalo indicado reproduce la cobertura utilizada en el experimento; la descarga completa requiere tiempo y espacio:

```bash
python descargar_goes_diario.py --grilla cpc_grilla.nc --inicio 2018-01-01 --fin 2025-04-07 --hora 15
```

La descarga se puede reanudar: omite archivos ya existentes. Por defecto exige 50 GiB libres y revisa el crecimiento de Git; ajustar `--min-libre-gib` según la máquina. El producto MCMIPF se obtiene del bucket público `noaa-goes16` de AWS sin credenciales. Algunos días pueden no estar disponibles: la notebook descarta ventanas de entrada incompletas. El tamaño del dataset, las fechas y las métricas pueden cambiar si cambia la cobertura.

2. Iniciar Jupyter desde la raíz del repositorio y abrir `TP_GOES16_lluvia_semanal.ipynb` con el kernel **Python (vision2-goes)**:

```bash
jupyter lab
```

3. Ejecutar las celdas en orden. La notebook descarga automáticamente los archivos anuales de precipitación diaria [CPC Global](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html) en `datos_crnn/cpc_diario/`, calcula la normalización solo con entrenamiento y entrena ambos modelos. No se suben datos ni pesos a Git.

## Particiones y evaluación

Entrenamiento: 2018–2023; validación: 2024; test: 2025. Una ventana se descarta si cruza los límites de una partición. La última imagen corresponde a las 15 UTC del día D y la etiqueta a D+1 hasta D+7 (00 UTC de D+1 a 00 UTC de D+8): queda un intervalo de 9 horas entre entrada y etiqueta. El test disponible en el experimento original comprende solo 12 semanas, por lo que no representa necesariamente todo un año. La comparación debe considerar BCE, precisión, recall, F1 e IoU, además de la referencia que siempre predice la clase mayoritaria. Las celdas vecinas están correlacionadas.

CPC es una estimación grillada basada en observaciones y tiene cobertura espacial incompleta. Las métricas se calculan solo sobre píxeles con lluvia válida durante toda la semana. En una corrida anterior, la variante con MaxPooling obtuvo F1 0,801 y recall 0,900 en validación; son resultados orientativos, no una garantía de reproducción exacta entre hardware y versiones.
