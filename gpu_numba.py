"""
gpu_numba.py
============
FASE 3 de la arquitectura: "CUDA / CuPy / Numba -> aceleración de cálculos".

Concepto clave para exponer
---------------------------
Una GPU tiene miles de núcleos pequeños. Si la MISMA operación se repite sobre
millones de datos independientes (distancias, z-scores...), la GPU lo hace en
paralelo masivo. En CUDA:

  * KERNEL  = función que corre en la GPU (aquí con el decorador @cuda.jit).
  * HILO (thread) = una copia del kernel; cada hilo procesa UN elemento.
  * BLOQUE (block) = grupo de hilos (usamos 256). GRID = conjunto de bloques.
  * cuda.grid(1)  = "¿qué número de hilo global soy?" -> es el índice i del dato.
  * Hay que COPIAR datos CPU->GPU (host->device) y regresar el resultado;
    esas copias cuestan tiempo, por eso la GPU solo gana con muchos datos.

Usamos:
  * Numba (@cuda.jit)   -> escribimos los kernels nosotros en Python.
  * CuPy                -> como NumPy pero en la GPU (media, desv. est., correlación).

Si no hay GPU (o no está instalado CuPy) todo cae automáticamente a NumPy, así
el código nunca se rompe: solo es más lento.
"""
import math
import time

import numpy as np

try:
    from numba import cuda
    HAY_GPU = cuda.is_available()
except Exception:                     # numba sin soporte CUDA
    cuda, HAY_GPU = None, False

try:
    import cupy as cp
except Exception:
    cp = None

R_TIERRA_KM = 6371.0
HILOS = 256                           # hilos por bloque (múltiplo de 32 = tamaño de un "warp")


# ===========================================================================
# KERNELS (corren en la GPU). Solo se definen si Numba + CUDA existen.
# ===========================================================================
if cuda is not None:

    @cuda.jit
    def k_haversine(lat1, lon1, lat2, lon2, out):
        """
        Distancia de círculo máximo (km) entre dos puntos de la Tierra.
        Cada hilo i calcula la distancia del par i -> out[i].
        """
        i = cuda.grid(1)                              # mi índice global
        if i < out.size:                              # el último bloque puede sobrar
            p1 = lat1[i] * math.pi / 180.0            # grados -> radianes
            p2 = lat2[i] * math.pi / 180.0
            dphi = p2 - p1
            dlam = (lon2[i] - lon1[i]) * math.pi / 180.0
            a = (math.sin(dphi / 2.0) ** 2
                 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2)
            out[i] = 2.0 * 6371.0 * math.asin(math.sqrt(a))

    @cuda.jit
    def k_zscore(x, media, desv, umbral, flags):
        """
        Detección de atípicos: z = (x - media)/desv. Si |z| > umbral -> atípico (1).
        """
        i = cuda.grid(1)
        if i < x.size:
            z = (x[i] - media) / desv
            flags[i] = 1 if abs(z) > umbral else 0

    @cuda.jit
    def k_ruta(lat1, lon1, lat2, lon2, pasos, out_lat, out_lon):
        """
        Simulación de rutas: para cada vuelo i y cada paso j calcula el punto
        intermedio sobre el círculo máximo (interpolación esférica).
        Usa una malla 2D: eje x = vuelo, eje y = paso.
        """
        i, j = cuda.grid(2)
        if i < lat1.size and j < pasos:
            t = j / (pasos - 1.0)                     # fracción del recorrido: 0 .. 1
            f1 = lat1[i] * math.pi / 180.0
            l1 = lon1[i] * math.pi / 180.0
            f2 = lat2[i] * math.pi / 180.0
            l2 = lon2[i] * math.pi / 180.0
            # distancia angular entre origen y destino
            d = 2.0 * math.asin(math.sqrt(math.sin((f2 - f1) / 2.0) ** 2 + math.cos(f1)
                                          * math.cos(f2) * math.sin((l2 - l1) / 2.0) ** 2))
            if d < 1e-9:                              # origen == destino
                A, B = 1.0 - t, t
            else:
                A = math.sin((1.0 - t) * d) / math.sin(d)
                B = math.sin(t * d) / math.sin(d)
            x = A * math.cos(f1) * math.cos(l1) + B * math.cos(f2) * math.cos(l2)
            y = A * math.cos(f1) * math.sin(l1) + B * math.cos(f2) * math.sin(l2)
            z = A * math.sin(f1) + B * math.sin(f2)
            out_lat[i, j] = math.atan2(z, math.sqrt(x * x + y * y)) * 180.0 / math.pi
            out_lon[i, j] = math.atan2(y, x) * 180.0 / math.pi


# ===========================================================================
# Versiones CPU (NumPy): sirven de comparación y de plan B sin GPU
# ===========================================================================
def haversine_cpu(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(a))


def zscore_cpu(x, umbral=3.0):
    return (np.abs((x - x.mean()) / x.std()) > umbral).astype(np.int32)


# ===========================================================================
# Envoltorios (host): copian datos a la GPU, lanzan el kernel y regresan
# ===========================================================================
def _bloques(n):
    """Cuántos bloques de HILOS hilos hacen falta para cubrir n datos."""
    return (n + HILOS - 1) // HILOS


def haversine(lat1, lon1, lat2, lon2):
    """Distancias en km. Usa la GPU si existe; si no, NumPy."""
    if not HAY_GPU:
        return haversine_cpu(*(np.asarray(a, float) for a in (lat1, lon1, lat2, lon2)))
    d = [cuda.to_device(np.asarray(a, dtype=np.float32)) for a in (lat1, lon1, lat2, lon2)]  # CPU->GPU
    out = cuda.device_array(len(lat1), dtype=np.float32)                                     # memoria en GPU
    k_haversine[_bloques(len(lat1)), HILOS](*d, out)       # lanzamiento: [bloques, hilos](args)
    return out.copy_to_host()                              # GPU->CPU


def detectar_atipicos(x, umbral=3.0):
    """Devuelve un arreglo de 0/1 (1 = atípico) usando z-score."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=float(np.nanmean(x)))         # NaN -> media (no cuenta como atípico)
    if not HAY_GPU:
        return zscore_cpu(x, umbral)
    if cp is not None:                                     # media y desv. est. en GPU con CuPy
        xd = cp.asarray(x); media, desv = float(xd.mean()), float(xd.std())
    else:
        media, desv = float(x.mean()), float(x.std())
    if desv == 0:
        return np.zeros(len(x), dtype=np.int32)
    flags = cuda.device_array(len(x), dtype=np.int32)
    k_zscore[_bloques(len(x)), HILOS](cuda.to_device(x), np.float32(media),
                                      np.float32(desv), np.float32(umbral), flags)
    return flags.copy_to_host()


def rutas_simuladas(lat1, lon1, lat2, lon2, pasos=50):
    """Matriz (n_vuelos x pasos) de latitudes y longitudes a lo largo de cada ruta."""
    n = len(lat1)
    if not HAY_GPU:                                        # versión NumPy vectorizada
        t = np.linspace(0, 1, pasos)[None, :]
        f1, l1, f2, l2 = (np.radians(np.asarray(a, float))[:, None] for a in (lat1, lon1, lat2, lon2))
        d = 2 * np.arcsin(np.sqrt(np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin((l2 - l1) / 2) ** 2))
        d = np.where(d < 1e-9, 1e-9, d)
        A, B = np.sin((1 - t) * d) / np.sin(d), np.sin(t * d) / np.sin(d)
        x = A * np.cos(f1) * np.cos(l1) + B * np.cos(f2) * np.cos(l2)
        y = A * np.cos(f1) * np.sin(l1) + B * np.cos(f2) * np.sin(l2)
        z = A * np.sin(f1) + B * np.sin(f2)
        return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))
    d = [cuda.to_device(np.asarray(a, dtype=np.float32)) for a in (lat1, lon1, lat2, lon2)]
    o_lat = cuda.device_array((n, pasos), dtype=np.float32)
    o_lon = cuda.device_array((n, pasos), dtype=np.float32)
    bloque = (16, 16)                                      # 16x16 = 256 hilos por bloque (malla 2D)
    grid = ((n + 15) // 16, (pasos + 15) // 16)
    k_ruta[grid, bloque](*d, pasos, o_lat, o_lon)
    return o_lat.copy_to_host(), o_lon.copy_to_host()


def matriz_correlacion(df, columnas, metodo="pearson"):
    """
    Matriz de correlación entre columnas. En GPU con CuPy (cp.corrcoef).
    Spearman = Pearson sobre los rangos (más robusto a atípicos).
    """
    datos = df[columnas].dropna()
    if metodo == "spearman":
        datos = datos.rank()
    m = datos.to_numpy(float).T                            # corrcoef pide variables en filas
    r = cp.asnumpy(cp.corrcoef(cp.asarray(m))) if (HAY_GPU and cp is not None) else np.corrcoef(m)
    import pandas as pd
    return pd.DataFrame(r, index=columnas, columns=columnas)


# ===========================================================================
# Benchmark CPU vs GPU
# ===========================================================================
def benchmark(n=5_000_000, repeticiones=3):
    """
    Compara haversine en NumPy (CPU) vs kernel CUDA (GPU) con n pares de puntos.
    Mide dos cosas en GPU: solo el kernel y el total (con copias CPU<->GPU).
    """
    rng = np.random.default_rng(0)
    lat1, lat2 = rng.uniform(14, 33, n).astype(np.float32), rng.uniform(14, 33, n).astype(np.float32)
    lon1, lon2 = rng.uniform(-118, -86, n).astype(np.float32), rng.uniform(-118, -86, n).astype(np.float32)

    t = time.perf_counter()
    for _ in range(repeticiones):
        ref = haversine_cpu(lat1, lon1, lat2, lon2)
    t_cpu = (time.perf_counter() - t) / repeticiones
    res = dict(n=n, cpu_s=t_cpu, gpu_disponible=bool(HAY_GPU))
    if not HAY_GPU:
        return res

    haversine(lat1[:1000], lon1[:1000], lat2[:1000], lon2[:1000])   # "calentamiento": compila el kernel (JIT)
    t = time.perf_counter()
    for _ in range(repeticiones):
        out = haversine(lat1, lon1, lat2, lon2)                      # incluye copias
    res["gpu_total_s"] = (time.perf_counter() - t) / repeticiones

    d = [cuda.to_device(a) for a in (lat1, lon1, lat2, lon2)]        # copias ANTES de medir
    o = cuda.device_array(n, dtype=np.float32)
    cuda.synchronize(); t = time.perf_counter()                      # synchronize: las llamadas a GPU son asíncronas
    for _ in range(repeticiones):
        k_haversine[_bloques(n), HILOS](*d, o)
    cuda.synchronize()
    res["gpu_kernel_s"] = (time.perf_counter() - t) / repeticiones
    res["max_error_km"] = float(np.max(np.abs(out - ref)))           # verificación de que ambos coinciden
    res["speedup_total"] = t_cpu / res["gpu_total_s"]
    res["speedup_kernel"] = t_cpu / res["gpu_kernel_s"]
    return res


if __name__ == "__main__":
    print("GPU disponible:", HAY_GPU)
    print(benchmark())
