"""
mpi_analisis.py
===============
FASE 2 de la arquitectura: "MPI4Py -> distribución del procesamiento".

Se ejecuta con:   mpirun -n 4 python mpi_analisis.py
(mpirun lanza 4 COPIAS de este mismo script; cada copia se llama "rank".)

Concepto clave para exponer
---------------------------
MPI = Message Passing Interface. Cada proceso tiene su PROPIA memoria; para
compartir datos se mandan mensajes. El patrón que usamos es el clásico:

      rank 0 reparte  --scatter-->  todos calculan su parte  --gather--> rank 0 junta

  * comm.Get_rank()  -> mi número de proceso (0, 1, 2...)
  * comm.Get_size()  -> cuántos procesos hay en total
  * comm.bcast       -> rank 0 manda el MISMO dato a todos (parámetros)
  * comm.scatter     -> rank 0 manda un PEDAZO DISTINTO a cada proceso
  * comm.gather      -> todos mandan su resultado a rank 0
  * comm.Reduce      -> combina arreglos de todos (aquí: suma) en rank 0

Qué calculamos: por aeropuerto, estadísticas de retraso + un intervalo de
confianza al 95% del retraso medio con BOOTSTRAP (remuestrear miles de veces).
El bootstrap es costoso en CPU y cada aeropuerto es independiente, así que es
un problema ideal para repartir entre procesos.
"""
import json
import os
import socket
import sys
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import limpieza_pandas as L


def bootstrap_ic95(x, n_boot, rng):
    """
    Intervalo de confianza 95% de la media de `x` por bootstrap:
    1) remuestrea con reemplazo `n_boot` veces, 2) saca la media de cada
    remuestreo, 3) toma los percentiles 2.5 y 97.5 de esas medias.
    """
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))     # índices al azar
    medias = x[idx].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

    # --- 1) Solo rank 0 lee el archivo y prepara el reparto ------------------
    if rank == 0:
        horarios = L.limpiar_horarios(pd.read_csv("datos/horarios.csv"))
        aeropuertos = sorted(horarios["dep_iata"].unique())
        # Reparto "round-robin": aeropuerto i -> proceso (i % size). Así la carga
        # queda equilibrada aunque haya más aeropuertos que procesos.
        trozos = [horarios[horarios["dep_iata"].isin(aeropuertos[r::size])]
                  for r in range(size)]
    else:
        trozos, n_boot = None, None

    comm.Barrier()                         # todos esperan para medir tiempo parejo
    t0 = MPI.Wtime()

    # --- 2) Distribución de datos -------------------------------------------
    n_boot = comm.bcast(n_boot if rank == 0 else None, root=0)   # mismo parámetro a todos
    mio = comm.scatter(trozos, root=0)                           # un pedazo distinto a cada uno

    # --- 3) Cálculo LOCAL (cada proceso trabaja solo con sus aeropuertos) ----
    rng = np.random.default_rng(1234 + rank)       # semilla distinta por proceso
    filas = []
    for iata, g in mio.groupby("dep_iata"):
        x = g["dep_delayed"].to_numpy(float)
        lo, hi = bootstrap_ic95(x, n_boot, rng)
        filas.append(dict(iata=iata, n=len(x), media=x.mean(), mediana=np.median(x),
                          ic95_inf=lo, ic95_sup=hi, rank_que_calculo=rank))
    horas_local = np.bincount(mio["hora"].dropna().astype(int), minlength=24).astype("int64")
    t_local = MPI.Wtime() - t0
    print(f"[rank {rank}/{size}] host={socket.gethostname()} aeropuertos="
          f"{len(filas)} filas={len(mio)} tiempo_local={t_local:.2f}s", flush=True)

    # --- 4) Agregación de resultados ----------------------------------------
    todas = comm.gather(filas, root=0)                  # lista de listas -> rank 0
    horas_total = np.zeros(24, dtype="int64") if rank == 0 else None
    comm.Reduce(horas_local, horas_total, op=MPI.SUM, root=0)   # suma de histogramas
    t_total = MPI.Wtime() - t0

    if rank == 0:
        df = pd.DataFrame([f for sub in todas for f in sub]).sort_values("media", ascending=False)
        os.makedirs("resultados", exist_ok=True)
        df.to_csv("resultados/mpi_estadisticas.csv", index=False)
        pd.Series(horas_total, name="salidas").rename_axis("hora").to_csv("resultados/mpi_salidas_por_hora.csv")
        # Guardamos el tiempo de esta corrida para comparar 1, 2, 4 procesos
        ruta = "resultados/mpi_tiempos.json"
        tiempos = json.load(open(ruta)) if os.path.exists(ruta) else {}
        tiempos[str(size)] = t_total
        json.dump(tiempos, open(ruta, "w"))
        print(f"\nTOTAL con {size} proceso(s): {t_total:.2f} s (bootstrap={n_boot:,})")
        print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
