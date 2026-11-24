"""
limpieza_pandas.py
==================
FASE de apoyo (la barra morada de la arquitectura): limpieza, transformación y
unificación de estructuras con PANDAS.

Pandas es la librería de tablas (DataFrames). Aquí hacemos 4 cosas típicas:
  1. convertir columnas a número (`pd.to_numeric`),
  2. quitar duplicados y filas inservibles,
  3. crear columnas nuevas (región, hora, bandera de retraso),
  4. unir tablas (`merge`) para cruzar retrasos con clima.
"""
import numpy as np
import pandas as pd


def limpiar_vuelos(df):
    """Vuelos en el aire: tipos numéricos, rangos físicos razonables y región."""
    df = df.copy()
    for c in ["lat", "lng", "alt", "speed", "dir", "v_speed"]:
        if c not in df:                       # el plan gratis de AirLabs omite algunos campos
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["flight_iata", "lat", "lng"])
    df = df.dropna(subset=["lat", "lng"])                      # sin posición no sirve
    # NOTA: NO borramos velocidades/alturas raras aquí; se detectan como ATÍPICOS
    # más adelante (GPU). Solo quitamos lo físicamente imposible (negativos).
    df = df[(df["speed"].fillna(0) >= 0) & (df["alt"].fillna(0) >= 0)]
    # Región por latitud (simple, suficiente para el análisis)
    df["region"] = pd.cut(df["lat"], bins=[-90, 19, 25, 90],
                          labels=["Sur-Sureste", "Centro", "Norte"])
    return df.reset_index(drop=True)


def limpiar_horarios(df):
    """Salidas programadas: retrasos numéricos, banderas de cancelado/retrasado, hora."""
    df = df.copy()
    for c in ["dep_delayed", "arr_delayed"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cancelado"] = df["status"].eq("cancelled")
    df["retrasado"] = df["dep_delayed"].fillna(0) > 15          # >15 min = retraso (criterio común)
    df["dep_delayed"] = df["dep_delayed"].fillna(0).clip(lower=0)  # adelantos -> 0
    df["hora"] = pd.to_datetime(df["dep_time"], errors="coerce").dt.hour
    return df.drop_duplicates(subset=["flight_iata", "dep_iata", "dep_time"]).reset_index(drop=True)


def resumen_por_aeropuerto(horarios, clima, aire):
    """
    Tabla final: UNA fila por aeropuerto con retrasos + clima + calidad del aire.
    `groupby` agrupa los vuelos por aeropuerto y `agg` calcula estadísticas;
    luego `merge` pega el clima y el aire usando el código IATA como llave.
    """
    r = (horarios.groupby("dep_iata")
         .agg(n_vuelos=("flight_iata", "size"),
              retraso_medio=("dep_delayed", "mean"),
              retraso_p90=("dep_delayed", lambda x: np.percentile(x, 90)),
              pct_retrasados=("retrasado", "mean"),
              cancelados=("cancelado", "sum"))
         .reset_index().rename(columns={"dep_iata": "iata"}))
    r["pct_retrasados"] *= 100
    r = r.merge(clima.drop(columns="fuente"), on="iata", how="left")
    r = r.merge(aire[["iata", "pm25", "pm10"]], on="iata", how="left")
    return r


def vuelos_por_hora(horarios):
    """Cuántas salidas hay en cada hora del día (para ver el ciclo diario)."""
    return horarios.groupby("hora").size().reindex(range(24), fill_value=0)
