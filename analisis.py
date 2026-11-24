"""
analisis.py
===========
Une todas las piezas: usa pandas (limpieza), la GPU (atípicos, distancias,
rutas, correlación) y responde las 5 PREGUNTAS DE ANÁLISIS del proyecto.
"""
import numpy as np
import pandas as pd

import gpu_numba as g
from config import AEROPUERTOS

VARS_CLIMA = ["temp", "humedad", "presion", "viento_vel", "visibilidad_m", "lluvia_mm", "nubosidad"]


def region_de_lat(lat):
    """Misma regla de región que usamos para los vuelos (por latitud)."""
    return "Norte" if lat >= 25 else ("Centro" if lat >= 19 else "Sur-Sureste")


def marcar_atipicos(vuelos, horarios, umbral=3.0):
    """
    Detección de valores atípicos (z-score > umbral) EN GPU en:
    velocidad y altitud de vuelos, y retraso de salidas.
    """
    v, h = vuelos.copy(), horarios.copy()
    for col in ["speed", "alt"]:
        v[f"atip_{col}"] = g.detectar_atipicos(v[col], umbral) if v[col].notna().any() else 0
    h["atip_retraso"] = g.detectar_atipicos(h["dep_delayed"], umbral)
    return v, h


def vuelos_cerca_de_aeropuertos(vuelos, radio_km=150):
    """
    Cuenta cuántos vuelos activos hay a menos de `radio_km` de cada aeropuerto.
    Es una matriz (vuelos x aeropuertos) de distancias -> trabajo ideal para GPU:
    aplanamos todas las parejas y mandamos UN solo kernel.
    """
    n, m = len(vuelos), len(AEROPUERTOS)
    la = np.repeat(vuelos["lat"].to_numpy(float), m)           # cada vuelo repetido m veces
    lo = np.repeat(vuelos["lng"].to_numpy(float), m)
    ap_la = np.tile([a[1] for a in AEROPUERTOS.values()], n)   # los m aeropuertos repetidos n veces
    ap_lo = np.tile([a[2] for a in AEROPUERTOS.values()], n)
    d = g.haversine(la, lo, ap_la, ap_lo).reshape(n, m)
    return pd.Series((d < radio_km).sum(axis=0), index=list(AEROPUERTOS), name="vuelos_cercanos")


def resumen_vuelos(v):
    """Vuelos activos, velocidad y altitud promedio por región."""
    return (v.groupby("region", observed=True)
            .agg(vuelos=("flight_iata", "size"), velocidad_media_kmh=("speed", "mean"),
                 altitud_media_m=("alt", "mean")).round(1))


def rutas_mas_frecuentes(v, top=10):
    """Pares origen-destino más repetidos."""
    r = v.dropna(subset=["dep_iata", "arr_iata"])
    return (r.assign(ruta=r["dep_iata"] + " → " + r["arr_iata"])["ruta"]
            .value_counts().head(top))


def simular_rutas(v, n_max=40, pasos=50):
    """Simula la trayectoria origen->destino de hasta n_max vuelos (GPU)."""
    ok = v[v["dep_iata"].isin(AEROPUERTOS) & v["arr_iata"].isin(AEROPUERTOS)
           & (v["dep_iata"] != v["arr_iata"])].head(n_max)
    c = lambda col, i: [AEROPUERTOS[x][i] for x in ok[col]]
    la, lo = g.rutas_simuladas(c("dep_iata", 1), c("dep_iata", 2), c("arr_iata", 1), c("arr_iata", 2), pasos)
    return ok.reset_index(drop=True), la, lo


def tabla_final(resumen, cercanos):
    """Agrega región y tráfico cercano a la tabla por aeropuerto."""
    t = resumen.copy()
    t["region"] = t["iata"].map(lambda i: region_de_lat(AEROPUERTOS[i][1]))
    t["vuelos_cercanos"] = t["iata"].map(cercanos)
    return t


def _r(t, a, b, metodo="spearman"):
    """Correlación entre dos columnas (NaN si alguna no varía)."""
    d = t[[a, b]].dropna()
    return np.nan if d[a].nunique() < 2 or d[b].nunique() < 2 else d[a].corr(d[b], method=metodo)


def responder_preguntas(t, h):
    """
    Calcula y redacta la respuesta a cada pregunta del proyecto.
    Devuelve una lista de textos (también se guarda en el reporte).
    """
    out = []
    # 1) ¿Relación clima-retrasos? Spearman por aeropuerto (n = nº de aeropuertos)
    rs = {v: _r(t, v, "retraso_medio") for v in VARS_CLIMA}
    mejor = max(rs, key=lambda k: abs(rs[k]) if not np.isnan(rs[k]) else -1)
    out.append("1) ¿Hay relación entre clima y retrasos? Correlación de Spearman con el retraso medio "
               f"(n={len(t)} aeropuertos): " + ", ".join(f"{k}={v:+.2f}" for k, v in rs.items())
               + f". La más fuerte: {mejor} ({rs[mejor]:+.2f}). Con tan pocos aeropuertos solo "
               "|r|>0.58 sería significativa (p<0.05).")
    # 2) Lluvia o baja visibilidad: comparamos retrasos de vuelos en aeropuertos con/sin condición adversa
    hh = h.merge(t[["iata", "lluvia_mm", "visibilidad_m"]], left_on="dep_iata", right_on="iata")
    adverso = (hh["lluvia_mm"] > 0.5) | (hh["visibilidad_m"] < 8000)
    if adverso.any() and (~adverso).any():
        a, b = hh.loc[adverso, "dep_delayed"], hh.loc[~adverso, "dep_delayed"]
        out.append(f"2) ¿Aumentan los retrasos con lluvia (>0.5 mm/h) o visibilidad <8 km? Retraso medio "
                   f"con condición adversa: {a.mean():.1f} min (n={len(a)}) vs. normal: {b.mean():.1f} min "
                   f"(n={len(b)}); diferencia {a.mean() - b.mean():+.1f} min.")
    else:
        out.append("2) No hubo aeropuertos con condición adversa en esta muestra: no se puede comparar.")
    # 3) Ranking de aeropuertos
    top = t.sort_values("retraso_medio", ascending=False).head(3)
    out.append("3) Aeropuertos con mayor retraso medio: " + "; ".join(
        f"{r.iata} ({r.retraso_medio:.1f} min, {r.pct_retrasados:.0f}% >15 min)" for r in top.itertuples()) + ".")
    # 4) Actividad por hora y región
    hh2 = h.dropna(subset=["hora"]).copy()
    hh2["region"] = hh2["dep_iata"].map(lambda i: region_de_lat(AEROPUERTOS[i][1]))
    pico = hh2.groupby("region")["hora"].agg(lambda s: int(s.value_counts().idxmax()))
    out.append("4) Hora con más salidas por región: " + ", ".join(f"{r}={int(x):02d}:00 h" for r, x in pico.items())
               + f". En todo el país el pico es a las {int(hh2['hora'].value_counts().idxmax()):02d}:00 h.")
    # 5) Condiciones ambientales vs tráfico cercano
    cols = ["temp", "humedad", "viento_vel", "lluvia_mm", "pm25"]
    r5 = {c: _r(t, c, "vuelos_cercanos") for c in cols}
    out.append("5) Condiciones que coinciden con más tráfico aéreo (vuelos a <150 km, Spearman): "
               + ", ".join(f"{k}={v:+.2f}" for k, v in r5.items()) + ".")
    return out
