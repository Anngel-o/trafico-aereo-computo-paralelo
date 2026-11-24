"""
visualizacion.py
================
FASE 4 (Dashboard / Reportes): gráficas con Matplotlib. Cada función guarda un
PNG en figuras/ y lo devuelve para mostrarlo en el notebook.

Colores: paleta Okabe-Ito (segura para daltonismo). Usamos pocos colores y
etiquetas directas para que las gráficas se entiendan al exponer.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from config import AEROPUERTOS, DIR_FIGURAS

AZUL, NARANJA, VERDE, ROJO, GRIS = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 110, "font.size": 10})


def _guardar(fig, nombre):
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{DIR_FIGURAS}/{nombre}.png", dpi=130)
    return fig


def fig_actividad_vuelos(v, rutas):
    """Vuelos por región, por aerolínea y rutas más frecuentes."""
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    v["region"].value_counts().plot.bar(ax=ax[0], color=AZUL, rot=0); ax[0].set_title("Vuelos activos por región")
    v["airline_iata"].value_counts().head(10).plot.bar(ax=ax[1], color=NARANJA, rot=0)
    ax[1].set_title("Vuelos por aerolínea (top 10)")
    rutas.iloc[::-1].plot.barh(ax=ax[2], color=VERDE); ax[2].set_title("Rutas más frecuentes")
    return _guardar(f, "01_actividad_vuelos")


def fig_retrasos_aeropuerto(t, mpi_df):
    """Retraso medio por aeropuerto con intervalo de confianza 95% (calculado con MPI)."""
    d = mpi_df.sort_values("media")
    f, ax = plt.subplots(figsize=(8, 5))
    err = np.vstack([d["media"] - d["ic95_inf"], d["ic95_sup"] - d["media"]])
    ax.barh(d["iata"], d["media"], xerr=err, color=AZUL, ecolor=GRIS, capsize=3)
    ax.set_xlabel("Retraso medio de salida (min) con IC 95% bootstrap (MPI)")
    ax.set_title("Retrasos por aeropuerto")
    return _guardar(f, "02_retrasos_aeropuerto")


def fig_clima_vs_retraso(t):
    """Dispersión clima vs retraso medio, un punto por aeropuerto, con recta de ajuste."""
    vars_ = [("lluvia_mm", "Lluvia (mm/h)"), ("visibilidad_m", "Visibilidad (m)"), ("viento_vel", "Viento (m/s)")]
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (c, nombre) in zip(ax, vars_):
        a.scatter(t[c], t["retraso_medio"], color=AZUL, s=40)
        for r in t.itertuples():
            a.annotate(r.iata, (getattr(r, c), r.retraso_medio), fontsize=8, xytext=(3, 3), textcoords="offset points")
        if t[c].nunique() > 1:
            m, b = np.polyfit(t[c], t["retraso_medio"], 1)
            xs = np.linspace(t[c].min(), t[c].max(), 10); a.plot(xs, m * xs + b, color=ROJO)
            a.set_title(f"{nombre}  (Pearson r={t[c].corr(t['retraso_medio']):+.2f})")
        a.set_xlabel(nombre); a.set_ylabel("Retraso medio (min)")
    return _guardar(f, "03_clima_vs_retraso")


def fig_actividad_hora(h, region_de_lat):
    """Salidas por hora del día, total y por región."""
    d = h.dropna(subset=["hora"]).copy()
    d["region"] = d["dep_iata"].map(lambda i: region_de_lat(AEROPUERTOS[i][1]))
    tab = d.groupby(["hora", "region"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
    f, ax = plt.subplots(figsize=(9, 4))
    for col, color in zip(tab.columns, [AZUL, NARANJA, VERDE]):
        ax.plot(tab.index, tab[col], marker="o", ms=3, color=color, label=col)
    ax.set_xlabel("Hora del día"); ax.set_ylabel("Salidas programadas"); ax.legend(title="Región")
    ax.set_title("Actividad aérea durante el día"); ax.set_xticks(range(0, 24, 2))
    return _guardar(f, "04_actividad_por_hora")


def fig_mapa(v, ok, la, lo):
    """Mapa simple: vuelos (rojo = atípico), aeropuertos y rutas simuladas."""
    f, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(la)):
        ax.plot(lo[i], la[i], color=GRIS, lw=.7, alpha=.5)
    atip = (v.get("atip_speed", 0) == 1) | (v.get("atip_alt", 0) == 1)
    ax.scatter(v.loc[~atip, "lng"], v.loc[~atip, "lat"], s=6, color=AZUL, label="Vuelos activos")
    ax.scatter(v.loc[atip, "lng"], v.loc[atip, "lat"], s=40, color=ROJO, marker="x", label="Atípicos (velocidad/altitud)")
    for i, (n, la_, lo_) in AEROPUERTOS.items():
        ax.scatter(lo_, la_, s=60, color=NARANJA, edgecolor="k", zorder=3); ax.annotate(i, (lo_, la_), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.plot([], [], color=GRIS, label="Rutas simuladas (GPU)")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.legend(loc="lower left"); ax.set_title("Vuelos, atípicos y rutas simuladas")
    return _guardar(f, "05_mapa_vuelos")


def fig_thingspeak(ts):
    """Serie de tiempo de las variables del sensor ThingSpeak."""
    cols = [c for c in ts.columns if c not in ("created_at", "fuente") and ts[c].notna().any()][:4]
    f, ax = plt.subplots(len(cols), 1, figsize=(10, 2 * len(cols) + 1), sharex=True)
    for a, c, color in zip(np.atleast_1d(ax), cols, [AZUL, NARANJA, VERDE, ROJO]):
        a.plot(ts["created_at"], ts[c], color=color, lw=1); a.set_ylabel(c[:18], fontsize=8)
    np.atleast_1d(ax)[0].set_title(f"Sensor ThingSpeak ({ts['fuente'].iloc[0]})")
    return _guardar(f, "06_thingspeak")


def fig_correlacion(corr):
    """Mapa de calor de correlaciones (divergente: azul negativo, rojo positivo)."""
    f, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index); ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    f.colorbar(im, label="Spearman"); ax.set_title("Correlación clima / calidad del aire / retrasos")
    return _guardar(f, "07_correlacion")


def fig_rendimiento(tiempos_mpi, bench, t_async):
    """Resumen de paralelismo: asyncio, MPI y GPU."""
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(["Secuencial", "asyncio"], t_async, color=[GRIS, AZUL]); ax[0].set_ylabel("segundos")
    ax[0].set_title("12 peticiones simuladas (0.5 s c/u)")
    ks = sorted(tiempos_mpi, key=int)
    ax[1].bar([f"{k} proc." for k in ks], [tiempos_mpi[k] for k in ks], color=NARANJA)
    ax[1].set_title("Bootstrap con MPI"); ax[1].set_ylabel("segundos")
    if bench.get("gpu_disponible"):
        ax[2].bar(["CPU NumPy", "GPU (con copias)", "GPU (solo kernel)"],
                  [bench["cpu_s"], bench["gpu_total_s"], bench["gpu_kernel_s"]], color=[GRIS, VERDE, AZUL])
        ax[2].set_yscale("log"); ax[2].set_title(f"Haversine, {bench['n']:,} pares"); ax[2].set_ylabel("segundos (log)")
    else:
        ax[2].text(.5, .5, "Sin GPU en este entorno", ha="center"); ax[2].set_axis_off()
    return _guardar(f, "08_rendimiento")
