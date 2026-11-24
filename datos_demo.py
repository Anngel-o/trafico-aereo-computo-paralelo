"""
datos_demo.py
=============
Generador de datos SIMULADOS (modo demo).

¿Para qué sirve? Si todavía no tienes una API key (AirLabs, OpenWeather,
OpenAQ) o la cuota se acabó, el proyecto sigue corriendo con datos sintéticos
que tienen EXACTAMENTE el mismo formato (columnas) que los reales.

IMPORTANTE (honestidad en el reporte): estos datos se marcan con la columna
`fuente = "SIMULADO"`. Las relaciones clima->retraso que aparecen aquí las
pusimos nosotros a propósito, así que NO son un hallazgo científico; sirven
para probar el pipeline. Los resultados reales salen al usar las API keys.
"""
import numpy as np
import pandas as pd
from config import AEROPUERTOS, BBOX_MEXICO

RNG = np.random.default_rng(42)       # semilla fija -> resultados reproducibles
AEROLINEAS = ["AM", "Y4", "VB", "VW", "AA", "UA", "DL", "AV"]


def clima_demo():
    """Un renglón de clima por aeropuerto (mismas columnas que OpenWeather)."""
    filas = []
    for iata, (_, lat, lon) in AEROPUERTOS.items():
        lluvia = max(0.0, RNG.normal(0.4, 1.2))               # mm en la última hora
        filas.append(dict(
            iata=iata, temp=RNG.normal(26, 6), humedad=RNG.uniform(25, 90),
            presion=RNG.normal(1012, 5), viento_vel=abs(RNG.normal(4, 2.5)),
            viento_dir=RNG.uniform(0, 360),
            visibilidad_m=float(np.clip(10000 - lluvia * 1500 + RNG.normal(0, 800), 800, 10000)),
            lluvia_mm=lluvia, nubosidad=RNG.uniform(0, 100),
            condicion="Rain" if lluvia > 0.5 else "Clear", fuente="SIMULADO"))
    return pd.DataFrame(filas)


def calidad_aire_demo():
    """PM2.5 y PM10 por aeropuerto (mismas columnas que OpenAQ)."""
    filas = []
    for iata in AEROPUERTOS:
        pm25 = abs(RNG.normal(18, 9))
        filas.append(dict(iata=iata, pm25=pm25, pm10=pm25 * RNG.uniform(1.4, 2.2),
                          estacion="demo", distancia_m=RNG.uniform(1000, 20000),
                          fuente="SIMULADO"))
    return pd.DataFrame(filas)


def horarios_demo(clima, n_por_aeropuerto=50):
    """
    Salidas programadas con retrasos. El retraso depende (a propósito) de la
    lluvia, la visibilidad y el viento del aeropuerto + ruido aleatorio.
    """
    filas = []
    for _, c in clima.iterrows():
        base = (3 + 12 * c.lluvia_mm + 0.004 * (10000 - c.visibilidad_m)
                + 1.5 * c.viento_vel)
        for _ in range(n_por_aeropuerto):
            hora = int(np.clip(RNG.normal(13, 4.5), 0, 23))   # más salidas a mediodía
            retraso = max(0.0, RNG.gamma(1.6, base / 1.6))
            if RNG.random() < 0.01:                            # algunos atípicos extremos
                retraso += RNG.uniform(150, 400)
            cancelado = RNG.random() < 0.015 + 0.01 * c.lluvia_mm
            filas.append(dict(
                flight_iata=f"{RNG.choice(AEROLINEAS)}{RNG.integers(100, 999)}",
                airline_iata=RNG.choice(AEROLINEAS), dep_iata=c.iata,
                arr_iata=RNG.choice([a for a in AEROPUERTOS if a != c.iata]),
                status="cancelled" if cancelado else "scheduled",
                dep_delayed=np.nan if cancelado else retraso, arr_delayed=np.nan,
                dep_time=f"2026-11-23 {hora:02d}:{RNG.integers(0, 59):02d}",
                fuente="SIMULADO"))
    return pd.DataFrame(filas)


def vuelos_demo(n=800):
    """Vuelos 'en el aire' con posición, altitud (m) y velocidad (km/h)."""
    s, w, nn, e = BBOX_MEXICO
    codigos = list(AEROPUERTOS)
    df = pd.DataFrame(dict(
        flight_iata=[f"{RNG.choice(AEROLINEAS)}{RNG.integers(100, 999)}" for _ in range(n)],
        airline_iata=RNG.choice(AEROLINEAS, n),
        dep_iata=RNG.choice(codigos, n), arr_iata=RNG.choice(codigos, n),
        lat=RNG.uniform(s, nn, n), lng=RNG.uniform(w, e, n),
        alt=np.clip(RNG.normal(9500, 2500, n), 300, 13000),
        speed=np.clip(RNG.normal(780, 120, n), 150, 1000),
        dir=RNG.uniform(0, 360, n), v_speed=RNG.normal(0, 3, n),
        status="en-route", aircraft_icao=RNG.choice(["B738", "A320", "A321", "B789"], n),
        fuente="SIMULADO"))
    k = max(3, n // 100)                       # inyectamos atípicos de velocidad/altitud
    idx = RNG.choice(n, k, replace=False)
    df.loc[idx, "speed"] = RNG.uniform(1400, 1800, k)
    df.loc[idx, "alt"] = RNG.uniform(17000, 22000, k)
    return df


def thingspeak_demo(n=500):
    """Serie de tiempo de un sensor ambiental (mismo formato que ThingSpeak)."""
    t = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="1min")
    ciclo = np.sin(np.linspace(0, 6 * np.pi, n))
    return pd.DataFrame(dict(
        created_at=t, temperatura=22 + 4 * ciclo + RNG.normal(0, .3, n),
        humedad=55 - 12 * ciclo + RNG.normal(0, 1, n),
        presion=30.0 + .05 * ciclo + RNG.normal(0, .01, n),
        velocidad_viento=abs(3 + 2 * ciclo + RNG.normal(0, .8, n)),
        fuente="SIMULADO"))
