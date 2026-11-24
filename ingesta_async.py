"""
ingesta_async.py
================
FASE 1 de la arquitectura: "Python + Asyncio -> lectura concurrente de APIs".

Concepto clave para exponer
---------------------------
Pedir datos a una API es tiempo de ESPERA (la red tarda), no de cálculo.
Con código normal (secuencial) esperamos una respuesta antes de mandar la
siguiente petición. Con `asyncio` mandamos muchas peticiones "al mismo tiempo"
y mientras una espera, el programa atiende a las otras. Es concurrencia
(no paralelismo de CPU): un solo hilo, muchas esperas traslapadas.

Piezas que usamos:
  * `async def`      -> define una "corrutina" (función que puede pausarse).
  * `await`          -> "pausa aquí hasta que llegue la respuesta, y mientras
                         tanto deja que otras corrutinas avancen".
  * `asyncio.gather` -> lanza varias corrutinas juntas y junta sus resultados.
  * `Semaphore`      -> límite de peticiones simultáneas (para no saturar ni
                         violar el rate-limit de las APIs gratuitas).
  * `aiohttp`        -> cliente HTTP asíncrono (equivalente async de `requests`).
"""
import asyncio
import time

import aiohttp
import pandas as pd

import datos_demo
from config import (AEROPUERTOS, BBOX_MEXICO, THINGSPEAK_CANAL, obtener_clave)

AIRLABS = "https://airlabs.co/api/v9"
OWM = "https://api.openweathermap.org/data/2.5/weather"
OPENAQ = "https://api.openaq.org/v3"
THINGSPEAK = "https://api.thingspeak.com/channels"
ULTIMA_INGESTA_S = None          # segundos que tardó la última ingesta (lo usa el reporte PDF)


# ---------------------------------------------------------------------------
# Función base: UNA petición GET con límite de concurrencia y reintentos
# ---------------------------------------------------------------------------
async def pedir_json(sesion, url, params=None, headers=None, sem=None, reintentos=2):
    """
    Hace GET a `url` y devuelve el JSON (o None si falla).

    - `sem` (Semaphore): `async with sem` deja pasar solo N peticiones a la vez.
    - Reintentamos con una pausa creciente (backoff) si la red falla.
    - Devolver None (en vez de lanzar error) permite que UNA API caída no
      tumbe todo el pipeline: después caemos a modo demo para esa fuente.
    """
    for intento in range(reintentos + 1):
        try:
            async with sem:
                async with sesion.get(url, params=params, headers=headers,
                                      timeout=aiohttp.ClientTimeout(total=25)) as r:
                    if r.status == 200:
                        datos = await r.json(content_type=None)
                        # AirLabs (y otras) a veces responden HTTP 200 pero con un
                        # campo "error" (ej. llave incorrecta): lo tratamos como fallo.
                        if isinstance(datos, dict) and datos.get("error"):
                            print(f"  ! {url} -> la API rechazó la petición: {datos['error']}")
                            return None
                        return datos
                    print(f"  ! {url.split('?')[0]} -> HTTP {r.status}"
                          + (" (llave rechazada o sin activar)" if r.status in (401, 403) else ""))
                    if r.status in (401, 403, 404):    # reintentar no ayuda con estos errores
                        return None
        except Exception as e:                      # timeout, DNS, etc.
            print(f"  ! error de red ({type(e).__name__}) en {url}")
        await asyncio.sleep(1.5 * (intento + 1))    # backoff antes de reintentar
    return None


# ---------------------------------------------------------------------------
# AirLabs: vuelos en el aire y horarios (retrasos)
# ---------------------------------------------------------------------------
async def obtener_vuelos(sesion, sem, clave):
    """Todos los vuelos activos dentro de la caja de México (endpoint /flights)."""
    bbox = ",".join(str(x) for x in BBOX_MEXICO)
    datos = await pedir_json(sesion, f"{AIRLABS}/flights",
                             {"api_key": clave, "bbox": bbox}, sem=sem)
    if not datos or "response" not in datos:
        return None
    df = pd.DataFrame(datos["response"])
    df["fuente"] = "AirLabs"
    return df


async def obtener_horarios(sesion, sem, clave, iata):
    """Salidas programadas de UN aeropuerto con retraso en minutos (/schedules)."""
    datos = await pedir_json(sesion, f"{AIRLABS}/schedules",
                             {"api_key": clave, "dep_iata": iata, "limit": 50}, sem=sem)
    if not datos or "response" not in datos:
        return None
    df = pd.DataFrame(datos["response"])
    df["fuente"] = "AirLabs"
    return df


# ---------------------------------------------------------------------------
# OpenWeather: clima actual en cada aeropuerto
# ---------------------------------------------------------------------------
async def obtener_clima(sesion, sem, clave, iata, lat, lon):
    """Clima actual en las coordenadas del aeropuerto (unidades métricas)."""
    d = await pedir_json(sesion, OWM,
                         {"lat": lat, "lon": lon, "appid": clave, "units": "metric"}, sem=sem)
    if not d:
        return None
    return dict(
        iata=iata, temp=d["main"]["temp"], humedad=d["main"]["humidity"],
        presion=d["main"]["pressure"], viento_vel=d["wind"].get("speed", 0),
        viento_dir=d["wind"].get("deg", 0), visibilidad_m=d.get("visibility", 10000),
        lluvia_mm=d.get("rain", {}).get("1h", 0.0),       # el campo solo existe si llueve
        nubosidad=d["clouds"]["all"], condicion=d["weather"][0]["main"], fuente="OpenWeather")


# ---------------------------------------------------------------------------
# OpenAQ: calidad del aire (PM2.5 / PM10) cerca del aeropuerto
# ---------------------------------------------------------------------------
async def obtener_calidad_aire(sesion, sem, clave, iata, lat, lon, radio_m=25000):
    """
    1) Busca estaciones a `radio_m` metros del aeropuerto (/locations).
    2) Pide la última medición de la más cercana que mida PM2.5 (/latest).
    """
    h = {"X-API-Key": clave}                              # OpenAQ v3 usa este header
    locs = await pedir_json(sesion, f"{OPENAQ}/locations",
                            {"coordinates": f"{lat},{lon}", "radius": radio_m, "limit": 10},
                            headers=h, sem=sem)
    if not locs or not locs.get("results"):
        return None
    for loc in locs["results"]:                           # ya vienen ordenadas por cercanía
        sensores = {s["id"]: s["parameter"]["name"] for s in loc.get("sensors", [])}
        if "pm25" not in sensores.values():
            continue
        ult = await pedir_json(sesion, f"{OPENAQ}/locations/{loc['id']}/latest",
                               headers=h, sem=sem)
        if not ult:
            continue
        medidas = {sensores.get(m["sensorsId"]): m["value"] for m in ult.get("results", [])}
        return dict(iata=iata, pm25=medidas.get("pm25"), pm10=medidas.get("pm10"),
                    estacion=loc.get("name"), distancia_m=loc.get("distance"),
                    fuente="OpenAQ")
    return None


# ---------------------------------------------------------------------------
# ThingSpeak: sensores ambientales (canal público, sin API key)
# ---------------------------------------------------------------------------
async def obtener_thingspeak(sesion, sem, canal=THINGSPEAK_CANAL, resultados=500):
    """
    Lee las últimas `resultados` lecturas del canal. Los nombres de los campos
    (field1..field8) se leen del propio canal, no los suponemos.

    NOTA: aquí NO se manda ninguna API key. El canal 12397 es PÚBLICO (estación
    meteorológica de MathWorks) y ThingSpeak permite leer canales públicos con un
    GET sin autenticación. La key de ThingSpeak solo se necesita para canales
    privados o para escribir datos (ver config.py).
    """
    d = await pedir_json(sesion, f"{THINGSPEAK}/{canal}/feeds.json",
                         {"results": resultados}, sem=sem)
    if not d:
        return None
    nombres = {f"field{i}": d["channel"].get(f"field{i}") for i in range(1, 9)
               if d["channel"].get(f"field{i}")}
    df = pd.DataFrame(d["feeds"]).rename(columns=nombres)
    df["created_at"] = pd.to_datetime(df["created_at"])
    for col in nombres.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fuente"] = f"ThingSpeak-{canal}"
    return df.drop(columns=["entry_id"], errors="ignore")


# ---------------------------------------------------------------------------
# Orquestador: lanza TODAS las peticiones a la vez con asyncio.gather
# ---------------------------------------------------------------------------
async def ingestar_todo(max_concurrentes=8):
    """
    Descarga todo en paralelo (concurrente) y devuelve un diccionario de
    DataFrames: vuelos, horarios, clima, aire, thingspeak.
    Si falta una API key o la API falla -> usa datos DEMO de esa fuente.
    """
    global ULTIMA_INGESTA_S
    t0 = time.perf_counter()
    k_air, k_owm, k_oaq = (obtener_clave(n) for n in
                           ("AIRLABS_KEY", "OPENWEATHER_KEY", "OPENAQ_KEY"))
    sem = asyncio.Semaphore(max_concurrentes)
    async with aiohttp.ClientSession() as s:
        # Se arman las "tareas" (todavía no se ejecutan)...
        t_ts = obtener_thingspeak(s, sem)
        t_vuelos = obtener_vuelos(s, sem, k_air) if k_air else asyncio.sleep(0)
        t_hor = [obtener_horarios(s, sem, k_air, i) if k_air else asyncio.sleep(0)
                 for i in AEROPUERTOS]
        t_clima = [obtener_clima(s, sem, k_owm, i, la, lo) if k_owm else asyncio.sleep(0)
                   for i, (_, la, lo) in AEROPUERTOS.items()]
        t_aire = [obtener_calidad_aire(s, sem, k_oaq, i, la, lo) if k_oaq else asyncio.sleep(0)
                  for i, (_, la, lo) in AEROPUERTOS.items()]
        # ...y aquí se lanzan TODAS juntas. `await` espera a que terminen todas.
        ts, vuelos, *resto = await asyncio.gather(t_ts, t_vuelos, *t_hor, *t_clima, *t_aire)

    n = len(AEROPUERTOS)
    hor, clima, aire = resto[:n], resto[n:2 * n], resto[2 * n:]

    # Armamos cada DataFrame; si no hay datos reales, caemos a demo.
    hor = [h for h in hor if isinstance(h, pd.DataFrame)]
    horarios = pd.concat(hor, ignore_index=True) if hor else None
    clima = [c for c in clima if isinstance(c, dict)]
    clima = pd.DataFrame(clima) if clima else None
    aire = [a for a in aire if isinstance(a, dict)]
    aire = pd.DataFrame(aire) if aire else None
    vuelos = vuelos if isinstance(vuelos, pd.DataFrame) else None

    clima_final = clima if clima is not None else datos_demo.clima_demo()
    out = dict(
        vuelos=vuelos if vuelos is not None else datos_demo.vuelos_demo(),
        horarios=horarios if horarios is not None else datos_demo.horarios_demo(clima_final),
        clima=clima_final,
        aire=aire if aire is not None else datos_demo.calidad_aire_demo(),
        thingspeak=ts if ts is not None else datos_demo.thingspeak_demo(),
    )
    ULTIMA_INGESTA_S = time.perf_counter() - t0
    print(f"Ingesta terminada en {ULTIMA_INGESTA_S:.2f} s")
    for nombre, df in out.items():
        if df is None:
            print(f"  {nombre:11s}: SIN DATOS")
        else:
            print(f"  {nombre:11s}: {len(df):5d} filas | fuente = {df['fuente'].iloc[0]}")
    return out


# ---------------------------------------------------------------------------
# Demostración didáctica: secuencial vs asyncio (sin gastar cuota de APIs)
# ---------------------------------------------------------------------------
async def demo_asyncio_vs_secuencial(n_peticiones=12, latencia=0.5):
    """
    Simula `n_peticiones` que tardan `latencia` s cada una (asyncio.sleep imita
    la espera de red). Secuencial ~ n*latencia; asyncio ~ latencia.
    Devuelve (t_secuencial, t_asyncio).
    """
    t0 = time.perf_counter()
    for _ in range(n_peticiones):
        await asyncio.sleep(latencia)               # una tras otra
    t_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[asyncio.sleep(latencia) for _ in range(n_peticiones)])
    t_async = time.perf_counter() - t0              # todas a la vez
    return t_sec, t_async


if __name__ == "__main__":
    # Fuera de un notebook se usa asyncio.run; en Colab basta `await ingestar_todo()`.
    datos = asyncio.run(ingestar_todo())
