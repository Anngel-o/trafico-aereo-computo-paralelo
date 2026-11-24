"""
verificar_claves.py
===================
AUDITORÍA de las API keys antes de usarlas.

Problema que resuelve: si una llave está mal copiada (espacio extra, letra
faltante), es de otro servicio o todavía no se activó, las APIs responden con
error y el sistema caería en silencio a datos simulados. Este módulo lo detecta
y lo dice claramente, sin imprimir nunca la llave completa.

Para CADA servicio hace 3 comprobaciones:
  1. EXISTE   -> la llave está en Secretos de Colab (o variable de entorno).
  2. FORMATO  -> tiene la forma esperada (ej. OpenWeather = 32 caracteres
                 hexadecimales; AirLabs = UUID). Un formato raro suele ser una
                 llave copiada incompleta o de otro servicio.
  3. FUNCIONA -> se hace UNA petición real y barata a la API y se interpreta la
                 respuesta (200 = válida, 401/403 = rechazada, 429 = límite...).

ThingSpeak se prueba SIN llave porque el canal 12397 es público (ver config.py).
"""
import asyncio
import re

import aiohttp
import pandas as pd

from config import THINGSPEAK_CANAL, obtener_clave

# nombre del secreto -> (servicio, patrón esperado, descripción del formato)
SERVICIOS = {
    "AIRLABS_KEY": ("AirLabs", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
                    "UUID (8-4-4-4-12 hexadecimales)"),
    "OPENWEATHER_KEY": ("OpenWeather", re.compile(r"^[0-9a-f]{32}$", re.I), "32 caracteres hexadecimales"),
    "OPENAQ_KEY": ("OpenAQ", re.compile(r"^[0-9a-f]{64}$", re.I), "64 caracteres hexadecimales"),
}


def enmascarar(clave):
    """Muestra solo el inicio y el final de la llave para poder compararla sin exponerla."""
    if not clave:
        return "-"
    return f"{clave[:4]}…{clave[-2:]} ({len(clave)} car.)" if len(clave) > 8 else "(muy corta)"


def revisar_formato(nombre, cruda):
    """
    Devuelve (clave_limpia, formato_ok, observaciones).
    Detecta espacios/saltos de línea/comillas invisibles y formato inesperado.
    """
    obs = []
    if not cruda:
        return None, False, ["no está configurada"]
    limpia = cruda.strip().strip("\"'").strip()
    if limpia != cruda:
        obs.append("tenía espacios/saltos de línea/comillas (se limpian automáticamente)")
    _, patron, descripcion = SERVICIOS[nombre]
    ok = bool(patron.match(limpia))
    if not ok:
        obs.append(f"formato inesperado: se esperaba {descripcion}")
    return limpia, ok, obs


# ---------------------------------------------------------------------------
# Interpretación de respuestas HTTP (función pura -> se puede probar sin red)
# ---------------------------------------------------------------------------
def clasificar(estado_http, cuerpo=None):
    """
    Traduce la respuesta de una API a (estado, detalle).
    estado_http: código HTTP (None si no hubo conexión). cuerpo: JSON si lo hay.
    """
    if estado_http is None:
        return "SIN CONEXIÓN", "no se pudo contactar a la API"
    if isinstance(cuerpo, dict) and cuerpo.get("error"):       # AirLabs: HTTP 200 + "error"
        err = cuerpo["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        return "INVÁLIDA", f"la API la rechazó: {msg}"
    if estado_http == 200:
        return "OK", "la API la aceptó"
    if estado_http in (401, 403):
        return "INVÁLIDA", f"HTTP {estado_http}: llave incorrecta, mal copiada o aún sin activar"
    if estado_http == 429:
        return "LÍMITE", "HTTP 429: se agotó la cuota/límite de peticiones (la llave sí es válida)"
    return "ERROR", f"HTTP {estado_http}"


async def _get(sesion, url, params=None, headers=None):
    """GET que devuelve (código_http, json_o_None) sin lanzar excepciones."""
    try:
        async with sesion.get(url, params=params, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
            try:
                cuerpo = await r.json(content_type=None)
            except Exception:
                cuerpo = None
            return r.status, cuerpo
    except Exception:
        return None, None


async def _probar(sesion, nombre, clave):
    """Una petición real y barata a la API del servicio."""
    if nombre == "AIRLABS_KEY":
        return await _get(sesion, "https://airlabs.co/api/v9/airports", {"api_key": clave, "iata_code": "CUU"})
    if nombre == "OPENWEATHER_KEY":
        return await _get(sesion, "https://api.openweathermap.org/data/2.5/weather",
                          {"lat": 28.70, "lon": -105.96, "appid": clave})
    if nombre == "OPENAQ_KEY":
        return await _get(sesion, "https://api.openaq.org/v3/locations", {"limit": 1}, {"X-API-Key": clave})


async def verificar_todo(mostrar=True):
    """
    Revisa las 3 llaves + el canal público de ThingSpeak.
    Devuelve un DataFrame con: servicio, llave (enmascarada), formato, estado, detalle.
    """
    filas = []
    async with aiohttp.ClientSession() as s:
        for nombre, (servicio, _, _) in SERVICIOS.items():
            cruda = obtener_clave(nombre, limpiar=False)
            limpia, formato_ok, obs = revisar_formato(nombre, cruda)
            if not limpia:
                filas.append(dict(servicio=servicio, secreto=nombre, llave="-", formato="-",
                                  estado="SIN KEY", detalle="no configurada -> esta fuente usará datos SIMULADOS"))
                continue
            http, cuerpo = await _probar(s, nombre, limpia)
            estado, detalle = clasificar(http, cuerpo)
            if obs:
                detalle += " | " + "; ".join(obs)
            filas.append(dict(servicio=servicio, secreto=nombre, llave=enmascarar(limpia),
                              formato="correcto" if formato_ok else "REVISAR", estado=estado, detalle=detalle))

        # ThingSpeak: canal público, sin llave
        http, cuerpo = await _get(s, f"https://api.thingspeak.com/channels/{THINGSPEAK_CANAL}/feeds.json", {"results": 1})
        estado, detalle = clasificar(http, cuerpo)
        if estado == "OK" and isinstance(cuerpo, dict):
            detalle = f"canal público «{cuerpo.get('channel', {}).get('name', '?')}» accesible sin llave"
        filas.append(dict(servicio="ThingSpeak", secreto="(sin llave: canal público 12397)", llave="no aplica",
                          formato="no aplica", estado=estado, detalle=detalle))

    df = pd.DataFrame(filas)
    if mostrar:
        icono = {"OK": "✅", "INVÁLIDA": "❌", "SIN KEY": "⚪", "LÍMITE": "⚠️", "SIN CONEXIÓN": "🔌", "ERROR": "❌"}
        print("VERIFICACIÓN DE API KEYS")
        for r in df.itertuples():
            print(f" {icono.get(r.estado, '?')} {r.servicio:11s} | llave: {r.llave:22s} | formato: {r.formato:9s} | {r.estado}: {r.detalle}")
        malas = df[df.estado.isin(["INVÁLIDA", "ERROR"])]
        if len(malas):
            print("\n -> Revisa las llaves marcadas ❌: copia de nuevo la llave completa desde el sitio del servicio, "
                  "sin espacios, y confirma el nombre exacto del secreto y el interruptor 'Acceso desde el cuaderno'.")
    return df


if __name__ == "__main__":
    asyncio.run(verificar_todo())
