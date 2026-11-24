"""
config.py
=========
Configuración central del proyecto "Análisis de tráfico aéreo y condiciones
ambientales" (CPD - 3ª Evaluación).

Aquí vive TODO lo que se comparte entre módulos:
  * la lista de aeropuertos que vamos a analizar (con sus coordenadas),
  * la caja geográfica (bounding box) de México para pedir vuelos a AirLabs,
  * la función que obtiene las API keys SIN escribirlas en el código.

Idea para exponer: separar la configuración del código evita repetir datos
y evita subir llaves secretas a GitHub.

REQUISITOS PREVIOS (cuentas y API keys)
---------------------------------------
Antes de ejecutar con datos reales hay que crear cuentas y generar llaves:

  * AirLabs      -> cuenta en airlabs.co            -> AIRLABS_KEY      (formato UUID)
  * OpenWeather  -> cuenta en openweathermap.org    -> OPENWEATHER_KEY  (32 hexadecimales;
                    las llaves nuevas pueden tardar un rato en activarse)
  * OpenAQ v3    -> registro en explore.openaq.org/register; la llave se ve en
                    explore.openaq.org/account     -> OPENAQ_KEY
  * ThingSpeak   -> NO necesita cuenta ni llave aquí (ver nota abajo).

Las llaves se guardan en Colab -> candado 🔑 (Secretos), con el interruptor
"Acceso desde el cuaderno" ACTIVADO. Nunca se escriben en el código.
El módulo verificar_claves.py comprueba que cada llave exista, tenga el formato
esperado y sea aceptada por su API.
"""
import os

# ---------------------------------------------------------------------------
# 1) Aeropuertos de México que analizaremos.
#    Formato:  código IATA -> (nombre, latitud, longitud)
#    Las coordenadas las usamos para pedirle a OpenWeather/OpenAQ el clima y
#    la calidad del aire justo en el aeropuerto.
# ---------------------------------------------------------------------------
AEROPUERTOS = {
    "MEX": ("CDMX Benito Juárez", 19.4363, -99.0721),
    "CUN": ("Cancún",             21.0365, -86.8771),
    "GDL": ("Guadalajara",        20.5218, -103.3112),
    "MTY": ("Monterrey",          25.7785, -100.1069),
    "TIJ": ("Tijuana",            32.5411, -116.9700),
    "CUU": ("Chihuahua",          28.7029, -105.9647),
    "CJS": ("Ciudad Juárez",      31.6361, -106.4287),
    "PVR": ("Puerto Vallarta",    20.6801, -105.2544),
    "SJD": ("Los Cabos",          23.1518, -109.7215),
    "MID": ("Mérida",             20.9370, -89.6577),
    "BJX": ("Del Bajío",          20.9935, -101.4808),
    "HMO": ("Hermosillo",         29.0959, -111.0479),
}

# Caja que cubre México: (lat_sur, lon_oeste, lat_norte, lon_este).
# AirLabs la pide en ese orden dentro del parámetro "bbox".
BBOX_MEXICO = (14.5, -118.5, 32.8, -86.5)

# Canal PÚBLICO de ThingSpeak: la estación meteorológica de MathWorks.
# ¿Por qué NO hace falta crear cuenta ni generar API key?
#   ThingSpeak organiza los datos en "canales". Un canal marcado como PÚBLICO se
#   puede leer con un simple GET a
#       https://api.thingspeak.com/channels/12397/feeds.json
#   sin autenticación (acceso gratuito de solo lectura). Las "API keys" de
#   ThingSpeak (Read/Write API Key) solo se necesitan para:
#     - leer un canal PRIVADO, o
#     - ESCRIBIR datos (publicar lecturas de nuestros propios sensores).
#   Como aquí solo leemos datos ambientales públicos (temperatura, humedad,
#   presión, viento...), usamos el canal 12397 tal cual.
THINGSPEAK_CANAL = 12397

# Carpetas de salida
DIR_DATOS = "datos"
DIR_RESULTADOS = "resultados"
DIR_FIGURAS = "figuras"


def obtener_clave(nombre, limpiar=True):
    """
    Busca una API key sin escribirla en el código.

    Orden de búsqueda:
      1. Secretos de Google Colab (el candado 🔑 de la barra izquierda).
      2. Variable de entorno (útil si lo corres en tu computadora).
    Devuelve None si no la encuentra -> el sistema entra en MODO DEMO
    para esa fuente (datos simulados, claramente etiquetados).

    limpiar=True quita espacios, saltos de línea y comillas al inicio/final: el
    error más común al copiar y pegar una llave es llevarse un espacio o un
    salto de línea invisible, y eso hace que la API la rechace.
    """
    valor = None
    try:
        from google.colab import userdata          # solo existe en Colab
        valor = userdata.get(nombre)
    except Exception:
        pass                                        # no estamos en Colab o no hay secreto
    if not valor:
        valor = os.environ.get(nombre)
    if valor and limpiar:
        valor = valor.strip().strip("\"'").strip()
    return valor or None
