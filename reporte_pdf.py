"""
reporte_pdf.py
==============
Genera el REPORTE PDF del proyecto de forma DINÁMICA: todo número, tabla y figura
sale de la ejecución actual. Si cambian las API keys, los datos o el código y
vuelves a ejecutar el notebook, el PDF se actualiza solo.

Se usa la librería ReportLab (Platypus): armamos una lista de "flowables"
(párrafos, tablas, imágenes) y `SimpleDocTemplate.build` los acomoda en páginas.

Entrada: un diccionario `ctx` que arma la última celda del notebook con los
resultados de cada fase (ver la celda "9. Reporte PDF").
"""
import datetime
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

AZUL = colors.HexColor("#0072B2")
REPO = "https://github.com/Anngel-o/trafico-aereo-computo-paralelo"
COLAB = "https://colab.research.google.com/drive/1a8DVBhEwQmpc0ToJ4koVQ1SSI4rfTell?usp=sharing"
EQUIPO = ["Jose Angel Ortiz Meraz (353195)", "Luz Mariam Garcia Castillo (348409)", "Carlos Sánchez Fuentes (368058)"]
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=15, spaceBefore=10, textColor=colors.HexColor("#0b3d6b"))
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=12, spaceBefore=8, textColor=AZUL)
B = ParagraphStyle("B", parent=_ss["BodyText"], fontSize=10, leading=14)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.grey)
CELDA = ParagraphStyle("CELDA", parent=B, fontSize=8.5, leading=10.5)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.8, leading=9.6,
                      backColor=colors.HexColor("#f3f5f7"), borderPadding=4, leftIndent=4)
OUT = ParagraphStyle("OUT", parent=CODE, backColor=colors.HexColor("#eef6ee"))


def _p(texto, estilo=B):
    return Paragraph(texto, estilo)


def _esc(t):
    """Escapa caracteres especiales para que ReportLab no los tome como etiquetas."""
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _codigo(t):
    return [Preformatted(t.strip("\n"), CODE), Spacer(1, 6)]


def _salida(t):
    return [_p("<b>Salida de esta ejecución:</b>", CAP), Preformatted(t.strip("\n"), OUT), Spacer(1, 6)]


def _figura(ruta, pie, ancho_cm=16.5):
    """Inserta una imagen conservando su proporción; si no existe, lo avisa."""
    if not os.path.exists(ruta):
        return [_p(f"(figura no disponible: {ruta})", CAP)]
    im = PILImage.open(ruta)
    return [Image(ruta, width=ancho_cm * cm, height=ancho_cm * cm * im.height / im.width),
            _p(pie, CAP), Spacer(1, 6)]


def _tabla(filas, anchos=None):
    """Tabla con encabezado azul. Las celdas de texto largo se envuelven en Paragraph."""
    filas = [[c if i == 0 else (_p(_esc(c), CELDA) if isinstance(c, str) and len(c) > 28 else c)
              for c in fila] for i, fila in enumerate(filas)]
    t = Table(filas, colWidths=anchos, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f7")])]))
    return [t, Spacer(1, 8)]


def _f(x, d=2):
    return f"{x:.{d}f}"


def generar_reporte(ctx, ruta="Reporte_Proyecto_CPD_Trafico_Aereo.pdf", dir_figuras="figuras"):
    """
    Arma el PDF completo. `ctx` debe traer:
      datos (dict de DataFrames), vuelos, horarios, tabla, reg (resumen por región),
      mpi_df, tiempos_mpi, bench, t_async (t_secuencial, t_asyncio), respuestas,
      atipicos (dict), ingesta_s, gpu_nombre, rutas_shape.
    Devuelve la ruta del PDF.
    """
    d, h, t = ctx["datos"], ctx["horarios"], ctx["tabla"]
    fuentes = {k: str(v["fuente"].iloc[0]) for k, v in d.items()}
    simuladas = [k for k, f in fuentes.items() if f == "SIMULADO"]
    reales = [k for k, f in fuentes.items() if f != "SIMULADO"]
    tm, bench = ctx["tiempos_mpi"], ctx["bench"]
    t_sec, t_as = ctx["t_async"]
    S = []

    # ---- Portada -----------------------------------------------------------
    S += [_p("Análisis de tráfico aéreo y condiciones ambientales usando sensores y APIs en la nube",
             ParagraphStyle("T", parent=_ss["Title"], fontSize=19, leading=23)),
          _p("Proyecto 3ª Evaluación · Cómputo paralelo y distribuido (CPD)",
             ParagraphStyle("s", parent=B, alignment=1, fontSize=11)),
          _p("<b>Equipo:</b> " + " · ".join(EQUIPO), ParagraphStyle("eq", parent=B, alignment=1, fontSize=9.5)),
          _p(f'<b>Repositorio:</b> <link href="{REPO}" color="blue">{REPO}</link>', ParagraphStyle("rp", parent=B, alignment=1, fontSize=8.5)),
          _p(f'<b>Colab:</b> <link href="{COLAB}" color="blue">{COLAB}</link>', ParagraphStyle("cb", parent=B, alignment=1, fontSize=8)),
          _p(f"Generado automáticamente el {datetime.datetime.now():%Y-%m-%d %H:%M} · "
             f"GPU: {ctx['gpu_nombre']}",
             ParagraphStyle("s2", parent=B, alignment=1, fontSize=9, textColor=colors.grey)),
          Spacer(1, 10)]

    # ---- Aviso dinámico sobre el origen de los datos -------------------------
    if simuladas:
        S += [_p("Aviso importante sobre los datos", H1),
              _p(f"En esta ejecución usaron datos <b>REALES</b>: {', '.join(reales) or 'ninguna fuente'}. "
                 f"Usaron datos <b>SIMULADOS</b> (modo demo, columna <i>fuente = SIMULADO</i>): <b>{', '.join(simuladas)}</b>, "
                 "porque falta su API key o la API no respondió. En los datos simulados el retraso depende por diseño de la lluvia, "
                 "la visibilidad y el viento, así que esas correlaciones <b>validan el pipeline, no son un hallazgo real</b>. "
                 "Al agregar las keys en Secretos de Colab y volver a ejecutar, este reporte se regenera con datos reales.")]
    else:
        S += [_p("Origen de los datos", H1),
              _p("Todas las fuentes de esta ejecución son <b>reales</b> (AirLabs, OpenWeather, OpenAQ y ThingSpeak).")]

    # ---- 1. Objetivo y arquitectura -----------------------------------------
    S += [_p("1. Objetivo y arquitectura", H1),
          _p("Aplicar cómputo paralelo para relacionar el tráfico aéreo en México con las condiciones ambientales. "
             "Arquitectura: <b>fuentes en la nube -> asyncio -> MPI4Py -> CUDA/CuPy/Numba -> dashboard</b>, con Pandas como apoyo.")]
    S += _tabla([["Fase", "Tecnología", "Paralelismo", "Qué hace aquí"],
                 ["1. Ingesta", "asyncio + aiohttp", "Concurrencia (E/S de red)", "Pide a 4 APIs ~40 peticiones a la vez en un solo hilo"],
                 ["2. Limpieza", "pandas", "Vectorización", "Tipos, duplicados, banderas de retraso, unión clima + retrasos"],
                 ["3. Distribución", "mpi4py (OpenMPI)", "Procesos (memoria distribuida)", "Reparte aeropuertos; bootstrap del retraso medio"],
                 ["4. Aceleración", "Numba CUDA + CuPy", "Miles de hilos en GPU", "Distancias, atípicos, rutas simuladas, correlaciones"],
                 ["5. Dashboard", "matplotlib", "-", "Figuras y respuestas a las 5 preguntas"]],
                [2.6 * cm, 3.4 * cm, 4 * cm, 7.4 * cm])

    # ---- 2. Requisitos previos: cuentas, keys y verificación ------------------
    S += [_p("2. Requisitos previos: cuentas y API keys", H1),
          _p("Antes de pedir datos reales hay un paso previo: <b>crear una cuenta en cada servicio y generar su API key</b>. "
             "Las llaves se guardan en Colab (Secretos, con 'Acceso desde el cuaderno' activado) y <b>nunca se escriben en el código</b> ni se suben a GitHub.")]
    S += _tabla([["Servicio", "¿Cuenta y key?", "Cómo se obtiene", "Secreto en Colab"],
                 ["AirLabs", "Sí", "Registro en airlabs.co; la key aparece en el panel de la cuenta (formato UUID)", "AIRLABS_KEY"],
                 ["OpenWeather", "Sí", "Registro en openweathermap.org; API keys en la cuenta (32 hexadecimales). Una key nueva puede tardar un rato en activarse", "OPENWEATHER_KEY"],
                 ["OpenAQ v3", "Sí", "Registro en explore.openaq.org/register; la key está en explore.openaq.org/account y se manda en el header X-API-Key", "OPENAQ_KEY"],
                 ["ThingSpeak", "<b>No</b>", "Canal público 12397 (estación meteorológica de MathWorks): ThingSpeak permite leer canales públicos con un GET sin autenticación y de forma gratuita", "(no aplica)"]],
                [2.4 * cm, 2.4 * cm, 9.4 * cm, 3.2 * cm])
    S += [_p("<b>Por qué ThingSpeak no necesita key:</b> las API keys de ThingSpeak (Read/Write) solo hacen falta para leer un canal <i>privado</i> o para "
             "<i>escribir</i> datos desde sensores propios. Aquí solo se leen datos ambientales públicos (temperatura, humedad, presión, viento), "
             "por eso basta con el número de canal (12397) y la URL <i>https://api.thingspeak.com/channels/12397/feeds.json</i>.")]
    S += [_p("Verificación automática de las keys (módulo verificar_claves.py)", H2),
          _p("Antes de ingestar, el sistema comprueba para cada key que <b>exista</b>, tenga el <b>formato</b> esperado (detecta llaves copiadas incompletas o con espacios) "
             "y que la API la <b>acepte</b> con una petición real de prueba. La llave nunca se imprime completa.")]
    cl = ctx.get("claves")
    if cl is not None:
        S += _tabla([["Servicio", "Llave (enmascarada)", "Formato", "Estado", "Detalle"]] +
                    [[r.servicio, r.llave, r.formato, r.estado, r.detalle] for r in cl.itertuples()],
                    [2.3 * cm, 3.4 * cm, 1.9 * cm, 2.7 * cm, 7.1 * cm])
    else:
        S += [_p("(La verificación no se ejecutó en esta corrida.)")]

    # ---- 3. Fuentes ----------------------------------------------------------
    S += [_p("3. Fuentes de datos de esta ejecución", H1)]
    S += _tabla([["Dato", "Filas", "Fuente"]] + [[k, len(v), fuentes[k]] for k, v in d.items()], [4 * cm, 3 * cm, 8 * cm])

    # ---- 3. Paso a paso -----------------------------------------------------
    S += [_p("4. Ejecución paso a paso", H1)]
    S += [_p("Paso 1 · Ingesta concurrente con asyncio (ingesta_async.py)", H2),
          _p("Pedir datos a una API es esperar a la red. Con <b>async/await</b> y <b>asyncio.gather</b> se lanzan todas las peticiones juntas; "
             "un <b>Semaphore</b> limita cuántas van a la vez y si una API falla se reintenta o se cae a modo demo.")]
    S += _codigo("datos = await ingesta_async.ingestar_todo()")
    S += _salida(f"Ingesta terminada en {_f(ctx['ingesta_s'])} s\n" +
                 "\n".join(f"  {k:11s}: {len(v):5d} filas | fuente = {fuentes[k]}" for k, v in d.items()) +
                 f"\n\nDemostración (12 peticiones de 0.5 s): secuencial {_f(t_sec)} s | asyncio {_f(t_as)} s "
                 f"-> {_f(t_sec / t_as, 1)}x más rápido")

    v, hh = ctx["vuelos"], ctx["horarios"]
    S += [_p("Paso 2 · Limpieza con pandas (limpieza_pandas.py)", H2),
          _p("Conversión a números, eliminación de duplicados y registros sin posición, banderas <i>cancelado</i> y <i>retrasado</i> (>15 min), "
             "hora de salida, región por latitud y unión (<b>groupby + merge</b>) para tener una fila por aeropuerto.")]
    S += _salida(f"Vuelos en el aire: {len(v)} | Salidas programadas: {len(hh)} | Cancelados: {int(hh.cancelado.sum())} "
                 f"| Retrasados >15 min: {int(hh.retrasado.sum())}")
    reg = ctx["reg"].reset_index()
    S += _tabla([["Región", "Vuelos activos", "Velocidad media (km/h)", "Altitud media (m)"]] +
                [[str(r.region), int(r.vuelos), _f(r.velocidad_media_kmh, 1), _f(r.altitud_media_m, 1)] for r in reg.itertuples()],
                [4 * cm, 3.5 * cm, 4.5 * cm, 4 * cm])

    S += [_p("Paso 3 · Distribución con MPI4Py (mpi_analisis.py)", H2),
          _p("<b>mpirun</b> lanza varias copias del script (<i>ranks</i>, cada una con su memoria). Rank 0 hace <b>bcast</b> del parámetro, "
             "<b>scatter</b> de los aeropuertos, cada proceso calcula un <b>bootstrap</b> del retraso medio y rank 0 junta con <b>gather</b> y <b>Reduce</b>.")]
    ks = sorted(tm, key=int)
    S += _tabla([["Procesos", "Tiempo total (s)", "Speedup vs 1 proceso"]] +
                [[k, _f(tm[k]), _f(tm["1"] / tm[k]) + "x"] for k in ks], [4 * cm, 4 * cm, 5 * cm])
    S += [_p("Colab ofrece pocos CPU virtuales (normalmente 2): con más procesos que núcleos no hay más aceleración y el arranque de procesos pesa. "
             "Lo importante es el patrón scatter/gather, que en un clúster sí escala.")]
    m = ctx["mpi_df"]
    S += _tabla([["Aeropuerto", "n", "Retraso medio", "IC95 inf", "IC95 sup", "Calculado por rank"]] +
                [[r.iata, int(r.n), _f(r.media, 1), _f(r.ic95_inf, 1), _f(r.ic95_sup, 1), int(r.rank_que_calculo)] for r in m.itertuples()])

    S += [_p("Paso 4 · GPU con Numba CUDA y CuPy (gpu_numba.py)", H2),
          _p("Un <b>kernel</b> es una función que ejecutan miles de hilos a la vez; cada hilo usa <b>cuda.grid()</b> para saber qué dato le toca. "
             "Kernels: <b>k_haversine</b> (distancia km), <b>k_zscore</b> (atípicos con |z|>3) y <b>k_ruta</b> (trayectoria de círculo máximo en malla 2D). "
             "CuPy calcula media, desviación y la matriz de correlación en GPU.")]
    a = ctx["atipicos"]
    S += _salida(f"GPU disponible: {bench['gpu_disponible']} ({ctx['gpu_nombre']})\n"
                 f"Atípicos de velocidad: {a['speed']} | de altitud: {a['alt']} | de retraso: {a['retraso']}\n"
                 f"Rutas simuladas: {ctx['rutas_shape']} (vuelos x pasos)")
    if bench["gpu_disponible"]:
        S += _tabla([[f"Haversine con {bench['n']:,} pares", "Tiempo (s)", "Speedup"],
                     ["CPU (NumPy)", _f(bench["cpu_s"], 4), "1.0x"],
                     ["GPU incluyendo copias CPU<->GPU", _f(bench["gpu_total_s"], 4), _f(bench["speedup_total"], 1) + "x"],
                     ["GPU solo kernel", _f(bench["gpu_kernel_s"], 4), _f(bench["speedup_kernel"], 1) + "x"]],
                    [8 * cm, 3.5 * cm, 3.5 * cm])
        S += [_p(f"Error máximo CPU vs GPU: {bench['max_error_km'] * 1000:.1f} m (la GPU usa float32). Las copias de memoria CPU<->GPU "
                 "cuestan tiempo, por eso la GPU conviene con muchos datos; la primera llamada compila el kernel (JIT), por eso hay un calentamiento.")]
    else:
        S += [_p("Esta ejecución no tuvo GPU: se usó NumPy. Activa T4 GPU en el tipo de entorno para ver la aceleración.")]
    S += _figura(f"{dir_figuras}/08_rendimiento.png", "Figura 1. Paralelismo en las tres capas: asyncio, MPI y GPU.")

    S += [_p("Paso 5 · Visualización", H2)]
    for n, pie, w in [("01_actividad_vuelos", "Figura 2. Vuelos por región, aerolínea y rutas más frecuentes.", 16.5),
                      ("02_retrasos_aeropuerto", "Figura 3. Retraso medio por aeropuerto con IC 95% (MPI).", 12),
                      ("05_mapa_vuelos", "Figura 4. Vuelos, atípicos (X rojas, GPU) y rutas simuladas (kernel CUDA).", 13),
                      ("04_actividad_por_hora", "Figura 5. Salidas por hora del día y región.", 13),
                      ("03_clima_vs_retraso", "Figura 6. Retraso medio contra lluvia, visibilidad y viento.", 16.5),
                      ("07_correlacion", "Figura 7. Correlación de Spearman (cp.corrcoef en GPU).", 12),
                      ("06_thingspeak", f"Figura 8. Sensor ThingSpeak ({fuentes['thingspeak']}).", 14)]:
        S += _figura(f"{dir_figuras}/{n}.png", pie, w)

    # ---- 6. Respuestas -------------------------------------------------------
    S += [_p("7. Respuestas a las preguntas de análisis", H1)]
    if simuladas:
        S += [_p("<b>Recordatorio:</b> parte de los datos es simulada; estas respuestas muestran el método, no conclusiones reales.")]
    for r in ctx["respuestas"]:
        S += [_p(_esc(r)), Spacer(1, 4)]
    top = t.sort_values("retraso_medio", ascending=False).head(6)
    S += [_p("Aeropuertos con más retraso:")]
    S += _tabla([["IATA", "Salidas", "Retraso medio (min)", "% >15 min", "Cancelados", "Lluvia (mm/h)", "Visibilidad (m)"]] +
                [[x.iata, int(x.n_vuelos), _f(x.retraso_medio, 1), _f(x.pct_retrasados, 0), int(x.cancelados),
                  _f(x.lluvia_mm, 1), _f(x.visibilidad_m, 0)] for x in top.itertuples()])

    # ---- 7. Conclusiones ----------------------------------------------------
    sp_mpi = max(tm["1"] / tm[k] for k in tm)
    S += [_p("8. Conclusiones y limitaciones", H1),
          _p(f"&bull; <b>asyncio</b>: {_f(t_sec / t_as, 1)}x menos espera en la demostración, porque las peticiones se traslapan.<br/>"
             f"&bull; <b>MPI</b>: mejor aceleración medida {_f(sp_mpi)}x, limitada por los núcleos de Colab y la poca carga.<br/>"
             + (f"&bull; <b>GPU</b>: {_f(bench['speedup_total'], 1)}x con copias y {_f(bench['speedup_kernel'], 1)}x solo cálculo con {bench['n']:,} datos.<br/>"
                if bench["gpu_disponible"] else "&bull; <b>GPU</b>: no disponible en esta ejecución.<br/>") +
             f"&bull; Con {len(t)} aeropuertos las correlaciones tienen poca potencia estadística: son indicios, no causalidad. El clima es el del "
             "momento de la consulta, no el de cada retraso.<br/>"
             "&bull; El plan gratuito de AirLabs limita campos y consultas; conviene guardar y acumular datos en el tiempo.")]
    S += [_p("9. Glosario de conceptos clave", H1),
          _p("<b>asyncio:</b> concurrencia (no paralelismo) para esperar a la red. <b>MPI:</b> procesos con memoria separada que se mandan mensajes (scatter, gather, Reduce). "
             "<b>CUDA:</b> kernel = función que corren miles de hilos; cuda.grid() da el índice; hay que copiar datos CPU<->GPU. "
             "<b>Atípico (z-score):</b> a más de 3 desviaciones estándar de la media. <b>Bootstrap:</b> remuestrear con reemplazo para estimar incertidumbre. "
             "<b>Haversine:</b> distancia entre dos puntos de una esfera.")]

    SimpleDocTemplate(ruta, pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Análisis de tráfico aéreo y condiciones ambientales").build(S)
    return ruta
