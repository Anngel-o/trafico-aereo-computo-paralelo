"""Genera Proyecto_CPD_Trafico_Aereo.ipynb a partir de los módulos .py (cada módulo -> celda %%writefile)."""
import nbformat as nbf
MODULOS = ["config", "datos_demo", "ingesta_async", "limpieza_pandas", "mpi_analisis", "gpu_numba", "analisis", "visualizacion", "reporte_pdf", "verificar_claves"]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Análisis de tráfico aéreo y condiciones ambientales con sensores y APIs en la nube
**CPD – Proyecto 3ª Evaluación** · Cómputo paralelo y distribuido

📦 **Repositorio GitHub:** https://github.com/Anngel-o/trafico-aereo-computo-paralelo
📓 **Notebook en Colab:** https://colab.research.google.com/drive/1a8DVBhEwQmpc0ToJ4koVQ1SSI4rfTell?usp=sharing

**Equipo:**
- Jose Angel Ortiz Meraz — 353195
- Luz Mariam Garcia Castillo — 348409
- Carlos Sánchez Fuentes — 368058


Arquitectura (según el enunciado): **APIs en la nube → asyncio → MPI4Py → CUDA/Numba/CuPy → Dashboard**, con **Pandas** como apoyo.

| Fase | Tecnología | Qué hace aquí |
|---|---|---|
| 1 | `asyncio` + `aiohttp` | Descarga concurrente de AirLabs, OpenWeather, OpenAQ y ThingSpeak |
| 2 | `pandas` | Limpieza y unión de tablas |
| 3 | `mpi4py` | Reparte aeropuertos entre procesos (bootstrap del retraso medio) |
| 4 | `numba.cuda` + `cupy` | Distancias, atípicos, rutas simuladas y correlaciones en GPU |
| 5 | `matplotlib` | Gráficas y respuestas a las preguntas de análisis |

**Antes de ejecutar:** Entorno de ejecución → *Cambiar tipo de entorno* → **GPU T4**. Opcional: agrega tus API keys en 🔑 *Secretos* (`AIRLABS_KEY`, `OPENWEATHER_KEY`, `OPENAQ_KEY`). Sin key, esa fuente usa **datos SIMULADOS** (columna `fuente = SIMULADO`)."""))
C.append(md("## 0. Preparación del entorno (MPI + GPU)"))
C.append(code("""# Instala OpenMPI (el "motor" de MPI) y el paquete de Python mpi4py que lo controla
!apt-get install -y -qq libopenmpi-dev openmpi-bin > /dev/null
!pip install -q mpi4py aiohttp
# Verifica que Colab nos dio una GPU NVIDIA
!nvidia-smi -L"""))
C.append(md("## 1. Módulos del proyecto\nCada celda guarda un archivo `.py` (`%%writefile`) para poder importarlo y subirlo tal cual a GitHub."))
for m in MODULOS:
    C.append(code(f"%%writefile {m}.py\n" + open(f"{m}.py").read().rstrip() + "\n"))
C.append(md("""## 1b. Paso previo – Verificación de las API keys
Antes de descargar datos se comprueba cada llave: **existe**, tiene el **formato** correcto y la API la **acepta** (petición real de prueba). La llave nunca se imprime completa. ThingSpeak no necesita llave: el canal 12397 es público."""))
C.append(code("""import verificar_claves
claves = await verificar_claves.verificar_todo()
claves"""))
C.append(md("""## 2. Paso 1 – Ingesta concurrente con `asyncio`
Primero una demostración sencilla del concepto (12 peticiones simuladas de 0.5 s) y luego la descarga real."""))
C.append(code("""import asyncio, os, json, numpy as np, pandas as pd
import ingesta_async, limpieza_pandas as L, analisis as A, gpu_numba as G, visualizacion as V
os.makedirs("datos", exist_ok=True); os.makedirs("resultados", exist_ok=True)

t_sec, t_async = await ingesta_async.demo_asyncio_vs_secuencial()
print(f"Secuencial: {t_sec:.2f} s   |   asyncio: {t_async:.2f} s   ->  {t_sec/t_async:.1f}x más rápido")"""))
C.append(code("""# Descarga REAL de todas las APIs al mismo tiempo (en Colab se puede usar `await` directo)
datos = await ingesta_async.ingestar_todo()
for nombre, df in datos.items():
    df.to_csv(f"datos/{nombre}.csv", index=False)   # los guardamos: MPI los leerá de aquí
datos["clima"].round(1).head()"""))
C.append(md("## 3. Paso 2 – Limpieza y análisis tabular con `pandas`"))
C.append(code("""vuelos = L.limpiar_vuelos(datos["vuelos"])
horarios = L.limpiar_horarios(datos["horarios"])
print(f"Vuelos en el aire: {len(vuelos)} | Salidas programadas: {len(horarios)} "
      f"| Cancelados: {horarios.cancelado.sum()} | Retrasados >15 min: {horarios.retrasado.sum()}")
resumen = L.resumen_por_aeropuerto(horarios, datos["clima"], datos["aire"])
resumen.round(1)"""))
C.append(code("""# Velocidad/altitud promedio y vuelos activos por región
A.resumen_vuelos(vuelos)"""))
C.append(md("""## 4. Paso 3 – Procesamiento distribuido con `MPI4Py`
`mpirun` lanza varias copias de `mpi_analisis.py`. Rank 0 reparte los aeropuertos (`scatter`), cada proceso calcula el **bootstrap** del retraso medio y rank 0 junta todo (`gather`/`Reduce`). Lo corremos con 1, 2 y 4 procesos para medir el tiempo (Colab tiene solo 2 CPU, así que más de 2 no acelera)."""))
C.append(code("""!rm -f resultados/mpi_tiempos.json
for n in (1, 2, 4):
    print(f"\\n================ mpirun con {n} proceso(s) ================")
    !mpirun --allow-run-as-root --oversubscribe -n {n} python mpi_analisis.py 200000 2>&1 | grep -E "rank|TOTAL" """))
C.append(code("""mpi_df = pd.read_csv("resultados/mpi_estadisticas.csv")
tiempos_mpi = json.load(open("resultados/mpi_tiempos.json"))
print("Speedup MPI:", {k: round(tiempos_mpi["1"]/v, 2) for k, v in tiempos_mpi.items()})
mpi_df.round(1)"""))
C.append(md("""## 5. Paso 4 – Aceleración con GPU (`Numba CUDA` + `CuPy`)
Kernels propios: distancia haversine, z-score de atípicos y trayectorias de ruta. CuPy: media/desviación y correlaciones."""))
C.append(code("""print("GPU disponible:", G.HAY_GPU)
vuelos, horarios = A.marcar_atipicos(vuelos, horarios)          # kernel z-score
print("Atípicos ->", vuelos.filter(like="atip_").sum().to_dict(), "| retrasos:", int(horarios.atip_retraso.sum()))
display(vuelos[(vuelos.atip_speed == 1) | (vuelos.atip_alt == 1)][["flight_iata", "speed", "alt", "lat", "lng"]].head())

cercanos = A.vuelos_cerca_de_aeropuertos(vuelos)                # kernel haversine (vuelos x aeropuertos)
tabla = A.tabla_final(resumen, cercanos)
ok, rut_lat, rut_lon = A.simular_rutas(vuelos)                  # kernel de rutas (malla 2D)
print("Rutas simuladas:", rut_lat.shape, "(vuelos x pasos)")"""))
C.append(code("""bench = G.benchmark(5_000_000)
print(bench)"""))
C.append(code("""cols = A.VARS_CLIMA + ["pm25", "retraso_medio", "pct_retrasados", "vuelos_cercanos"]
corr = G.matriz_correlacion(tabla, cols, "spearman")            # cp.corrcoef en GPU
corr.round(2)"""))
C.append(md("## 6. Paso 5 – Visualización (Dashboard con Matplotlib)"))
C.append(code("""import matplotlib.pyplot as plt
rutas = A.rutas_mas_frecuentes(vuelos)
V.fig_actividad_vuelos(vuelos, rutas); plt.show()
V.fig_retrasos_aeropuerto(tabla, mpi_df); plt.show()
V.fig_clima_vs_retraso(tabla); plt.show()
V.fig_actividad_hora(horarios, A.region_de_lat); plt.show()
V.fig_mapa(vuelos, ok, rut_lat, rut_lon); plt.show()
V.fig_thingspeak(datos["thingspeak"]); plt.show()
V.fig_correlacion(corr); plt.show()
V.fig_rendimiento(tiempos_mpi, bench, (t_sec, t_async)); plt.show()"""))
C.append(md("## 7. Respuestas a las preguntas de análisis"))
C.append(code("""respuestas = A.responder_preguntas(tabla, horarios)
for r in respuestas: print(r, "\\n")
fuentes = {k: v["fuente"].iloc[0] for k, v in datos.items()}
print("Fuentes de datos usadas:", fuentes)
json.dump({"respuestas": respuestas, "fuentes": fuentes, "benchmark": bench, "mpi": tiempos_mpi},
          open("resultados/resumen.json", "w"), ensure_ascii=False, indent=1, default=float)"""))
C.append(md("""## 8. Conclusiones y limitaciones
* **asyncio** reduce el tiempo de espera de red porque las peticiones se traslapan.
* **MPI** reparte trabajo independiente por aeropuerto; la aceleración está limitada por los 2 núcleos de Colab.
* **GPU**: gana cuando hay millones de datos; con pocos datos las copias CPU↔GPU dominan.
* Con 12 aeropuertos las correlaciones tienen poco poder estadístico: son indicios, no prueba causal. El clima se toma *en el momento de la consulta*, no en el instante de cada retraso.
* Si alguna fuente aparece como `SIMULADO`, esas conclusiones solo validan el pipeline, no el mundo real."""))
C.append(md("""## 9. Reporte PDF dinámico
Esta celda arma el PDF con **los resultados de esta ejecución** (tablas, tiempos, figuras y respuestas). Si cambias keys, datos o código y vuelves a ejecutar todo, el PDF se actualiza solo."""))
C.append(code("""!pip install -q reportlab
import reporte_pdf
from numba import cuda
ctx = dict(
    claves=claves, datos=datos, vuelos=vuelos, horarios=horarios, tabla=tabla, reg=A.resumen_vuelos(vuelos),
    mpi_df=mpi_df, tiempos_mpi=tiempos_mpi, bench=bench, t_async=(t_sec, t_async), respuestas=respuestas,
    atipicos=dict(speed=int(vuelos.atip_speed.sum()), alt=int(vuelos.atip_alt.sum()), retraso=int(horarios.atip_retraso.sum())),
    ingesta_s=ingesta_async.ULTIMA_INGESTA_S, rutas_shape=rut_lat.shape,
    gpu_nombre=(str(cuda.get_current_device().name, "utf-8") if G.HAY_GPU and isinstance(cuda.get_current_device().name, bytes) else cuda.get_current_device().name if G.HAY_GPU else "sin GPU (NumPy)"))
ruta_pdf = reporte_pdf.generar_reporte(ctx)
print("PDF generado:", ruta_pdf)
try:                                   # descarga automática a tu computadora
    from google.colab import files
    files.download(ruta_pdf)
except Exception as e:
    print("Descárgalo desde el panel de archivos (carpeta izquierda):", e)"""))
nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "Proyecto_CPD_Trafico_Aereo.ipynb"); print(len(C), "celdas")
