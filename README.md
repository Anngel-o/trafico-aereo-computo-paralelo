# Análisis de tráfico aéreo y condiciones ambientales (CPD – 3ª Evaluación)

- 📦 **Repositorio:** https://github.com/Anngel-o/trafico-aereo-computo-paralelo
- 📓 **Notebook en Colab:** https://colab.research.google.com/drive/1a8DVBhEwQmpc0ToJ4koVQ1SSI4rfTell?usp=sharing

**Equipo**
| Nombre | Matrícula |
|---|---|
| Jose Angel Ortiz Meraz | 353195 |
| Luz Mariam Garcia Castillo | 348409 |
| Carlos Sánchez Fuentes | 368058 |

Pipeline de cómputo paralelo: **APIs en la nube → asyncio → MPI4Py → CUDA/Numba/CuPy → Matplotlib**, con Pandas como apoyo.

## Requisitos previos: cuentas y API keys
| Servicio | ¿Key? | Cómo | Secreto en Colab |
|---|---|---|---|
| AirLabs | Sí | Cuenta en airlabs.co | `AIRLABS_KEY` |
| OpenWeather | Sí | Cuenta en openweathermap.org (una key nueva puede tardar en activarse) | `OPENWEATHER_KEY` |
| OpenAQ v3 | Sí | Registro en explore.openaq.org/register; la key está en explore.openaq.org/account | `OPENAQ_KEY` |
| ThingSpeak | **No** | Canal público 12397: se lee con un GET sin autenticación (las keys de ThingSpeak solo sirven para canales privados o para escribir datos) | - |

Las keys van en 🔑 Secretos de Colab (interruptor *Acceso desde el cuaderno* activado), nunca en el código. `verificar_claves.py` comprueba existencia, formato y que la API las acepte.

## Cómo ejecutarlo (recomendado: Google Colab con GPU T4)
1. Sube `Proyecto_CPD_Trafico_Aereo.ipynb` a Colab → *Entorno de ejecución → Cambiar tipo → GPU T4*.
2. (Opcional) En 🔑 *Secretos* agrega `AIRLABS_KEY`, `OPENWEATHER_KEY`, `OPENAQ_KEY`. ThingSpeak (canal público 12397) no necesita key.
3. *Entorno de ejecución → Ejecutar todas*. La última celda genera y descarga `Reporte_Proyecto_CPD_Trafico_Aereo.pdf` actualizado con esa ejecución.

Sin key, esa fuente usa **datos simulados** (`fuente = SIMULADO`) para poder probar el pipeline.

## Archivos
| Archivo | Fase |
|---|---|
| `config.py` | Aeropuertos, caja geográfica, lectura de API keys |
| `ingesta_async.py` | asyncio + aiohttp: lectura concurrente de AirLabs, OpenWeather, OpenAQ, ThingSpeak |
| `datos_demo.py` | Datos simulados (modo demo) |
| `limpieza_pandas.py` | Limpieza, banderas de retraso/cancelación, unión de tablas |
| `mpi_analisis.py` | MPI: scatter/gather/Reduce, bootstrap por aeropuerto (`mpirun -n 4 python mpi_analisis.py`) |
| `gpu_numba.py` | Kernels CUDA (haversine, z-score, rutas) + CuPy (correlación) + benchmark |
| `analisis.py` | Atípicos, tráfico cercano, respuestas a las 5 preguntas |
| `visualizacion.py` | Gráficas |
| `verificar_claves.py` | Auditoría de API keys (existencia, formato, prueba real contra cada API) |
| `reporte_pdf.py` | Genera el PDF del reporte con los resultados de la ejecución (se llama en la última celda) |
| `build_notebook.py` | Regenera el notebook desde los módulos |
