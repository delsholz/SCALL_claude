# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comando para correr la app

```bash
streamlit run app_scall.py
```

La app corre en `http://localhost:8501`. No hay tests ni linter configurados.

## Arquitectura

Aplicación Streamlit de una sola página con 4 módulos:

### Flujo de datos
```
data/cr2_prDaily_2020.txt.gz
        ↓
data_loader.py  (carga y filtra 2000–2020, cacheado con @st.cache_data)
        ↓
simulator.py    (simulación del estanque, análisis de escenarios)
        ↓
app_scall.py    (UI Streamlit, gráficos Plotly, generación PDF)
```

### Módulos

**`data_loader.py`**
- `cargar_datos_crudos(anio_inicio, anio_fin)` — lee el `.txt.gz` del CR², parsea metadatos de estaciones (filas 0–14), filtra datos diarios por rango de años. Retorna `df_estaciones`, `df_diario`, `df_mensual`, `codigos`.
- `aplicar_filtro_calidad(umbral_calidad, anio_inicio, anio_fin)` — filtra estaciones según % de meses válidos. Los parámetros de año son la clave de caché; cambiarlos invalida automáticamente el caché de Streamlit.

**`simulator.py`**
- `simular_continua(...)` — simula el balance diario del estanque para todo el período histórico usando `itertools.accumulate`.
- `encontrar_anios_extremos(df_sim_completa, codigo_estacion)` — identifica año seco (P5), normal (mediana) y lluvioso (P95) sobre años con ≥300 días de datos.
- `simular_escenario(df_sim_completa, anio_sim, ...)` — re-simula un año específico desde estanque vacío (nivel=0). Retorna DataFrame diario con columnas: Lluvia, Captado, Demanda, Estanque Final, Rebalse, Déficit Diario, Agua Acumulada Teórica.
- `calcular_curva_optimizacion(df_slice, capacidad_maxima)` — prueba capacidades de 1.000 en 1.000 L hasta `max(30.000, capacidad×2)`. **Vectorizado con NumPy**: simula todas las capacidades en paralelo en cada día (no doble bucle Python). Se llama 3 veces por simulación (una por tab de escenario).

**`utils.py`**
- `formato_chileno(valor, decimales)` — formatea números con punto como separador de miles y coma como decimal.
- `calcular_distancia_vectorizada(...)` — distancia efectiva estación-proyecto con corrección altitudinal: `sqrt(d_horiz² + (k × Δh_km)²)`. El coeficiente `k` es ajustable por el usuario.
- `arreglar_coordenada(val, es_longitud)` — parsea coordenadas mal formateadas del archivo CR².

**`app_scall.py`**
- Todo el estado de sesión vive en `st.session_state`: `simulacion_calculada`, `resultado`, `informe_datos`, `alt_auto`.
- La simulación completa se ejecuta dentro del bloque `if st.button(...)` y guarda resultados en `st.session_state` para que Tab 2 y el PDF los consuman sin re-calcular.
- `generar_informe_pdf(d)` usa fpdf2 para construir un PDF de una página en memoria (`io.BytesIO`), incluye logo `Logo_Amulen.png` si existe.
- `mostrar_detalles_escenario(df_slice, nombre, key_suffix)` es la función de visualización principal: KPIs, gráfico de nivel, curva de optimización, seguridad hídrica, tabla diaria y botón de exportación Excel.

### Datos de entrada
El archivo `data/cr2_prDaily_2020.txt.gz` tiene un formato especial: las primeras 15 filas son metadatos (fila 0=códigos, 3=nombres, 4=altitudes, 5=latitudes, 6=longitudes), y desde la fila 16 en adelante son datos diarios con columnas por estación. El valor `-9999` indica dato faltante.

### Gradiente orográfico
El ajuste de lluvia por altitud se aplica multiplicando todos los valores de precipitación por `max(0, 1 + (coef_orografico/100) × (desnivel_m/100))`. Este multiplicador se propaga a `simular_continua` y `simular_escenario`.
