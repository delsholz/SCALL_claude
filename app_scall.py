import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

from fpdf import FPDF

from utils import formato_chileno, calcular_distancia_vectorizada
from data_loader import cargar_datos_crudos, aplicar_filtro_calidad
from simulator import (simular_continua, simular_escenario,
                       encontrar_anios_extremos, calcular_curva_optimizacion)


# ===============================================================
# GENERADOR DE INFORME PDF
# ===============================================================
def generar_informe_pdf(d):
    from datetime import date

    # Colores
    AZUL_OSC  = (26,  58,  92)
    AZUL_MED  = (26, 111, 163)
    AZUL_CLAR = (212, 237, 255)
    GRIS      = (245, 245, 245)
    BLANCO    = (255, 255, 255)
    NEGRO     = (30,  30,  30)
    AMARILLO  = (255, 243, 205)
    CELESTE   = (209, 236, 241)
    VERDE     = (212, 237, 218)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(0, 0, 0)

    W = 210  # ancho A4

    # ── HEADER ────────────────────────────────────────────────
    pdf.set_fill_color(*AZUL_OSC)
    pdf.rect(0, 0, W, 26, "F")
    
    # --- INSERCIÓN DEL LOGO AMULEN ---
    try:
        # Colocamos el logo a la derecha (x=165), arriba (y=4), con un ancho de 35mm
        pdf.image("Logo_Amulen.png", x=165, y=4, w=35)
    except:
        # Si la imagen no existe, el PDF se genera igual sin el logo
        pass

    pdf.set_text_color(*BLANCO)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(12, 5)
    pdf.cell(0, 9, "SIMULADOR SCALL", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 15)
    pdf.cell(0, 6, "Informe de Viabilidad de Cosecha de Aguas Lluvias")

    # ── BARRA NOMBRE PROYECTO ─────────────────────────────────
    pdf.set_fill_color(*AZUL_MED)
    pdf.rect(0, 26, W, 11, "F")
    pdf.set_text_color(*BLANCO)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_xy(12, 27)
    pdf.cell(0, 9, d["nombre_proyecto"])

    # ── LÍNEA FECHA ───────────────────────────────────────────
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_xy(12, 39)
    pdf.cell(0, 5,
             f"Fecha: {date.today().strftime('%d/%m/%Y')}   |   "
             f"Periodo analizado: {d['anio_inicio']}-{d['anio_fin']}   |   "
             f"Fuente climatica: CR2 Chile")

    y = 47

    def sec_title(titulo, y):
        pdf.set_fill_color(*AZUL_MED)
        pdf.set_text_color(*BLANCO)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(10, y)
        pdf.cell(190, 7, f"  {titulo}", fill=True)
        return y + 9

    def kv(label, valor, y, x=10, w_label=62, w_val=83, alt=False):
        bg = GRIS if alt else BLANCO
        pdf.set_fill_color(*bg)
        pdf.set_text_color(*NEGRO)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_xy(x, y)
        pdf.cell(w_label, 6, label, fill=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(w_val, 6, str(valor), fill=True)
        return y + 6

    # ── SECCIÓN: PARÁMETROS ───────────────────────────────────
    y = sec_title("PARAMETROS DEL SISTEMA", y)
    y_params = y

    params = [
        ("Superficie del Techo",    f"{d['techo']} m2"),
        ("Capacidad del Estanque",  f"{d['capacidad_maxima']:,.0f} Litros"),
        ("Eficiencia de Captacion", f"{d['eficiencia']*100:.0f}%"),
        ("Numero de Personas",      f"{d['numero_personas']}"),
        ("Consumo por Persona",     f"{d['litros_persona_dia']} L/persona/dia"),
        ("Consumo Mensual Total",   f"{d['consumo_mensual']:,.0f} L/mes"),
        ("Tipo de Uso",             d["tipo_uso"].split("(")[0].strip()),
        ("Meses de Operacion",      ", ".join(d["meses_seleccionados"])),
        ("Coordenadas",             f"{d['lat_proyecto']:.4f} / {d['lon_proyecto']:.4f}"),
        ("Altitud del Proyecto",    f"{d['alt_proyecto']} m.s.n.m."),
    ]
    for i, (lbl, val) in enumerate(params):
        kv(lbl, val, y_params + i * 6, x=10, w_label=55, w_val=80, alt=(i % 2 == 0))

    y_after_params = y_params + len(params) * 6 + 4

    # ── SECCIÓN: ESTACIÓN ─────────────────────────────────────
    y_est = sec_title("ESTACION METEOROLOGICA", y_after_params)
    station = [
        ("Nombre",             d["est_nombre"]),
        ("Codigo",             d["est_codigo"]),
        ("Altitud",            f"{d['est_altitud']} m.s.n.m."),
        ("Desnivel Proyecto",  f"{d['desnivel']} m"),
        ("Distancia Efectiva", f"{d['distancia']:.1f} km (con correccion altitudinal)"),
        ("Periodo con datos",  f"{d['anio_inicio']} - {d['anio_fin']}"),
        ("Calidad de Datos",   f"{d['pct_calidad']:.1f}% de meses validos"),
    ]
    for i, (lbl, val) in enumerate(station):
        kv(lbl, val, y_est + i * 6, x=10, w_label=55, w_val=135, alt=(i % 2 == 0))

    y = y_est + len(station) * 6 + 5

    # ── SECCIÓN: TABLA RESULTADOS ─────────────────────────────
    y = sec_title("RESULTADOS POR ESCENARIO CLIMATICO", y)

    col_ws  = [52, 14, 26, 28, 28, 22, 20]
    headers = ["Escenario", "Ano", "Lluvia (mm)", "Captado (L)",
               "Demanda (L)", "Cobertura %", "Dias sin agua"]

    pdf.set_fill_color(*AZUL_OSC)
    pdf.set_text_color(*BLANCO)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_xy(10, y)
    for h, w in zip(headers, col_ws):
        pdf.cell(w, 8, h, border=1, align="C", fill=True)
    y += 8

    escenarios = [
        ("Ano Seco (P5)",        d["anio_seco"],     d["df_seco"],     AMARILLO),
        ("Ano Normal (Mediana)", d["anio_mediano"],  d["df_normal"],   CELESTE),
        ("Ano Lluvioso (P95)",   d["anio_lluvioso"], d["df_lluvioso"], VERDE),
    ]
    for nombre_esc, anio_esc, df_esc, color in escenarios:
        td   = df_esc["Demanda (L)"].sum()
        # Manejo de tildes/nombres de columnas
        col_def = "Deficit Diario (L)" if "Deficit Diario (L)" in df_esc else "Déficit Diario (L)"
        def_ = df_esc[col_def].sum()
        
        ts   = td + def_
        pct  = (ts / td * 100) if td > 0 else 100
        dsag = int((df_esc.filter(like="ficit").iloc[:, 0] < 0).sum())
        tc   = df_esc["Captado (L)"].sum()
        lluv = d["totales_anio"].get(anio_esc, 0)

        pdf.set_fill_color(*color)
        pdf.set_text_color(*NEGRO)
        pdf.set_font("Helvetica", "B" if "Normal" in nombre_esc else "", 8)
        pdf.set_xy(10, y)
        vals = [nombre_esc, str(anio_esc), f"{lluv:,.0f}", f"{tc:,.0f}",
                f"{td:,.0f}", f"{pct:.1f}%", str(dsag)]
        aligns = ["L", "C", "C", "C", "C", "C", "C"]
        for v, w, al in zip(vals, col_ws, aligns):
            pdf.cell(w, 8, v, border=1, align=al, fill=True)
        y += 8

    y += 6

    # ── SECCIÓN: TAMAÑO OPTIMO ────────────────────────────────
    y = sec_title("TAMANO OPTIMO DEL ESTANQUE (Ano Normal)", y)

    # Nota: Asegúrate de tener importada calcular_curva_optimizacion
    from simulator import calcular_curva_optimizacion
    caps, efs, cap_opt, ef_act, ef_opt = calcular_curva_optimizacion(
        d["df_normal"], d["capacidad_maxima"]
    )

    kpis = [
        ("Estanque actual",            f"{d['capacidad_maxima']:,.0f} L",
         "Tamano optimo estimado",     f"{cap_opt:,.0f} L"),
        ("Cobertura con estanque actual", f"{ef_act:.1f}%",
         "Cobertura con tamano optimo",   f"{ef_opt:.1f}%"),
    ]
    for i, (l1, v1, l2, v2) in enumerate(kpis):
        bg = GRIS if i % 2 == 0 else BLANCO
        pdf.set_fill_color(*bg)
        pdf.set_text_color(*NEGRO)
        pdf.set_xy(10, y)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(55, 6, l1, fill=True)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*AZUL_MED)
        pdf.cell(40, 6, v1, fill=True)
        pdf.set_text_color(*NEGRO)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(55, 6, l2, fill=True)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*AZUL_MED)
        pdf.cell(40, 6, v2, fill=True)
        y += 6

    y += 5

    # ── SECCIÓN: CONCLUSIÓN ───────────────────────────────────
    y = sec_title("CONCLUSION", y)

    td_n  = d["df_normal"]["Demanda (L)"].sum()
    col_def_n = "Deficit Diario (L)" if "Deficit Diario (L)" in d["df_normal"] else "Déficit Diario (L)"
    def_n = d["df_normal"][col_def_n].sum()
    pct_n = ((td_n + def_n) / td_n * 100) if td_n > 0 else 100
    
    col_def_s = "Deficit Diario (L)" if "Deficit Diario (L)" in d["df_seco"] else "Déficit Diario (L)"
    dsag_s = int((d["df_seco"][col_def_s] < 0).sum())

    lineas = [
        f"En el ano normal ({d['anio_mediano']}), el sistema cubre el {pct_n:.1f}% de la demanda anual con el estanque actual de {d['capacidad_maxima']:,.0f} L.",
        f"Para alcanzar maxima eficiencia se recomienda un estanque de {cap_opt:,.0f} L ({ef_opt:.1f}% de cobertura).",
        f"En el ano seco (P5, {d['anio_seco']}), hay {dsag_s} dias sin suministro. Se recomienda considerar una fuente complementaria para esos periodos.",
        f"Analisis basado en {d['anio_fin'] - d['anio_inicio'] + 1} anos de datos historicos de la estacion {d['est_nombre']}.",
    ]
    pdf.set_text_color(*NEGRO)
    for linea in lineas:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_xy(10, y)
        pdf.multi_cell(190, 5.5, linea)
        y += 6

    # ── FOOTER ────────────────────────────────────────────────
    pdf.set_y(-12)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5,
             "Generado por Simulador SCALL  |  Datos climaticos: Centro de Ciencia del Clima y la Resiliencia (CR)2  |  www.cr2.cl",
             align="C")

    return io.BytesIO(pdf.output())

# ===============================================================
# CSS
# ===============================================================
st.markdown("""
<style>
button[kind="primary"] {
    color: black !important;
    font-weight: bold !important;
}
span[data-baseweb="tag"] {
    background-color: #3498db !important;
}
span[data-baseweb="tag"] span {
    color: black !important;
    font-weight: bold !important;
}
span[data-baseweb="tag"] svg {
    fill: black !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# SIDEBAR
# ===============================================================
st.sidebar.header("1. Datos del Proyecto")
nombre_proyecto  = st.sidebar.text_input("Nombre del Proyecto/Lugar", "Mi Proyecto SCALL")
techo            = st.sidebar.number_input("Superficie del Techo (m2)", min_value=10.0, value=120.0)
capacidad_maxima = st.sidebar.number_input("Capacidad Máxima del Estanque (Litros)", min_value=100.0, value=5000.0, step=500.0)
eficiencia       = st.sidebar.slider("Eficiencia de Captación (Escorrentía)", 0.5, 1.0, 0.85, 0.05,
                                     help="0.9 para metal/vidrio, 0.7 para tejas.")

st.sidebar.subheader("Consumo de Agua")
numero_personas    = st.sidebar.number_input("Número de personas", min_value=1, value=4, step=1)
litros_persona_dia = st.sidebar.number_input("Consumo (Litros/persona/día)", min_value=1.0, value=50.0)

tipo_uso = st.sidebar.radio(
    "Tipo de Uso (Días operativos)",
    options=["Colegio / Oficina (Lun-Vie)", "Casa / Residencia (Lun-Dom)"],
    index=0,
    help="Define si el lugar tiene consumo de agua durante los fines de semana."
)

consumo_fines_semana = "Casa / Residencia" in tipo_uso
dias_mes             = 30 if consumo_fines_semana else 22
patron_texto         = "Lunes a Domingo (30 días)" if consumo_fines_semana else "Lunes a Viernes (22 días)"
consumo_mensual      = numero_personas * litros_persona_dia * dias_mes

lista_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
meses_dict  = {m: f"{i+1:02d}" for i, m in enumerate(lista_meses)}

meses_seleccionados = st.sidebar.multiselect(
    "Meses de operación (con demanda)",
    options=lista_meses,
    default=lista_meses[2:],
    help="Meses en que el lugar está habitado. En el resto de meses solo se acumulará agua."
)
meses_num_seleccionados = [meses_dict[m] for m in meses_seleccionados]

st.sidebar.info(
    f"💧 Consumo mensual estimado:\n**{formato_chileno(consumo_mensual, 0)} Litros/mes**\n\n"
    f"📅 **Patrón Semanal:**\n{patron_texto}."
)

st.sidebar.header("2. Calidad de Datos")
umbral_calidad = st.sidebar.slider(
    "Mínimo de registros válidos (%)",
    min_value=0, max_value=100, value=80, step=5,
    help="Solo se usarán estaciones con este % de datos válidos (1990 a 2020)."
)

st.sidebar.header("3. Coordenadas de Ubicación")
st.sidebar.write("Ingresa la latitud, longitud y altitud de tu proyecto.")
lat_proyecto = st.sidebar.number_input("Latitud",  value=-33.4500, format="%.4f")
lon_proyecto = st.sidebar.number_input("Longitud", value=-70.6500, format="%.4f")

if st.sidebar.button("📍 Obtener altitud automáticamente"):
    with st.sidebar:
        with st.spinner("Consultando altitud..."):
            try:
                resp = requests.get(
                    f"https://api.opentopodata.org/v1/srtm90m?locations={lat_proyecto},{lon_proyecto}",
                    timeout=8
                )
                data = resp.json()
                if data.get("status") == "OK":
                    elev = data["results"][0]["elevation"]
                    st.session_state['alt_auto'] = int(round(elev))
                else:
                    st.sidebar.warning("No se pudo obtener la altitud.")
            except Exception:
                st.sidebar.warning("Error al conectar con el servicio de altitud. Ingresa el valor manualmente.")

alt_default = int(st.session_state.get('alt_auto', 500))
alt_proyecto = st.sidebar.number_input("Altitud (m.s.n.m.)", value=alt_default, step=10)

st.sidebar.subheader("Corrección por Altitud")
coef_altitud = st.sidebar.slider(
    "Coeficiente de corrección altitudinal (k)",
    min_value=1, max_value=20, value=5, step=1,
    help=(
        "Controla cuánto 'pesa' la diferencia de altura al buscar la estación más cercana.\n\n"
        "k=1 → 1 km de desnivel = 1 km horizontal\n"
        "k=5 → 1 km de desnivel = 5 km horizontales\n"
        "k=20 → prioriza fuertemente estaciones a altitud similar\n\n"
        "Sube el valor si tu zona tiene fuerte variación de lluvia con la altura."
    )
)
st.sidebar.subheader("Gradiente Orográfico (Lluvia vs Altura)")
coef_orografico = st.sidebar.slider(
    "Variación de lluvia (% por cada 100m de desnivel)",
    min_value=-15.0, max_value=15.0, value=0.0, step=1.0,
    help="Ajusta físicamente los milímetros de lluvia. Si tu proyecto está más alto que la estación meteorológica, usualmente llueve más (usa valores positivos). Si está a sotavento, podría llover menos (usa negativos)."
)

st.title("💧 Simulador de Cosecha de Aguas Lluvias SCALL")
st.write("Calcula la viabilidad de tu estanque analizando la **realidad climática continua** desde 1990 a 2020 "
         "(utilizando datos oficiales del **(CR)²**) y extrayendo los años reales extremos.")

# ===============================================================
# TABS
# ===============================================================
tab1, tab2 = st.tabs([" Simulación Histórica y Escenarios", " Resumen Mensual Histórico"])

if 'simulacion_calculada' not in st.session_state:
    st.session_state.simulacion_calculada = False

with tab1:
    if not st.session_state.simulacion_calculada:
        st.info(" **¡Bienvenido al Simulador SCALL!**\n\n"
                "1. Configura los parámetros de tu techo y consumo en la barra lateral (izquierda).\n"
                "2. Ingresa las coordenadas exactas de tu proyecto.\n"
                "3. Presiona el botón para iniciar el análisis climático.")

    if st.button("Buscar Estación y Calcular Balance", type="primary"):
        st.session_state.simulacion_calculada = True

    if st.session_state.simulacion_calculada:
        try:
            with st.spinner("Construyendo modelos y ejecutando la simulación..."):

                try:
                    df_est_crudas, df_diario, df_mensual, codigos = cargar_datos_crudos()
                except ValueError as e:
                    st.error(f"⚠️ **Error en los datos:** {e}")
                    st.stop()

                df_estaciones, df_diario, df_mensual = aplicar_filtro_calidad(umbral_calidad)

                if df_estaciones.empty:
                    st.error(f"⚠️ Ninguna estación cumple con el estándar de {umbral_calidad}% de calidad.")
                    st.stop()

                df_est_crudas = df_est_crudas.copy()
                alts_estaciones = df_est_crudas['Altitud'].fillna(alt_proyecto).values.astype(float)
                df_est_crudas['Dist'] = calcular_distancia_vectorizada(
                    lat_proyecto, lon_proyecto, alt_proyecto,
                    df_est_crudas['Latitud'].values.astype(float),
                    df_est_crudas['Longitud'].values.astype(float),
                    alts_estaciones,
                    coef_altitud=coef_altitud
                )

                df_est_limpias = df_est_crudas[['Codigo', 'Dist']].drop_duplicates(subset=['Codigo'])
                df_estaciones  = df_estaciones.merge(df_est_limpias, on='Codigo', how='left')

                estacion_cercana = df_estaciones.loc[df_estaciones['Dist'].idxmin()]
                distancia_minima = estacion_cercana['Dist']
                cinco_cercanas   = df_est_crudas[df_est_crudas['Codigo'] != estacion_cercana['Codigo']].nsmallest(5, 'Dist')
                codigo_estacion  = str(estacion_cercana['Codigo'])

                lluvias_mensuales = df_mensual[['Año_Mes', 'Año', codigo_estacion]].copy()
                lluvias_mensuales.rename(columns={'Año_Mes': 'Fecha'}, inplace=True)
                fechas_dt_m = pd.to_datetime(lluvias_mensuales['Fecha'] + '-01', format='%Y-%m-%d', errors='coerce')
                lluvias_mensuales['Mes']     = fechas_dt_m.dt.strftime('%m')
                lluvias_mensuales['Mes_num'] = fechas_dt_m.dt.month

                meses_con_datos    = lluvias_mensuales[codigo_estacion].notna().sum()
                porcentaje_calidad = (meses_con_datos / len(lluvias_mensuales)) * 100

                # Rango real con datos de la estación seleccionada
                datos_reales = lluvias_mensuales.dropna(subset=[codigo_estacion])
                anio_inicio  = int(datos_reales['Año'].min()) if not datos_reales.empty else int(lluvias_mensuales['Año'].min())
                anio_fin     = int(datos_reales['Año'].max()) if not datos_reales.empty else int(lluvias_mensuales['Año'].max())

                alt_estacion = estacion_cercana.get('Altitud', float('nan'))
                # Calculamos el desnivel real (positivo si el proyecto está más alto)
                desnivel_real = (alt_proyecto - alt_estacion) if not pd.isna(alt_estacion) else 0.0
                desnivel_abs = abs(desnivel_real)
                desnivel = desnivel_abs  # <--- AGREGA ESTA LÍNEA
                
                # Fórmula del Gradiente Orográfico
                variacion_pct = (coef_orografico / 100.0) * (desnivel_real / 100.0)
                multiplicador_lluvia = max(0.0, 1.0 + variacion_pct) # Evita multiplicadores negativos
                
                alt_txt = f" | Altitud: **{formato_chileno(alt_estacion, 0)} m.s.n.m.** (desnivel: {formato_chileno(desnivel_abs, 0)} m)" \
                               if desnivel_abs is not None else ""
                st.info(f"📍 **Estación más cercana:** {estacion_cercana['Nombre']} "
                        f"*(Distancia efectiva: **{formato_chileno(distancia_minima, 1)} km**)*{alt_txt}")
                
                if multiplicador_lluvia != 1.0:
                    signo = "+" if multiplicador_lluvia > 1 else ""
                    st.warning(f" **Ajuste Orográfico Aplicado:** Debido a los {formato_chileno(desnivel_real, 0)}m de diferencia de altura, "
                               f"las lluvias originales de esta estación se ajustaron en un **{signo}{(multiplicador_lluvia - 1)*100:.1f}%** para tu proyecto.")

                # --- Mapa ---
                fig_mapa = go.Figure()
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=cinco_cercanas['Latitud'].tolist(), lon=cinco_cercanas['Longitud'].tolist(),
                    mode='markers+text', marker=dict(size=10, color='#888888'),
                    text=cinco_cercanas['Nombre'].tolist(), textposition='top right',
                    name='Estaciones cercanas', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=[estacion_cercana['Latitud']], lon=[estacion_cercana['Longitud']],
                    mode='markers+text', marker=dict(size=14, color='#0000FF'),
                    text=[estacion_cercana['Nombre']], textposition='top right',
                    name='Estación seleccionada', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=[lat_proyecto], lon=[lon_proyecto],
                    mode='markers+text', marker=dict(size=14, color='#FF0000'),
                    text=[nombre_proyecto], textposition='top right',
                    name='Tu Proyecto', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.update_layout(
                    mapbox=dict(
                        style='open-street-map',
                        center=dict(lat=(lat_proyecto + estacion_cercana['Latitud']) / 2,
                                    lon=(lon_proyecto + estacion_cercana['Longitud']) / 2),
                        zoom=8
                    ),
                    margin=dict(l=0, r=0, t=0, b=0), height=350,
                    legend=dict(orientation='h', y=-0.08)
                )
                st.plotly_chart(fig_mapa, use_container_width=True,
                                config={'scrollZoom': False, 'displayModeBar': True})

                # --- Simulación continua ---
                st.markdown("---")
                st.write(f"### Simulación Histórica Continua ({anio_inicio} - {anio_fin})")
                st.write("Gráfico continuo que muestra el volumen de agua acumulado en el estanque a lo largo "
                         "de la década. Refleja el balance entre la lluvia captada en escenarios históricos y "
                         "la demanda diaria cubierta, asumiendo un inicio en 0 Litros.")

                df_sim_completa = simular_continua(
                    df_diario, codigo_estacion, anio_inicio, anio_fin,
                    meses_num_seleccionados, consumo_fines_semana,
                    numero_personas, litros_persona_dia, capacidad_maxima,
                    techo, eficiencia, multiplicador_lluvia  # <--- AGREGA ESTO AQUÍ
                )

                anio_seco, anio_mediano, anio_lluvioso, totales_por_anio = encontrar_anios_extremos(
                    df_sim_completa, codigo_estacion
                )

                fig_cont = px.area(df_sim_completa, x="Fecha", y="Estanque Final (L)",
                                   color_discrete_sequence=['#5b9bd5'])
                fig_cont.add_hline(y=capacidad_maxima, line_dash="dash", line_color="red",
                                   annotation_text="Capacidad del Estanque",
                                   annotation_position="top left", annotation_font=dict(color="red"))
                fig_cont.add_vrect(x0=f"{anio_seco}-01-01", x1=f"{anio_seco}-12-31",
                                   fillcolor="#f39c12", opacity=0.15,
                                   annotation_text="Año Seco", annotation_position="top left")
                fig_cont.add_vrect(x0=f"{anio_mediano}-01-01", x1=f"{anio_mediano}-12-31",
                                   fillcolor="blue", opacity=0.15,
                                   annotation_text="Año Normal", annotation_position="top left")
                fig_cont.add_vrect(x0=f"{anio_lluvioso}-01-01", x1=f"{anio_lluvioso}-12-31",
                                   fillcolor="green", opacity=0.15,
                                   annotation_text="Año Lluvioso", annotation_position="top left")
                fig_cont.update_layout(yaxis_title="Volumen (Litros)",
                                       xaxis_title="Línea de Tiempo Histórica",
                                       margin=dict(b=40))
                fig_cont.update_traces(hovertemplate="<b>%{x}</b><br>Nivel: %{y:,.0f} L<extra></extra>")
                st.plotly_chart(fig_cont, use_container_width=True,
                                config={'scrollZoom': False, 'displayModeBar': True})

                # --- Escenarios reales ---
                st.markdown("---")
                st.write("### Análisis del Estanque en Escenarios Reales")
                st.write("Evaluación del rendimiento del estanque en años climáticos representativos "
                         "(Seco, Normal y Lluvioso). Cada escenario se simula de forma independiente, "
                         "asumiendo la condición más exigente: un estanque inicialmente vacío (0 Litros).")

                col1, col2, col3 = st.columns(3)
                col1.warning(f" **Año Seco (P5): {anio_seco}**\n\n"
                             f"Total Lluvias: **{formato_chileno(totales_por_anio[anio_seco], 1)} mm**")
                col2.info(f" **Año Normal (Mediana): {anio_mediano}**\n\n"
                          f"Total Lluvias: **{formato_chileno(totales_por_anio[anio_mediano], 1)} mm**")
                col3.success(f" **Año Lluvioso (P95): {anio_lluvioso}**\n\n"
                             f"Total Lluvias: **{formato_chileno(totales_por_anio[anio_lluvioso], 1)} mm**")

                sim_params = dict(
                    codigo_estacion=codigo_estacion,
                    meses_num_seleccionados=meses_num_seleccionados,
                    consumo_fines_semana=consumo_fines_semana,
                    numero_personas=numero_personas,
                    litros_persona_dia=litros_persona_dia,
                    capacidad_maxima=capacidad_maxima,
                    techo=techo,
                    eficiencia=eficiencia,
                    multiplicador_lluvia=multiplicador_lluvia  # <--- Y ESTO AQUÍ
                )

                df_seco     = simular_escenario(df_sim_completa, anio_seco,     **sim_params)
                df_normal   = simular_escenario(df_sim_completa, anio_mediano,  **sim_params)
                df_lluvioso = simular_escenario(df_sim_completa, anio_lluvioso, **sim_params)

                st.write("### Comparación del Volumen Acumulado del Estanque en Años Extremos")

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=df_seco["Eje X"], y=df_seco["Agua Acumulada Teórica (L)"],
                    mode='lines', name=f"Año Seco ({anio_seco})",
                    line=dict(color='#f39c12', width=2),
                    hovertemplate="<b>%{x|%d %b}</b><br>%{y:,.0f} L<extra>Año Seco</extra>"
                ))
                fig_comp.add_trace(go.Scatter(
                    x=df_normal["Eje X"], y=df_normal["Agua Acumulada Teórica (L)"],
                    mode='lines', name=f"Año Normal ({anio_mediano})",
                    line=dict(color='#3498db', width=2.5),
                    hovertemplate="<b>%{x|%d %b}</b><br>%{y:,.0f} L<extra>Año Normal</extra>"
                ))
                fig_comp.add_trace(go.Scatter(
                    x=df_lluvioso["Eje X"], y=df_lluvioso["Agua Acumulada Teórica (L)"],
                    mode='lines', name=f"Año Lluvioso ({anio_lluvioso})",
                    line=dict(color='#2ecc71', width=2),
                    hovertemplate="<b>%{x|%d %b}</b><br>%{y:,.0f} L<extra>Año Lluvioso</extra>"
                ))
                fig_comp.add_hline(
                    y=capacidad_maxima, line_dash="dash", line_color="red", line_width=2,
                    annotation_text=f"Capacidad máxima: {formato_chileno(capacidad_maxima, 0)} L",
                    annotation_position="top left", annotation_font=dict(color="red", size=12)
                )
                todos_y         = pd.concat([df_seco["Agua Acumulada Teórica (L)"],
                                             df_normal["Agua Acumulada Teórica (L)"],
                                             df_lluvioso["Agua Acumulada Teórica (L)"]])
                y_min_comp      = todos_y.min()
                y_max_comp      = todos_y.max()
                margen_inf_comp = max(abs(y_min_comp) * 0.4, capacidad_maxima * 0.20)
                margen_sup_comp = max((y_max_comp - capacidad_maxima) * 0.25, capacidad_maxima * 0.15)
                fig_comp.update_layout(
                    yaxis=dict(title="Volumen (Litros)",
                               range=[y_min_comp - margen_inf_comp, y_max_comp + margen_sup_comp],
                               tickformat=","),
                    xaxis=dict(title="Meses del Año", dtick="M1", tickformat="%b",
                               showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
                    height=420, margin=dict(t=20, b=60, l=60, r=20),
                    legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_comp, use_container_width=True,
                                config={'scrollZoom': False, 'displayModeBar': True})

                # --- Detalle por escenario ---
                st.write("###  Tablas y Diseño por Escenario")
                tab_n, tab_s, tab_ll = st.tabs([" Año Normal", "Año Seco", " Año Lluvioso"])

                def colorear_filas_diarias(row):
                    estilos = [""] * len(row)
                    idx_sd  = list(row.index).index("Déficit Diario (L)")
                    if row["Déficit Diario (L)"] == 0:
                        estilos[idx_sd] = "background-color: #d4edda; color: #155724; font-weight: bold"
                    else:
                        estilos[idx_sd] = "background-color: #f8d7da; color: #721c24; font-weight: bold"
                    return estilos

                def mostrar_detalles_escenario(df_slice, nombre):
                    total_captado      = df_slice['Captado (L)'].sum()
                    total_demanda      = df_slice['Demanda (L)'].sum()
                    deficit_total      = df_slice['Déficit Diario (L)'].sum()
                    total_suministrado = total_demanda + deficit_total
                    pct_cubierto       = (total_suministrado / total_demanda * 100) if total_demanda > 0 else 100
                    dias_sin_agua      = (df_slice['Déficit Diario (L)'] < 0).sum()

                    st.write("#### Indicadores Clave de Desempeño (KPIs)")
                    st.caption("**Fórmula de Balance:** *Total Potencial a Captar = "
                               "Agua Consumida + Agua Almacenada + Agua Perdida por Rebalse*")

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("💧 Total Potencial a Captar", f"{formato_chileno(total_captado, 0)} L")
                    kpi2.metric("🎯 Demanda Cubierta",         f"{formato_chileno(pct_cubierto, 1)} %")
                    kpi3.metric("🚨 Días sin agua",            f"{dias_sin_agua} días")
                    kpi4.metric("🌊 Rebalse",                  f"{formato_chileno(df_slice['Rebalse (L)'].sum(), 0)} L")
                    st.markdown("---")

                    st.write(f"#### Nivel del Estanque: {nombre}")
                    y_vals     = df_slice["Agua Acumulada Teórica (L)"]
                    y_min, y_max = y_vals.min(), y_vals.max()
                    margen_inf = max(abs(y_min) * 0.4, capacidad_maxima * 0.20)
                    margen_sup = max((y_max - capacidad_maxima) * 0.25, capacidad_maxima * 0.15)

                    fig_nivel = go.Figure()
                    fig_nivel.add_trace(go.Scatter(
                        x=df_slice["Eje X"], y=y_vals, mode='lines', fill='tozeroy',
                        name="Acumulado Teórico", line=dict(color='#3498db', width=2),
                        fillcolor='rgba(52, 152, 219, 0.25)',
                        hovertemplate="<b>%{x|%d %b}</b><br>Nivel: %{y:,.0f} L<extra></extra>"
                    ))
                    fig_nivel.add_hline(
                        y=capacidad_maxima, line_dash="dash", line_color="red", line_width=2,
                        annotation_text=f"Capacidad máxima: {formato_chileno(capacidad_maxima, 0)} L",
                        annotation_position="top left", annotation_font=dict(color="red", size=12)
                    )
                    fig_nivel.add_hline(
                        y=0, line_dash="dot", line_color="orange", line_width=1.5,
                        annotation_text="Sin agua", annotation_position="bottom right",
                        annotation_font=dict(color="orange", size=11)
                    )
                    fig_nivel.update_layout(
                        yaxis=dict(title="Volumen (Litros)",
                                   range=[y_min - margen_inf, y_max + margen_sup], tickformat=","),
                        xaxis=dict(title="Meses del Año", dtick="M1", tickformat="%b",
                                   showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
                        height=400, margin=dict(t=20, b=50, l=60, r=20),
                        plot_bgcolor="rgba(0,0,0,0)", showlegend=False
                    )
                    st.plotly_chart(fig_nivel, use_container_width=True,
                                    config={'scrollZoom': False, 'displayModeBar': True})
                    st.markdown("---")

                    st.write("####  Curva de Optimización del Estanque")
                    capacidades_prueba, eficiencias, cap_optima, ef_actual, ef_optima = \
                        calcular_curva_optimizacion(df_slice, capacidad_maxima)

                    col_opt1, col_opt2 = st.columns(2)
                    col_opt1.metric(
                        " Tamaño Óptimo del Estanque",
                        f"{formato_chileno(cap_optima, 0)} L",
                        help="Punto eficiente donde la curva se estabiliza o alcanza el 95%."
                    )
                    col_opt2.metric(
                        " Cobertura con Tu Estanque",
                        f"{formato_chileno(ef_actual, 1)} %",
                        delta=f"{formato_chileno(ef_actual - ef_optima, 1)} % vs óptimo"
                    )

                    fig_opt = go.Figure()
                    fig_opt.add_trace(go.Scatter(
                        x=capacidades_prueba, y=eficiencias, mode='lines+markers',
                        name='Cobertura (%)', line=dict(color='#3498db', width=3)
                    ))
                    fig_opt.add_vline(x=capacidad_maxima, line_dash="dash", line_color="#e74c3c",
                                      line_width=2,
                                      annotation_text=f"Tu Estanque\n({capacidad_maxima:,.0f} L)",
                                      annotation_position="bottom right")
                    fig_opt.add_vline(x=cap_optima, line_dash="dash", line_color="#2ecc71",
                                      line_width=2,
                                      annotation_text=f"Óptimo\n({cap_optima:,.0f} L)",
                                      annotation_position="top left")
                    fig_opt.update_layout(yaxis_title="% Demanda Anual Cubierta",
                                          xaxis_title="Capacidad Estanque (L)", margin=dict(b=40))
                    st.plotly_chart(fig_opt, use_container_width=True,
                                    config={'scrollZoom': False, 'displayModeBar': True})

                    # Balance humano
                    st.markdown("---")
                    st.write("#### Resumen de Abastecimiento Humano")
                    aporte_real_persona = (total_suministrado / total_demanda) * litros_persona_dia \
                        if total_demanda > 0 else 0

                    col_tank, col_persona = st.columns(2)
                    with col_tank:
                        st.metric(
                            label="Volumen Total Cubierto",
                            value=f"{formato_chileno(total_suministrado, 0)} L",
                            delta=f"de {formato_chileno(total_demanda, 0)} L demandados en el año",
                            delta_color="off"
                        )
                        porcentaje_limite = min(pct_cubierto, 100.0)
                        st.markdown(f"""
                        <div style="margin-top: 15px;">
                            <div style="font-weight: bold; font-size: 14.5px; margin-bottom: 6px;
                                        color: #d4f7ff; letter-spacing: 0.5px;">
                                El estanque cubrió el {formato_chileno(pct_cubierto, 1)}% de la necesidad
                            </div>
                            <div style="background-color: rgba(255,255,255,0.08); border-radius: 8px;
                                        height: 22px; width: 100%; position: relative; overflow: hidden;
                                        border: 1px solid rgba(255,255,255,0.15);
                                        box-shadow: inset 0px 4px 6px rgba(0,0,0,0.4);">
                                <div style="background-color: #3498db; width: {porcentaje_limite}%;
                                            height: 100%; position: absolute; left: 0; top: 0;
                                            transition: width 1s ease-in-out;
                                            box-shadow: 2px 0px 4px rgba(0,0,0,0.3);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_persona:
                        st.metric(
                            label=" Agua Asegurada por Persona",
                            value=f"{formato_chileno(aporte_real_persona, 1)} L/día",
                            delta=f"Meta original: {litros_persona_dia} L/día",
                            delta_color="off"
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Seguridad hídrica
                    st.markdown("---")
                    st.write("#### Análisis de Seguridad y Disponibilidad")

                    total_dias    = len(df_slice)
                    dias_criticos = (df_slice['Estanque Final (L)'] < (capacidad_maxima * 0.10)).sum()
                    dias_optimos  = (df_slice['Estanque Final (L)'] > (capacidad_maxima * 0.80)).sum()
                    dias_medios   = total_dias - dias_criticos - dias_optimos
                    p_critico     = (dias_criticos / total_dias) * 100
                    p_medio       = (dias_medios   / total_dias) * 100
                    p_optimo      = (dias_optimos  / total_dias) * 100

                    color_gris   = "#95a5a6"
                    color_cielo  = "#a5e4ff"
                    color_oscuro = "#2e68b1"

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<h3 style='color:{color_gris}; margin-bottom:0;'> {formato_chileno(p_critico, 1)}%</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_gris}; font-weight:bold; margin-bottom:0;'>Crítico (<10%)</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_gris}; font-size:0.8rem;'>Reserva mínima: {dias_criticos} días.</p>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<h3 style='color:{color_cielo}; margin-bottom:0;'> {formato_chileno(p_medio, 1)}%</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_cielo}; font-weight:bold; margin-bottom:0;'>Estado Operativo</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_cielo}; font-size:0.8rem;'>Nivel funcional: {dias_medios} días.</p>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<h3 style='color:{color_oscuro}; margin-bottom:0;'> {formato_chileno(p_optimo, 1)}%</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_oscuro}; font-weight:bold; margin-bottom:0;'>Seguridad (>80%)</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color_oscuro}; font-size:0.8rem;'>Autonomía total: {dias_optimos} días.</p>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="display: flex; width: 100%; height: 26px; border-radius: 13px;
                                overflow: hidden; margin-top: 15px;
                                border: 1px solid rgba(255,255,255,0.1);
                                box-shadow: inset 0px 2px 4px rgba(0,0,0,0.3);">
                        <div style="width: {p_critico}%; background-color: {color_gris};"></div>
                        <div style="width: {p_medio}%; background-color: {color_cielo};"></div>
                        <div style="width: {p_optimo}%; background-color: {color_oscuro};"></div>
                    </div>
                    <div style="display: flex; width: 100%; font-size: 11px; font-weight: bold;
                                color: {color_cielo}; padding-top: 8px;">
                        <div style="width: {p_critico}%;"></div>
                        <div style="width: {p_medio}%; text-align: center; letter-spacing: 1px;">ESTADO OPERATIVO</div>
                        <div style="width: {p_optimo}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Tabla diaria
                    df_view = df_slice.drop(columns=['Fecha Pura', 'Mes_Dia', 'Eje X',
                                                     'Agua Acumulada Teórica (L)'])
                    df_estilizado = df_view.style.apply(colorear_filas_diarias, axis=1).format({
                        "Lluvia (mm)":        lambda x: formato_chileno(x, 2),
                        "Captado (L)":        lambda x: formato_chileno(x, 1),
                        "Demanda (L)":        lambda x: formato_chileno(x, 1),
                        "Estanque Final (L)": lambda x: formato_chileno(x, 0),
                        "Rebalse (L)":        lambda x: formato_chileno(x, 0),
                        "Déficit Diario (L)": lambda x: formato_chileno(x, 0),
                    }).hide(axis="index")

                    st.write(f"####  Tabla Diaria de la Simulación")
                    st.dataframe(df_estilizado, use_container_width=True, height=400)

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_view.to_excel(writer, index=False, sheet_name=nombre)
                    buffer.seek(0)
                    st.download_button(
                        label=f"📥 Exportar {nombre}", data=buffer,
                        file_name=f"balance_{nombre.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"btn_{nombre}"
                    )

                with tab_n:  mostrar_detalles_escenario(df_normal,   "Año Normal")
                with tab_s:  mostrar_detalles_escenario(df_seco,     "Año Seco")
                with tab_ll: mostrar_detalles_escenario(df_lluvioso, "Año Lluvioso")

                promedios_m = lluvias_mensuales.groupby('Mes')[codigo_estacion].mean()
                promedios_m = promedios_m.reindex([f"{i:02d}" for i in range(1, 13)]).fillna(0)

                st.session_state['resultado'] = {
                    'estacion_cercana':         estacion_cercana,
                    'lluvias_estacion':         lluvias_mensuales,
                    'codigo_estacion':          codigo_estacion,
                    'anio_inicio':              anio_inicio,
                    'anio_fin':                 anio_fin,
                    'precipitaciones_promedio': promedios_m.tolist(),
                }

                st.session_state['informe_datos'] = {
                    'nombre_proyecto':        nombre_proyecto,
                    'lat_proyecto':           lat_proyecto,
                    'lon_proyecto':           lon_proyecto,
                    'alt_proyecto':           alt_proyecto,
                    'techo':                  techo,
                    'capacidad_maxima':       capacidad_maxima,
                    'eficiencia':             eficiencia,
                    'numero_personas':        numero_personas,
                    'litros_persona_dia':     litros_persona_dia,
                    'consumo_mensual':        consumo_mensual,
                    'tipo_uso':               tipo_uso,
                    'meses_seleccionados':    meses_seleccionados,
                    'est_nombre':             estacion_cercana['Nombre'],
                    'est_codigo':             codigo_estacion,
                    'est_altitud':            alt_estacion if not pd.isna(alt_estacion) else "—",
                    'desnivel':               round(desnivel) if desnivel else "—",
                    'distancia':              distancia_minima,
                    'anio_inicio':            anio_inicio,
                    'anio_fin':               anio_fin,
                    'pct_calidad':            porcentaje_calidad,
                    'anio_seco':              anio_seco,
                    'anio_mediano':           anio_mediano,
                    'anio_lluvioso':          anio_lluvioso,
                    'totales_anio':           totales_por_anio.to_dict(),
                    'df_seco':                df_seco,
                    'df_normal':              df_normal,
                    'df_lluvioso':            df_lluvioso,
                    'lluvias_mensuales':      lluvias_mensuales,
                    'precipitaciones_promedio': promedios_m.tolist(),
                }

        except Exception as e:
            st.error(f"⚠️ **Error inesperado en el cálculo:** {e}")

# ===============================================================
# TAB 2 — RESUMEN PRECIPITACIONES
# ===============================================================
with tab2:
    if 'resultado' not in st.session_state:
        st.info("<-- Primero presiona **Buscar Estación y Calcular Balance** en la primera pestaña.")
    else:
        r = st.session_state['resultado']
        estacion_cercana         = r['estacion_cercana']
        lluvias_estacion         = r['lluvias_estacion']
        codigo_estacion          = r['codigo_estacion']
        anio_inicio, anio_fin    = r['anio_inicio'], r['anio_fin']
        precipitaciones_promedio = r['precipitaciones_promedio']

        st.write(f"###  Resumen Histórico ({anio_inicio}–{anio_fin}) — Estación {estacion_cercana['Nombre']}")
        st.write("*(Esta pestaña resume el comportamiento general del clima mes a mes)*")

        nombres_mes = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May',  6:'Jun',
                       7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}

        df_pp = lluvias_estacion[['Fecha', 'Año', 'Mes_num', codigo_estacion]].copy()
        df_pp.rename(columns={codigo_estacion: 'PP (mm)'}, inplace=True)

        meses_validos_por_anio = df_pp.dropna(subset=['PP (mm)']).groupby('Año')['Mes_num'].nunique()
        anios_completos        = meses_validos_por_anio[meses_validos_por_anio == 12].index
        if len(anios_completos) > 0:
            total_anual_promedio = df_pp[df_pp['Año'].isin(anios_completos)].groupby('Año')['PP (mm)'].sum().mean()
        else:
            total_anual_promedio = df_pp.groupby('Año')['PP (mm)'].sum().mean()

        st.markdown("####  Precipitación mensual por año (mm)")
        pivot = df_pp.pivot(index='Año', columns='Mes_num', values='PP (mm)')
        pivot = pivot.reindex(index=range(anio_inicio, anio_fin + 1), columns=range(1, 13))
        pivot.columns        = [nombres_mes[m] for m in pivot.columns]
        pivot['Total Anual'] = pivot.sum(axis=1, min_count=1)

        fila_promedio      = pd.Series(precipitaciones_promedio + [sum(precipitaciones_promedio)],
                                       index=pivot.columns, name='📊 Promedio')
        pivot_con_promedio = pd.concat([pivot, fila_promedio.to_frame().T])

        def estilo_pivot(val):
            return 'background-color: #f0f0f0; color: #999999; font-style: italic' if pd.isna(val) else ''

        def estilo_fila_promedio(row):
            style = 'background-color: #1a6fa3; color: #ffffff; font-weight: bold; border-top: 2px solid #000'
            return [style] * len(row) if row.name == '📊 Promedio' else [''] * len(row)

        pivot_estilizado = (pivot_con_promedio.style
                            .map(estilo_pivot)
                            .apply(estilo_fila_promedio, axis=1)
                            .format(lambda x: formato_chileno(x, 0) if not pd.isna(x) else '—'))
        st.dataframe(pivot_estilizado, use_container_width=True)

        st.markdown("####  Total anual de precipitación por año")
        totales_anuales = df_pp.groupby('Año')['PP (mm)'].sum(min_count=1).reset_index().dropna(subset=['PP (mm)'])
        fig2 = px.bar(totales_anuales, x='Año', y='PP (mm)', color='PP (mm)',
                      color_continuous_scale='Blues')
        fig2.add_hline(y=total_anual_promedio, line_dash='dash', line_color='red',
                       annotation_text=f"Promedio: {formato_chileno(total_anual_promedio, 1)} mm",
                       annotation_position="top left")
        fig2.update_layout(yaxis_title="Precipitación total (mm)", xaxis_title="",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True})

        st.markdown("####  Promedio mensual histórico (Climograma)")
        promedios_mes = pd.DataFrame({'Mes': list(nombres_mes.values()), 'PP (mm)': precipitaciones_promedio})
        fig3 = px.bar(promedios_mes, x='Mes', y='PP (mm)', color='PP (mm)',
                      color_continuous_scale='Blues')
        fig3.update_layout(yaxis_title="Precipitación promedio (mm)", xaxis_title="",
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True})

# ===============================================================
# BOTÓN DE INFORME FINAL
# ===============================================================
if 'informe_datos' in st.session_state:
    st.markdown("---")
    st.markdown("### 📄 Informe del Proyecto")
    st.write("Descarga un informe PDF con el resumen ejecutivo del proyecto: parámetros, estación, escenarios climáticos y recomendación de tamaño óptimo.")
    buf = generar_informe_pdf(st.session_state['informe_datos'])
    nombre_archivo = st.session_state['informe_datos']['nombre_proyecto'].replace(' ', '_')
    st.download_button(
        label="📥 Descargar Informe Completo (.pdf)",
        data=buf,
        file_name=f"Informe_SCALL_{nombre_archivo}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
