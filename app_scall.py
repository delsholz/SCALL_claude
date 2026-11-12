import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from informe_pdf import generar_informe_pdf
from utils import formato_chileno, calcular_distancia_vectorizada
from data_loader import cargar_datos_crudos, aplicar_filtro_calidad
from simulator import (simular_continua, simular_escenario,
                       encontrar_anios_extremos, calcular_curva_optimizacion)

try:
    from history import (guardar_simulacion, cargar_historial_con_estado, normalizar_usuario,
                         cargar_simulacion_completa, eliminar_simulacion)
    _HISTORIAL_DISPONIBLE = True
except Exception:
    _HISTORIAL_DISPONIBLE = False

st.set_page_config(page_title="Simulador SCALL", page_icon="💧", layout="wide")


@st.cache_data
def _calcular_pp_media_nacional():
    """Precipitación anual media (mm/año) por estación CR2, solo años con ≥300 días válidos."""
    df_est, df_diario, _, codigos = cargar_datos_crudos()
    codigos_ok = [c for c in codigos if c in df_diario.columns]

    anios  = df_diario['Fecha'].str.slice(0, 4)
    subset = df_diario[codigos_ok].copy()
    subset['_anio'] = anios

    sum_anual   = subset.groupby('_anio')[codigos_ok].sum(min_count=1)
    count_anual = subset.groupby('_anio')[codigos_ok].count()
    sum_anual[count_anual < 300] = np.nan

    pp_media  = sum_anual.mean()
    df_est_ok = df_est[df_est['Codigo'].isin(codigos_ok)].copy()
    df_est_ok['PP_media'] = df_est_ok['Codigo'].map(pp_media)
    return df_est_ok.dropna(subset=['PP_media', 'Latitud', 'Longitud'])


@st.cache_data
def _triangular_delaunay():
    """
    Triangulación de Delaunay sobre las estaciones CR2.
    Devuelve GeoJSON de triángulos + lista de PP media por triángulo.
    Filtra aristas largas para evitar triángulos que crucen océano o zonas sin datos.
    """
    from scipy.spatial import Delaunay

    df   = _calcular_pp_media_nacional()
    pts  = df[['Longitud', 'Latitud']].values.astype(float)
    vals = df['PP_media'].values.astype(float)

    tri      = Delaunay(pts)
    MAX_EDGE = 2.2  # grados — filtra triángulos que cruzan zonas sin estaciones

    features  = []
    tri_vals  = []

    for simplex in tri.simplices:
        coords = pts[simplex]
        edges  = [
            np.linalg.norm(coords[0] - coords[1]),
            np.linalg.norm(coords[1] - coords[2]),
            np.linalg.norm(coords[0] - coords[2]),
        ]
        if max(edges) > MAX_EDGE:
            continue

        ring = [[float(c[0]), float(c[1])] for c in coords]
        ring.append(ring[0])  # cerrar polígono

        features.append({
            "type": "Feature",
            "id": str(len(features)),
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {}
        })
        tri_vals.append(float(vals[simplex].mean()))

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, tri_vals, df


# ===============================================================
# CSS
# ===============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, p, label, div {
    font-family: 'Poppins', sans-serif !important;
}


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
try:
    import base64
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "Logo_Amulen_blanco.png"), "rb") as _f_logo:
        _logo_b64 = base64.b64encode(_f_logo.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="background-color:#151434; border-radius:8px; padding:16px 12px 12px 12px; margin-bottom:12px;">
            <img src="data:image/png;base64,{_logo_b64}" style="width:100%;">
        </div>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    st.sidebar.title("💧 SCALL — Fundación Amulén")
st.sidebar.markdown("---")
usuario = st.sidebar.text_input(
    "👤 Tu nombre (para el historial)",
    placeholder="Ej: Diego, María, Equipo Amulén...",
    help="Cada persona ve solo sus propias simulaciones en el historial."
)
st.sidebar.markdown("---")
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
    help="Solo se usarán estaciones con este % de datos válidos (2000 a 2020)."
)

st.sidebar.header("3. Coordenadas de Ubicación")

_modo_coord = st.sidebar.radio(
    "Método de ingreso",
    ["Coordenadas manuales", "Link de Google Maps"],
    horizontal=True,
)

def _extraer_coords_gmaps(url: str):
    """Extrae latitud y longitud desde un link de Google Maps."""
    import re
    # Formato: @lat,lon o /place/.../@lat,lon o ?q=lat,lon
    patrones = [
        r'@(-?\d+\.\d+),(-?\d+\.\d+)',
        r'[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)',
        r'll=(-?\d+\.\d+),(-?\d+\.\d+)',
    ]
    for pat in patrones:
        m = re.search(pat, url)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None, None

if _modo_coord == "Link de Google Maps":
    _gmaps_url = st.sidebar.text_input(
        "Pega el link de Google Maps",
        placeholder="https://maps.google.com/...",
    )
    _lat_gmaps, _lon_gmaps = _extraer_coords_gmaps(_gmaps_url) if _gmaps_url else (None, None)
    if _gmaps_url and _lat_gmaps is None:
        st.sidebar.warning("No se pudieron extraer coordenadas del link. Verifica que sea un link de Google Maps con ubicación.")
    _lat_default  = _lat_gmaps  if _lat_gmaps  is not None else -33.4500
    _lon_default  = _lon_gmaps  if _lon_gmaps  is not None else -70.6500
    if _lat_gmaps is not None:
        st.sidebar.success(f"Coordenadas detectadas: {_lat_gmaps:.4f}, {_lon_gmaps:.4f}")
    lat_proyecto = st.sidebar.number_input("Latitud",  value=_lat_default, format="%.4f")
    lon_proyecto = st.sidebar.number_input("Longitud", value=_lon_default, format="%.4f")
else:
    st.sidebar.write("Ingresa la latitud y longitud de tu proyecto.")
    lat_proyecto = st.sidebar.number_input("Latitud",  value=-33.4500, format="%.4f")
    lon_proyecto = st.sidebar.number_input("Longitud", value=-70.6500, format="%.4f")

if st.sidebar.button("📍 Obtener altitud automáticamente"):
    with st.sidebar:
        with st.spinner("Consultando altitud..."):
            elev = None

            # Intento 1: OpenTopoData (hasta 2 intentos)
            for _ in range(2):
                try:
                    resp = requests.get(
                        f"https://api.opentopodata.org/v1/srtm90m?locations={lat_proyecto},{lon_proyecto}",
                        timeout=6
                    )
                    data = resp.json()
                    if data.get("status") == "OK":
                        elev = int(round(data["results"][0]["elevation"]))
                        break
                except Exception:
                    pass

            # Intento 2: Open-Meteo como respaldo
            if elev is None:
                try:
                    resp2 = requests.get(
                        f"https://api.open-meteo.com/v1/elevation?latitude={lat_proyecto}&longitude={lon_proyecto}",
                        timeout=6
                    )
                    data2 = resp2.json()
                    if "elevation" in data2 and data2["elevation"]:
                        elev = int(round(data2["elevation"][0]))
                except Exception:
                    pass

            if elev is not None:
                st.session_state['alt_auto'] = elev
            else:
                st.sidebar.warning("No se pudo obtener la altitud automáticamente. Ingresa el valor manualmente.")

alt_default = int(st.session_state.get('alt_auto', 500))
alt_proyecto = st.sidebar.number_input("Altitud (m.s.n.m.)", value=alt_default, step=10)

coef_altitud = 5
st.sidebar.subheader("Gradiente Orográfico (Lluvia vs Altura)")
coef_orografico = st.sidebar.slider(
    "Variación de lluvia (% por cada 100m de desnivel)",
    min_value=-15.0, max_value=15.0, value=0.0, step=1.0,
    help=(
        "Ajusta los mm de lluvia de la estación para reflejar la diferencia de altura con tu proyecto. "
        "A diferencia del coeficiente k, este parámetro SÍ modifica la lluvia simulada.\n\n"
        "Referencia rápida:\n"
        "• Valle / costa plana → 0%\n"
        "• Precordillera (proyecto más alto) → +3% a +7%\n"
        "• Cordillera alta (proyecto más alto) → +5% a +12%\n"
        "• Sotavento / sombra de lluvia → −5% a −10%\n\n"
        "Default 0%: sin ajuste. Úsalo solo si conoces el comportamiento del terreno; "
        "un ajuste incorrecto introduce más error que dejarlo en 0."
    )
)

st.title("💧 Simulador de Cosecha de Aguas Lluvias SCALL")
st.write("Calcula la viabilidad de tu estanque analizando la **realidad climática continua** desde 2000 a 2020 "
         "(utilizando datos oficiales del **(CR)²**) y extrayendo los años reales extremos.")

# ===============================================================
# FUNCIONES DE VISUALIZACIÓN (compartidas entre tabs)
# ===============================================================
def colorear_filas_diarias(row):
    estilos = [""] * len(row)
    if "Déficit Diario (L)" not in row.index:
        return estilos
    idx_sd = list(row.index).index("Déficit Diario (L)")
    if row["Déficit Diario (L)"] == 0:
        estilos[idx_sd] = "background-color: #d4edda; color: #155724; font-weight: bold"
    else:
        estilos[idx_sd] = "background-color: #f8d7da; color: #721c24; font-weight: bold"
    return estilos


def mostrar_detalles_escenario(df_slice, nombre, key_suffix="", curva_opt=None):
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
    eje_col = 'Eje X'
    y_vals  = df_slice["Agua Acumulada Teórica (L)"]
    y_min, y_max = y_vals.min(), y_vals.max()
    margen_inf   = max(abs(y_min) * 0.4, capacidad_maxima * 0.20)
    margen_sup   = max((y_max - capacidad_maxima) * 0.25, capacidad_maxima * 0.15)

    fig_nivel = go.Figure()
    fig_nivel.add_trace(go.Scatter(
        x=df_slice[eje_col], y=y_vals, mode='lines', fill='tozeroy',
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
    st.plotly_chart(fig_nivel, width="stretch",
                    config={'scrollZoom': False, 'displayModeBar': True},
                    key=f"fig_nivel{key_suffix}")
    st.markdown("---")

    st.write("####  Curva de Optimización del Estanque")
    if curva_opt is None:
        curva_opt = calcular_curva_optimizacion(df_slice, capacidad_maxima)
    capacidades_prueba, eficiencias, cap_optima, ef_actual, ef_optima = curva_opt

    col_opt1, col_opt2 = st.columns(2)
    col_opt1.metric(
        " Tamaño Óptimo del Estanque",
        f"{formato_chileno(cap_optima, 0)} L",
        help="Primera capacidad que alcanza el 95% de cobertura o, si no se llega a 95%, "
             "el codo de la curva (método Kneedle) donde la ganancia marginal se estabiliza."
    )
    col_opt2.metric(
        " Cobertura con Tu Estanque",
        f"{formato_chileno(ef_actual, 1)} %",
        delta=f"{formato_chileno(ef_actual - ef_optima, 1)} % vs óptimo",
        delta_color="normal" if ef_actual >= ef_optima else "inverse"
    )

    fig_opt = go.Figure()

    # Curva principal con puntos en cada simulación
    fig_opt.add_trace(go.Scatter(
        x=capacidades_prueba, y=eficiencias,
        mode='lines+markers', name='Cobertura (%)',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6, color='#3498db', symbol='circle',
                    line=dict(color='white', width=1)),
        fill='tozeroy', fillcolor='rgba(52,152,219,0.10)',
        hovertemplate="<b>%{x:,.0f} L</b><br>Cobertura: %{y:.1f}%<extra></extra>"
    ))

    # Punto marcado: Tu Estanque actual
    idx_actual = int(np.argmin(np.abs(np.array(capacidades_prueba) - capacidad_maxima)))
    fig_opt.add_trace(go.Scatter(
        x=[capacidades_prueba[idx_actual]], y=[ef_actual],
        mode='markers',
        marker=dict(color='#e74c3c', size=13, symbol='circle',
                    line=dict(color='white', width=2)),
        name=f'Tu Estanque ({formato_chileno(capacidad_maxima, 0)} L)',
        hovertemplate=f"Tu Estanque: {capacidad_maxima:,.0f} L<br>Cobertura: {ef_actual:.1f}%<extra></extra>"
    ))

    # Punto marcado: Óptimo (solo si difiere del actual)
    if abs(cap_optima - capacidad_maxima) > 1000 and cap_optima in capacidades_prueba:
        fig_opt.add_trace(go.Scatter(
            x=[cap_optima], y=[ef_optima],
            mode='markers',
            marker=dict(color='#2ecc71', size=13, symbol='diamond',
                        line=dict(color='white', width=2)),
            name=f'Óptimo ({formato_chileno(cap_optima, 0)} L)',
            hovertemplate=f"Óptimo: {cap_optima:,.0f} L<br>Cobertura: {ef_optima:.1f}%<extra></extra>"
        ))

    # Líneas verticales con anotaciones escalonadas verticalmente para evitar superposición
    if abs(cap_optima - capacidad_maxima) > 1000:
        opt_es_mayor = cap_optima > capacidad_maxima
        fig_opt.add_vline(
            x=capacidad_maxima, line_dash="solid", line_color="#e74c3c", line_width=1.5
        )
        fig_opt.add_annotation(
            x=capacidad_maxima, xref="x",
            y=0.97, yref="paper",
            text=f"Tu Estanque<br>{capacidad_maxima:,.0f} L | {ef_actual:.1f}%",
            showarrow=False,
            font=dict(color="#e74c3c", size=10),
            xanchor="right" if opt_es_mayor else "left",
            yanchor="top",
        )
        fig_opt.add_vline(
            x=cap_optima, line_dash="dash", line_color="#2ecc71", line_width=1.5
        )
        fig_opt.add_annotation(
            x=cap_optima, xref="x",
            y=0.74, yref="paper",
            text=f"Óptimo<br>{cap_optima:,.0f} L | {ef_optima:.1f}%",
            showarrow=False,
            font=dict(color="#2ecc71", size=10),
            xanchor="left" if opt_es_mayor else "right",
            yanchor="top",
        )
    else:
        fig_opt.add_vline(
            x=capacidad_maxima, line_dash="dash", line_color="#2ecc71", line_width=1.5
        )
        fig_opt.add_annotation(
            x=capacidad_maxima, xref="x",
            y=0.97, yref="paper",
            text=f"Tu Estanque ≈ Óptimo<br>{capacidad_maxima:,.0f} L | {ef_actual:.1f}%",
            showarrow=False,
            font=dict(color="#2ecc71", size=10),
            xanchor="left",
            yanchor="top",
        )

    max_ef_vis = max(eficiencias) if eficiencias else 100
    fig_opt.update_layout(
        yaxis=dict(title="% Demanda Anual Cubierta",
                   range=[0, min(max_ef_vis * 1.18, 105)],
                   ticksuffix="%"),
        xaxis=dict(title="Capacidad del Estanque (L)", tickformat=","),
        height=400, margin=dict(t=30, b=60, l=65, r=20),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_opt, width="stretch",
                    config={'scrollZoom': False, 'displayModeBar': True},
                    key=f"fig_opt{key_suffix}")

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
                        color: #3498db; letter-spacing: 0.5px;">
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
    color_cielo  = "#a5e4ff"   # segmento de la barra (fondo)
    color_cielo_txt = "#2596cf"  # texto legible en tema claro y oscuro
    color_oscuro = "#2e68b1"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<h3 style='color:{color_gris}; margin-bottom:0;'> {formato_chileno(p_critico, 1)}%</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_gris}; font-weight:bold; margin-bottom:0;'>Crítico (<10%)</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_gris}; font-size:0.8rem;'>Reserva mínima: {dias_criticos} días.</p>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<h3 style='color:{color_cielo_txt}; margin-bottom:0;'> {formato_chileno(p_medio, 1)}%</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_cielo_txt}; font-weight:bold; margin-bottom:0;'>Estado Operativo</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_cielo_txt}; font-size:0.8rem;'>Nivel funcional: {dias_medios} días.</p>", unsafe_allow_html=True)
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
                color: {color_cielo_txt}; padding-top: 8px;">
        <div style="width: {p_critico}%;"></div>
        <div style="width: {p_medio}%; text-align: center; letter-spacing: 1px;">ESTADO OPERATIVO</div>
        <div style="width: {p_optimo}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabla diaria
    cols_a_eliminar = ['Fecha Pura', 'Mes_Dia', 'Eje X', 'Eje X Real', 'Agua Acumulada Teórica (L)', 'tipo']
    df_view = df_slice.drop(columns=[c for c in cols_a_eliminar if c in df_slice.columns])
    df_estilizado = df_view.style.apply(colorear_filas_diarias, axis=1).format({
        "Lluvia (mm)":        lambda x: formato_chileno(x, 2),
        "Captado (L)":        lambda x: formato_chileno(x, 1),
        "Demanda (L)":        lambda x: formato_chileno(x, 1),
        "Estanque Final (L)": lambda x: formato_chileno(x, 0),
        "Rebalse (L)":        lambda x: formato_chileno(x, 0),
        "Déficit Diario (L)": lambda x: formato_chileno(x, 0),
    }).hide(axis="index")

    st.write(f"####  Tabla Diaria de la Simulación")
    st.dataframe(df_estilizado, width="stretch", height=400)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_view.to_excel(writer, index=False, sheet_name=nombre[:31])
    buffer.seek(0)
    st.download_button(
        label=f"📥 Exportar {nombre}", data=buffer,
        file_name=f"balance_{nombre.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"btn_{nombre}{key_suffix}"
    )


# ===============================================================
# TABS
# ===============================================================
tab1, tab2, tab3, tab4 = st.tabs([" Simulación Histórica y Escenarios", " Resumen Mensual Histórico", " Mapa de Factibilidad Nacional", "Historial"])

if 'simulacion_calculada' not in st.session_state:
    st.session_state.simulacion_calculada = False

with tab1:
    if not st.session_state.simulacion_calculada and not st.session_state.get('simulacion_desde_historial', False):
        st.info(" **¡Bienvenido al Simulador SCALL!**\n\n"
                "1. Configura los parámetros de tu techo y consumo en la barra lateral (izquierda).\n"
                "2. Ingresa las coordenadas exactas de tu proyecto.\n"
                "3. Presiona el botón para iniciar el análisis climático.")

    if st.button("Buscar Estación y Calcular Balance", type="primary"):
        st.session_state.simulacion_calculada = True
        st.session_state['simulacion_desde_historial'] = False
        st.session_state['_sim_guardada'] = False

    if st.session_state.get('simulacion_desde_historial', False) and 'informe_datos' in st.session_state:
        _d = st.session_state['informe_datos']
        st.success(f"📂 Simulación cargada desde historial: **{_d.get('nombre_proyecto', '')}**")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Estación", _d.get('est_nombre', '—'))
        _c2.metric("Período analizado", f"{_d.get('anio_inicio', '')}–{_d.get('anio_fin', '')}")
        _c3.metric("Años seco / normal / húmedo",
                   f"{_d.get('anio_seco','?')} / {_d.get('anio_mediano','?')} / {_d.get('anio_lluvioso','?')}")
        st.info("Puedes ver el **Resumen Mensual** en la pestaña 2 y descargar el **PDF** al final de la página. "
                "Presiona el botón de arriba para recalcular con los parámetros actuales del panel lateral.")

    if st.session_state.simulacion_calculada:
        try:
            with st.spinner("Construyendo modelos y ejecutando la simulación..."):

                try:
                    df_estaciones, df_diario, df_mensual, df_est_crudas, codigos = aplicar_filtro_calidad(umbral_calidad, "2000", "2020")
                except (FileNotFoundError, ValueError) as e:
                    st.error(f"⚠️ **Error en los datos:** {e}")
                    st.stop()

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
                cinco_cercanas   = df_estaciones[df_estaciones['Codigo'] != estacion_cercana['Codigo']].nsmallest(5, 'Dist')
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
                desnivel = desnivel_real
                
                # Fórmula del Gradiente Orográfico
                variacion_pct = (coef_orografico / 100.0) * (desnivel_real / 100.0)
                multiplicador_lluvia = max(0.0, 1.0 + variacion_pct) # Evita multiplicadores negativos
                
                alt_txt = f" | Altitud: **{formato_chileno(alt_estacion, 0)} m.s.n.m.** (desnivel: {formato_chileno(desnivel_abs, 0)} m)" \
                               if not pd.isna(alt_estacion) else ""
                st.info(f"📍 **Estación más cercana:** {estacion_cercana['Nombre']} "
                        f"*(Distancia efectiva: **{formato_chileno(distancia_minima, 1)} km**)*{alt_txt}")
                
                TABLA_GRADIENTE = (
                    "| Terreno | Gradiente |\n"
                    "|---|---|\n"
                    "| Valle / costa plana | 0% |\n"
                    "| Precordillera (300–600 m sobre estación) | +4% |\n"
                    "| Precordillera alta (600–1.000 m) | +6% |\n"
                    "| Cordillera (> 1.000 m) | +8% |\n"
                    "| Sotavento / sombra de lluvia | −5% a −10% |"
                )
                if multiplicador_lluvia != 1.0:
                    signo = "+" if multiplicador_lluvia > 1 else ""
                    st.info(
                        f"⚙️ **Gradiente orográfico activo:** las lluvias de la estación se ajustaron "
                        f"**{signo}{(multiplicador_lluvia - 1)*100:.1f}%** por los "
                        f"{formato_chileno(desnivel_real, 0)} m de desnivel. "
                        f"Si quieres cambiar el ajuste, modifica el slider en la barra lateral.\n\n"
                        f"{TABLA_GRADIENTE}"
                    )
                elif coef_orografico == 0.0 and desnivel_real > 300:
                    if desnivel_real > 1000:
                        gradiente_sugerido = 8
                    elif desnivel_real > 600:
                        gradiente_sugerido = 6
                    else:
                        gradiente_sugerido = 4
                    st.info(
                        f"💡 **Sugerencia:** Tu proyecto está **{formato_chileno(desnivel_real, 0)} m más alto** "
                        f"que la estación — probablemente llueve más allí. "
                        f"Considera usar **+{gradiente_sugerido}%** en el slider de Gradiente Orográfico y recalcular.\n\n"
                        f"{TABLA_GRADIENTE}"
                    )

                # --- Mapa ---
                fig_mapa = go.Figure()
                fig_mapa.add_trace(go.Scattermap(
                    lat=cinco_cercanas['Latitud'].tolist(), lon=cinco_cercanas['Longitud'].tolist(),
                    mode='markers+text', marker=dict(size=10, color='#888888'),
                    text=cinco_cercanas['Nombre'].tolist(), textposition='top right',
                    name='Estaciones cercanas', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.add_trace(go.Scattermap(
                    lat=[estacion_cercana['Latitud']], lon=[estacion_cercana['Longitud']],
                    mode='markers+text', marker=dict(size=14, color='#0000FF'),
                    text=[estacion_cercana['Nombre']], textposition='top right',
                    name='Estación seleccionada', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.add_trace(go.Scattermap(
                    lat=[lat_proyecto], lon=[lon_proyecto],
                    mode='markers+text', marker=dict(size=14, color='#FF0000'),
                    text=[nombre_proyecto], textposition='top right',
                    name='Tu Proyecto', hovertemplate='%{text}<extra></extra>'
                ))
                fig_mapa.update_layout(
                    map=dict(
                        style='open-street-map',
                        center=dict(lat=(lat_proyecto + estacion_cercana['Latitud']) / 2,
                                    lon=(lon_proyecto + estacion_cercana['Longitud']) / 2),
                        zoom=8
                    ),
                    margin=dict(l=0, r=0, t=0, b=0), height=350,
                    legend=dict(orientation='h', y=-0.08)
                )
                st.plotly_chart(fig_mapa, width="stretch",
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
                    techo, eficiencia, multiplicador_lluvia
                )

                anio_seco, anio_mediano, anio_lluvioso, totales_por_anio = encontrar_anios_extremos(
                    df_sim_completa
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
                st.plotly_chart(fig_cont, width="stretch",
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
                    multiplicador_lluvia=multiplicador_lluvia
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
                st.plotly_chart(fig_comp, width="stretch",
                                config={'scrollZoom': False, 'displayModeBar': True})

                # --- Detalle por escenario ---
                st.write("###  Tablas y Diseño por Escenario")
                tab_n, tab_s, tab_ll = st.tabs([" Año Normal", "Año Seco", " Año Lluvioso"])

                curva_normal   = calcular_curva_optimizacion(df_normal,   capacidad_maxima)
                curva_seco     = calcular_curva_optimizacion(df_seco,     capacidad_maxima)
                curva_lluvioso = calcular_curva_optimizacion(df_lluvioso, capacidad_maxima)

                with tab_n:  mostrar_detalles_escenario(df_normal,   "Año Normal",   key_suffix="_normal",   curva_opt=curva_normal)
                with tab_s:  mostrar_detalles_escenario(df_seco,     "Año Seco",     key_suffix="_seco",     curva_opt=curva_seco)
                with tab_ll: mostrar_detalles_escenario(df_lluvioso, "Año Lluvioso", key_suffix="_lluvioso", curva_opt=curva_lluvioso)

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
                    'curva_normal':           curva_normal,
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
                    'desnivel':               round(desnivel) if not pd.isna(alt_estacion) else "—",   # con signo: + = proyecto sobre la estación
                    'distancia':              distancia_minima,
                    'anio_inicio':            anio_inicio,
                    'anio_fin':               anio_fin,
                    'pct_calidad':            porcentaje_calidad,
                    'multiplicador_lluvia':   multiplicador_lluvia,
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
                st.session_state.pop('_pdf_cache', None)

                if _HISTORIAL_DISPONIBLE and not st.session_state.get('_sim_guardada', False):
                    try:
                        if guardar_simulacion(st.session_state['informe_datos'], usuario=usuario):
                            st.session_state['_sim_guardada'] = True
                            st.toast("✓ Simulación guardada en historial", icon="💾")
                            st.session_state.pop('_hist_cache', None)
                        else:
                            st.toast("No se pudo guardar en el historial (la simulación sí se calculó).", icon="⚠️")
                    except Exception:
                        st.toast("No se pudo guardar en el historial (la simulación sí se calculó).", icon="⚠️")

        except Exception as e:
            st.error(f"⚠️ **Error inesperado en el cálculo:** {e}")

# ===============================================================
# TAB 2 — RESUMEN PRECIPITACIONES
# ===============================================================
with tab2:
    _r2 = st.session_state.get('resultado')
    if _r2 is None:
        st.info("<-- Primero presiona **Buscar Estación y Calcular Balance** en la primera pestaña.")
    elif _r2.get('lluvias_estacion') is None:
        st.info("Esta simulación cargada del historial no incluye el detalle mensual histórico. "
                "Presiona **Buscar Estación y Calcular Balance** en la pestaña 1 para regenerarlo.")
    else:
        r = _r2
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
        pivot_con_promedio.index = pivot_con_promedio.index.astype(str)   # años + «📊 Promedio»: sin tipos mezclados (pyarrow)

        def estilo_pivot(val):
            return 'background-color: #f0f0f0; color: #999999; font-style: italic' if pd.isna(val) else ''

        def estilo_fila_promedio(row):
            style = 'background-color: #1a6fa3; color: #ffffff; font-weight: bold; border-top: 2px solid #000'
            return [style] * len(row) if row.name == '📊 Promedio' else [''] * len(row)

        pivot_estilizado = (pivot_con_promedio.style
                            .map(estilo_pivot)
                            .apply(estilo_fila_promedio, axis=1)
                            .format(lambda x: formato_chileno(x, 0) if not pd.isna(x) else '—'))
        st.dataframe(pivot_estilizado, width="stretch")

        st.markdown("####  Total anual de precipitación por año")
        totales_anuales = df_pp.groupby('Año')['PP (mm)'].sum(min_count=1).reset_index().dropna(subset=['PP (mm)'])
        fig2 = px.bar(totales_anuales, x='Año', y='PP (mm)', color='PP (mm)',
                      color_continuous_scale='Blues')
        fig2.add_hline(y=total_anual_promedio, line_dash='dash', line_color='red',
                       annotation_text=f"Promedio: {formato_chileno(total_anual_promedio, 1)} mm",
                       annotation_position="top left")
        fig2.update_layout(yaxis_title="Precipitación total (mm)", xaxis_title="",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch",
                        config={'scrollZoom': False, 'displayModeBar': True})

        st.markdown("####  Promedio mensual histórico (Climograma)")
        promedios_mes = pd.DataFrame({'Mes': list(nombres_mes.values()), 'PP (mm)': precipitaciones_promedio})
        fig3 = px.bar(promedios_mes, x='Mes', y='PP (mm)', color='PP (mm)',
                      color_continuous_scale='Blues')
        fig3.update_layout(yaxis_title="Precipitación promedio (mm)", xaxis_title="",
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, width="stretch",
                        config={'scrollZoom': False, 'displayModeBar': True})

# ===============================================================
# TAB 3 — MAPA DE FACTIBILIDAD NACIONAL
# ===============================================================
with tab3:
    st.markdown("### 🗺️ Mapa de Factibilidad Nacional — Cosecha de Aguas Lluvias")
    st.write(
        "Zonas trianguladas entre ~400 estaciones CR2 (2000–2020). "
        "Cada triángulo representa la zona entre estaciones reales con el valor interpolado de su área."
    )

    if not st.session_state.get('_mapa_nacional_activo', False):
        if st.button("🗺️ Generar mapa nacional", type="primary", key="btn_mapa_nac"):
            st.session_state['_mapa_nacional_activo'] = True
            st.rerun()
        st.caption("El mapa procesa las ~400 estaciones CR2 (2000-2020); tarda unos segundos "
                   "solo la primera vez por sesión.")
    else:
        with st.spinner("Construyendo triangulación de Delaunay..."):
            _geojson_tri, _vals_tri, df_mapa_nac = _triangular_delaunay()

        _ids_tri = [str(i) for i in range(len(_vals_tri))]

        # Colorscale precipitación: blanco → celeste → azul base → azul oscuro Amulén
        CS_PP = [
            [0.00, "#f0f8ff"],   # casi blanco (zonas muy secas)
            [0.15, "#cce9fa"],   # azul muy claro
            [0.35, "#76c2f5"],   # celeste Amulén
            [0.65, "#2e68b1"],   # azul base Amulén
            [1.00, "#151434"],   # azul oscuro Amulén
        ]

        # Colorscale semáforo para cobertura (rojo → verde)
        CS_COBERTURA = [
            [0.00, "#d73027"],
            [0.25, "#f46d43"],
            [0.50, "#fee08b"],
            [0.75, "#66bd63"],
            [1.00, "#1a9850"],
        ]

        LAYOUT_MAPA = dict(
            map=dict(style='open-street-map', center=dict(lat=-37.0, lon=-71.5), zoom=4),
            margin=dict(l=0, r=0, t=0, b=0), height=680,
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)", bordercolor="#cccccc",
                borderwidth=1, font=dict(size=11),
            ),
        )

        hover_est = (
            "<b>" + df_mapa_nac['Nombre'] + "</b><br>"
            + "PP media: " + df_mapa_nac['PP_media'].round(0).astype(int).astype(str) + " mm/año"
        )

        maptab1, maptab2 = st.tabs(["Precipitación Media Anual", "Factibilidad según tus parámetros"])

        with maptab1:
            fig_nac = go.Figure()
            fig_nac.add_trace(go.Choroplethmap(
                geojson=_geojson_tri,
                locations=_ids_tri,
                z=_vals_tri,
                colorscale=CS_PP,
                zmin=0, zmax=2500,
                marker_opacity=0.72,
                marker_line_width=0.3,
                marker_line_color="rgba(255,255,255,0.25)",
                colorbar=dict(
                    title="mm/año",
                    tickvals=[0, 200, 500, 1000, 2000, 2500],
                    ticktext=["0", "200", "500", "1.000", "2.000", "2.500+"],
                    thickness=14, len=0.65,
                ),
                hovertemplate="<b>PP media:</b> %{z:.0f} mm/año<extra></extra>",
                name="Precipitación",
            ))
            # Estaciones como puntos de referencia
            fig_nac.add_trace(go.Scattermap(
                lat=df_mapa_nac['Latitud'], lon=df_mapa_nac['Longitud'],
                mode='markers',
                marker=dict(size=4, color='rgba(21,20,52,0.55)'),
                text=hover_est,
                hovertemplate='%{text}<extra></extra>',
                name="Estaciones CR2",
            ))
            fig_nac.update_layout(**LAYOUT_MAPA)
            st.plotly_chart(fig_nac, width="stretch", config={'scrollZoom': True})

            st.info(
                "**¿Cómo leer el mapa?**  Cada zona triangulada entre estaciones muestra la "
                "precipitación media anual interpolada (2000–2020).\n\n"
                "| Color | PP media anual | Zona típica |\n"
                "|-------|---------------|-------------|\n"
                "| ⬜ Blanco | < 100 mm | Norte Grande (Tarapacá, Antofagasta) |\n"
                "| 🔵 Azul muy claro | 100–300 mm | Norte Chico (Atacama, Coquimbo) |\n"
                "| 🔵 Celeste | 300–800 mm | Zona Central (Valparaíso, Maule) |\n"
                "| 🔵 Azul medio | 800–1.500 mm | La Araucanía, Los Ríos |\n"
                "| 🟣 Azul oscuro | > 1.500 mm | Los Lagos, Aysén, Magallanes |\n\n"
                "*Triangulación de Delaunay sobre ~400 estaciones CR2. Los puntos negros son las "
                "estaciones reales. Las zonas sin triángulos no tienen cobertura de datos.*"
            )

        with maptab2:
            consumo_diario_mapa = numero_personas * litros_persona_dia
            dias_op_año_mapa    = len(meses_num_seleccionados) * (30 if consumo_fines_semana else 22)
            demanda_anual_mapa  = consumo_diario_mapa * dias_op_año_mapa

            if demanda_anual_mapa <= 0 or len(meses_num_seleccionados) == 0:
                st.warning("Define el consumo y los meses de operación en la barra lateral para ver este mapa.")
            else:
                # Cobertura por triángulo usando la PP media de cada uno
                vals_cob = [
                    min(v * techo * eficiencia / demanda_anual_mapa * 100, 100)
                    for v in _vals_tri
                ]

                st.write(
                    f"Cobertura estimada con techo **{techo:.0f} m²**, eficiencia **{eficiencia*100:.0f}%**, "
                    f"consumo **{consumo_diario_mapa:.0f} L/día** y **{len(meses_num_seleccionados)} meses** "
                    f"de operación. *Estimación simplificada — no incluye el efecto de almacenamiento del estanque.*"
                )

                fig_dyn = go.Figure()
                fig_dyn.add_trace(go.Choroplethmap(
                    geojson=_geojson_tri,
                    locations=_ids_tri,
                    z=vals_cob,
                    colorscale=CS_PP,
                    zmin=0, zmax=100,
                    marker_opacity=0.72,
                    marker_line_width=0.3,
                    marker_line_color="rgba(255,255,255,0.25)",
                    colorbar=dict(
                        title="Cobertura %",
                        tickvals=[0, 20, 40, 60, 80, 100],
                        ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                        thickness=14, len=0.65,
                    ),
                    hovertemplate="<b>Cobertura estimada:</b> %{z:.1f}%<extra></extra>",
                    name="Cobertura",
                ))
                fig_dyn.add_trace(go.Scattermap(
                    lat=df_mapa_nac['Latitud'], lon=df_mapa_nac['Longitud'],
                    mode='markers',
                    marker=dict(size=4, color='rgba(21,20,52,0.55)'),
                    text=hover_est,
                    hovertemplate='%{text}<extra></extra>',
                    name="Estaciones CR2",
                ))

                if 'informe_datos' in st.session_state:
                    id_ = st.session_state['informe_datos']
                    fig_dyn.add_trace(go.Scattermap(
                        lat=[id_['lat_proyecto']], lon=[id_['lon_proyecto']],
                        mode='markers+text',
                        marker=dict(size=14, color='#151434'),
                        text=[id_['nombre_proyecto']],
                        textposition="top right",
                        hovertemplate=f"<b>{id_['nombre_proyecto']}</b><extra></extra>",
                        name="Tu proyecto",
                    ))

                fig_dyn.update_layout(**LAYOUT_MAPA)
                st.plotly_chart(fig_dyn, width="stretch", config={'scrollZoom': True})

                st.info(
                    "La cobertura se estima como **(PP media × techo × eficiencia) ÷ demanda anual**. "
                    "No incluye el efecto de amortiguación del estanque — zonas con lluvia concentrada "
                    "en pocos meses pueden aparecer con cobertura más baja de la real."
                )


# ===============================================================
# TAB 4 — HISTORIAL DE SIMULACIONES
# ===============================================================
with tab4:
    st.markdown("### 🕓 Historial de Simulaciones")

    def _hay(v):
        """True si el valor existe (la base devuelve NaN cuando falta un dato)."""
        return v is not None and not pd.isna(v)

    if not _HISTORIAL_DISPONIBLE:
        st.warning("⚠️ Historial no disponible. Configura las credenciales de Supabase en `.streamlit/secrets.toml`.")
    elif not (usuario or "").strip():
        st.info("👤 Ingresa tu nombre en el panel izquierdo para ver tu historial de simulaciones.")
    else:
        usuario_n = normalizar_usuario(usuario)
        st.markdown(f"Mostrando simulaciones de **{usuario_n}** (no importan las mayúsculas ni los espacios sobrantes).")

        if st.session_state.get('_hist_flash'):
            st.success(st.session_state.pop('_hist_flash'))

        f1, f2 = st.columns([3, 2])
        busqueda = f1.text_input("🔎 Buscar por proyecto o estación", key="hist_busqueda", placeholder="Ej: Crucero, Quilaco…")
        orden = f2.selectbox("Ordenar por", ["Más recientes primero", "Más antiguas primero",
                                             "Mayor cobertura", "Nombre del proyecto"], key="hist_orden")
        g1, g2 = st.columns([3, 2])
        incluir_anon = g1.checkbox(
            "Incluir simulaciones guardadas sin nombre («Anónimo»)", key="hist_anon",
            help="Si corriste una simulación antes de escribir tu nombre en el panel izquierdo, quedó guardada como «Anónimo».")
        if g2.button("🔄 Actualizar historial", key="btn_refresh_hist"):
            st.session_state.pop('_hist_cache', None)
            st.rerun()

        # Se vuelve a pedir a la base solo si cambió el nombre o el interruptor de «Anónimo»
        clave_hist = (usuario_n.lower(), bool(incluir_anon))
        if st.session_state.get('_hist_clave') != clave_hist:
            st.session_state.pop('_hist_cache', None)
            st.session_state['_hist_clave'] = clave_hist
        if '_hist_cache' not in st.session_state:
            with st.spinner("Cargando historial..."):
                st.session_state['_hist_cache'] = cargar_historial_con_estado(usuario_n, incluir_anon)
        df_hist, error_hist = st.session_state['_hist_cache']

        if error_hist:
            st.error(f"No se pudo cargar el historial: {error_hist}")
            st.caption("Tus simulaciones no se perdieron. Pulsa «Actualizar historial» para reintentar.")
        elif df_hist.empty:
            st.info("No hay simulaciones guardadas con este nombre. Ejecuta una simulación y se guardará sola. "
                    "Si ya corriste una sin escribir tu nombre, marca «Incluir simulaciones guardadas sin nombre».")
        else:
            df_v = df_hist.copy()
            if busqueda.strip():
                q = busqueda.strip().lower()
                df_v = df_v[df_v['nombre_proyecto'].fillna('').str.lower().str.contains(q, regex=False)
                            | df_v['est_nombre'].fillna('').str.lower().str.contains(q, regex=False)]
            if orden == "Más antiguas primero":
                df_v = df_v.sort_values('fecha_simulacion', ascending=True)
            elif orden == "Mayor cobertura":
                df_v = df_v.sort_values('pct_cubierto_normal', ascending=False, na_position='last')
            elif orden == "Nombre del proyecto":
                df_v = df_v.sort_values('nombre_proyecto', key=lambda c: c.fillna('').str.lower())

            col_titulo, col_zip = st.columns([3, 1])
            col_titulo.write(f"**{len(df_v)} de {len(df_hist)} simulación(es)**")

            ZIP_MAX = 25
            with col_zip:
                if st.button(f"📦 ZIP de PDFs (máx. {ZIP_MAX})", key="btn_zip", disabled=df_v.empty):
                    import zipfile, io as _io
                    zip_buf = _io.BytesIO()
                    lote = df_v.head(ZIP_MAX)
                    barra = st.progress(0.0, text="Generando PDFs…")
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for k, (_, row_z) in enumerate(lote.iterrows(), 1):
                            datos_z = cargar_simulacion_completa(int(row_z['id']))
                            if datos_z:
                                try:
                                    pdf_z = generar_informe_pdf(datos_z)
                                    nombre_z = (row_z.get('nombre_proyecto') or 'proyecto').replace(' ', '_')
                                    zf.writestr(f"{nombre_z}_{int(row_z['id'])}.pdf", pdf_z.read())
                                except Exception as e_z:
                                    st.warning(f"No se pudo generar el PDF de {row_z.get('nombre_proyecto','?')}: {e_z}")
                            barra.progress(k / len(lote), text=f"Generando PDFs… {k} de {len(lote)}")
                    barra.empty()
                    st.session_state['_zip_bytes'] = zip_buf.getvalue()
                    st.session_state['_zip_nombre'] = f"SCALL_{usuario_n.replace(' ', '_')}_informes.zip"
                if st.session_state.get('_zip_bytes'):
                    st.download_button("⬇️ Guardar ZIP", data=st.session_state['_zip_bytes'],
                                       file_name=st.session_state.get('_zip_nombre', 'SCALL_informes.zip'),
                                       mime="application/zip", key="btn_zip_download")
            if len(df_v) > ZIP_MAX:
                st.caption(f"El ZIP incluye las primeras {ZIP_MAX} de la lista actual (filtra o busca para elegir cuáles).")

            n_mostrar = st.session_state.get('_hist_n', 15)
            for _, row in df_v.head(n_mostrar).iterrows():
                sim_id   = int(row['id'])
                fecha    = row['fecha_simulacion'].strftime('%d/%m/%Y %H:%M') if _hay(row['fecha_simulacion']) else '—'
                proyecto = row.get('nombre_proyecto') or '—'
                estacion = row.get('est_nombre') or '—'
                pct      = row.get('pct_cubierto_normal')
                dias     = row.get('dias_sin_agua_normal')
                cap_opt  = row.get('cap_optima')

                pct_str  = f"{pct:.1f}%" if _hay(pct) else "—"
                dias_str = str(int(dias)) if _hay(dias) else "—"
                cap_str  = f"{formato_chileno(cap_opt, 0)} L" if _hay(cap_opt) else "—"
                sin_nombre = " · 👤 sin nombre" if str(row.get('usuario', '')).strip().lower() == 'anónimo' else ""

                with st.expander(f"**{proyecto}** — {fecha} · {estacion}{sin_nombre}", expanded=False):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Cobertura año normal", pct_str)
                    m2.metric("Días sin agua (normal)", dias_str)
                    m3.metric("Estanque óptimo", cap_str)

                    b1, b2, b3 = st.columns(3)

                    with b1:
                        if st.button("📂 Cargar simulación", key=f"cargar_{sim_id}"):
                            with st.spinner("Cargando desde historial..."):
                                datos = cargar_simulacion_completa(sim_id)
                            if datos:
                                st.session_state['informe_datos'] = datos
                                est_codigo = datos.get('est_codigo')
                                lluvias_m  = datos.get('lluvias_mensuales')
                                anio_ini   = datos.get('anio_inicio')
                                anio_fin_  = datos.get('anio_fin')
                                pp_prom    = datos.get('precipitaciones_promedio', [])
                                st.session_state['resultado'] = {
                                    'estacion_cercana':         {'Nombre': datos.get('est_nombre', '—')},
                                    'lluvias_estacion':         lluvias_m,
                                    'codigo_estacion':          est_codigo,
                                    'anio_inicio':              anio_ini,
                                    'anio_fin':                 anio_fin_,
                                    'precipitaciones_promedio': pp_prom,
                                }
                                st.session_state['simulacion_desde_historial'] = True
                                st.session_state.simulacion_calculada = False
                                st.session_state['_hist_flash'] = (
                                    f"📂 Cargada «{proyecto}». Ve a la pestaña «Simulación Histórica y Escenarios» "
                                    "para verla; el PDF se descarga al final de esa página.")
                                st.session_state.pop('_pdf_cache', None)
                                st.rerun()
                            else:
                                st.error("No se pudo cargar la simulación (la base no respondió o el registro está dañado). "
                                         "Intenta de nuevo; si sigue igual, vuelve a correrla.")

                    with b2:
                        if st.button("📄 Generar PDF", key=f"pdf_btn_{sim_id}"):
                            st.session_state['pdf_solicitado'] = sim_id
                            st.session_state.pop(f'pdf_bytes_{sim_id}', None)

                        if st.session_state.get('pdf_solicitado') == sim_id:
                            if f'pdf_bytes_{sim_id}' not in st.session_state:
                                with st.spinner("Generando PDF..."):
                                    datos_pdf = cargar_simulacion_completa(sim_id)
                                if datos_pdf:
                                    st.session_state[f'pdf_bytes_{sim_id}'] = generar_informe_pdf(datos_pdf).getvalue()
                                    st.session_state[f'pdf_nombre_{sim_id}'] = datos_pdf.get('nombre_proyecto', 'proyecto')
                                else:
                                    st.error("No se pudo leer la simulación para armar el PDF.")
                            pdf_bytes = st.session_state.get(f'pdf_bytes_{sim_id}')
                            if pdf_bytes:
                                nombre_pdf = (st.session_state.get(f'pdf_nombre_{sim_id}') or 'proyecto').replace(' ', '_')
                                st.download_button(
                                    label="⬇️ Descargar PDF",
                                    data=pdf_bytes,
                                    file_name=f"Informe_SCALL_{nombre_pdf}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{sim_id}",
                                )

                    with b3:
                        if st.button("🗑️ Eliminar", key=f"del_{sim_id}"):
                            st.session_state[f'confirmar_eliminar_{sim_id}'] = True
                        if st.session_state.get(f'confirmar_eliminar_{sim_id}', False):
                            if st.button("¿Confirmar eliminación?", key=f"confirm_{sim_id}", type="primary"):
                                if eliminar_simulacion(sim_id):
                                    st.toast("Simulación eliminada", icon="🗑️")
                                else:
                                    st.toast("No se pudo eliminar. Intenta de nuevo.", icon="⚠️")
                                st.session_state.pop(f'confirmar_eliminar_{sim_id}', None)
                                st.session_state.pop('_hist_cache', None)
                                st.rerun()

            if len(df_v) > n_mostrar:
                if st.button(f"Mostrar más ({len(df_v) - n_mostrar} restantes)", key="btn_hist_mas"):
                    st.session_state['_hist_n'] = n_mostrar + 15
                    st.rerun()


# ===============================================================
# BOTÓN DE INFORME FINAL
# ===============================================================
if 'informe_datos' in st.session_state:
    st.markdown("---")
    st.markdown("### 📄 Informe del Proyecto")
    st.write("Descarga un informe PDF con el resumen ejecutivo del proyecto: parámetros, estación, escenarios climáticos y recomendación de tamaño óptimo.")
    # El PDF se genera una sola vez por simulación y se cachea en la sesión;
    # el cache se invalida al recalcular o al cargar otra simulación del historial.
    if st.session_state.get('_pdf_cache') is None:
        with st.spinner("Generando informe PDF..."):
            st.session_state['_pdf_cache'] = generar_informe_pdf(st.session_state['informe_datos']).getvalue()
    nombre_archivo = st.session_state['informe_datos']['nombre_proyecto'].replace(' ', '_')
    st.download_button(
        label="📥 Descargar Informe Completo (.pdf)",
        data=st.session_state['_pdf_cache'],
        file_name=f"Informe_SCALL_{nombre_archivo}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )
