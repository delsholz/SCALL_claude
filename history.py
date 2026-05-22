import io
import numpy as np
import pandas as pd
import streamlit as st


def _get_client():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None


def _df_a_json(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    return df.to_json(orient='records', date_format='iso', force_ascii=False)


def _json_a_df(json_str, cols_fecha=None):
    if not json_str:
        return None
    df = pd.read_json(io.StringIO(json_str), orient='records')
    if cols_fecha:
        for col in cols_fecha:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _serializar_informe(d):
    curva = d.get('curva_normal')
    curva_s = None
    if curva is not None:
        caps, efics, cap_opt, ef_act, ef_opt = curva
        curva_s = {
            'capacidades': [float(x) for x in caps],
            'eficiencias': [float(x) for x in efics],
            'cap_optima': float(cap_opt) if cap_opt is not None else None,
            'ef_actual': float(ef_act) if ef_act is not None else None,
            'ef_optima': float(ef_opt) if ef_opt is not None else None,
        }

    est_alt = d.get('est_altitud')
    desnivel = d.get('desnivel')

    return {
        'curva_normal': curva_s,
        'nombre_proyecto': d.get('nombre_proyecto'),
        'lat_proyecto': d.get('lat_proyecto'),
        'lon_proyecto': d.get('lon_proyecto'),
        'alt_proyecto': d.get('alt_proyecto'),
        'techo': d.get('techo'),
        'capacidad_maxima': d.get('capacidad_maxima'),
        'eficiencia': d.get('eficiencia'),
        'numero_personas': d.get('numero_personas'),
        'litros_persona_dia': float(d['litros_persona_dia']) if d.get('litros_persona_dia') is not None else None,
        'consumo_mensual': d.get('consumo_mensual'),
        'tipo_uso': d.get('tipo_uso'),
        'meses_seleccionados': d.get('meses_seleccionados'),
        'est_nombre': d.get('est_nombre'),
        'est_codigo': d.get('est_codigo'),
        'est_altitud': float(est_alt) if isinstance(est_alt, (int, float)) else None,
        'desnivel': int(desnivel) if isinstance(desnivel, (int, float)) else None,
        'distancia': float(d['distancia']) if d.get('distancia') is not None else None,
        'anio_inicio': d.get('anio_inicio'),
        'anio_fin': d.get('anio_fin'),
        'pct_calidad': d.get('pct_calidad'),
        'anio_seco': d.get('anio_seco'),
        'anio_mediano': d.get('anio_mediano'),
        'anio_lluvioso': d.get('anio_lluvioso'),
        'totales_anio': {str(k): v for k, v in (d.get('totales_anio') or {}).items()},
        'precipitaciones_promedio': d.get('precipitaciones_promedio'),
        'df_seco': _df_a_json(d.get('df_seco')),
        'df_normal': _df_a_json(d.get('df_normal')),
        'df_lluvioso': _df_a_json(d.get('df_lluvioso')),
        'lluvias_mensuales': _df_a_json(d.get('lluvias_mensuales')),
    }


def _deserializar_informe(j):
    COLS_FECHA_SIM = ['Fecha', 'Eje X']

    curva_s = j.get('curva_normal')
    curva = None
    if curva_s:
        curva = (
            np.array(curva_s['capacidades']),
            np.array(curva_s['eficiencias']),
            curva_s.get('cap_optima'),
            curva_s.get('ef_actual'),
            curva_s.get('ef_optima'),
        )

    totales_raw = j.get('totales_anio') or {}
    totales_anio = {int(k): v for k, v in totales_raw.items()}

    est_altitud = j.get('est_altitud')
    if est_altitud is None:
        est_altitud = '—'

    desnivel = j.get('desnivel')
    if desnivel is None:
        desnivel = '—'

    return {
        'curva_normal': curva,
        'nombre_proyecto': j.get('nombre_proyecto'),
        'lat_proyecto': j.get('lat_proyecto'),
        'lon_proyecto': j.get('lon_proyecto'),
        'alt_proyecto': j.get('alt_proyecto'),
        'techo': j.get('techo'),
        'capacidad_maxima': j.get('capacidad_maxima'),
        'eficiencia': j.get('eficiencia'),
        'numero_personas': j.get('numero_personas'),
        'litros_persona_dia': j.get('litros_persona_dia'),
        'consumo_mensual': j.get('consumo_mensual'),
        'tipo_uso': j.get('tipo_uso'),
        'meses_seleccionados': j.get('meses_seleccionados'),
        'est_nombre': j.get('est_nombre'),
        'est_codigo': j.get('est_codigo'),
        'est_altitud': est_altitud,
        'desnivel': desnivel,
        'distancia': j.get('distancia'),
        'anio_inicio': j.get('anio_inicio'),
        'anio_fin': j.get('anio_fin'),
        'pct_calidad': j.get('pct_calidad'),
        'anio_seco': j.get('anio_seco'),
        'anio_mediano': j.get('anio_mediano'),
        'anio_lluvioso': j.get('anio_lluvioso'),
        'totales_anio': totales_anio,
        'precipitaciones_promedio': j.get('precipitaciones_promedio'),
        'df_seco': _json_a_df(j.get('df_seco'), COLS_FECHA_SIM),
        'df_normal': _json_a_df(j.get('df_normal'), COLS_FECHA_SIM),
        'df_lluvioso': _json_a_df(j.get('df_lluvioso'), COLS_FECHA_SIM),
        'lluvias_mensuales': _json_a_df(j.get('lluvias_mensuales')),
    }


def _extraer_metricas(d):
    pct_cubierto_normal = None
    dias_sin_agua_normal = None
    cap_optima = None

    df_n = d.get('df_normal')
    if df_n is not None and not df_n.empty:
        total_demanda = df_n['Demanda (L)'].sum()
        deficit_total = df_n['Déficit Diario (L)'].sum()
        if total_demanda > 0:
            pct_cubierto_normal = round((total_demanda + deficit_total) / total_demanda * 100, 1)
        dias_sin_agua_normal = int((df_n['Déficit Diario (L)'] < 0).sum())

    curva = d.get('curva_normal')
    if curva is not None and len(curva) >= 3 and curva[2] is not None:
        cap_optima = float(curva[2])

    return pct_cubierto_normal, dias_sin_agua_normal, cap_optima


def guardar_simulacion(informe_datos):
    """Saves a simulation to Supabase. Returns True on success."""
    client = _get_client()
    if client is None:
        return False

    try:
        pct, dias, cap_opt = _extraer_metricas(informe_datos)
        serial = _serializar_informe(informe_datos)
        d = informe_datos
        est_alt = d.get('est_altitud')

        row = {
            'nombre_proyecto': d.get('nombre_proyecto'),
            'lat_proyecto': d.get('lat_proyecto'),
            'lon_proyecto': d.get('lon_proyecto'),
            'alt_proyecto': d.get('alt_proyecto'),
            'techo': d.get('techo'),
            'capacidad_maxima': d.get('capacidad_maxima'),
            'eficiencia': d.get('eficiencia'),
            'numero_personas': d.get('numero_personas'),
            'litros_persona_dia': float(d['litros_persona_dia']) if d.get('litros_persona_dia') is not None else None,
            'tipo_uso': d.get('tipo_uso'),
            'meses_seleccionados': d.get('meses_seleccionados'),
            'est_nombre': d.get('est_nombre'),
            'est_codigo': d.get('est_codigo'),
            'est_altitud': float(est_alt) if isinstance(est_alt, (int, float)) else None,
            'distancia': float(d['distancia']) if d.get('distancia') is not None else None,
            'anio_seco': d.get('anio_seco'),
            'anio_mediano': d.get('anio_mediano'),
            'anio_lluvioso': d.get('anio_lluvioso'),
            'pct_cubierto_normal': pct,
            'dias_sin_agua_normal': dias,
            'cap_optima': cap_opt,
            'informe_datos_json': serial,
        }

        client.table('simulaciones').insert(row).execute()
        return True
    except Exception:
        return False


def cargar_historial():
    """Returns a DataFrame with simulation metadata rows (no JSON blobs)."""
    client = _get_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = client.table('simulaciones').select(
            'id,fecha_simulacion,nombre_proyecto,est_nombre,est_codigo,'
            'techo,capacidad_maxima,numero_personas,pct_cubierto_normal,'
            'dias_sin_agua_normal,cap_optima,anio_seco,anio_mediano,anio_lluvioso'
        ).order('fecha_simulacion', desc=True).execute()

        if not resp.data:
            return pd.DataFrame()

        df = pd.DataFrame(resp.data)
        df['fecha_simulacion'] = pd.to_datetime(df['fecha_simulacion'])
        return df
    except Exception:
        return pd.DataFrame()


def cargar_simulacion_completa(sim_id):
    """Returns the full informe_datos dict for a simulation ID."""
    client = _get_client()
    if client is None:
        return None

    try:
        resp = (
            client.table('simulaciones')
            .select('informe_datos_json')
            .eq('id', sim_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return _deserializar_informe(resp.data[0]['informe_datos_json'])
    except Exception:
        return None


def eliminar_simulacion(sim_id):
    """Deletes a simulation by ID. Returns True on success."""
    client = _get_client()
    if client is None:
        return False

    try:
        client.table('simulaciones').delete().eq('id', sim_id).execute()
        return True
    except Exception:
        return False
