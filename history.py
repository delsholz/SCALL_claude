"""
Historial de simulaciones SCALL.

Almacén permanente: la API REST de Supabase (via requests, sin supabase-py).
Credenciales (en orden de prioridad):
  1. .streamlit/secrets.toml  → [supabase] url / key
  2. Variables de entorno      → SUPABASE_URL / SUPABASE_KEY

Respaldo: si Supabase no está configurado o no responde (el proyecto se borró, no hay internet, la clave venció…),
el historial sigue funcionando en un archivo SQLite del servidor (~/.scall_amulen/historial.db; se puede cambiar con
SCALL_HISTORIAL_DB). Es temporal en Streamlit Cloud: se pierde cuando la app se reinicia. Tras un fallo se deja de
insistir con Supabase 5 minutos, así la app no se queda esperando una dirección muerta.
"""
import io
import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st


def _get_supabase_config():
    """Retorna (url, key) desde secrets.toml o variables de entorno."""
    try:
        return st.secrets["supabase"]["url"], st.secrets["supabase"]["key"]
    except Exception:
        pass
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if url and key:
        return url, key
    return None, None


def _headers():
    _, key = _get_supabase_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _base_url():
    url, _ = _get_supabase_config()
    return f"{url.rstrip('/')}/rest/v1/simulaciones"


def _supabase_disponible():
    url, key = _get_supabase_config()
    return bool(url and key)


ANONIMO = "Anónimo"


def normalizar_usuario(usuario):
    """Nombre sin espacios sobrantes (inicio, fin y repetidos). Vacío -> «Anónimo»."""
    limpio = " ".join(str(usuario or "").split())
    return limpio or ANONIMO


def _filtro_usuario(usuario, incluir_anonimas=False):
    """Parámetros PostgREST para buscar las simulaciones de `usuario` SIN distinguir mayúsculas ni tildes de caso
    (ilike), y opcionalmente también las guardadas sin nombre. Los caracteres que PostgREST usa como sintaxis
    (* % , ( ) y la barra invertida) se reemplazan por espacios para que un nombre raro no rompa la consulta."""
    nombre = normalizar_usuario(usuario)
    seguro = re.sub(r'[*%,()\\]', ' ', nombre)
    seguro = " ".join(seguro.split())
    if incluir_anonimas and nombre != ANONIMO:
        return {'or': f'(usuario.ilike.{seguro},usuario.eq.{ANONIMO})'}
    return {'usuario': f'ilike.{seguro}'}


# ---------------------------------------------------------------------------
# Serialización / Deserialización
# ---------------------------------------------------------------------------

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
        'multiplicador_lluvia': d.get('multiplicador_lluvia'),
        'direccion': d.get('direccion'),
        'comuna': d.get('comuna'),
        'region': d.get('region'),
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
        'multiplicador_lluvia': j.get('multiplicador_lluvia'),
        'direccion': j.get('direccion') or '',
        'comuna': j.get('comuna') or '',
        'region': j.get('region') or '',
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


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

# --- Circuito de seguridad: tras un fallo de Supabase se usa el respaldo local un rato ---
_SB_PAUSA_S = 300
_sb_fallo_hasta = 0.0
_sb_motivo = ""


def _sb_activo():
    return _supabase_disponible() and time.time() >= _sb_fallo_hasta


def _sb_fallo(exc_o_texto):
    """Anota que Supabase falló y deja de usarlo por unos minutos."""
    global _sb_fallo_hasta, _sb_motivo
    if isinstance(exc_o_texto, requests.exceptions.ConnectionError):
        _sb_motivo = "no se pudo conectar (el proyecto ya no existe o no hay internet)"
    elif isinstance(exc_o_texto, requests.exceptions.Timeout):
        _sb_motivo = "tardó demasiado en responder"
    else:
        _sb_motivo = str(exc_o_texto)[:160]
    _sb_fallo_hasta = time.time() + _SB_PAUSA_S


def estado_historial():
    """('supabase' | 'local', motivo). 'local' = el historial se guarda temporalmente en el servidor."""
    if _sb_activo():
        return "supabase", ""
    if not _supabase_disponible():
        return "local", "no hay credenciales de Supabase configuradas"
    return "local", _sb_motivo or "Supabase no responde"


# --- Respaldo local (SQLite) ---
_COLS_LOCAL = ("fecha_simulacion", "usuario", "nombre_proyecto", "est_nombre", "est_codigo", "techo",
               "capacidad_maxima", "numero_personas", "pct_cubierto_normal", "dias_sin_agua_normal",
               "cap_optima", "anio_seco", "anio_mediano", "anio_lluvioso", "informe_datos_json")

_SQL_TABLA = """CREATE TABLE IF NOT EXISTS simulaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_simulacion TEXT, usuario TEXT, nombre_proyecto TEXT,
    est_nombre TEXT, est_codigo TEXT, techo REAL, capacidad_maxima REAL, numero_personas INTEGER,
    pct_cubierto_normal REAL, dias_sin_agua_normal INTEGER, cap_optima REAL,
    anio_seco INTEGER, anio_mediano INTEGER, anio_lluvioso INTEGER, informe_datos_json TEXT)"""


def _ruta_db():
    env = os.environ.get("SCALL_HISTORIAL_DB")
    return Path(env) if env else Path.home() / ".scall_amulen" / "historial.db"


def _conectar_local():
    ultimo = None
    for ruta in (_ruta_db(), Path(tempfile.gettempdir()) / "scall_historial.db"):
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(ruta), timeout=10)
            con.execute(_SQL_TABLA)
            return con
        except Exception as e:
            ultimo = e
    raise ultimo


def _a_nativo(v):
    """Tipos numpy/pandas -> tipos que SQLite entiende (NaN -> NULL)."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _local_guardar(row, serial):
    try:
        con = _conectar_local()
        try:
            datos = {k: _a_nativo(row.get(k)) for k in _COLS_LOCAL}
            datos["fecha_simulacion"] = datetime.now(timezone.utc).isoformat()
            datos["informe_datos_json"] = json.dumps(serial, ensure_ascii=False, default=_json_default)
            con.execute(f"INSERT INTO simulaciones ({','.join(datos)}) VALUES ({','.join('?' * len(datos))})",
                        list(datos.values()))
            con.commit()
        finally:
            con.close()
        return True
    except Exception:
        return False


def _local_listar(usuario, incluir_anonimas, limite):
    con = _conectar_local()
    try:
        sql = ("SELECT id,fecha_simulacion,usuario,nombre_proyecto,est_nombre,est_codigo,techo,capacidad_maxima,"
               "numero_personas,pct_cubierto_normal,dias_sin_agua_normal,cap_optima,anio_seco,anio_mediano,"
               "anio_lluvioso FROM simulaciones")
        params = []
        if usuario:
            nombre = normalizar_usuario(usuario).lower()
            if incluir_anonimas and nombre != ANONIMO.lower():
                sql += " WHERE lower(usuario) = ? OR usuario = ?"
                params = [nombre, ANONIMO]
            else:
                sql += " WHERE lower(usuario) = ?"
                params = [nombre]
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limite))
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def _fechas_chile(df):
    fechas = pd.to_datetime(df['fecha_simulacion'], utc=True, errors='coerce')
    try:
        fechas = fechas.dt.tz_convert('America/Santiago')       # se guarda en UTC: mostrar la hora de Chile
    except Exception:
        pass
    df['fecha_simulacion'] = fechas
    return df


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def guardar_simulacion(informe_datos, usuario=None):
    """Guarda una simulación (Supabase y, si no responde, el respaldo local). Retorna True si quedó guardada."""
    try:
        pct, dias, cap_opt = _extraer_metricas(informe_datos)
        serial = _serializar_informe(informe_datos)
        d = informe_datos
        est_alt = d.get('est_altitud')

        row = {
            'usuario': normalizar_usuario(usuario),
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
    except Exception:
        return False

    if _sb_activo():
        try:
            payload = json.dumps(row, ensure_ascii=False, default=_json_default)
            resp = requests.post(_base_url(), headers=_headers(), data=payload, timeout=10)
            if resp.status_code in (200, 201):
                return True
            _sb_fallo(f"Supabase respondió {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            _sb_fallo(e)
    return _local_guardar(row, serial)


def _json_default(obj):
    """Convierte tipos numpy/pandas a tipos Python nativos para json.dumps."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"No serializable: {type(obj)}")


def cargar_historial_con_estado(usuario=None, incluir_anonimas=False, limite=500):
    """Metadatos de las simulaciones de `usuario` (sin distinguir mayúsculas), de la más nueva a la más antigua.

    Retorna (DataFrame, error). `error` es None si todo salió bien y un texto en castellano si falló. Si Supabase no
    responde se lee el respaldo local (consulta `estado_historial()` para avisarlo en pantalla).
    """
    if _sb_activo():
        try:
            cols = (
                'id,fecha_simulacion,usuario,nombre_proyecto,est_nombre,est_codigo,'
                'techo,capacidad_maxima,numero_personas,pct_cubierto_normal,'
                'dias_sin_agua_normal,cap_optima,anio_seco,anio_mediano,anio_lluvioso'
            )
            params = {'select': cols, 'order': 'fecha_simulacion.desc', 'limit': str(limite)}
            if usuario:
                params.update(_filtro_usuario(usuario, incluir_anonimas))
            resp = requests.get(_base_url(), headers={**_headers(), "Prefer": ""}, params=params, timeout=10)
            if resp.ok:
                data = resp.json()
                if not data:
                    return pd.DataFrame(), None
                return _fechas_chile(pd.DataFrame(data)), None
            _sb_fallo(f"Supabase respondió {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            _sb_fallo(e)
    try:
        df = _local_listar(usuario, incluir_anonimas, limite)
        return (_fechas_chile(df) if not df.empty else df), None
    except Exception as e:
        return pd.DataFrame(), f"No se pudo leer el historial: {e}"


def cargar_historial(usuario=None):
    """Compatibilidad: solo el DataFrame (vacío si hubo error)."""
    df, _ = cargar_historial_con_estado(usuario)
    return df


def cargar_simulacion_completa(sim_id):
    """Retorna el informe_datos completo de una simulación por ID."""
    if _sb_activo():
        try:
            resp = requests.get(
                _base_url(),
                headers={**_headers(), "Prefer": ""},
                params={'select': 'informe_datos_json', 'id': f'eq.{sim_id}', 'limit': '1'},
                timeout=20,
            )
            if resp.ok:
                if not resp.json():
                    return None
                return _deserializar_informe(resp.json()[0]['informe_datos_json'])
            _sb_fallo(f"Supabase respondió {resp.status_code}")
        except Exception as e:
            _sb_fallo(e)
    try:
        con = _conectar_local()
        try:
            fila = con.execute("SELECT informe_datos_json FROM simulaciones WHERE id = ?", (int(sim_id),)).fetchone()
        finally:
            con.close()
        return _deserializar_informe(json.loads(fila[0])) if fila and fila[0] else None
    except Exception:
        return None


def eliminar_simulacion(sim_id):
    """Elimina una simulación por ID. Retorna True si tuvo éxito."""
    if _sb_activo():
        try:
            resp = requests.delete(_base_url(), headers=_headers(), params={'id': f'eq.{sim_id}'}, timeout=10)
            if resp.status_code in (200, 204):
                return True
            _sb_fallo(f"Supabase respondió {resp.status_code}")
        except Exception as e:
            _sb_fallo(e)
    try:
        con = _conectar_local()
        try:
            cur = con.execute("DELETE FROM simulaciones WHERE id = ?", (int(sim_id),))
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()
    except Exception:
        return False
