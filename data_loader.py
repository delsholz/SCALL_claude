import streamlit as st
import pandas as pd
import numpy as np

from utils import arreglar_coordenada


@st.cache_data(ttl=3600)
def cargar_datos_crudos(anio_inicio="2000", anio_fin="2020"):
    archivo_cr2 = "data/cr2_prDaily_2020.txt.gz"
    try:
        df_meta = pd.read_csv(archivo_cr2, sep=',', nrows=15, header=None, dtype=str, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo '{archivo_cr2}'.")

    codigos_completos  = df_meta.iloc[0, 1:].values.astype(str)
    nombres_completos  = df_meta.iloc[3, 1:].values.astype(str)
    altitudes_crudas   = df_meta.iloc[4, 1:].values.astype(str)
    latitudes_crudas   = df_meta.iloc[5, 1:].values.astype(str)
    longitudes_crudas  = df_meta.iloc[6, 1:].values.astype(str)

    indices_validos = [i for i, cod in enumerate(codigos_completos)
                       if cod != 'nan' and str(cod).strip() != '']
    codigos    = [codigos_completos[i] for i in indices_validos]
    nombres    = [nombres_completos[i] for i in indices_validos]
    latitudes  = [arreglar_coordenada(latitudes_crudas[i],  es_longitud=False) for i in indices_validos]
    longitudes = [arreglar_coordenada(longitudes_crudas[i], es_longitud=True)  for i in indices_validos]

    def _parsear_altura(val):
        try:
            return float(str(val).strip().replace(',', '.'))
        except (ValueError, TypeError):
            return np.nan

    altitudes = [_parsear_altura(altitudes_crudas[i]) for i in indices_validos]

    df_estaciones = pd.DataFrame({
        'Codigo': codigos, 'Nombre': nombres,
        'Latitud': latitudes, 'Longitud': longitudes,
        'Altitud': altitudes,
    }).dropna(subset=['Latitud', 'Longitud'])

    columnas_originales = ['Fecha'] + list(codigos_completos)
    cols_unicas, conteo_cols = [], {}
    for c in columnas_originales:
        original = c
        if c in conteo_cols:
            conteo_cols[original] += 1
            c = f"{original}_dup{conteo_cols[original]}"
        else:
            conteo_cols[original] = 0
        cols_unicas.append(c)

    chunks_filtrados = []
    for chunk in pd.read_csv(archivo_cr2, sep=',', skiprows=15, names=cols_unicas,
                              chunksize=10000, dtype=str, low_memory=False):
        chunk['Anio_str'] = chunk.iloc[:, 0].astype(str).str.slice(0, 4)
        chunk_filtrado = chunk[(chunk['Anio_str'] >= anio_inicio) & (chunk['Anio_str'] <= anio_fin)].copy()
        if not chunk_filtrado.empty:
            chunks_filtrados.append(chunk_filtrado)

    if not chunks_filtrados:
        raise ValueError(f"No se encontraron datos entre {anio_inicio} y {anio_fin} en el archivo.")

    df_data = pd.concat(chunks_filtrados, ignore_index=True)
    df_data.drop(columns=['Anio_str'], inplace=True)

    columnas_a_mantener = ['Fecha'] + [c for c in codigos if c in df_data.columns]
    df_data = df_data[columnas_a_mantener]
    df_data.replace([-9999, '-9999', '-9999.0', -9999.0, 'nan', ''], pd.NA, inplace=True)

    df_data['Fecha_str'] = df_data['Fecha'].astype(str)
    df_data['Año_Mes']   = df_data['Fecha_str'].str.slice(0, 7)
    df_data['Mes_Dia']   = df_data['Fecha_str'].str.slice(5, 10)

    df_data[codigos] = df_data[codigos].astype(str).replace({',': '.'}, regex=True)
    df_data[codigos] = df_data[codigos].apply(pd.to_numeric, errors='coerce')
    df_data[codigos] = df_data[codigos].clip(lower=0)

    df_diario  = df_data.copy()
    df_mensual = df_data.groupby('Año_Mes')[codigos].sum(min_count=1).reset_index()
    df_mensual['Año'] = df_mensual['Año_Mes'].str.slice(0, 4).astype(int)

    return df_estaciones, df_diario, df_mensual, codigos


@st.cache_data(ttl=3600)
def aplicar_filtro_calidad(umbral_calidad, anio_inicio="2000", anio_fin="2020"):
    df_estaciones, df_diario, df_mensual, codigos = cargar_datos_crudos(anio_inicio, anio_fin)

    total_meses      = len(df_mensual)
    minimo_requerido = total_meses * (umbral_calidad / 100)
    conteo_validos   = df_mensual[codigos].notna().sum()

    estaciones_validas       = conteo_validos[conteo_validos >= minimo_requerido].index.tolist()
    df_estaciones_filtradas  = df_estaciones[df_estaciones['Codigo'].isin(estaciones_validas)].copy()

    return df_estaciones_filtradas, df_diario, df_mensual, df_estaciones, codigos


def _obtener_datos_2026_desactivado(lat, lon, totales_anio_historico):  # DESACTIVADO
    """
    Obtiene precipitación diaria para 2026 y calcula la tendencia histórica.

    - ERA5 2011-2025   : descargado para construir tendencia y factor de corrección
    - ERA5 2026        : datos observados hasta anteayer
    - CFS v2           : pronóstico estacional hasta Dic 2026

    Factor de corrección: últimos 10 años completos vs los 10 años anteriores
    (usa ERA5 para 2011-2025 y CR2 para años anteriores).

    Retorna
    -------
    df_2026         : DataFrame diario con columnas Fecha, PP, tipo
    factor_sequia   : float, razón prom_reciente / prom_antiguo
    prom_antiguo    : float, mm/año promedio del período base
    prom_reciente   : float, mm/año promedio del período reciente
    serie_tendencia : dict {año: mm_anuales} con CR2 + ERA5 + estimado 2026
    rango_reciente  : tuple (año_inicio, año_fin) del período reciente
    rango_antiguo   : tuple (año_inicio, año_fin) del período base
    """
    import requests
    from datetime import date, timedelta

    hoy        = date.today()
    corte_era5 = (hoy - timedelta(days=2)).strftime('%Y-%m-%d')

    # ── PARTE 1: ERA5 histórico 2011-2025 (tendencia + factor) ───
    totales_era5 = {}
    try:
        url_e15 = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date=2011-01-01&end_date=2025-12-31"
            f"&daily=precipitation_sum&timezone=UTC"
        )
        r15  = requests.get(url_e15, timeout=30)
        d15  = r15.json()
        df_e = pd.DataFrame({
            'Fecha': d15['daily']['time'],
            'PP':    [v if v is not None else np.nan
                      for v in d15['daily']['precipitation_sum']]
        })
        df_e['Anio'] = pd.to_datetime(df_e['Fecha']).dt.year
        totales_era5 = df_e.groupby('Anio')['PP'].sum().to_dict()
    except Exception:
        pass

    # ── PARTE 2: ERA5 datos reales 2026 ───────────────────────────
    df_real = pd.DataFrame(columns=['Fecha', 'PP', 'tipo'])
    try:
        url_r26 = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date=2026-01-01&end_date={corte_era5}"
            f"&daily=precipitation_sum&timezone=UTC"
        )
        r26  = requests.get(url_r26, timeout=15)
        d26  = r26.json()
        pp_r = [v if v is not None else 0.0 for v in d26['daily']['precipitation_sum']]
        df_real = pd.DataFrame({'Fecha': d26['daily']['time'], 'PP': pp_r, 'tipo': 'real'})
    except Exception:
        pass

    # ── PARTE 3: Pronóstico estacional CFS v2 ────────────────────
    df_fc = pd.DataFrame(columns=['Fecha', 'PP', 'tipo'])
    try:
        inicio_fc = hoy.strftime('%Y-%m-%d')
        url_fc = (
            f"https://seasonal-api.open-meteo.com/v1/seasonal?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date={inicio_fc}&end_date=2026-12-31"
            f"&daily=precipitation_sum&models=cfs_v2&timezone=UTC"
        )
        r_fc      = requests.get(url_fc, timeout=20)
        d_fc      = r_fc.json()
        fechas_fc = d_fc['daily']['time']
        members   = [v for k, v in d_fc['daily'].items()
                     if k != 'time' and v is not None]
        if members:
            pp_matrix = np.array([[x if x is not None else 0.0 for x in m]
                                   for m in members], dtype=float)
            pp_fc = pp_matrix.mean(axis=0)
        else:
            pp_fc = np.zeros(len(fechas_fc))
        df_fc = pd.DataFrame({'Fecha': fechas_fc, 'PP': pp_fc, 'tipo': 'pronostico'})
    except Exception:
        pass

    # ── PARTE 4: Serie combinada CR2 + ERA5 y factor de sequía ───
    factor_sequia  = 1.0
    prom_antiguo   = np.nan
    prom_reciente  = np.nan
    rango_reciente = (2016, 2025)
    rango_antiguo  = (2006, 2015)
    serie_tendencia = {}

    try:
        totales_cr2 = pd.Series(totales_anio_historico)
        totales_cr2.index = totales_cr2.index.astype(int)

        # ERA5 tiene prioridad en los años que se superpone con CR2 (más actualizado)
        anios_todos = sorted(set(totales_cr2.index) | set(totales_era5.keys()))
        serie_comb  = {}
        for a in anios_todos:
            if a in totales_era5:
                serie_comb[a] = float(totales_era5[a])
            elif a in totales_cr2.index:
                serie_comb[a] = float(totales_cr2[a])

        serie_s = pd.Series(serie_comb).sort_index()

        # Últimos 10 años completos disponibles (máx 2025)
        anios_ok = [a for a in serie_s.index
                    if a <= 2025 and np.isfinite(serie_s[a])]
        if len(anios_ok) >= 10:
            a_max = anios_ok[-1]         # ej. 2025
            a_r0  = a_max - 9            # ej. 2016  (inicio período reciente)
            a_a1  = a_r0  - 1            # ej. 2015  (fin período antiguo)
            a_a0  = a_a1  - 9            # ej. 2006  (inicio período antiguo)

            rango_reciente = (a_r0, a_max)
            rango_antiguo  = (a_a0, a_a1)

            prom_reciente = float(serie_s[
                (serie_s.index >= a_r0) & (serie_s.index <= a_max)].mean())
            prom_antiguo  = float(serie_s[
                (serie_s.index >= a_a0) & (serie_s.index <= a_a1)].mean())
        else:
            # Fallback si no hay suficientes años con ERA5
            prom_antiguo  = float(totales_cr2[
                (totales_cr2.index >= 1990) & (totales_cr2.index <= 2009)].mean())
            prom_reciente = float(totales_cr2[
                (totales_cr2.index >= 2010) & (totales_cr2.index <= 2020)].mean())

        if np.isfinite(prom_antiguo) and prom_antiguo > 0:
            factor_sequia = float(np.clip(prom_reciente / prom_antiguo, 0.3, 1.2))

        # Serie de tendencia = CR2 + ERA5 (sin 2026 todavía)
        serie_tendencia = dict(serie_comb)

    except Exception:
        pass

    # ── Aplicar factor al pronóstico ─────────────────────────────
    if not df_fc.empty:
        df_fc['PP'] = df_fc['PP'] * factor_sequia

    # ── Combinar y completar AÑO ENTERO 2026 (365 días) ─────────
    df_parcial = pd.concat([df_real, df_fc], ignore_index=True)
    df_parcial['PP'] = pd.to_numeric(df_parcial['PP'], errors='coerce').fillna(0).clip(lower=0)
    df_parcial = df_parcial.drop_duplicates(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)

    # Índice completo 1 Ene – 31 Dic 2026
    todas_fechas = pd.date_range('2026-01-01', '2026-12-31', freq='D')
    df_anio = pd.DataFrame({'Fecha': todas_fechas.strftime('%Y-%m-%d')})
    df_anio = df_anio.merge(df_parcial[['Fecha', 'PP', 'tipo']], on='Fecha', how='left')

    # Interpolar los huecos de PP (normalmente 1-2 días entre ERA5 y CFS v2)
    df_anio['PP'] = pd.to_numeric(df_anio['PP'], errors='coerce') \
                      .interpolate(method='linear').fillna(0).clip(lower=0)

    # Tipo de dato: 'real' si la fecha ya pasó, 'pronostico' si es futura
    hoy_str = hoy.strftime('%Y-%m-%d')
    df_anio['tipo'] = df_anio.apply(
        lambda r: r['tipo'] if pd.notna(r['tipo'])
                  else ('real' if r['Fecha'] <= hoy_str else 'pronostico'),
        axis=1
    )

    df_2026 = df_anio.reset_index(drop=True)

    # Agregar 2026 a la serie de tendencia (total anual completo)
    pp_2026_total = float(df_2026['PP'].sum())
    if pp_2026_total > 0:
        serie_tendencia[2026] = pp_2026_total

    return (df_2026, factor_sequia, prom_antiguo, prom_reciente,
            serie_tendencia, rango_reciente, rango_antiguo)
