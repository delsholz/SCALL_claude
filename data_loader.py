import streamlit as st
import pandas as pd
import numpy as np

from utils import arreglar_coordenada


@st.cache_data(ttl=3600)
def cargar_datos_crudos():
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
        chunk_filtrado = chunk[(chunk['Anio_str'] >= "1990") & (chunk['Anio_str'] <= "2020")].copy()
        if not chunk_filtrado.empty:
            chunks_filtrados.append(chunk_filtrado)

    if not chunks_filtrados:
        raise ValueError("No se encontraron datos entre 1990 y 2020 en el archivo.")

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
def aplicar_filtro_calidad(umbral_calidad):
    df_estaciones, df_diario, df_mensual, codigos = cargar_datos_crudos()

    total_meses      = len(df_mensual)
    minimo_requerido = total_meses * (umbral_calidad / 100)
    conteo_validos   = df_mensual[codigos].notna().sum()

    estaciones_validas       = conteo_validos[conteo_validos >= minimo_requerido].index.tolist()
    df_estaciones_filtradas  = df_estaciones[df_estaciones['Codigo'].isin(estaciones_validas)].copy()

    return df_estaciones_filtradas, df_diario, df_mensual
