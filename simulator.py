import pandas as pd
import numpy as np
from itertools import accumulate

def simular_continua(df_diario, codigo_estacion, anio_inicio, anio_fin,
                     meses_num_seleccionados, consumo_fines_semana,
                     numero_personas, litros_persona_dia, capacidad_maxima,
                     techo, eficiencia, multiplicador_lluvia=1.0):
    df = df_diario[['Fecha', codigo_estacion]].copy().dropna(subset=['Fecha']).sort_values('Fecha')
    df['Anio'] = df['Fecha'].str.slice(0, 4).astype(int)
    df = df[(df['Anio'] >= anio_inicio) & (df['Anio'] <= anio_fin)].copy()

    fechas_dt        = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Mes_Num']    = fechas_dt.dt.strftime('%m')
    df['Dia_Semana'] = fechas_dt.dt.dayofweek

    consumo_diario = numero_personas * litros_persona_dia
    mask_mes = df['Mes_Num'].isin(meses_num_seleccionados)
    mask_dia = True if consumo_fines_semana else (df['Dia_Semana'] < 5)

    df['Demanda (L)'] = np.where(mask_mes & mask_dia, consumo_diario, 0)
    # AQUI SE APLICA EL GRADIENTE OROGRAFICO A LA LLUVIA:
    df['Captado (L)'] = (df[codigo_estacion].fillna(0) * multiplicador_lluvia) * techo * eficiencia

    cap_arr = df['Captado (L)'].values
    dem_arr = df['Demanda (L)'].values

    def _paso(nivel, cd):
        captado, demanda = cd
        disp  = min(nivel + captado, capacidad_maxima)
        usada = min(disp, demanda)
        return max(0.0, disp - usada)

    estanque_arr = np.fromiter(
        accumulate(zip(cap_arr, dem_arr), _paso, initial=0.0),
        dtype=float, count=len(cap_arr) + 1
    )[1:]

    df['Estanque Final (L)'] = estanque_arr
    # Guardamos la lluvia corregida para usarla después
    df['Lluvia_Corregida'] = df[codigo_estacion].fillna(0) * multiplicador_lluvia
    return df

def encontrar_anios_extremos(df_sim_completa, codigo_estacion):
    # Ahora sumamos la lluvia corregida por el gradiente orográfico
    totales_por_anio  = df_sim_completa.groupby('Anio')['Lluvia_Corregida'].sum()
    dias_por_anio     = df_sim_completa.groupby('Anio')['Lluvia_Corregida'].count()
    anios_completos   = dias_por_anio[dias_por_anio >= 300].index
    totales_completos = totales_por_anio[totales_por_anio.index.isin(anios_completos)]

    if len(totales_completos) >= 3:
        p05 = totales_completos.quantile(0.05)
        p95 = totales_completos.quantile(0.95)
        anio_seco     = (totales_completos - p05).abs().idxmin()
        anio_lluvioso = (totales_completos - p95).abs().idxmin()

        mediana_pp = totales_completos.median()
        candidatos = (totales_completos - mediana_pp).abs().nsmallest(3).index
        df_validos = df_sim_completa[df_sim_completa['Anio'].isin(anios_completos)]
        prom_mens  = df_validos.groupby('Mes_Num')['Lluvia_Corregida'].sum() / len(anios_completos)

        mejor_anio, menor_error = candidatos[0], float('inf')
        for anio_cand in candidatos:
            meses_cand = df_validos[df_validos['Anio'] == anio_cand].groupby('Mes_Num')['Lluvia_Corregida'].sum()
            error = ((meses_cand - prom_mens) ** 2).sum()
            if error < menor_error:
                menor_error = error
                mejor_anio  = anio_cand
        anio_mediano = mejor_anio

    elif len(totales_completos) >= 1:
        anio_seco = anio_mediano = anio_lluvioso = totales_completos.index[0]
    else:
        anio_seco = anio_mediano = anio_lluvioso = totales_por_anio.index[0]

    return anio_seco, anio_mediano, anio_lluvioso, totales_por_anio

def simular_escenario(df_sim_completa, anio_sim, codigo_estacion,
                      meses_num_seleccionados, consumo_fines_semana,
                      numero_personas, litros_persona_dia,
                      capacidad_maxima, techo, eficiencia, multiplicador_lluvia=1.0):
    df_anio = df_sim_completa[df_sim_completa['Anio'] == anio_sim].copy()

    mask_mes     = df_anio['Mes_Num'].isin(meses_num_seleccionados).values
    mask_dia     = np.ones(len(df_anio), dtype=bool) if consumo_fines_semana else (df_anio['Dia_Semana'].values < 5)
    consumo_base = numero_personas * litros_persona_dia
    demanda      = np.where(mask_mes & mask_dia, consumo_base, 0.0)

    # AQUI SE APLICA EL GRADIENTE OROGRAFICO A LA LLUVIA DIARIA:
    pp_vals = pd.to_numeric(df_anio[codigo_estacion], errors='coerce').fillna(0).values * multiplicador_lluvia
    captado = np.clip(pp_vals, 0, None) * techo * eficiencia

    n                  = len(df_anio)
    estanque_final     = np.empty(n)
    rebalse_arr        = np.empty(n)
    agua_usada_arr     = np.empty(n)
    acumulado_teorico  = np.empty(n)
    deficit_arr        = np.empty(n)

    nivel = 0.0
    for i in range(n):
        teorico    = nivel + captado[i]
        disponible = min(teorico, capacidad_maxima)
        reb        = max(0.0, teorico - capacidad_maxima)
        usada      = min(disponible, demanda[i])
        nivel      = max(0.0, disponible - demanda[i])

        estanque_final[i]    = nivel
        rebalse_arr[i]       = reb
        agua_usada_arr[i]    = usada
        acumulado_teorico[i] = teorico - demanda[i]
        deficit_arr[i]       = 0.0 if usada >= demanda[i] else usada - demanda[i]

    fechas    = df_anio['Fecha'].values
    mes_dia   = df_anio['Fecha'].str.slice(5, 10).values
    es_finde  = df_anio['Dia_Semana'].values >= 5
    etiquetas = np.where(
        ~mask_mes, " (Vacaciones)",
        np.where(~consumo_fines_semana & es_finde, " (Finde)", "")
    )

    df_result = pd.DataFrame({
        "Fecha Pura":                 fechas,
        "Mes_Dia":                    mes_dia,
        "Día":                        [f"{f}{e}" for f, e in zip(fechas, etiquetas)],
        "Lluvia (mm)":                np.clip(pp_vals, 0, None),
        "Captado (L)":                captado,
        "Demanda (L)":                demanda,
        "Estanque Final (L)":         estanque_final,
        "Rebalse (L)":                rebalse_arr,
        "Agua Acumulada Teórica (L)": acumulado_teorico,
        "Déficit Diario (L)":         deficit_arr,
    })
    df_result['Eje X'] = pd.to_datetime('2024-' + df_result['Mes_Dia'], errors='coerce')
    return df_result

def calcular_curva_optimizacion(df_slice, capacidad_maxima):
    limite_curva       = int(max(30000, capacidad_maxima * 2.0))
    paso_curva         = 1000
    capacidades_prueba = list(range(paso_curva, limite_curva + paso_curva, paso_curva))

    cap_arr = df_slice['Captado (L)'].values
    dem_arr = df_slice['Demanda (L)'].values
    dem_tot = dem_arr.sum()

    # Vectorizado: simula todas las capacidades en paralelo con NumPy
    caps    = np.array(capacidades_prueba, dtype=np.float64)
    nivel   = np.zeros(len(caps))
    sum_tot = np.zeros(len(caps))
    for captado_i, demanda_i in zip(cap_arr, dem_arr):
        disp     = np.minimum(nivel + captado_i, caps)
        usada    = np.minimum(disp, demanda_i)
        nivel    = np.maximum(0.0, disp - demanda_i)
        sum_tot += usada
    eficiencias = (sum_tot / dem_tot * 100).tolist() if dem_tot > 0 else [100.0] * len(caps)

    max_ef     = max(eficiencias)
    cap_optima = capacidades_prueba[-1]
    if max_ef >= 95:
        # Primer estanque que alcanza el 95% de cobertura
        for cp, ef in zip(capacidades_prueba, eficiencias):
            if ef >= 95:
                cap_optima = cp
                break
    else:
        # Algoritmo Kneedle para curva cóncava (ganancias decrecientes):
        # normalizamos ambos ejes a [0,1] y buscamos el punto donde la curva
        # se desvía más por encima de la diagonal → max(y_norm - x_norm).
        # Ese punto es donde la ganancia marginal relativa es máxima antes
        # de que la curva empiece a aplanarse.
        x = np.array(capacidades_prueba, dtype=float)
        y = np.array(eficiencias, dtype=float)
        rng_x = x[-1] - x[0]
        rng_y = y[-1] - y[0]
        if rng_x > 0 and rng_y > 0:
            x_n = (x - x[0]) / rng_x
            y_n = (y - y[0]) / rng_y
            # Curva cóncava: y_n > x_n en la zona de máxima ganancia relativa
            distancias = y_n - x_n
            cap_optima = capacidades_prueba[int(np.argmax(distancias))]
        else:
            cap_optima = capacidades_prueba[-1]

    idx_cercano = (np.abs(np.array(capacidades_prueba) - capacidad_maxima)).argmin()
    ef_actual   = eficiencias[idx_cercano]
    ef_optima   = eficiencias[capacidades_prueba.index(cap_optima)]

    return capacidades_prueba, eficiencias, cap_optima, ef_actual, ef_optima


def _simular_desde_serie_desactivado(df_pp, meses_num_seleccionados, consumo_fines_semana,  # DESACTIVADO
                        numero_personas, litros_persona_dia,
                        capacidad_maxima, techo, eficiencia):
    """
    Simula el balance del estanque a partir de una serie externa de precipitaciones
    (ej: ERA5 + pronóstico CFS v2 para 2026).

    df_pp debe tener columnas: Fecha (str YYYY-MM-DD), PP (float mm/día), tipo (str)
    """
    df = df_pp.copy()
    df['Fecha_dt']   = pd.to_datetime(df['Fecha'], errors='coerce')
    df               = df.dropna(subset=['Fecha_dt']).sort_values('Fecha_dt').reset_index(drop=True)
    df['Mes_Num']    = df['Fecha_dt'].dt.strftime('%m')
    df['Dia_Semana'] = df['Fecha_dt'].dt.dayofweek

    consumo_base = numero_personas * litros_persona_dia
    mask_mes     = df['Mes_Num'].isin(meses_num_seleccionados).values
    mask_dia     = np.ones(len(df), dtype=bool) if consumo_fines_semana \
                   else (df['Dia_Semana'].values < 5)
    demanda      = np.where(mask_mes & mask_dia, consumo_base, 0.0)

    pp_vals = pd.to_numeric(df['PP'], errors='coerce').fillna(0).clip(lower=0).values
    captado = pp_vals * techo * eficiencia

    n                 = len(df)
    estanque_final    = np.empty(n)
    rebalse_arr       = np.empty(n)
    acumulado_teorico = np.empty(n)
    deficit_arr       = np.empty(n)

    nivel = 0.0
    for i in range(n):
        teorico    = nivel + captado[i]
        disponible = min(teorico, capacidad_maxima)
        reb        = max(0.0, teorico - capacidad_maxima)
        usada      = min(disponible, demanda[i])
        nivel      = max(0.0, disponible - demanda[i])

        estanque_final[i]    = nivel
        rebalse_arr[i]       = reb
        acumulado_teorico[i] = teorico - demanda[i]
        deficit_arr[i]       = 0.0 if usada >= demanda[i] else usada - demanda[i]

    mes_dia   = df['Fecha_dt'].dt.strftime('%m-%d').values
    es_finde  = df['Dia_Semana'].values >= 5
    etiquetas = np.where(
        ~mask_mes,  " (Vacaciones)",
        np.where(~consumo_fines_semana & es_finde, " (Finde)", "")
    )
    fecha_strs = df['Fecha_dt'].dt.strftime('%Y-%m-%d').values

    df_result = pd.DataFrame({
        "Fecha Pura":                 fecha_strs,
        "Mes_Dia":                    mes_dia,
        "Día":                        [f"{f}{e}" for f, e in zip(fecha_strs, etiquetas)],
        "Lluvia (mm)":                pp_vals,
        "Captado (L)":                captado,
        "Demanda (L)":                demanda,
        "Estanque Final (L)":         estanque_final,
        "Rebalse (L)":                rebalse_arr,
        "Agua Acumulada Teórica (L)": acumulado_teorico,
        "Déficit Diario (L)":         deficit_arr,
        "tipo":                       df['tipo'].values,
    })
    # Eje X normalizado a 2024 para que los gráficos sean idénticos a los escenarios históricos
    df_result['Eje X']      = pd.to_datetime('2024-' + df_result['Mes_Dia'], errors='coerce')
    # Eje X Real con fechas 2026 para el gráfico real/pronóstico
    df_result['Eje X Real'] = pd.to_datetime(df_result['Fecha Pura'], errors='coerce')
    return df_result
