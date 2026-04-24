import pandas as pd
import numpy as np


def formato_chileno(valor, decimales=0):
    if pd.isna(valor):
        return ""
    texto = f"{valor:,.{decimales}f}"
    return texto.translate(str.maketrans(',.', '.,'))


def calcular_distancia_vectorizada(lat1, lon1, alt1_m, lats, lons, alts_m, coef_altitud=5.0):
    R = 6371.0
    phi1    = np.radians(lat1)
    phi2    = np.radians(lats)
    dphi    = np.radians(lats - lat1)
    dlambda = np.radians(lons - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    d_horiz_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    delta_h_km = np.abs(alts_m - alt1_m) / 1000.0
    return np.sqrt(d_horiz_km ** 2 + (coef_altitud * delta_h_km) ** 2)


def arreglar_coordenada(val, es_longitud=False):
    val = str(val).strip().replace(',', '.')
    if val == 'nan' or not val:
        return None
    val_clean = val.replace('.', '')
    if not val_clean.startswith('-'):
        return None
    if len(val_clean) <= 3:
        return float(val_clean)
    if es_longitud and val_clean.startswith('-10'):
        fixed = val_clean[:4] + '.' + val_clean[4:]
    else:
        fixed = val_clean[:3] + '.' + val_clean[3:]
    return float(fixed)
