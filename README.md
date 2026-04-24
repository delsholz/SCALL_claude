# Simulador SCALL — Cosecha de Aguas Lluvias

Herramienta web para calcular la viabilidad de un sistema de cosecha de aguas lluvias (SCALL), basada en datos climáticos reales del período 2010–2020 del [(CR)²](https://www.cr2.cl/).

## ¿Qué hace?

- Encuentra la estación meteorológica más cercana a tu proyecto
- Simula el balance hídrico diario según tu techo, estanque y consumo
- Identifica años secos, normales y lluviosos reales
- Genera una curva de optimización del tamaño del estanque

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app_scall.py
```

La app se abre en `http://localhost:8501`.

## Datos

Los archivos de datos están en la carpeta `data/`:
- `cr2_prDaily_2020.txt.gz` — Precipitaciones diarias CR² (2010–2020)
- `BBDD precipitaciones.csv` — Base de datos auxiliar de precipitaciones
