"""Informe PDF del Simulador SCALL — identidad visual Fundación Amulén.

Estructura:
  1. Portada tipo afiche: fondo azul oscuro, dibujo del sistema y 4 tarjetas de resultado.
  2. Datos del análisis, escenarios y nivel real del estanque.
  3. Tamaño óptimo del estanque y conclusiones.

Punto de entrada: generar_informe_pdf(d) -> io.BytesIO  (d = st.session_state['informe_datos']).
"""
import io
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from fpdf import FPDF
from PIL import Image

from simulator import calcular_curva_optimizacion
from utils import clasificar_viabilidad, formato_chileno

_BASE = os.path.dirname(os.path.abspath(__file__))
_FONTS = os.path.join(_BASE, "fonts")
_LOGO = os.path.join(_BASE, "assets", "Logo_Amulen_blanco_hd.png")
_LOGO_RESPALDO = os.path.join(_BASE, "assets", "Logo_Amulen_blanco.png")

# ── Paleta (la misma de los esquemas de proyecto) ─────────────────────
OSC = (21, 19, 51)            # #151333 fondo oscuro
OSC2 = (35, 32, 102)          # tarjetas sobre fondo oscuro
AZUL = (46, 103, 176)         # #2E67B0
LINEA = (42, 95, 180)         # #2A5FB4
CIELO = (118, 194, 245)       # #76C2F5
AGUA = (95, 174, 236)         # #5FAEEC
AMARILLO = (255, 226, 122)    # #FFE27A
LIMA = (225, 255, 188)        # #E1FFBC
ARENA = (248, 243, 234)       # #F8F3EA
BLANCO = (255, 255, 255)
TEXTO = OSC
GRIS = (110, 119, 130)
ROJO_SUAVE = (255, 179, 167)

C_SECO = (255, 243, 205)
C_NORMAL = (209, 232, 248)
C_LLUVIOSO = (225, 255, 188)

_MESES_ES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}


def _hex(c):
    return "#%02X%02X%02X" % tuple(c)


# ── Utilidades de formato ─────────────────────────────────────────────
def _vacio(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() in ("", "—", "-")
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _num(v, dec=0, suf="", vacio="sin dato"):
    """Número en formato chileno (5.000 · 68,0). Si falta el dato, escribe «sin dato»."""
    if _vacio(v):
        return vacio
    try:
        return formato_chileno(float(v), dec) + suf
    except (TypeError, ValueError):
        return str(v)


def _col_deficit(df):
    return "Déficit Diario (L)" if "Déficit Diario (L)" in df.columns else "Deficit Diario (L)"


def _metricas_escenario(df, totales, anio):
    """Demanda, captado, cobertura (%), días sin agua y lluvia anual de un escenario."""
    col = _col_deficit(df)
    td = float(df["Demanda (L)"].sum())
    deficit = float(df[col].sum())
    cobertura = ((td + deficit) / td * 100) if td > 0 else 100.0
    return {
        "demanda": td,
        "captado": float(df["Captado (L)"].sum()),
        "cobertura": cobertura,
        "dias_sin_agua": int((df[col] < 0).sum()),
        "dias_operacion": int((df["Demanda (L)"] > 0).sum()),
        "lluvia": totales.get(anio, 0),
    }


def _desnivel_texto(desnivel):
    if _vacio(desnivel):
        return "sin dato"
    n = float(desnivel)
    if abs(n) < 0.5:
        return "a la misma altura"
    return f"{formato_chileno(abs(n), 0)} m " + ("sobre" if n > 0 else "bajo") + " la estación"


# ── Gráficos (matplotlib) ─────────────────────────────────────────────
def _registrar_fuentes_mpl():
    for f in ("Poppins-Regular.ttf", "Poppins-Bold.ttf", "Poppins-Italic.ttf"):
        try:
            font_manager.fontManager.addfont(os.path.join(_FONTS, f))
        except Exception:
            pass
    plt.rcParams["font.family"] = "Poppins"


def _png(fig, ancho_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.04,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    w, h = Image.open(buf).size
    buf.seek(0)
    return buf, ancho_mm * h / w


def _estilo_eje(ax):
    ax.set_facecolor(_hex(ARENA))
    ax.grid(axis="y", color=_hex(AZUL), alpha=0.14)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(_hex(AZUL))
    ax.tick_params(axis="both", labelsize=8, colors=_hex(OSC))


def _grafico_nivel(df, capacidad):
    """Nivel REAL del estanque (Estanque Final): entre 0 y su capacidad."""
    _registrar_fuentes_mpl()
    x = pd.to_datetime(df["Eje X"])
    y = df["Estanque Final (L)"].astype(float).values
    cap = float(capacidad)

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    fig.patch.set_facecolor("white")
    _estilo_eje(ax)

    if "Día" in df.columns:
        vac = df["Día"].astype(str).str.contains("Vacaciones").values
        if vac.any():
            ax.fill_between(x, 0, 1, where=vac, transform=ax.get_xaxis_transform(),
                            color="#CBD3DD", alpha=0.55, linewidth=0, label="Sin clases")

    ax.fill_between(x, y, color=_hex(CIELO), alpha=0.35, linewidth=0)
    ax.plot(x, y, color=_hex(AZUL), linewidth=1.8, label="Nivel del estanque")
    ax.axhline(cap, color="#E0A800", linestyle="--", linewidth=1.4,
               label=f"Capacidad: {formato_chileno(cap, 0)} L")

    ax.set_ylim(-cap * 0.13, cap * 1.14)
    col = _col_deficit(df)
    sin_agua = (df[col] < 0).values
    if sin_agua.any():
        ax.vlines(x[sin_agua], -cap * 0.115, -cap * 0.02, color="#E5533D", linewidth=1.3,
                  label="Día sin agua")
    if "Rebalse (L)" in df.columns:
        reb = (df["Rebalse (L)"] > 0).values
        if reb.any():
            ax.vlines(x[reb], cap * 1.03, cap * 1.12, color=_hex(AGUA), linewidth=1.3,
                      label="Rebalse")

    ax.set_ylim(-cap * 0.13, cap * 1.14)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= cap * 1.001])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: _MESES_ES.get(mdates.num2date(v).month, "")))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: formato_chileno(v, 0)))
    ax.set_ylabel("Volumen (L)", fontsize=8.5, color=_hex(OSC))
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5,
              frameon=False)
    fig.tight_layout(pad=0.4)
    return _png(fig, 190)


def _grafico_curva(caps, efs, cap_opt, ef_act, ef_opt, capacidad):
    _registrar_fuentes_mpl()
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    fig.patch.set_facecolor("white")
    _estilo_eje(ax)
    ax.grid(axis="both", color=_hex(AZUL), alpha=0.14)

    ax.plot(caps, efs, color=_hex(AGUA), linewidth=2.4, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgecolor=_hex(AZUL), label="Cobertura simulada")
    if not _vacio(ef_act):
        ax.axvline(capacidad, color="#E0A800", linestyle="--", linewidth=1.5,
                   label=f"Estanque actual ({formato_chileno(capacidad, 0)} L)")
        ax.scatter([capacidad], [ef_act], color="#E0A800", s=70, zorder=5)
    if not _vacio(cap_opt) and not _vacio(ef_opt):
        ax.axvline(cap_opt, color=_hex(AZUL), linestyle="--", linewidth=1.5,
                   label=f"Óptimo recomendado ({formato_chileno(cap_opt, 0)} L)")
        ax.scatter([cap_opt], [ef_opt], color=_hex(AZUL), s=90, marker="D", zorder=6)

    ax.set_xlabel("Capacidad del estanque (L)", fontsize=8.5, color=_hex(OSC))
    ax.set_ylabel("Cobertura de la demanda (%)", fontsize=8.5, color=_hex(OSC))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_chileno(v, 0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95, facecolor="white",
              edgecolor=_hex(CIELO))
    fig.tight_layout(pad=0.4)
    return _png(fig, 190)


def _dibujo_portada(m, d):
    """Ilustración plana del sistema: lluvia → techo → estanque → uso. 210 × 94 mm."""
    _registrar_fuentes_mpl()
    W, H = 210.0, 94.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(_hex(OSC))
    ax.set_facecolor(_hex(OSC))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    LINE, PIPE, WATER, SOLIDO, WALL = "#2A5FB4", "#62B3F2", "#5FAEEC", "#5DAEEE", "#EEF0F0"
    GY = 18  # nivel del suelo

    # suelo
    ax.add_patch(Rectangle((0, 0), W, GY, fc="#8A6440", ec="none", zorder=1))
    ax.add_patch(Rectangle((0, GY - 1.6), W, 1.6, fc="#9DF1DB", ec="none", zorder=1.1))

    # nube y lluvia
    for cx, cy, r in ((24, 80, 6.5), (32, 83, 8.5), (41, 80, 6.5), (32, 77, 7)):
        ax.add_patch(Circle((cx, cy), r, fc="#EEF3FA", ec="none", zorder=3))
    for i, gx in enumerate(np.linspace(18, 46, 8)):
        y0 = 70 - (i % 2) * 4
        ax.plot([gx, gx - 2.5], [y0, y0 - 6], color=PIPE, lw=2.4, solid_capstyle="round", zorder=3)
    ax.text(52, 83, "Lluvia año normal", color="white", fontsize=8, va="center", zorder=6)
    ax.text(52, 78, f"{_num(m['normal']['lluvia'])} mm", color=_hex(AMARILLO), fontsize=11,
            fontweight="bold", va="center", zorder=6)

    # escuela: muro, techo, puerta y ventanas
    ax.add_patch(Rectangle((12, GY), 56, 26, fc=WALL, ec=LINE, lw=2, zorder=3))
    ax.add_patch(Polygon([(8, GY + 26), (72, GY + 26), (62, GY + 36), (18, GY + 36)],
                         closed=True, fc=SOLIDO, ec=LINE, lw=2, zorder=4))
    ax.add_patch(Rectangle((35, GY), 10, 14, fc="#98D2F8", ec=LINE, lw=1.6, zorder=4))
    for vx in (18, 52):
        ax.add_patch(Rectangle((vx, GY + 9), 10, 9, fc="#98D2F8", ec=LINE, lw=1.6, zorder=4))

    # canaleta y bajada → tubería enterrada → estanque
    ax.plot([72, 76, 76], [GY + 26, GY + 26, GY], color=PIPE, lw=4, solid_capstyle="round", zorder=5)
    ax.plot([76, 76, 98, 98], [GY, 9, 9, 26], color=PIPE, lw=4, solid_capstyle="round",
            solid_joinstyle="round", zorder=2)

    # estanque
    tx, tw, th = 100, 34, 38
    ax.add_patch(Rectangle((tx, GY), tw, th, fc="#EAF1F7", ec=LINE, lw=2, zorder=3))
    nivel = min(max(m["normal"]["cobertura"], 12), 100) / 100  # decorativo, según cobertura
    ax.add_patch(Rectangle((tx + 1, GY + 1), tw - 2, (th - 2) * nivel, fc=WATER, ec="none", zorder=3.2))
    ax.add_patch(Ellipse((tx + tw / 2, GY + th), tw, 7, fc="#98D2F8", ec=LINE, lw=2, zorder=4))
    ax.text(tx + tw / 2, GY + th * 0.4, f"{_num(d['capacidad_maxima'])} L", color="white",
            fontsize=12, fontweight="bold", ha="center", va="center", zorder=6)

    # salida del estanque → llave
    ax.plot([tx + tw, 150, 150, 168, 168], [GY + 6, GY + 6, 9, 9, GY + 14], color=PIPE, lw=4,
            solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.plot([168, 168, 180, 180], [GY + 14, GY + 22, GY + 22, GY + 18], color=PIPE, lw=4,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    for k, dy in enumerate((0, 4.5, 9)):
        ax.add_patch(Ellipse((180, GY + 14 - dy), 2.2, 3.2, fc=WATER, ec="none", zorder=5))

    # niñas y niños
    for k, (cx, cam) in enumerate(((190, "#3B6FD6"), (197.5, "#FFD43B"), (205, "#7ACB8A"))):
        ax.add_patch(Rectangle((cx - 2.6, GY), 5.2, 9, fc=cam, ec=LINE, lw=1.4, zorder=4))
        ax.add_patch(Circle((cx, GY + 12.2), 2.9, fc="#F2C7A5", ec=LINE, lw=1.4, zorder=4))

    # insignias numeradas
    def insignia(x, y, n):
        ax.add_patch(Circle((x, y), 4.4, fc=_hex(AMARILLO), ec="white", lw=1.6, zorder=8))
        ax.text(x, y - 0.1, str(n), color=_hex(OSC), fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=9)

    insignia(12, 86, 1)
    insignia(40, GY + 44, 2)
    insignia(tx + tw / 2, GY + th + 12, 3)
    insignia(197, GY + 25, 4)

    # leyendas sobre el suelo
    pers = _num(d["numero_personas"])
    ax.text(40, 9, f"Techo {_num(d['techo'], 1)} m²", color="white", fontsize=9.5,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(40, 4.2, "capta la lluvia", color="#F2E4D0", fontsize=7.5, ha="center", va="center", zorder=6)
    ax.text(tx + tw / 2, 4.2, "la guarda", color="#F2E4D0", fontsize=7.5, ha="center", va="center", zorder=6)
    ax.text(tx + tw / 2, 9, f"Estanque {_num(d['capacidad_maxima'])} L", color="white", fontsize=9.5,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(197, 9, f"{pers} personas", color="white", fontsize=9.5, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(197, 4.2, f"{_num(d['litros_persona_dia'])} L/persona/día", color="#F2E4D0", fontsize=7.5,
            ha="center", va="center", zorder=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _logo(ancho_px):
    """Logo blanco de Amulén, en alta resolución (con respaldo al antiguo)."""
    ruta = _LOGO if os.path.exists(_LOGO) else _LOGO_RESPALDO
    try:
        img = Image.open(ruta).convert("RGBA")
        if img.width != ancho_px:
            img = img.resize((ancho_px, int(img.height * ancho_px / img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf, img.width / img.height
    except Exception:
        return None, 3.1


# ── PDF ───────────────────────────────────────────────────────────────
class _PDF(FPDF):
    logo_pie = None

    def footer(self):
        if self.page_no() == 1:      # la portada va sin pie
            return
        self.set_fill_color(*OSC)
        self.rect(0, 285, 210, 12, "F")
        self.set_fill_color(*CIELO)
        self.rect(0, 285, 210, 1.4, "F")
        self.set_xy(10, 288)
        self.set_font("Poppins", "I", 7)
        self.set_text_color(*ARENA)
        self.cell(140, 6, "Simulador SCALL  ·  Fundación Amulén  ·  Datos climáticos: CR²  ·  www.cr2.cl")
        if self.logo_pie is not None:
            try:
                self.image(self.logo_pie[0], x=198 - 8 * self.logo_pie[1], y=287.4, h=8)
            except Exception:
                pass


def generar_informe_pdf(d):
    hoy = date.today().strftime("%d/%m/%Y")
    totales = d.get("totales_anio") or {}

    m = {
        "seco": _metricas_escenario(d["df_seco"], totales, d["anio_seco"]),
        "normal": _metricas_escenario(d["df_normal"], totales, d["anio_mediano"]),
        "lluvioso": _metricas_escenario(d["df_lluvioso"], totales, d["anio_lluvioso"]),
    }
    curva = d.get("curva_normal") or calcular_curva_optimizacion(d["df_normal"], d["capacidad_maxima"])
    caps, efs, cap_opt, ef_act, ef_opt = curva
    pct_n = m["normal"]["cobertura"]
    veredicto = clasificar_viabilidad(pct_n)
    anios_validos = len(totales) or (d["anio_fin"] - d["anio_inicio"] + 1)
    n_meses = max(len(d.get("meses_seleccionados") or []), 1)
    mult = d.get("multiplicador_lluvia")
    # el signo se calcula con las altitudes (los historiales viejos guardaron el desnivel sin signo)
    if not _vacio(d.get("alt_proyecto")) and not _vacio(d.get("est_altitud")):
        desnivel = float(d["alt_proyecto"]) - float(d["est_altitud"])
    else:
        desnivel = d.get("desnivel")

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(0, 0, 0)
    for estilo, archivo in (("", "Poppins-Regular.ttf"), ("B", "Poppins-Bold.ttf"),
                            ("I", "Poppins-Italic.ttf"), ("BI", "Poppins-BoldItalic.ttf")):
        pdf.add_font("Poppins", estilo, os.path.join(_FONTS, archivo))
    pdf.logo_pie = _logo(900)
    logo_cab = _logo(1600)
    pdf.set_title(f"Informe SCALL — {d['nombre_proyecto']}")
    pdf.set_author("Fundación Amulén")

    def celda(w, h, txt, estilo="", size=8, relleno=True, align="L", color=None, x=None, y=None):
        """Celda que achica la letra si el texto no cabe."""
        if x is not None:
            pdf.set_xy(x, y)
        if color:
            pdf.set_text_color(*color)
        sz = size
        pdf.set_font("Poppins", estilo, sz)
        while sz > 5.5 and pdf.get_string_width(str(txt)) > w - 2:
            sz -= 0.5
            pdf.set_font("Poppins", estilo, sz)
        pdf.cell(w, h, str(txt), fill=relleno, align=align)

    def logo(buf_ratio, x, y, h):
        if buf_ratio[0] is None:
            return
        try:
            pdf.image(buf_ratio[0], x=x, y=y, h=h)
            buf_ratio[0].seek(0)
        except Exception:
            pass

    # ══ PÁGINA 1 — PORTADA ═══════════════════════════════════════════
    pdf.add_page()
    pdf.set_fill_color(*OSC)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*CIELO)
    pdf.rect(0, 0, 210, 3, "F")
    pdf.set_fill_color(*AMARILLO)
    pdf.rect(0, 3, 45, 1.2, "F")

    logo(logo_cab, 12, 12, 17)
    pdf.set_xy(110, 14)
    pdf.set_font("Poppins", "", 8)
    pdf.set_text_color(*CIELO)
    pdf.cell(88, 5, "INFORME DE VIABILIDAD", align="R")
    pdf.set_xy(110, 19.5)
    pdf.set_text_color(*ARENA)
    pdf.cell(88, 5, hoy, align="R")

    pdf.set_xy(12, 40)
    pdf.set_font("Poppins", "B", 30)
    pdf.set_text_color(*BLANCO)
    pdf.cell(186, 13, "Simulador SCALL")
    pdf.set_xy(12, 54)
    pdf.set_font("Poppins", "", 13)
    pdf.set_text_color(*CIELO)
    pdf.cell(186, 7, "Cosecha de aguas lluvias")

    # nombre del proyecto
    pdf.set_fill_color(*OSC2)
    pdf.rect(10, 68, 190, 22, "F", round_corners=True, corner_radius=3)
    pdf.set_fill_color(*AMARILLO)
    pdf.rect(10, 68, 2.2, 22, "F")
    nombre = str(d["nombre_proyecto"])
    sz = 21
    pdf.set_font("Poppins", "B", sz)
    while sz > 11 and pdf.get_string_width(nombre) > 176:
        sz -= 1
        pdf.set_font("Poppins", "B", sz)
    pdf.set_xy(17, 70)
    pdf.set_text_color(*BLANCO)
    pdf.cell(180, 11, nombre)
    pdf.set_xy(17, 81)
    pdf.set_font("Poppins", "", 8.5)
    pdf.set_text_color(*CIELO)
    pdf.cell(180, 6, f"Estación {d['est_nombre']}  ·  Datos {d['anio_inicio']}–{d['anio_fin']}  ·  "
                     f"{_num(d['alt_proyecto'], 0, ' m.s.n.m.')}")

    # dibujo del sistema
    try:
        pdf.image(_dibujo_portada(m, d), x=0, y=94, w=210)
    except Exception:
        pass

    # tarjetas de resultado
    color_v = {"VIABLE": LIMA, "PARCIALMENTE VIABLE": AMARILLO}.get(veredicto, ROJO_SUAVE)
    explica = {
        "VIABLE": "La lluvia cubre la mayor parte de la demanda de un año normal.",
        "PARCIALMENTE VIABLE": "La lluvia cubre solo una parte de la demanda; conviene una fuente complementaria.",
    }.get(veredicto, "La lluvia no alcanza por sí sola: se necesita otra fuente de agua.")

    def tarjeta(x, y, w, h, etiqueta, valor, sub, fondo, c_etiq, c_valor, c_sub, tam_valor=24):
        pdf.set_fill_color(*fondo)
        pdf.rect(x, y, w, h, "F", round_corners=True, corner_radius=3)
        pdf.set_xy(x + 5, y + 3.5)
        pdf.set_font("Poppins", "B", 7.5)
        pdf.set_text_color(*c_etiq)
        pdf.cell(w - 10, 4, etiqueta.upper())
        sz_v = tam_valor
        pdf.set_font("Poppins", "B", sz_v)
        while sz_v > 12 and pdf.get_string_width(valor) > w - 10:
            sz_v -= 1
            pdf.set_font("Poppins", "B", sz_v)
        pdf.set_xy(x + 5, y + 9)
        pdf.set_text_color(*c_valor)
        pdf.cell(w - 10, 11, valor)
        pdf.set_xy(x + 5, y + h - 7.5)
        pdf.set_font("Poppins", "", 7.5)
        pdf.set_text_color(*c_sub)
        pdf.cell(w - 10, 4.5, sub)

    cy = 192
    tarjeta(10, cy, 92, 30, "Veredicto", veredicto, "con el estanque ingresado", color_v,
            OSC, OSC, OSC, tam_valor=19)
    tarjeta(108, cy, 92, 30, "Cobertura · año normal", _num(pct_n, 1, " %"),
            f"con estanque de {_num(d['capacidad_maxima'])} L", OSC2, CIELO, BLANCO, ARENA)
    tarjeta(10, cy + 35, 92, 30, "Estanque recomendado", _num(cap_opt, 0, " L"),
            f"cobertura de {_num(ef_opt, 1, ' %')} en año normal", OSC2, CIELO, AMARILLO, ARENA)
    tarjeta(108, cy + 35, 92, 30, "Días sin agua · año seco", _num(m["seco"]["dias_sin_agua"]),
            f"de {_num(m['seco']['dias_operacion'])} días de operación", OSC2, CIELO, BLANCO, ARENA)

    pdf.set_xy(10, 264)
    pdf.set_font("Poppins", "I", 8.5)
    pdf.set_text_color(*ARENA)
    pdf.multi_cell(190, 5, explica, align="C")
    pdf.set_xy(10, 283)
    pdf.set_font("Poppins", "", 7)
    pdf.set_text_color(*CIELO)
    pdf.cell(190, 4, "Datos climáticos: Centro de Ciencia del Clima y la Resiliencia (CR²)  ·  Fundación Amulén",
             align="C")

    # ══ PÁGINA 2 — DATOS Y RESULTADOS ════════════════════════════════
    def nueva_pagina(titulo_pagina):
        pdf.add_page()
        pdf.set_fill_color(*OSC)
        pdf.rect(0, 0, 210, 20, "F")
        pdf.set_fill_color(*CIELO)
        pdf.rect(0, 20, 210, 1.6, "F")
        logo(logo_cab, 168, 4.5, 11)
        pdf.set_xy(10, 4.5)
        pdf.set_font("Poppins", "B", 12)
        pdf.set_text_color(*BLANCO)
        pdf.cell(150, 6, titulo_pagina)
        pdf.set_xy(10, 11.5)
        pdf.set_font("Poppins", "", 7.5)
        pdf.set_text_color(*CIELO)
        nom = str(d["nombre_proyecto"])
        pdf.cell(150, 5, nom if len(nom) < 70 else nom[:67] + "…")
        return 27

    def titulo_seccion(txt, y):
        pdf.set_fill_color(*AMARILLO)
        pdf.rect(10, y, 2.4, 7, "F")
        pdf.set_fill_color(*AZUL)
        pdf.set_xy(12.4, y)
        pdf.set_font("Poppins", "B", 9)
        pdf.set_text_color(*BLANCO)
        pdf.cell(187.6, 7, f"  {txt}", fill=True)
        return y + 7

    def hay_lugar(y, alto):
        return y + alto <= 283

    y = nueva_pagina("Datos del análisis")

    y = titulo_seccion("PARÁMETROS DEL SISTEMA", y)
    demanda_anual = m["normal"]["demanda"]
    filas = [
        ("Superficie del techo", f"{_num(d['techo'], 2)} m²",
         "Capacidad del estanque", f"{_num(d['capacidad_maxima'])} L"),
        ("Eficiencia de captación", _num(float(d["eficiencia"]) * 100, 0, " %"),
         "Tipo de uso", str(d.get("tipo_uso") or "sin dato").split("(")[0].strip()),
        ("Número de personas", _num(d["numero_personas"]),
         "Consumo por persona", f"{_num(d['litros_persona_dia'])} L/persona/día"),
        ("Demanda anual (año normal)", _num(demanda_anual, 0, " L"),
         "Consumo mensual promedio", _num(demanda_anual / n_meses, 0, " L/mes")),
        ("Coordenadas", f"{_num(d['lat_proyecto'], 4)} / {_num(d['lon_proyecto'], 4)}",
         "Altitud del proyecto", _num(d["alt_proyecto"], 0, " m.s.n.m.")),
    ]
    for i, (l1, v1, l2, v2) in enumerate(filas):
        bg = ARENA if i % 2 == 0 else BLANCO
        pdf.set_fill_color(*bg)
        celda(46, 6.8, l1, "B", 7.5, True, color=TEXTO, x=10, y=y)
        celda(49, 6.8, v1, "", 7.5, True, color=TEXTO)
        celda(46, 6.8, l2, "B", 7.5, True, color=TEXTO)
        celda(49, 6.8, v2, "", 7.5, True, color=TEXTO)
        y += 6.8
    pdf.set_fill_color(*BLANCO)
    celda(46, 6.8, "Meses de operación", "B", 7.5, True, color=TEXTO, x=10, y=y)
    celda(144, 6.8, ", ".join(d.get("meses_seleccionados") or []) or "sin dato", "", 7.5, True, color=TEXTO)
    y += 6.8
    if not _vacio(mult) and abs(float(mult) - 1.0) > 1e-6:
        pdf.set_fill_color(*ARENA)
        celda(46, 6.8, "Ajuste orográfico", "B", 7.5, True, color=TEXTO, x=10, y=y)
        celda(144, 6.8, f"la lluvia de la estación se multiplicó por {_num(mult, 2)} "
                        f"({'+' if mult > 1 else ''}{_num((float(mult) - 1) * 100, 1)} %)",
              "", 7.5, True, color=TEXTO)
        y += 6.8
    y += 5

    y = titulo_seccion("ESTACIÓN METEOROLÓGICA UTILIZADA", y)
    filas = [
        ("Nombre de la estación", str(d["est_nombre"]),
         "Código", str(d["est_codigo"])),
        ("Altitud de la estación", _num(d["est_altitud"], 0, " m.s.n.m."),
         "Desnivel con el proyecto", _desnivel_texto(desnivel)),
        ("Distancia efectiva", _num(d["distancia"], 1, " km (corrección altitudinal)"),
         "Calidad de los datos", _num(d["pct_calidad"], 1, " % de meses válidos")),
    ]
    for i, (l1, v1, l2, v2) in enumerate(filas):
        bg = ARENA if i % 2 == 0 else BLANCO
        pdf.set_fill_color(*bg)
        celda(46, 6.8, l1, "B", 7.5, True, color=TEXTO, x=10, y=y)
        celda(49, 6.8, v1, "", 7.5, True, color=TEXTO)
        celda(46, 6.8, l2, "B", 7.5, True, color=TEXTO)
        celda(49, 6.8, v2, "", 7.5, True, color=TEXTO)
        y += 6.8
    y += 5

    # tabla de escenarios
    y = titulo_seccion("RESULTADOS POR ESCENARIO CLIMÁTICO", y)
    anchos = [46, 14, 24, 28, 28, 24, 26]
    cabeza = ["Escenario", "Año", "Lluvia (mm)", "Captado (L)", "Demanda (L)", "Cobertura", "Días sin agua"]
    pdf.set_fill_color(*OSC)
    pdf.set_xy(10, y)
    for h, w in zip(cabeza, anchos):
        celda(w, 8, h, "B", 8, True, "C", BLANCO)
    y += 8
    for nombre, anio, clave, color in (("Año seco (P5)", d["anio_seco"], "seco", C_SECO),
                                       ("Año normal (mediana)", d["anio_mediano"], "normal", C_NORMAL),
                                       ("Año lluvioso (P95)", d["anio_lluvioso"], "lluvioso", C_LLUVIOSO)):
        e = m[clave]
        negrita = "B" if clave == "normal" else ""
        pdf.set_fill_color(*color)
        pdf.set_xy(10, y)
        valores = [nombre, str(anio), _num(e["lluvia"]), _num(e["captado"]), _num(e["demanda"]),
                   _num(e["cobertura"], 1, " %"), _num(e["dias_sin_agua"])]
        for k, (v, w) in enumerate(zip(valores, anchos)):
            celda(w, 8, v, negrita, 8, True, "L" if k == 0 else "C", TEXTO)
        y += 8
    pdf.set_draw_color(*CIELO)
    pdf.line(10, y, 200, y)
    y += 6

    # gráfico del nivel real del estanque
    y = titulo_seccion(f"NIVEL DEL ESTANQUE · AÑO NORMAL ({d['anio_mediano']})", y)
    try:
        img, alto = _grafico_nivel(d["df_normal"], d["capacidad_maxima"])
        if not hay_lugar(y + 2, alto):
            y = nueva_pagina("Datos del análisis")
        pdf.image(img, x=10, y=y + 2, w=190)
        y += alto + 7
    except Exception as e:
        pdf.set_xy(10, y + 3)
        pdf.set_font("Poppins", "I", 8)
        pdf.set_text_color(*GRIS)
        pdf.cell(190, 8, f"[Gráfico no disponible: {type(e).__name__}]", align="C")
        y += 14

    # ══ PÁGINA 3 — TAMAÑO ÓPTIMO Y CONCLUSIONES ═════════════════════
    y = nueva_pagina("Tamaño óptimo y conclusiones")
    y = titulo_seccion("TAMAÑO ÓPTIMO DEL ESTANQUE · AÑO NORMAL", y)
    ancho_t = 46.5
    y += 3
    for k, (et, val, sub, fondo, c_val) in enumerate((
            ("Estanque ingresado", _num(d["capacidad_maxima"], 0, " L"),
             f"cobertura {_num(ef_act, 1, ' %')}", ARENA, AZUL),
            ("Estanque recomendado", _num(cap_opt, 0, " L"),
             f"cobertura {_num(ef_opt, 1, ' %')}", C_NORMAL, AZUL),
    )):
        x = 10 + k * 95
        pdf.set_fill_color(*fondo)
        pdf.rect(x, y, 92, 18, "F", round_corners=True, corner_radius=2.5)
        pdf.set_xy(x + 4, y + 2)
        pdf.set_font("Poppins", "B", 7.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(84, 4, et.upper())
        pdf.set_xy(x + 4, y + 6)
        pdf.set_font("Poppins", "B", 15)
        pdf.set_text_color(*c_val)
        pdf.cell(50, 8, val)
        pdf.set_xy(x + 4, y + 13)
        pdf.set_font("Poppins", "", 7.5)
        pdf.set_text_color(*TEXTO)
        pdf.cell(84, 4, sub)
    y += 23

    try:
        img, alto = _grafico_curva(caps, efs, cap_opt, ef_act, ef_opt, d["capacidad_maxima"])
        pdf.image(img, x=10, y=y, w=190)
        y += alto + 6
    except Exception as e:
        pdf.set_xy(10, y)
        pdf.set_font("Poppins", "I", 8)
        pdf.set_text_color(*GRIS)
        pdf.cell(190, 8, f"[Gráfico no disponible: {type(e).__name__}]", align="C")
        y += 14

    # conclusiones
    y = titulo_seccion("CONCLUSIONES Y RECOMENDACIONES", y)
    seco = m["seco"]
    parrafos = [
        (f"Viabilidad del sistema: {veredicto} ({_num(pct_n, 1, ' %')} de cobertura en año normal)", "B"),
        (f"En el año normal ({d['anio_mediano']}), el sistema cubre el {_num(pct_n, 1, ' %')} de la demanda "
         f"anual con el estanque ingresado de {_num(d['capacidad_maxima'])} L. Los datos corresponden a "
         f"{anios_validos} años de registros históricos de la estación {d['est_nombre']}.", ""),
        (f"Para aprovechar mejor la lluvia se recomienda un estanque de {_num(cap_opt)} L, con el que se "
         f"alcanza una cobertura de {_num(ef_opt, 1, ' %')} en el año normal." if not _vacio(cap_opt) else
         "No fue posible calcular un tamaño óptimo de estanque con los datos disponibles.", ""),
        (f"En el año seco (percentil 5, año {d['anio_seco']}) se registran {_num(seco['dias_sin_agua'])} días "
         f"sin suministro. Se recomienda disponer de una fuente de abastecimiento complementaria para "
         f"cubrir esos períodos críticos.", ""),
        (f"Dotación utilizada: {_num(d['litros_persona_dia'])} L/persona/día (RIDAA, DS MOP 50/2002, Anexo 4). "
         f"Techo de {_num(d['techo'], 2)} m² y eficiencia de captación de "
         f"{_num(float(d['eficiencia']) * 100, 0, ' %')}.", "I"),
        (f"Informe generado el {hoy} con datos del Centro de Ciencia del Clima y la Resiliencia (CR²).", "I"),
    ]
    # medir antes de dibujar: la caja tiene el alto del texto
    alto_total = 5
    for txt, est in parrafos:
        pdf.set_font("Poppins", est, 8.5)
        lineas = pdf.multi_cell(180, 5.2, txt, dry_run=True, output="LINES")
        alto_total += len(lineas) * 5.2 + 2.5
    if not hay_lugar(y + 2, alto_total):
        y = nueva_pagina("Tamaño óptimo y conclusiones")
    pdf.set_fill_color(*ARENA)
    pdf.rect(10, y, 190, alto_total, "F")
    pdf.set_fill_color(*AMARILLO)
    pdf.rect(10, y, 2.4, alto_total, "F")
    yy = y + 3
    pdf.set_text_color(*TEXTO)
    for txt, est in parrafos:
        pdf.set_font("Poppins", est, 8.5)
        pdf.set_xy(16, yy)
        pdf.multi_cell(180, 5.2, txt)
        yy = pdf.get_y() + 2.5

    # cómo leer el informe
    y = y + alto_total + 6
    notas = [
        "Año seco, normal y lluvioso son el percentil 5, la mediana y el percentil 95 de la lluvia anual "
        "de la estación; cada uno se simula por separado.",
        "El estanque comienza vacío (0 L) en cada escenario: es la condición más exigente.",
        "Los días en que la estación no midió se simulan como 0 mm de lluvia, así que el resultado es "
        "conservador.",
        "El tamaño recomendado es el punto donde agrandar el estanque deja de dar ganancias importantes.",
    ]
    alto_notas = 7 + 3
    for txt in notas:
        pdf.set_font("Poppins", "", 8)
        alto_notas += len(pdf.multi_cell(176, 4.8, txt, dry_run=True, output="LINES")) * 4.8 + 1.5
    if hay_lugar(y, alto_notas):
        y = titulo_seccion("CÓMO LEER ESTE INFORME", y)
        yy = y + 3
        for txt in notas:
            pdf.set_fill_color(*AMARILLO)
            pdf.rect(12, yy + 1.6, 1.6, 1.6, "F")
            pdf.set_font("Poppins", "", 8)
            pdf.set_text_color(*TEXTO)
            pdf.set_xy(17, yy)
            pdf.multi_cell(176, 4.8, txt)
            yy = pdf.get_y() + 1.5

    return io.BytesIO(bytes(pdf.output()))
