"""Esquema «Proyecto de agua potable» de una escuela (APA) o un SSR (CR) con el tipo de dibujo de la ilustración de Canva
(`references/ilustracion_canva_ssr.png`), aprobado por Diego el 2026-11-09 con Crucero Nuevo.

Uso: python -I esquema.py config.json salida.png

Lee la configuración (ver references/config_ejemplo.json) y dibuja de izquierda a derecha, sin tuberías que se
devuelvan: fuente → tramos (casetas con sus equipos, estanques, macromedidor) → destino (escuela o casas del SSR).
Numera cada etapa. Abajo va «número · nombre · descripción breve» (sin marcas, litros ni HP). Las medidas van chicas
DENTRO de cada equipo (campo "medida" de la configuración).
Celeste con contorno azul = lo instala Amulén · todo gris = ya existe. Tamaños en proporción real aproximada
(1 unidad ≈ 9 cm). Salida de 200 px por unidad (2250 px de alto).
"""
import sys, json, math, random, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyBboxPatch, Ellipse
from matplotlib.transforms import Affine2D
from matplotlib import font_manager as fm
from PIL import Image

AQUI = Path(__file__).resolve().parent.parent
for _fd in (AQUI / "assets" / "fonts", Path(__file__).resolve().parent / "fonts"):
    for f in sorted(_fd.glob("*.ttf")):
        fm.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Poppins"
LOGOS = Path(r"C:/Users/Diego Elsholz/OneDrive - Universidad Católica de Chile/Escritorio/Logos")

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
out = sys.argv[2]
ES_SSR = cfg.get("tipo", "escuela") == "ssr"
TE = float(cfg.get("texto_etapas", 1.0))                  # tamaño del texto de la lista de etapas (1 = normal)
SD = bool(cfg.get("solo_dibujo"))      # solo el dibujo y la lista de etapas (sin título, leyenda ni logos): para incrustarlo en otros informes

# ---------- colores (ilustración de Canva + kit de marca) ----------
OSC, ARENA, DEST = "#151333", "#F7F2E9", "#1C6ADD"
LINE, PIPE, WATER, LIGHT = "#2A5FB4", "#62B3F2", "#5FAEEC", "#98D2F8"       # azules de lo NUEVO más marcados
GRASS, GROUND, WALL, LIME, WELL = "#9DF1DB", "#9A7048", "#EEF0F0", "#E1FFBC", "#2E67B0"   # GROUND = tierra café (cálida, media)
SOLIDO = "#5DAEEE"                                   # relleno opaco de filtros e hidroneumáticos nuevos
# lo que ya existe se dibuja TODO en gris (relleno, contorno, agua y texto)
OLD_F, OLD_E, OLD_W, OLD_WALL, OLD_T = "#B4BBC2", "#6E7782", "#9EA6AE", "#CDD2D6", "#3F4953"   # existente: RELLENO gris (contorno continuo)
LW = 2.6
GY, P = 36, 40.5                                      # suelo y altura de la tubería
H = 112.5
RAD = 1.8                                             # espesor del radier
WT = 28.4                                             # nivel de la napa (agua subterránea)
YU = (GY + WT) / 2                                    # profundidad de la tubería enterrada: mitad de la parte gris

def col(nuevo):
    return (LIGHT, LINE) if nuevo else (OLD_F, OLD_E)

def nuevo_de(d, defecto=True):
    return d.get("nuevo", defecto) if isinstance(d, dict) else defecto

def equipo(e):
    return (e, {}) if isinstance(e, str) else (e["tipo"], e)

# ---------- 1. medir el ancho ----------
ANCHO = {"tablero": 15, "cloracion": 15, "filtro": 9, "bombas_paralelo": 22, "hidroneumatico": 10, "presostato": 0, "generador": 12,
         "bomba_periferica": 18, "filtro_cartucho": 11, "uv": 12}
EXT_W, GAP = 34, 8            # EXT_W: espacio a la derecha de la caseta para el estanque exterior con su radier
fuente = cfg.get("fuente", {"tipo": "pozo"})
tramos = cfg.get("tramos", [])

POZO_DENTRO = bool(fuente.get("dentro_caseta"))      # el pozo está ADENTRO de la caseta (se agrega su ancho a la primera)
POZO_W = 11
anchos = []
for i_t, t in enumerate(tramos):
    k, d = next(iter(t.items()))
    if k == "caseta":
        nombres = [equipo(e)[0] for e in d.get("equipos", [])]
        ancho_eq = sum(ANCHO.get(n, 12) for n in nombres)
        if "generador" in nombres and "tablero" in nombres and nombres.index("generador") < nombres.index("tablero"):
            ancho_eq -= ANCHO["tablero"]          # el tablero va ENCIMA del generador: no suma ancho (caseta más angosta)
        base_w = max(5 + ancho_eq + (POZO_W if (POZO_DENTRO and i_t == 0) else 0), d.get("ancho_min", 26))
        anchos.append(base_w + (EXT_W if d.get("estanque_exterior") else 0))
    elif k == "estanque":
        if d.get("forma") == "vertical":
            _l = re.sub(r"[^\d]", "", str(d.get("medida", "") or "")) or "5000"
            anchos.append(21 * (int(_l) / 5000) ** (1 / 3) + 10)
        else:
            anchos.append(30 * d.get("cantidad", 1))
    elif k == "planta":
        _l = re.sub(r"[^\d]", "", str(d.get("estanque_potable", {}).get("medida", "") or "")) or "5000"
        anchos.append(d.get("ancho", 82) * d.get("escala", 0.62) + 21 * (int(_l) / 5000) ** (1 / 3) + 9)
    elif k == "contador":
        anchos.append(14)
X0 = 38 if fuente.get("tipo") == "lluvia" else (13 if POZO_DENTRO else 31)              # donde empiezan los tramos (con el pozo adentro de la caseta, más a la izquierda)
dest_w = 58 if ES_SSR else (68 if fuente.get("tipo") == "lluvia" else 41)
W = cfg.get("ancho_dibujo") or max(X0 + sum(anchos) + GAP * len(tramos) + dest_w + 6, 200)      # ancho_dibujo: escuela más angosta

fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc=OSC, ec="none", zorder=-10))

# ---------- utilidades de dibujo ----------
def poly(pts, fc, z, ec=LINE, lw=LW):
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round"))

def rbox(x, y, w, h, r, fc, z, ec=LINE, lw=LW):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

def pipe(xs, ys, z=3, lw=7):
    ax.plot(xs, ys, color=LINE, lw=lw + 3.4, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(xs, ys, color=PIPE, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.05)

def flecha(x, y, ang=0, s=0.55, z=6.6):
    """Flecha de dirección del flujo, muy discreta: un chevron «>» chico sobre la tubería (ang en grados: 0 → derecha,
    90 → arriba, 180 → izquierda)."""
    a = math.radians(ang); c, sn = math.cos(a), math.sin(a)
    pts = [(-s * 0.55, s * 0.8), (s * 0.45, 0), (-s * 0.55, -s * 0.8)]
    ax.plot([x + px * c - py * sn for px, py in pts], [y + px * sn + py * c for px, py in pts], color=LINE, lw=1.3,
            solid_capstyle="round", solid_joinstyle="round", zorder=z)

def num(x, y, n, z=20):
    ax.add_patch(Circle((x, y), 2.05, fc=WELL, ec="white", lw=1.6, zorder=z))
    ax.text(x, y - 0.1, str(n), ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=z + 1)

def medida(x, y, txt, nuevo=True, rot=0, fs=8.5, z=9):
    """Medida chica DENTRO del equipo (60 L, 1 HP, 10 kVA…). Si no hay dato no se dibuja nada."""
    if txt:
        ax.text(x, y, txt, ha="center", va="center", rotation=rot, fontsize=fs, fontweight="bold",
                color=LINE if nuevo else OLD_T, zorder=z)

def onda(x0, x1, y, amp=0.55, n=2):
    xs = np.linspace(x0, x1, 80)
    return list(zip(xs, y + amp * np.sin((xs - x0) / (x1 - x0) * 2 * math.pi * n)))

def cerro(cx, base, w, h, fc, nieve=False):
    """Cerro tipo VOLCÁN: triángulo ANCHO y bajo con laderas casi rectas y la cumbre redondeada."""
    xs = np.linspace(-1, 1, 121)
    e = 0.11                                                                   # redondez de la cumbre
    f = np.sqrt(xs ** 2 + e ** 2) - e
    ys = h * (1 - f / (math.sqrt(1 + e ** 2) - e))
    pts = [(cx + w / 2 * a, base + b) for a, b in zip(xs, ys)]
    poly(pts, fc, 0.5)
    if nieve:
        sn = [(x, y) for x, y in pts if y > base + h * 0.78]
        xl, xr, y0 = sn[0][0], sn[-1][0], base + h * 0.78
        tt = np.linspace(0, 1, 40)
        ond = [(xr - (xr - xl) * q, y0 - 2.1 * abs(math.sin(math.pi * 2.5 * q))) for q in tt]
        poly(sn + ond, "#F7F0DF", 0.55)

def arbol(x, s=1.0):
    ax.add_patch(Rectangle((x - 0.9 * s, GY + 3), 1.8 * s, 6 * s, fc="#8A5A3C", ec=LINE, lw=2, zorder=1.0))
    cx, cy, r = x, GY + 12.5 * s, 5.4 * s
    ang = np.linspace(0, 2 * math.pi, 400)
    rr = r * (1 + 0.075 * np.abs(np.cos(4 * ang)))                       # 8 bultos redondos con una punta entre cada uno
    poly(list(zip(cx + rr * np.cos(ang), cy + rr * np.sin(ang))), "#58B06A", 1.1)
    ax.add_patch(Polygon([(cx + rr[i] * 0.92 * np.cos(a), cy + rr[i] * 0.92 * np.sin(a)) for i, a in enumerate(ang) if math.pi / 2 < a < 3 * math.pi / 2]
                         + [(cx, cy - r * 0.85)], fc="#8CC88A", ec="none", zorder=1.15))

def recipiente(x, y, w, h, z=5, agua=0.6, nuevo=True, puntos=False, solido=False, liquido=None, cuerpo=None):
    """Estanque o filtro con tapa redondeada. Con agua de borde ondulado (`liquido` cambia su color, p. ej. blanco para
    el cloro) o, con solido=True, relleno opaco sin agua. Lo que ya existe va todo en gris."""
    ec = LINE if nuevo else OLD_E
    fc = (SOLIDO if nuevo else OLD_W) if solido else (LIGHT if nuevo else OLD_F)
    if cuerpo and not solido and nuevo:
        fc = cuerpo                                      # cuerpo de otro tono para que el líquido se destaque
    r = w / 2
    top = [(x + r + r * math.cos(a), y + h - r + r * 0.75 * math.sin(a)) for a in np.linspace(0, math.pi, 40)]
    poly([(x, y), (x + w, y), (x + w, y + h - r)] + top + [(x, y + h - r)], fc, z, ec=ec)
    if agua and not solido:
        ya = y + (h - r * 0.3) * agua
        wp = onda(x + 0.6, x + w - 0.6, ya)
        poly(wp + [(x + w - 0.6, y + 0.6), (x + 0.6, y + 0.6)], liquido or (WATER if nuevo else OLD_W), z + 0.1, ec="none", lw=0)
        ax.plot([p[0] for p in wp], [p[1] for p in wp], color=ec, lw=1.4, zorder=z + 0.15)
    if puntos:
        random.seed(int(x * 10))
        for _ in range(int(w * h / 6)):
            ax.add_patch(Circle((random.uniform(x + 0.9, x + w - 0.9), random.uniform(y + 1.2, y + (h - r) * 0.95)), 0.2, fc="white", ec="none", zorder=z + 0.2))

def radier(x, w, nuevo=True, z=3.9):
    """Losa de hormigón bajo un estanque."""
    fc, ec = col(nuevo)
    ax.add_patch(Rectangle((x, GY), w, RAD, fc=fc, ec=ec, lw=LW, zorder=z))

def manguera(pts, ec, z=5.1, lw=1.0):
    """Manguerita muy fina y flexible: línea delgada con las esquinas redondeadas (suavizado de Chaikin)."""
    q = [tuple(p) for p in pts]
    for _ in range(3):
        n = [q[0]]
        for (x0, y0), (x1, y1) in zip(q[:-1], q[1:]):
            n += [(0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1), (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)]
        q = n + [q[-1]]
    ax.plot([p[0] for p in q], [p[1] for p in q], color=ec, lw=lw, solid_capstyle="round", zorder=z)

def pildora(x, y, w, h, nuevo=True, z=5):
    """Hidroneumático (tanque de membrana vertical) como el del plano: cuerpo redondeado arriba y abajo, cordón de
    soldadura, válvula de aire arriba y TRES PATAS abiertas. Va levantado: su ÚNICA conexión (abajo, al centro) baja hasta
    la línea de agua y se une con una TE; la línea sigue de largo. (y = piso; h = altura total, patas incluidas.)"""
    fc, ec = (SOLIDO, LINE) if nuevo else (OLD_W, OLD_E)
    yb = y + 6.3                                                                     # fondo del cuerpo
    cxm = x + w / 2
    ax.add_patch(Rectangle((cxm - 0.4, y), 0.8, yb - y + 0.4, fc=fc, ec=ec, lw=1.4, zorder=z - 0.2))           # pata de atrás
    for sg in (-1, 1):                                                               # patas delanteras, abiertas
        poly([(cxm + sg * (w * 0.30), yb + 0.4), (cxm + sg * (w * 0.30 + 0.8), yb + 0.4),
              (cxm + sg * (w * 0.46 + 0.9), y), (cxm + sg * (w * 0.46), y)], fc, z - 0.1, ec=ec, lw=1.5)
    ax.add_patch(Rectangle((cxm - 0.5, P + 0.5), 1.0, yb - P - 0.3, fc=fc, ec=ec, lw=1.4, zorder=z + 0.05))   # conexión que baja a la línea
    ax.add_patch(Rectangle((cxm - 0.95, P - 0.95), 1.9, 1.9, fc="white", ec=ec, lw=1.5, zorder=z + 0.1))      # te sobre la línea
    rbox(x, yb, w, h - (yb - y), w / 2, fc, z, ec=ec)
    ax.plot([x + 0.12, x + w - 0.12], [yb + (h - (yb - y)) * 0.58] * 2, color=ec, lw=1.6, zorder=z + 0.1)      # cordón de soldadura
    ax.add_patch(Rectangle((cxm - 0.45, y + h - 0.1), 0.9, 0.75, fc=fc, ec=ec, lw=1.3, zorder=z - 0.1))        # válvula de aire
    ax.add_patch(Rectangle((cxm - 0.7, y + h + 0.55), 1.4, 0.4, fc=ec, ec=ec, lw=1.0, zorder=z - 0.1))
    ax.plot([x + w * 0.24, x + w * 0.24], [yb + 4.4, yb + 6.0], color="white", lw=2.0, solid_capstyle="round", alpha=0.7, zorder=z + 0.1)

AMAR = "#FFE27A"                                      # bombas nuevas: amarillas completas

def capsula(cx, cy, nuevo=True, txt=None, z=6, patas=False):
    """Bomba COMPLETA y AMARILLA (existente: gris), SIN mesa: lo importante es el MOTOR RECTANGULAR (aletas, tapa
    trasera, caja de bornes, patitas) con la potencia escrita; la voluta redonda es solo un cabezal mínimo a la izquierda
    con su brida de descarga hacia arriba."""
    fc, ec = col(nuevo)
    fa = AMAR if nuevo else OLD_F
    vx, vr = cx - 2.7, 1.0                                                                                   # cabezal redondo mínimo
    for fx in (cx - 0.4, cx + 3.0):                                                                          # patitas del motor
        poly([(fx, cy - 1.6), (fx + 0.9, cy - 1.6), (fx + 1.3, cy - 2.15), (fx - 0.4, cy - 2.15)], fa, z - 0.1, ec=ec, lw=1.4)
    ax.add_patch(Rectangle((vx - 0.4, cy + 0.8), 0.8, 1.0, fc=fa, ec=ec, lw=1.4, zorder=z - 0.1))             # descarga
    ax.add_patch(Rectangle((vx - 0.75, cy + 1.7), 1.5, 0.4, fc=fa, ec=ec, lw=1.3, zorder=z))                # brida de descarga
    ax.add_patch(Rectangle((vx - vr - 0.45, cy - 0.65), 0.45, 1.3, fc=fa, ec=ec, lw=1.3, zorder=z))          # brida de succión
    ax.add_patch(Circle((vx, cy), vr, fc=fa, ec=ec, lw=1.8, zorder=z))                                        # cabezal redondo
    ax.add_patch(Rectangle((vx + vr - 0.1, cy - 0.5), 0.9, 1.0, fc=fa, ec=ec, lw=1.3, zorder=z))             # acople
    ax.add_patch(Rectangle((cx + 0.7, cy + 1.5), 1.8, 0.9, fc=fa, ec=ec, lw=1.4, zorder=z - 0.1))            # caja de bornes
    ax.add_patch(Circle((cx + 1.6, cy + 1.95), 0.28, fc="white", ec=ec, lw=1.0, zorder=z + 0.2))
    rbox(cx - 0.8, cy - 1.6, 5.4, 3.2, 0.55, fa, z, ec=ec, lw=2.0)                                           # cuerpo del motor
    for fy in (-1.05, 1.05):
        ax.plot([cx - 0.8, cx + 4.6], [cy + fy] * 2, color=ec, lw=1.2, zorder=z + 0.1)                      # aletas
    rbox(cx + 4.6, cy - 1.4, 1.1, 2.8, 0.45, fa, z, ec=ec, lw=2.0)                                           # tapa trasera
    medida(cx + 1.8, cy, txt, nuevo, fs=7, z=z + 0.3)

def macromedidor(mx, nuevo=True, z=6):
    """Macromedidor chico, como el ícono de medidor de agua: caja redonda de doble anillo con visor «00123» y gota,
    cuellos y bridas a ambos lados sobre la tubería, apoyado en un dado de hormigón (no flota)."""
    fc, ec = col(nuevo)
    R, f = 2.3, 0.74                                                                                         # radio de la caja y escala
    ax.add_patch(Rectangle((mx - 1.9, GY), 3.8, P - R - GY + 0.1, fc=fc, ec=ec, lw=1.8, zorder=z - 0.3))   # dado de apoyo
    for sg in (-1, 1):                                                                                        # cuellos y bridas
        ax.add_patch(Rectangle((mx + sg * (R + 0.1) - 0.35, P - 0.7), 0.7, 1.4, fc=fc, ec=ec, lw=1.4, zorder=z - 0.1))
        rbox(mx + sg * (R + 1.25) - 0.7, P - 1.5, 1.4, 3.0, 0.4, fc, z, ec=ec, lw=1.8)
    ax.add_patch(Circle((mx, P), R, fc=fc, ec=ec, lw=2.2, zorder=z))                                         # caja (anillo exterior)
    ax.add_patch(Circle((mx, P), R - 0.5, fc="white", ec=ec, lw=1.5, zorder=z + 0.1))                        # esfera interior
    ax.plot([mx - 0.6, mx + 0.6], [P + 1.5, P + 1.5], color=ec, lw=1.4, solid_capstyle="round", zorder=z + 0.2)
    rbox(mx - 1.55, P - 0.1, 3.1, 1.35, 0.3, "#F4F6F7", z + 0.2, ec=ec, lw=1.2)                              # visor del contador
    ax.text(mx, P + 0.6, "00123", ha="center", va="center", fontsize=4.9, fontweight="bold", color=ec, zorder=z + 0.3)
    gx, gy = mx, P - 1.15                                                                                     # gota
    poly([(gx, gy + 0.7), (gx - 0.42, gy + 0.08), (gx + 0.42, gy + 0.08)], WATER if nuevo else OLD_W, z + 0.25, ec=ec, lw=0.9)
    ax.add_patch(Circle((gx, gy - 0.03), 0.44, fc=WATER if nuevo else OLD_W, ec=ec, lw=0.9, zorder=z + 0.25))

def casa(x, w, h, z=3, existente=True):
    """Casa de la escuela o de un comité: lo que ya existe va en gris."""
    fc, ec = (OLD_WALL, OLD_E) if existente else (WALL, LINE)
    sombra = "#CDD2D7" if existente else "#CFE9F2"
    ax.add_patch(Rectangle((x, GY), w, h, fc=fc, ec=ec, lw=LW, zorder=z))
    poly([(x - 3, GY + h), (x + w / 2, GY + h + w * 0.54), (x + w + 3, GY + h)], fc, z + 0.1, ec=ec)
    poly([(x + 0.2, GY + h), (x + w / 2, GY + h + w * 0.46), (x + w / 2, GY + h)], sombra, z + 0.15, ec="none", lw=0)
    ax.add_patch(Rectangle((x + 0.2, GY + 0.2), w * 0.13, h - 0.4, fc=sombra, ec="none", zorder=z + 0.05))
    for wx in (x + w * 0.44, x + w * 0.71):
        ax.add_patch(Rectangle((wx, GY + h * 0.41), w * 0.21, h * 0.32, fc="white", ec=ec, lw=2, zorder=z + 0.3))
        ax.plot([wx + w * 0.105] * 2, [GY + h * 0.41, GY + h * 0.73], color=ec, lw=1.6, zorder=z + 0.4)
        ax.plot([wx, wx + w * 0.21], [GY + h * 0.57] * 2, color=ec, lw=1.6, zorder=z + 0.4)

def despiche(xr, y, nuevo=True, z=7.3, dir=1, baja=2.4):
    """Despiche (vaciado) del estanque o retrolavado del filtro: tubito corto que sale de la pared (dir=1 a la
    derecha, -1 a la izquierda), con una llave de paso muy chica (dos triángulos) y que baja un poco.
    xr = pared del equipo, y = altura de la salida."""
    ec, fc = (LINE, PIPE) if nuevo else (OLD_E, OLD_W)
    xs_, ys_ = [xr - 0.3 * dir, xr + 3.0 * dir, xr + 3.0 * dir], [y, y, y - baja]
    ax.plot(xs_, ys_, color=ec, lw=4.4, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(xs_, ys_, color=fc, lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.05)
    vx = xr + 1.4 * dir
    poly([(vx - 0.7, y + 0.6), (vx, y), (vx - 0.7, y - 0.6)], "white", z + 0.2, ec=ec, lw=1.1)
    poly([(vx + 0.7, y + 0.6), (vx, y), (vx + 0.7, y - 0.6)], "white", z + 0.2, ec=ec, lw=1.1)

def caida(xc, ytop, ya, nuevo=True, z=7.6):
    """Agua cayendo desde la boca de la tubería que entra al estanque hasta la superficie (ya): chorro, salpicaduras
    y ondas donde pega en el agua."""
    ec, fc = (LINE, WATER) if nuevo else (OLD_E, OLD_W)
    poly([(xc - 0.5, ytop), (xc + 0.5, ytop), (xc + 0.7, ya), (xc - 0.7, ya)], fc, z, ec=ec, lw=0.9)
    for dx, dy in ((-1.6, 0.9), (1.7, 1.4), (0.2, 2.2)):
        ax.add_patch(Circle((xc + dx, ya + dy), 0.26, fc="white", ec=ec, lw=0.7, zorder=z))
    for ew_, eh_, al in ((4.2, 0.9, 1.0), (7.0, 1.5, 0.6)):
        ax.add_patch(Ellipse((xc, ya), ew_, eh_, fc="none", ec="white", lw=1.2, alpha=al, zorder=z))

def pera(x0, w, y0, h, xs_, ya, nuevo=True, z=7.5, topy=None):
    """Pera (interruptor de nivel): cable que cuelga de un prensacable en la tapa del estanque, con un poco de holgura,
    y una pera flotando en la superficie (mitad sumergida) con una banda y una tapita; ondas alrededor."""
    ec = LINE if nuevo else OLD_E
    amar, azul = ("#FFD43B", "#3B6FD6") if nuevo else (OLD_F, OLD_W)
    negro = LINE if nuevo else OLD_E                                                   # cable en la gama de colores (azul), no negro
    r = w / 2; cx0 = x0 + r
    if topy is None:
        topy = y0 + h - r + 0.75 * r * math.sqrt(max(0.0, 1 - ((xs_ - cx0) / r) ** 2))
    ax.add_patch(Rectangle((xs_ - 0.75, topy - 0.55), 1.5, 1.2, fc=negro, ec=negro, lw=1.0, zorder=z))       # prensacable
    xf, yc = xs_ + 2.0, ya + 0.1                                                                            # centro del cuerpo (medio sumergido)
    ytop = yc + 2.2                                                                                         # arriba del cuerpo
    t = np.linspace(0, 1, 50)
    y_ini, y_fin = topy - 0.5, ytop + 1.3
    cab_x = xs_ + (xf - xs_) * t + 1.3 * np.sin(2 * math.pi * t) * (1 - t)                                 # cable negro con holgura
    cab_y = y_ini + (y_fin - y_ini) * t
    ax.plot(cab_x, cab_y, color=negro, lw=2.0, solid_capstyle="round", zorder=z)
    rbox(xf - 0.55, ytop - 0.1, 1.1, 1.5, 0.3, amar, z + 0.05, ec=ec, lw=1.4)                               # cuello amarillo
    for yy in (ytop + 0.35, ytop + 0.7, ytop + 1.05):
        ax.plot([xf - 0.4, xf + 0.4], [yy, yy], color=ec, lw=0.9, zorder=z + 0.2)                           # estrías del cuello
    rbox(xf - 1.75, yc - 2.2, 3.5, 4.5, 1.0, amar, z + 0.1, ec=ec, lw=1.8)                                  # cuerpo amarillo
    rbox(xf - 1.2, yc - 1.65, 2.4, 3.4, 0.6, azul, z + 0.2, ec=ec, lw=1.2)                                  # placa azul
    rbox(xf - 0.8, yc - 1.15, 1.6, 2.4, 0.4, "none", z + 0.25, ec="white", lw=0.8)
    for ew_, eh_, al in ((4.6, 0.8, 1.0), (7.0, 1.3, 0.55)):
        ax.add_patch(Ellipse((xf, ya), ew_, eh_, fc="none", ec="white", lw=1.1, alpha=al, zorder=z + 0.3))

def tubo_blanco(xs, ys, z, lw):
    ax.plot(xs, ys, color=LINE, lw=lw + 3.0, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(xs, ys, color="white", lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.05)

def gota(cx, cy, s, z):
    """Gota de agua bien formada: punta arriba, panza redonda abajo, con un brillo."""
    u = np.linspace(0, math.pi, 50)
    hw = 0.55 * s * np.sin(u) * (u / math.pi) ** 0.7
    yy = cy + 1.4 * s - 2.4 * s * (1 - np.cos(u)) / 2
    pts = [(cx - w_, y_) for w_, y_ in zip(hw, yy)] + [(cx + w_, y_) for w_, y_ in zip(hw[::-1], yy[::-1])]
    poly(pts, WATER, z, ec=LINE, lw=1.3)
    ax.plot([cx - 0.28 * s, cx - 0.28 * s], [cy - 0.35 * s, cy + 0.15 * s], color="white", lw=1.5, solid_capstyle="round", zorder=z + 0.1)

def grifo(x, z=4.4):
    """Llave de agua de la escuela como la del cuadro, pero BLANCA con contorno azul: cuerpo y pico, vástago con manilla
    y UNA sola gota de agua bien formada cayendo."""
    tubo_blanco([x, x], [GY + 6.0, GY + 10.6], z, 4.6)
    tubo_blanco([x, x + 3.2, x + 3.2], [GY + 10.6, GY + 10.6, GY + 9.0], z, 4.6)                              # pico
    tubo_blanco([x, x], [GY + 10.8, GY + 12.2], z + 0.1, 2.2)                                                 # vástago
    tubo_blanco([x - 1.5, x + 1.5], [GY + 12.4, GY + 12.4], z + 0.1, 1.8)                                     # manilla
    gota(x + 3.2, GY + 5.0, 1.7, z)                                                                            # gota

def escuela(x, w, h, z=3, rh=5.5, nwin=None, bandera=True):
    """Escuela muy simple (existente, toda en gris): un solo piso largo con techo bajo a dos aguas, puerta central con un
    frontón chico y escalón, dos ventanas y un mástil con bandera sobre el techo."""
    fc, ec = OLD_WALL, OLD_E
    som = "#CDD2D7"
    ax.add_patch(Rectangle((x, GY), w, h, fc=fc, ec=ec, lw=LW, zorder=z))                                  # muro
    ax.add_patch(Rectangle((x + 0.2, GY + 0.2), w * 0.1, h - 0.4, fc=som, ec="none", zorder=z + 0.05))     # sombra lateral
    poly([(x - 2.5, GY + h), (x + w / 2, GY + h + rh), (x + w + 2.5, GY + h)], fc, z + 0.1, ec=ec)         # techo
    poly([(x - 2.2, GY + h), (x + w / 2, GY + h + rh), (x + w / 2, GY + h)], som, z + 0.15, ec="none", lw=0)
    ax.add_patch(Rectangle((x + 10.4, GY), 5.6, 0.9, fc=som, ec=ec, lw=1.6, zorder=z + 0.2))              # escalón
    ax.add_patch(Rectangle((x + 11.2, GY + 0.9), 4.0, 8.4, fc="#C9CED3", ec=ec, lw=2.0, zorder=z + 0.25))  # puerta
    ax.add_patch(Circle((x + 14.4, GY + 4.8), 0.4, fc=ec, ec="none", zorder=z + 0.3))
    poly([(x + 10.4, GY + 10.4), (x + 13.2, GY + 13.0), (x + 16.0, GY + 10.4)], "white", z + 0.25, ec=ec, lw=1.8)   # frontón
    _nw = nwin if nwin else max(3, int((w - w * 0.5625 - 2) / 4.9))
    for wx in [x + w * 0.5625 + i_w * 4.9 for i_w in range(_nw)]:                                           # ventanas
        ax.add_patch(Rectangle((wx, GY + 5.8), 3.4, 5.0, fc="white", ec=ec, lw=2.0, zorder=z + 0.3))
        ax.plot([wx + 1.7] * 2, [GY + 5.8, GY + 10.8], color=ec, lw=1.5, zorder=z + 0.4)
        ax.plot([wx, wx + 3.4], [GY + 8.3] * 2, color=ec, lw=1.5, zorder=z + 0.4)
    yp = GY + h + rh                                                                                       # mástil con bandera
    if bandera:
        ax.plot([x + w / 2] * 2, [yp - 0.3, yp + 8.0], color=ec, lw=2.2, zorder=z + 0.05)
        poly([(x + w / 2, yp + 8.0), (x + w / 2 + 5.5, yp + 6.5), (x + w / 2, yp + 5.0)], "#C9CED3", z + 0.05, ec=ec, lw=1.6)

def generador_ext(cx, nv=False, med=None):
    """Grupo electrógeno a la intemperie (existente = gris), al costado de la escuela."""
    fc, ec = col(nv)
    fg = LIME if nv else OLD_F
    ax.add_patch(Rectangle((cx + 0.6, GY), 10.3, 1.0, fc=ec, ec=ec, lw=1.2, zorder=2.5))
    rbox(cx + 1, GY + 1.0, 9.5, 7.2, 1.0, fg, 2.6, ec=ec)
    for k in range(3):
        ax.plot([cx + 2.0, cx + 5.6], [GY + 5.8 + k * 0.85] * 2, color=ec, lw=1.4, zorder=2.8)
    ax.add_patch(Rectangle((cx + 6.6, GY + 5.5), 3.2, 2.2, fc="white", ec=ec, lw=1.4, zorder=2.8))
    ax.plot([cx + 8.6, cx + 8.1, cx + 8.9, cx + 8.4], [GY + 7.3, GY + 6.6, GY + 6.6, GY + 5.8], color=ec, lw=1.2, zorder=2.9)
    for kx in (7.2, 9.3):
        ax.add_patch(Circle((cx + kx, GY + 6.1), 0.28, fc=ec, ec="none", zorder=2.9))
    ax.add_patch(Rectangle((cx + 2.6, GY + 8.1), 1.0, 2.1, fc=ec, ec=ec, lw=1.0, zorder=2.5))
    ax.add_patch(Rectangle((cx + 2.2, GY + 10.0), 1.8, 0.55, fc=ec, ec=ec, lw=1.0, zorder=2.5))
    medida(cx + 5.75, GY + 2.5, med, nv, z=2.8)

def tablero_cara(bx, by, nv=True, z=4):
    """Tablero eléctrico chico (celeste/verde = nuevo)."""
    fc, ec = col(nv)
    ax.add_patch(Rectangle((bx, by), 8.0, 12, fc=LIME if nv else OLD_F, ec=ec, lw=LW, zorder=z))
    for c in range(3):
        ax.add_patch(Circle((bx + 1.9 + c * 2.1, by + 10.2), 0.6, fc="white", ec=ec, lw=1.4, zorder=z + 0.2))
    poly([(bx + 4, by + 8.0), (bx + 2.3, by + 4.4), (bx + 5.7, by + 4.4)], "none", z + 0.2, ec=ec, lw=1.6)
    ax.plot([bx + 4.2, bx + 3.7, bx + 4.5, bx + 4.0], [by + 7.1, by + 6.1, by + 6.1, by + 5.0], color=ec, lw=1.3, zorder=z + 0.3)
    ax.add_patch(Rectangle((bx + 2.6, by + 1.0), 2.8, 1.4, fc="white", ec=ec, lw=1.3, zorder=z + 0.2))

def lluvia_flecha(x, y, l=4.2, z=8):
    """Flecha de lluvia (celeste, hacia abajo y a la derecha) sobre el techo."""
    dx, dy = 0.45, -1.0
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    x1, y1 = x + dx * l, y + dy * l
    ax.plot([x, x1 - dx * 0.8], [y, y1 - dy * 0.8], color=LINE, lw=4.6, solid_capstyle="round", zorder=z)
    ax.plot([x, x1 - dx * 0.8], [y, y1 - dy * 0.8], color=PIPE, lw=2.4, solid_capstyle="round", zorder=z + 0.05)
    px_, py_ = -dy, dx
    poly([(x1, y1), (x1 - dx * 1.5 + px_ * 0.9, y1 - dy * 1.5 + py_ * 0.9), (x1 - dx * 1.5 - px_ * 0.9, y1 - dy * 1.5 - py_ * 0.9)], PIPE, z + 0.1, ec=LINE, lw=1.3)

def cloracion_std(cx, ly, base, nv=True):
    """La cloración ESTÁNDAR de los proyectos: estanque de cloro con líquido blanco + bomba dosificadora MINI colgada de la pared
    (placa con tornillos, cuerpo, pantalla, botones, cabezal), con su manguerita de succión y la de inyección en una unión sobre la
    línea (altura ly). base = piso del estanque. Devuelve la x de la unión."""
    fc, ec = col(nv)
    recipiente(cx + 0.8, base, 5.2, 7.8, z=5, nuevo=nv, agua=0.82, liquido="white", cuerpo="#BCD9EF")
    s_ = 0.5
    px0, py0 = cx + 8.4, ly + 4.5
    ax.add_patch(Rectangle((px0, py0), 6.4 * s_, 10.4 * s_, fc="#F4F6F7" if nv else OLD_WALL, ec=ec, lw=1.3, zorder=5.0))
    for sx_, sy_ in ((0.8, 0.8), (5.6, 0.8), (0.8, 9.6), (5.6, 9.6)):
        ax.add_patch(Circle((px0 + sx_ * s_, py0 + sy_ * s_), 0.18, fc=ec, ec="none", zorder=5.1))
    rbox(px0 + 1.0 * s_, py0 + 3.6 * s_, 4.4 * s_, 6.0 * s_, 0.5, LIME if nv else OLD_F, 5.4, ec=ec, lw=1.6)
    ax.add_patch(Rectangle((px0 + 1.8 * s_, py0 + 7.3 * s_), 2.8 * s_, 1.5 * s_, fc="white", ec=ec, lw=1.0, zorder=5.5))
    for bxx in (2.2, 4.2):
        ax.add_patch(Circle((px0 + bxx * s_, py0 + 5.3 * s_), 0.28, fc="white", ec=ec, lw=1.0, zorder=5.5))
    rbox(px0 + 1.7 * s_, py0 + 1.3 * s_, 3.0 * s_, 2.3 * s_, 0.35, "white", 5.4, ec=ec, lw=1.5)
    ys_ = py0 + 2.45 * s_
    manguera([(cx + 3.4, base + 7.4), (cx + 3.4, ys_), (px0 + 1.7 * s_, ys_)], ec, z=5.6)
    xi = px0 + 6.4 * s_ + 1.2
    manguera([(px0 + 4.7 * s_, ys_), (xi, ys_), (xi, ly + 1.9)], ec, z=5.6)
    ax.add_patch(Rectangle((xi - 0.25, ly + 0.8), 0.5, 1.2, fc="white", ec=ec, lw=1.1, zorder=5.2))
    ax.add_patch(Rectangle((xi - 0.7, ly - 1.0), 1.4, 2.0, fc="white", ec=ec, lw=1.3, zorder=5.2))
    return xi

def ventana(x, y, w, h, z=2.6):
    """Ventana del muro de la escuela (existente: gris): vidrio claro con cruz."""
    ax.add_patch(Rectangle((x, y), w, h, fc="#EAF1F7", ec=OLD_E, lw=2.0, zorder=z))
    ax.plot([x + w / 2] * 2, [y, y + h], color=OLD_E, lw=1.5, zorder=z + 0.1)
    ax.plot([x, x + w], [y + h * 0.55] * 2, color=OLD_E, lw=1.5, zorder=z + 0.1)

def ventana_alumnos(x, y, w, h, z=2.6):
    """Ventana GRANDE de la sala: se ven alumnos sentados en sus pupitres estudiando, y la pizarra al fondo."""
    ax.add_patch(Rectangle((x, y), w, h, fc="#EAF1F7", ec=OLD_E, lw=2.6, zorder=z))                       # vidrio
    ax.add_patch(Rectangle((x + w * 0.62, y + h * 0.50), w * 0.34, h * 0.36, fc="#2E4A6B", ec=OLD_E, lw=1.4, zorder=z + 0.1))   # pizarra
    for k_ in range(3):
        ax.plot([x + w * 0.66, x + w * (0.86 - 0.05 * k_)], [y + h * (0.78 - 0.1 * k_)] * 2, color="white", lw=1.0, alpha=0.8, zorder=z + 0.2)
    ax.plot([x, x + w], [y + 1.4, y + 1.4], color=OLD_E, lw=1.2, alpha=0.6, zorder=z + 0.1)                 # piso de la sala
    pelo = "#3A3F55"
    piel = "#F2C7A5"
    for i_, (cam, col_) in enumerate((("#3B6FD6", 0), ("#FFD43B", 1), ("#7ACB8A", 2))):
        dx = x + 2.2 + i_ * (w * 0.2)
        ax.add_patch(Rectangle((dx - 0.2, y + 1.5), 0.5, 3.2, fc=OLD_E, ec="none", zorder=z + 0.2))          # patas del pupitre
        ax.add_patch(Rectangle((dx + 3.8, y + 1.5), 0.5, 3.2, fc=OLD_E, ec="none", zorder=z + 0.2))
        rbox(dx + 0.6, y + 5.0, 2.8, 3.6, 0.8, cam, z + 0.3, ec=OLD_E, lw=1.2)                                # torso
        ax.add_patch(Circle((dx + 2.0, y + 9.6), 1.35, fc=piel, ec=OLD_E, lw=1.2, zorder=z + 0.35))           # cabeza
        ax.add_patch(Polygon([(dx + 0.65, y + 9.8), (dx + 2.0, y + 11.2), (dx + 3.35, y + 9.8), (dx + 2.0, y + 10.1)], fc=pelo, ec="none", zorder=z + 0.4))   # pelo
        ax.add_patch(Rectangle((dx - 0.4, y + 4.3), 4.8, 0.7, fc="#B88A5A", ec=OLD_E, lw=1.0, zorder=z + 0.5))   # tapa del pupitre
        ax.add_patch(Rectangle((dx + 0.8, y + 5.0), 2.4, 0.5, fc="white", ec=OLD_E, lw=0.8, zorder=z + 0.55))    # cuaderno

def litros(txt):
    d_ = re.sub(r"[^\d]", "", str(txt or ""))
    return int(d_) if d_ else None

def tanque_vertical(x0, y0, w, bh, nv=True, medida=None, z=7, nivel=0.9):
    """Estanque vertical acanalado (forma de la imagen de Diego), TRANSPARENTE: se ve el nivel del agua. Cuerpo cilíndrico con 4 anillos
    en relieve, hombro abombado, cuello con tapa y asa; base PLANA apoyada en el radier. Devuelve (cx, cima, y del agua)."""
    from matplotlib.colors import to_rgba
    fc, ec = (to_rgba(LIGHT, 0.38), LINE) if nv else (to_rgba(OLD_F, 0.5), OLD_E)
    cx = x0 + w / 2
    ribs = [y0 + bh * f for f in (0.2, 0.4, 0.6, 0.8)]
    dome = 0.26 * bh

    def rad_f(y):
        return (w / 2) * (1 + 0.035 * sum(math.exp(-((y - yr) / 0.5) ** 2) for yr in ribs))

    ys = np.linspace(y0, y0 + bh, 240)
    izq = [(cx - rad_f(y_), y_) for y_ in ys]
    der = [(cx + rad_f(y_), y_) for y_ in ys[::-1]]
    cima = [(cx + (w / 2) * u_ * 0.97, y0 + bh + dome * (1 - u_ ** 2) ** 0.6) for u_ in np.linspace(-1, 1, 80)]
    ya = y0 + bh * nivel
    ys_w = np.linspace(y0, ya, 60)
    agua = [(cx - rad_f(y_), y_) for y_ in ys_w] + onda(cx - rad_f(ya) + 0.3, cx + rad_f(ya) - 0.3, ya, amp=0.5, n=3) \
        + [(cx + rad_f(y_), y_) for y_ in ys_w[::-1]]
    poly(agua, WATER if nv else OLD_W, z - 0.02, ec="none", lw=0)                                          # agua adentro
    poly(izq + cima + der, fc, z, ec=ec, lw=2.4)                                                            # cuerpo transparente
    for yr in ribs + [y0 + bh]:
        r_ = (w / 2) * (1.02 if yr != y0 + bh else 0.97)
        xs_ = np.linspace(cx - r_, cx + r_, 60)
        ax.plot(xs_, [yr - 0.9 * math.sqrt(max(0.0, 1 - ((a_ - cx) / r_) ** 2)) for a_ in xs_], color=ec, lw=1.6, zorder=z + 0.1)
    ytop = y0 + bh + dome
    ax.add_patch(Rectangle((cx - 2.5, ytop - 0.4), 5.0, 1.6, fc=fc, ec=ec, lw=1.8, zorder=z + 0.1))        # cuello (la «puerta» de arriba)
    ax.add_patch(Ellipse((cx, ytop + 1.2), 5.0, 1.0, fc=to_rgba(WATER if nv else OLD_W, 0.9), ec=ec, lw=1.6, zorder=z + 0.2))   # boca ABIERTA, sin tapa
    if medida:
        ax.text(cx, y0 + bh * 0.3, medida, ha="center", va="center", fontsize=10, fontweight="bold",
                color=LINE if nv else OLD_T, zorder=z + 0.4)
    return cx, ytop + 1.2, ya

def sup_hombro(x0, y0, w, bh, xq):
    """Altura de la superficie del hombro del estanque vertical en la posición xq."""
    cx = x0 + w / 2
    u_ = max(-0.97, min(0.97, (xq - cx) / ((w / 2) * 0.97)))
    return y0 + bh + 0.26 * bh * (1 - u_ ** 2) ** 0.6

def bomba_per(cx, nv=True, med=None):
    """Electrobomba periférica (amarilla) con su controlador electrónico de presión sobre la línea. Devuelve (pcx, x del controlador)."""
    fc, ec = col(nv)
    pcx = cx + 6.6
    fa_ = AMAR if nv else OLD_F
    ax.add_patch(Rectangle((pcx - 0.7, GY), 5.2, P - 2.15 - GY + 0.1, fc=fa_, ec=ec, lw=1.6, zorder=5.8))
    capsula(pcx, P, nuevo=nv, txt=med)
    xc_ = pcx + 8.2
    ax.add_patch(Rectangle((xc_ - 0.95, P - 0.95), 1.9, 1.9, fc="white", ec=ec, lw=1.5, zorder=5.2))
    ax.add_patch(Rectangle((xc_ - 0.4, P + 0.9), 0.8, 0.8, fc=fc, ec=ec, lw=1.2, zorder=5.1))
    rbox(xc_ - 1.4, P + 1.6, 2.8, 3.6, 0.6, "white" if nv else OLD_F, 5.4, ec=ec, lw=1.7)
    ax.add_patch(Rectangle((xc_ - 0.9, P + 3.6), 1.8, 1.0, fc=fc, ec=ec, lw=1.1, zorder=5.5))
    for dx_ in (-0.45, 0.45):
        ax.add_patch(Circle((xc_ + dx_, P + 2.5), 0.28, fc=fc, ec=ec, lw=0.9, zorder=5.5))
    return pcx, xc_

def filtro_manual(hx, ly, simbolo="gota", med=None, base=None, nv=True):
    """Filtro como en el manual/foto: vaso acanalado colgado de la repisa y CABEZAL ancho con puertos laterales y símbolo, sobre la línea ly."""
    fc, ec = col(nv)
    fcc_ = PIPE if nv else OLD_W
    yb_ = base if base is not None else GY + 3.0
    rbox(hx - 2.1, yb_, 4.2, ly - 1.3 - yb_, 1.2, SOLIDO if nv else OLD_W, 5, ec=ec, lw=2.0)
    for k_ in range(1, 5):
        ax.plot([hx - 2.1 + k_ * 0.84] * 2, [yb_ + 0.9, ly - 2.2], color=ec, lw=1.0, alpha=0.75, zorder=5.2)
    rbox(hx - 2.6, ly - 1.4, 5.2, 2.8, 0.7, fcc_, 5.4, ec=ec, lw=2.0)
    for nx_ in (hx - 3.4, hx + 2.6):
        ax.add_patch(Rectangle((nx_, ly - 0.55), 0.8, 1.1, fc=fcc_, ec=ec, lw=1.3, zorder=5.3))
    ax.add_patch(Circle((hx, ly), 0.95, fc="white", ec=ec, lw=1.3, zorder=5.6))
    if simbolo == "puntos":
        for dx_, dy_ in ((-0.35, 0.2), (0.35, 0.2), (0.0, -0.35)):
            ax.add_patch(Circle((hx + dx_, ly + dy_), 0.17, fc=ec, ec="none", zorder=5.7))
    else:
        gota(hx, ly - 0.45, 0.45, 5.7)
    if med:
        ax.text(hx, (yb_ + ly) / 2 - 0.4, med, rotation=90, ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=ec if nv else OLD_T, zorder=5.8, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

def nube_lluvia(cx, yb, w, techo_y, x_ini, x_fin):
    """Nube (como las del cuadro) y MUCHAS gotas que caen hasta el techo. techo_y(x) = altura del techo en x."""
    cre = "#F7F0DF"
    circ = [(-0.34, 0.0, 0.08), (-0.2, 0.6, 0.10), (0.0, 1.0, 0.12), (0.19, 0.6, 0.10), (0.34, 0.0, 0.08)]
    for ox, oy, rr in circ:
        ax.add_patch(Circle((cx + ox * w, yb + 1.6 + oy), rr * w + 0.45, fc=LINE, ec=LINE, lw=1.0, zorder=8))
    ax.add_patch(Rectangle((cx - 0.38 * w, yb + 0.35), 0.78 * w, 1.6, fc=LINE, ec=LINE, lw=1.0, zorder=8))
    for ox, oy, rr in circ:
        ax.add_patch(Circle((cx + ox * w, yb + 1.6 + oy), rr * w, fc=cre, ec="none", zorder=8.1))
    ax.add_patch(Rectangle((cx - 0.38 * w, yb + 0.8), 0.78 * w, 1.2, fc=cre, ec="none", zorder=8.1))
    fila = 0
    y = yb - 0.8
    while y > yb - 16:
        xg = x_ini + (1.5 if fila % 2 else 0.0)
        while xg < x_fin:
            if y > techo_y(xg) + 1.4 and abs(xg - cx) < 0.46 * w:
                gota(xg, y, 0.9, 8.2)
            xg += 2.6
        y -= 2.2
        fila += 1

def llave(x, z=4.3):
    ax.plot([x, x, x + 2.7, x + 2.7], [GY + 6.5, GY + 8.6, GY + 8.6, GY + 7.2], color=LINE, lw=3, zorder=z)
    poly([(x + 2.7, GY + 6.3), (x + 2.0, GY + 4.9), (x + 3.4, GY + 4.9)], WATER, z, lw=1.2)
    ax.add_patch(Circle((x + 2.7, GY + 4.6), 0.75, fc=WATER, ec=LINE, lw=1.2, zorder=z))

etapas = []
def etapa(nombre, dato=""):
    etapas.append((nombre, dato)); return len(etapas)

# ---------- 2. fondo ----------
random.seed(11)
cols = ["#C5DCC1", "#A6C8A1", "#88B287", "#A6C8A1", "#C5DCC1"]
hx, i = 4, 0
while hx < W and fuente.get("tipo") != "lluvia":
    w = random.uniform(50, 68)
    cerro(hx + w / 2, GY + 10, w, random.uniform(11, 17), cols[i % len(cols)])
    hx += w * 0.72; i += 1
if fuente.get("tipo") != "lluvia":
    cerro(W * 0.48, GY + 10, 84, 30, "#9CBB99", nieve=True)
ax.add_patch(Rectangle((0, GY), W, 12, fc=GRASS, ec="none", zorder=0.8))
ax.plot([0, W], [GY, GY], color=LINE, lw=LW, zorder=0.9)
BORDE = lambda x: 18.5 + 1.1 * math.sin(x * 0.21) + 0.6 * math.sin(x * 0.77)
poly([(0, GY), (0, 19)] + [(x, BORDE(x)) for x in np.linspace(0, W, 400)] + [(W, 19), (W, GY)], GROUND, 0.7, ec="none", lw=0)
xs_n = np.linspace(0, W, 400)
topn = [(x, WT + 0.5 * math.sin(x * 0.16)) for x in xs_n]                                # superficie de la napa
poly(topn + [(x, BORDE(x)) for x in xs_n[::-1]], "#8FCBF2", 0.72, ec="none", lw=0)       # agua subterránea (napa)
ax.plot([q[0] for q in topn], [q[1] for q in topn], color=LINE, lw=1.6, zorder=0.73)
for dy_, ph_ in ((3.2, 1.0), (6.0, 2.6)):                                                # ondas dentro de la napa
    ax.plot(xs_n, WT - dy_ + 0.5 * np.sin(xs_n * 0.2 + ph_), color="white", lw=1.2, alpha=0.5, zorder=0.735)
random.seed(4)
for _ in range(int(W * 1.15)):                                                           # piedras: en la parte GRIS del terreno
    ax.add_patch(Ellipse((random.uniform(1, W - 1), random.uniform(WT + 1.5, GY - 1.1)), random.uniform(0.7, 1.6),
                         random.uniform(0.5, 1.0), angle=random.uniform(0, 180), fc="#EBDCC4", ec="none", zorder=0.75))
if fuente.get("tipo") != "lluvia":
    arbol(6, 0.95)                                  # (con lluvia, a la izquierda va el generador de la escuela)
if fuente.get("tipo") != "lluvia":
    arbol(W - 3.5, 0.9)

# ---------- 3. fuente ----------
ft = fuente.get("tipo", "pozo")
XF = X0 - 4 if tramos and next(iter(tramos[0])) == "estanque" else X0
if ft == "pozo":
    px = X0 + 1.5 if POZO_DENTRO else 15
    bn = fuente.get("bomba_nueva", True)
    # pozo profundo: tubo de ademe (pared azul) hasta DENTRO de la napa; adentro, el agua de la napa y la bomba sumergida
    ax.add_patch(Rectangle((px, 20.0), 7, GY - 20.0, fc="#EAF1F7", ec=LINE, lw=2.2, zorder=1.5))                 # ademe
    ax.add_patch(Rectangle((px + 0.15, 20.15), 6.7, WT - 20.15, fc="#8FCBF2", ec="none", zorder=1.52))           # agua dentro del pozo
    ax.plot([px + 0.15, px + 6.85], [WT, WT], color=LINE, lw=1.4, zorder=1.55)
    ax.add_patch(Rectangle((px - 0.7, GY - 0.2), 8.4, 0.8, fc="#C9CED3", ec=LINE, lw=1.6, zorder=1.6))           # brocal
    if POZO_DENTRO:
        pipe([px + 3.5, px + 3.5], [28.0, P], z=3, lw=5.5)                    # el pozo está dentro de la caseta: la columna sube directo a la línea
    else:
        pipe([px + 3.5, px + 3.5, XF, XF], [28.0, YU, YU, P], z=3, lw=5.5)        # va ENTERRADA y solo sale (al pie de la caseta) para conectarse
        flecha((px + 7.5 + XF) / 2, YU, 0, s=0.5)
    # bomba sumergible de pozo profundo (amarilla si es nueva): cable, motor abajo, rejilla de succión, cuerpo de etapas
    # con anillos, cabezal de descarga y la columna que sube
    fa = AMAR if bn else OLD_F
    eb = LINE if bn else OLD_E
    ax.plot([px + 5.9, px + 5.9], [22.0, GY - 0.2], color=LINE if bn else OLD_E, lw=1.4, zorder=3.1)                       # cable de alimentación
    ax.plot([px + 5.1, px + 5.9], [22.0, 22.0], color=LINE if bn else OLD_E, lw=1.4, zorder=3.1)
    rbox(px + 1.9, 20.6, 3.2, 3.5, 0.7, fa, 3.3, ec=eb, lw=1.6)                                                  # motor sumergible
    ax.add_patch(Rectangle((px + 2.1, 24.1), 2.8, 0.65, fc=eb, ec="none", zorder=3.35))                          # rejilla de succión
    for yy in (24.2, 24.5):
        ax.plot([px + 2.2, px + 4.8], [yy, yy], color="white", lw=0.7, zorder=3.4)
    rbox(px + 1.9, 24.8, 3.2, 2.9, 0.5, fa, 3.3, ec=eb, lw=1.6)                                                  # cuerpo de la bomba (etapas)
    for yy in (25.8, 26.75):
        ax.plot([px + 1.9, px + 5.1], [yy, yy], color=eb, lw=1.1, zorder=3.4)                                   # anillos de las etapas
    rbox(px + 2.5, 27.6, 2.0, 0.8, 0.3, fa, 3.3, ec=eb, lw=1.4)                                                  # cabezal de descarga
    medida(px + 3.5, 22.35, fuente.get("medida"), bn, rot=90, fs=6.0, z=3.5)
    num(px + 3.5, GY + (9.5 if POZO_DENTRO else 3.4), etapa("Pozo", fuente.get("dato", "fuente de agua")))      # sobre el brocal (la tubería ya no sube por el pozo)
elif ft == "lluvia":
    # SCALL: LA ESCUELA en 3D (muro frontal + superficie del techo + costado) es el telón grande, y TODO el sistema está en perspectiva
    # al frente de su pared. El agua cae por la altura: de la canaleta directo al primer estanque (sin bajada larga). Es la misma
    # escuela la que capta y la que recibe el agua (la red vuelve enterrada a su llave, en este mismo muro).
    ge = fuente.get("generador_exterior")
    xa, xb = 18.0, W - 20.0                                   # muro: de xa a xb
    wh = 31.0                                                 # alto del muro
    yw = GY + wh                                              # altura del alero
    ddx, ddy = 8.0, 9.0                                       # profundidad (oblicua): el techo se ve como superficie
    _p1 = tramos[1].get("planta") if len(tramos) > 1 else None                          # sin planta: el estanque entrega directo a la llave
    _x1 = X0 + anchos[0] + GAP
    if _p1 is not None:
        _tw2 = 21 * ((litros(_p1["estanque_potable"].get("medida")) or 5000) / 5000) ** (1 / 3)
        xr_ = _x1 + _p1.get("ancho", 88) * _p1.get("escala", 0.62) + 7 + _tw2             # borde derecho del estanque de agua potable
    else:
        xr_ = X0 + anchos[0] + 4                                                          # borde derecho del único estanque
    SCH = dict(xa=xa, xb=xb, yw=yw, xf=xr_ + 8.0, xr=xr_)
    ax.add_patch(Rectangle((xa - 3, GY), xb - xa + 3 + ddx, 3.0, fc="#D7D2C8", ec="none", zorder=1.6))                 # piso de adelante
    ax.add_patch(Rectangle((xa, GY + 3.0), xb - xa, wh - 3.0, fc=OLD_WALL, ec=OLD_E, lw=LW, zorder=2))                  # muro frontal
    for sxx in np.arange(xa + 12, xb - 2, 12):
        ax.plot([sxx, sxx], [GY + 3.6, yw - 0.8], color="#C2C8CD", lw=1.2, zorder=2.05)                               # tablas del muro
    poly([(xb, GY + 3.0), (xb + ddx, GY + 3.0 + 6), (xb + ddx, yw + ddy), (xb, yw)], "#A8B0B8", 2.0, ec=OLD_E, lw=LW)    # costado
    poly([(xa - 3, yw), (xb + 3, yw), (xb + 3 + ddx, yw + ddy), (xa - 3 + ddx, yw + ddy)], OLD_F, 2.1, ec=OLD_E, lw=LW)   # SUPERFICIE DEL TECHO
    for k_ in range(1, 6):                                                                                              # ondas de la plancha de techo
        f_ = k_ / 6
        ax.plot([xa - 3 + ddx * f_, xb + 3 + ddx * f_], [yw + ddy * f_] * 2, color=OLD_E, lw=1.3, alpha=0.55, zorder=2.2)
    ax.add_patch(Rectangle((xa - 3, yw - 0.7), xb - xa + 6, 0.7, fc=OLD_E, ec="none", zorder=2.3))                        # frente del alero
    num(xa + 90, yw + ddy * 0.5, etapa("Techo y canaletas", fuente.get("dato", "captación de agua lluvia")))
    if ge is not None:                       # generador (existente) afuera, al costado de la escuela; tablero (nuevo) colgado del muro
        gcx = 3.0
        generador_ext(gcx, ge.get("nuevo", False), ge.get("medida"))
        by_ = GY + 7.5
        tablero_cara(xa + 1.6, by_, True, z=4)
        manguera([(gcx + 10.7, GY + 4.8), (xa - 2.2, GY + 4.8), (xa - 0.8, by_ + 1.7), (xa + 1.3, by_ + 1.7)], LINE, z=4.2, lw=1.7)
        ax.add_patch(Rectangle((gcx + 10.5, GY + 4.3), 0.7, 1.0, fc=OLD_E, ec=OLD_E, lw=1.0, zorder=2.7))
        rbox(xa + 0.5, by_ + 1.2, 1.1, 1.0, 0.25, "white", 4.3, ec=LINE, lw=1.2)
    # canaleta a lo largo del alero, con soportes, que termina sobre el primer estanque
    _t0 = tramos[0]["estanque"]
    _tw0 = 21 * ((litros(_t0.get("medida")) or 5000) / 5000) ** (1 / 3)
    xin0 = X0 + 2 + _tw0 / 2
    SCH["xin"] = xin0
    pipe([xin0, xb + 2.5], [yw - 0.6, yw - 0.6], z=4, lw=4.4)
    ax.plot([xin0, xb + 2.5], [yw + 0.55, yw + 0.55], color=LINE, lw=2.2, zorder=4.2)
    for bxx in np.arange(xin0 + 6, xb, 10):
        ax.plot([bxx] * 2, [yw - 1.4, yw - 0.1], color=LINE, lw=2.0, zorder=4.1)
    flecha((xin0 + xb) / 2, yw - 0.6, 180, z=4.4)
    # nube con MUCHAS gotas sobre el techo (en el hueco libre entre la leyenda y los logos)
    nubes = fuente.get("nubes") or [118.0]                                                          # centros de las nubes (varias = escuela más angosta)
    nw = float(fuente.get("nube_ancho", 46))
    for cxc in nubes:
        nube_lluvia(cxc, yw + ddy + 8.2, nw, lambda xq: yw + 1.0, xa + (2 if fuente.get("nubes") else 30), xb - 6)                  # nubes más arriba
    for yy_ in np.arange(yw + 2.6, yw + ddy - 0.6, 2.2):                                           # lluvia en TODO el techo
        for xx_ in np.arange(xa - 1 + ddx * (yy_ - yw) / ddy, xb + 2 + ddx * (yy_ - yw) / ddy, 3.0):
            if all(abs(xx_ - cxc) >= 0.46 * nw for cxc in nubes):
                gota(xx_, yy_, 0.85, 8.2)
    # ventanas de la escuela (y la ventana GRANDE de la sala con alumnos estudiando, a la derecha)
    ventana(xa + 11.8, GY + 9.0, 8.0, 10.0)
    xfv = SCH["xf"]
    ventana(xfv + 9.0, GY + 10.0, 8.0, 11.0)
    ventana_alumnos(xfv + 21.0, GY + 7.0, 24.0, 17.0)
    ventana(xfv + 49.0, GY + 10.0, 7.0, 11.0)
elif ft == "vertiente":
    cerro(16, GY, 30, 14, "#8BBF8B")
    fcv, ecv = col(fuente.get("nuevo", False))
    rbox(12, GY + 1, 9, 6, 1, fcv, 2, ec=ecv)
    pipe([21, XF], [P, P])
    flecha((21 + XF) / 2, P, 0)
    num(16.5, GY + 11, etapa("Vertiente", fuente.get("dato", "fuente de agua")))
else:
    pipe([2, XF], [P, P])
    ax.add_patch(Circle((16, P), 3.4, fc=OLD_F, ec=OLD_E, lw=LW, zorder=6))
    ax.text(16, P, "APR", ha="center", va="center", fontsize=9, fontweight="bold", color=OLD_T, zorder=6.2)
    num(16, P + 7.5, etapa("Red del APR", fuente.get("dato", "conexión existente")))

# ---------- 4. tramos ----------
def fin(i, xfin):
    """La tubería que llega a un estanque suelto sube 4 unidades antes (entra por arriba)."""
    return xfin - 4 if i + 1 < len(tramos) and next(iter(tramos[i + 1])) == "estanque" else xfin

x = X0
prev_buried = False                      # la red de salida ya viene enterrada hasta el macromedidor
for idx, t in enumerate(tramos):
    k, d = next(iter(t.items()))
    w = anchos[idx]
    if k == "caseta":
        equipos = [equipo(e) for e in d.get("equipos", [])]
        ext = d.get("estanque_exterior")
        presost = any(n == "presostato" for n, _ in equipos)
        ch = d.get("alto", 27)                  # alto del muro (una caseta chica puede ser más baja)
        fcw, ecw = ((("#D3ECFC", LINE)) if not d.get("existente", True) else (OLD_WALL, OLD_E))   # caseta nueva: celeste claro
        wc = w - EXT_W if ext else w       # ancho de la caseta (una sola, con todo adentro menos el estanque exterior)
        xw = x + wc                        # pared derecha de la caseta
        if d.get("vidrio"):                # caseta de VIDRIO transparente: se ve todo lo de adentro
            ax.add_patch(Rectangle((x, GY), wc, ch, fc=(1, 1, 1, 0.13), ec=LINE, lw=LW, zorder=2))
            for mx_ in np.arange(x + 15, x + wc - 3, 15):
                ax.plot([mx_] * 2, [GY, GY + ch], color=LINE, lw=1.5, alpha=0.75, zorder=2.2)
            for gx_ in np.arange(x + 4, x + wc - 8, 15):
                ax.plot([gx_, gx_ + 5], [GY + ch - 2, GY + ch - 9], color="white", lw=1.7, alpha=0.4, zorder=2.2)
            ax.add_patch(Rectangle((x - 2.5, GY + ch), wc + 5, 2.6, fc=(1, 1, 1, 0.55), ec=LINE, lw=LW, zorder=2.1))
        else:
            ax.add_patch(Rectangle((x, GY), wc, ch, fc=fcw, ec=ecw, lw=LW, zorder=2))
            ax.add_patch(Rectangle((x - 2.5, GY + ch), wc + 5, 2.6, fc="white", ec=ecw, lw=LW, zorder=2.1))
        _pz = POZO_DENTRO and idx == 0
        cx = x + 2.5 + (POZO_W if _pz else 0)
        abierta = (px + 3.5) if _pz else x                        # desde dónde corre la tubería principal sin cortar
        filtro_x = b0 = b1 = hid_cx = None   # posiciones para las tuberías del estanque exterior
        lifts = []                         # (xa, xb, yh): la línea sube a la altura yh para entrar/salir por el cabezal de un filtro
        ext_src = None                     # x desde donde la línea sube hacia el estanque exterior
        b_top = P + 10                     # altura a la que la succión del estanque llega al colector de las bombas
        _et = d.get("etapa")               # una sola etapa para toda la caseta: se reserva el número ANTES de dibujar los equipos
        _n_et = etapa(_et["nombre"], _et.get("dato", "")) if _et else None
        gen_der = None                     # borde derecho del generador (para su cable al tablero)
        orden = []                         # el estanque exterior va justo antes de las bombas (o al final)
        for nombre, e in equipos:
            if ext and (nombre == "bombas_paralelo" or e.get("succion_tanque")) and "ext" not in [o[0] for o in orden]:
                orden.append(("ext", ext))
            orden.append((nombre, e))
        if ext and "ext" not in [o[0] for o in orden]:
            orden.append(("ext", ext))
        for nombre, e in orden:
            nv = nuevo_de(e)
            fc, ec = col(nv)
            if nombre == "generador":     # adentro de la caseta, en el piso, del mismo color que el tablero
                # grupo electrógeno: base (patines), carcasa con rejillas de ventilación, panel de control con rayo y
                # chimenea de escape; la potencia va abajo de la carcasa
                fg = LIME if nv else OLD_F
                ax.add_patch(Rectangle((cx + 0.6, GY), 10.3, 1.0, fc=ec, ec=ec, lw=1.2, zorder=2.5))
                rbox(cx + 1, GY + 1.0, 9.5, 7.2, 1.0, fg, 2.6, ec=ec)
                for k in range(3):
                    ax.plot([cx + 2.0, cx + 5.6], [GY + 5.8 + k * 0.85] * 2, color=ec, lw=1.4, zorder=2.8)
                ax.add_patch(Rectangle((cx + 6.6, GY + 5.5), 3.2, 2.2, fc="white", ec=ec, lw=1.4, zorder=2.8))
                ax.plot([cx + 8.6, cx + 8.1, cx + 8.9, cx + 8.4], [GY + 7.3, GY + 6.6, GY + 6.6, GY + 5.8], color=ec, lw=1.2, zorder=2.9)
                for kx in (7.2, 9.3):
                    ax.add_patch(Circle((cx + kx, GY + 6.1), 0.28, fc=ec, ec="none", zorder=2.9))
                ax.add_patch(Rectangle((cx + 2.6, GY + 8.1), 1.0, 2.1, fc=ec, ec=ec, lw=1.0, zorder=2.5))      # escape
                ax.add_patch(Rectangle((cx + 2.2, GY + 10.0), 1.8, 0.55, fc=ec, ec=ec, lw=1.0, zorder=2.5))
                medida(cx + 5.75, GY + 2.5, e.get("medida"), nv, z=2.8)
                gen_der = cx + 10.5
                gen_cx = cx + 1                # x donde parte el generador (el tablero, si hay, va encima)
                cx += ANCHO["generador"]
            elif nombre == "tablero":
                apilado = gen_der is not None         # con generador antes: el tablero va ENCIMA del generador (caseta angosta)
                by = GY + 13.5                        # tablero chico, colgado alto en la pared: no toca la tubería
                bx = gen_cx + 0.75 if apilado else cx + 3.0
                if apilado:   # cable con ENCHUFE en cada punta: toma del generador → toma del tablero
                    xg = bx + 4.0
                    ax.add_patch(Rectangle((xg - 0.8, GY + 8.2), 1.6, 0.6, fc=ec, ec=ec, lw=1.0, zorder=4.1))       # toma del generador
                    rbox(xg - 0.65, GY + 8.8, 1.3, 1.0, 0.25, "white", 4.2, ec=ec, lw=1.3)                           # enchufe (generador)
                    ax.add_patch(Rectangle((xg - 0.8, by - 0.6), 1.6, 0.6, fc=ec, ec=ec, lw=1.0, zorder=4.1))        # toma del tablero
                    rbox(xg - 0.65, by - 1.6, 1.3, 1.0, 0.25, "white", 4.2, ec=ec, lw=1.3)                           # enchufe (tablero)
                    manguera([(xg, GY + 9.8), (xg - 1.3, GY + 10.7), (xg + 1.3, by - 2.4), (xg, by - 1.6)], ec, z=4.15, lw=1.7)   # cable
                ax.add_patch(Rectangle((bx, by), 8.0, 12, fc=LIME if nv else OLD_F, ec=ec, lw=LW, zorder=4))
                for c in range(3):
                    ax.add_patch(Circle((bx + 1.9 + c * 2.1, by + 10.2), 0.6, fc="white", ec=ec, lw=1.4, zorder=4.2))
                poly([(bx + 4, by + 8.0), (bx + 2.3, by + 4.4), (bx + 5.7, by + 4.4)], "none", 4.2, ec=ec, lw=1.6)
                ax.plot([bx + 4.2, bx + 3.7, bx + 4.5, bx + 4.0], [by + 7.1, by + 6.1, by + 6.1, by + 5.0], color=ec, lw=1.3, zorder=4.3)
                ax.add_patch(Rectangle((bx + 2.6, by + 1.0), 2.8, 1.4, fc="white", ec=ec, lw=1.3, zorder=4.2))
                if not apilado:
                    flecha(cx + 11.0, P, 0)
                    cx += ANCHO["tablero"]
            elif nombre == "cloracion":
                # estanque de cloro a piso con líquido blanco, por detrás de la tubería principal
                recipiente(cx + 0.8, GY, 5.2, 7.8, z=2.5, nuevo=nv, agua=0.82, liquido="white", cuerpo="#BCD9EF")   # líquido sobre la tubería: se ve
                medida(cx + 3.4, GY + 2.3, e.get("medida"), nv, z=2.9)
                # bomba dosificadora COLGADA EN LA PARED (placa de montaje con tornillos): succiona del estanque por una
                # manguera e INYECTA el cloro en la tubería por otra, con válvula de inyección
                # (la dosificadora es MINI: la mitad del tamaño base, factor s = 0.5)
                s_ = 0.5
                px0, py0 = cx + 8.4, GY + 9.0
                ax.add_patch(Rectangle((px0, py0), 6.4 * s_, 10.4 * s_, fc="#F4F6F7" if nv else OLD_WALL, ec=ec, lw=1.3, zorder=2.4))
                for sx_, sy_ in ((0.8, 0.8), (5.6, 0.8), (0.8, 9.6), (5.6, 9.6)):
                    ax.add_patch(Circle((px0 + sx_ * s_, py0 + sy_ * s_), 0.18, fc=ec, ec="none", zorder=2.5))
                rbox(px0 + 1.0 * s_, py0 + 3.6 * s_, 4.4 * s_, 6.0 * s_, 0.5, LIME if nv else OLD_F, 4.6, ec=ec, lw=1.6)   # cuerpo
                ax.add_patch(Rectangle((px0 + 1.8 * s_, py0 + 7.3 * s_), 2.8 * s_, 1.5 * s_, fc="white", ec=ec, lw=1.0, zorder=4.8))  # pantalla
                for bxx in (2.2, 4.2):
                    ax.add_patch(Circle((px0 + bxx * s_, py0 + 5.3 * s_), 0.28, fc="white", ec=ec, lw=1.0, zorder=4.8))           # botones
                rbox(px0 + 1.7 * s_, py0 + 1.3 * s_, 3.0 * s_, 2.3 * s_, 0.35, "white", 4.7, ec=ec, lw=1.5)                      # cabezal
                ys_ = py0 + 2.45 * s_                                                                                       # altura de las mangueras
                manguera([(cx + 3.4, GY + 7.4), (cx + 3.4, ys_), (px0 + 1.7 * s_, ys_)], ec)                           # succión: manguerita del estanque
                xi = px0 + 6.4 * s_ + 1.2                                                                                  # unión en la tubería
                manguera([(px0 + 4.7 * s_, ys_), (xi, ys_), (xi, P + 1.9)], ec)                                          # descarga: inyecta en la unión
                ax.add_patch(Rectangle((xi - 0.25, P + 0.8), 0.5, 1.2, fc="white", ec=ec, lw=1.1, zorder=5.2))            # toma de la unión (te)
                ax.add_patch(Rectangle((xi - 0.7, P - 1.0), 1.4, 2.0, fc="white", ec=ec, lw=1.3, zorder=5.2))              # unión (tuerca) sobre la tubería
                flecha(cx + 9.4, P, 0)
                if not e.get("sin_numero"):
                    num(px0 + 1.6, GY + 17.6, etapa("Cloración", e.get("dato", "inyección de cloro al agua")))
                cx += ANCHO["cloracion"]
            elif nombre == "filtro":
                recipiente(cx + 2, GY, 5.2, 18, nuevo=nv, solido=True, puntos=True)       # delgado, sólido, a piso
                # cabezal (válvula multivía): ENTRADA por la boquilla izquierda y SALIDA por la derecha; arriba, la perilla y
                # el RETROLAVADO (desagüe corto con llave, casi no ocupa espacio)
                fcc = PIPE if nv else OLD_W
                rbox(cx + 2.8, GY + 17.6, 3.6, 2.6, 0.7, fcc, 5.4, ec=ec, lw=2.2)
                for nx_ in (cx + 2.2, cx + 6.3):
                    ax.add_patch(Rectangle((nx_, GY + 18.1), 0.7, 1.3, fc=fcc, ec=ec, lw=1.4, zorder=5.3))
                ax.add_patch(Circle((cx + 4.6, GY + 18.9), 0.75, fc="white", ec=ec, lw=1.3, zorder=5.6))
                ax.plot([cx + 4.6, cx + 4.6], [GY + 18.9, GY + 19.5], color=ec, lw=1.3, zorder=5.7)
                # retrolavado: tubito que sale por ARRIBA del cabezal, con llave, y baja a la izquierda
                manguera([(cx + 4.6, GY + 20.0), (cx + 4.6, GY + 21.9), (cx + 2.3, GY + 21.9), (cx + 2.3, GY + 20.6)], ec, z=5.0)
                poly([(cx + 3.0, GY + 22.5), (cx + 3.5, GY + 21.9), (cx + 3.0, GY + 21.3)], "white", 5.2, ec=ec, lw=1.0)
                poly([(cx + 4.0, GY + 22.5), (cx + 3.5, GY + 21.9), (cx + 4.0, GY + 21.3)], "white", 5.2, ec=ec, lw=1.0)
                medida(cx + 4.6, GY + 13.8, e.get("medida"), nv, rot=90, fs=7.5, z=5.5)
                num(cx + 4.6, GY + 7.2, etapa(e.get("nombre", "Filtro de hierro"), e.get("dato", "remoción de hierro")))
                filtro_x = cx + 4.6
                cx += ANCHO["filtro"]
            elif nombre == "ext":
                ew, eh = 24, 25
                ex = xw + 7                   # afuera de la caseta, a su derecha; las tuberías se dibujan después del bucle
                nvx = nuevo_de(ext)
                radier(ex - 2, ew + 4, nvx)
                recipiente(ex, GY + RAD, ew, eh, z=7, agua=0.66, nuevo=nvx)           # afuera, sobre su radier
                ecx = LINE if nvx else OLD_E
                despiche(ex + ew, GY + RAD + 2.0, nvx)                                  # despiche con llave, a ras del fondo
                if ext.get("sonda_nivel"):
                    pera(ex, ew, GY + RAD, eh, ex + ew - 7, GY + RAD + (eh - ew / 2 * 0.3) * 0.66, nvx, z=7.4)
                medida(ex + ew / 2 - 3, GY + 8.5, ext.get("medida"), nvx, fs=10, z=7.6)
                num(ex + 6.5, GY + 12.6, etapa(ext.get("nombre", "Estanque de acumulación"), ext.get("dato", "almacenamiento de agua tratada")))
                ext_src = cx - 0.6
            elif nombre == "bombas_paralelo":
                n = e.get("cantidad", 2); exist = e.get("existentes", 0)
                b0, b1 = cx + 2, cx + 19
                b_top = P + 10 * (n - 1)
                pipe([b0, b0], [P, P + 10 * (n - 1)]); pipe([b1, b1], [P, P + 10 * (n - 1)])
                for i in range(n):
                    yy = P + 10 * i
                    pipe([b0, b1], [yy, yy])
                    capsula((b0 + b1) / 2 + 0.6, yy, nuevo=i >= exist, txt=e.get("medida"), patas=(i == 0))
                    flecha(b0 + 1.9, yy, 0)
                    if presost:
                        px2 = (b0 + b1) / 2 + 7.2
                        ax.plot([px2, px2], [yy + 0.7, yy + 3.2], color=LINE, lw=1.6, zorder=6.2)
                        ax.add_patch(Circle((px2, yy + 4.3), 1.1, fc="white", ec=LINE, lw=1.5, zorder=6.3))
                dato = e.get("dato", "bombeo a la red de distribución")
                num((b0 + b1) / 2 + 1.0, P + 5.0 if n == 2 else P + 10 * (n - 1) + 6.2, etapa("Impulsión", dato))   # con 2 bombas, entre ellas
                cx += ANCHO["bombas_paralelo"]
            elif nombre == "bomba_periferica":
                # electrobomba periférica (amarilla) con su controlador electrónico de presión sobre la línea
                pcx = cx + 6.6
                fa_ = AMAR if nv else OLD_F
                ax.add_patch(Rectangle((pcx - 0.7, GY), 5.2, P - 2.15 - GY + 0.1, fc=fa_, ec=ec, lw=1.6, zorder=5.8))   # soporte al piso
                capsula(pcx, P, nuevo=nv, txt=e.get("medida"))
                xc_ = pcx + 8.2
                ax.add_patch(Rectangle((xc_ - 0.95, P - 0.95), 1.9, 1.9, fc="white", ec=ec, lw=1.5, zorder=5.2))         # te sobre la línea
                ax.add_patch(Rectangle((xc_ - 0.4, P + 0.9), 0.8, 0.8, fc=fc, ec=ec, lw=1.2, zorder=5.1))
                rbox(xc_ - 1.4, P + 1.6, 2.8, 3.6, 0.6, "white" if nv else OLD_F, 5.4, ec=ec, lw=1.7)                 # controlador de presión
                ax.add_patch(Rectangle((xc_ - 0.9, P + 3.6), 1.8, 1.0, fc=fc, ec=ec, lw=1.1, zorder=5.5))                 # pantalla
                for dx_ in (-0.45, 0.45):
                    ax.add_patch(Circle((xc_ + dx_, P + 2.5), 0.28, fc=fc, ec=ec, lw=0.9, zorder=5.5))
                if e.get("succion_tanque"):                                                   # bomba que chupa del estanque exterior
                    b0, b1, b_top = cx + 0.4, pcx + 9.9, P
                    num(pcx + 1.0, GY + 11.6, etapa(e.get("nombre", "Bombeo a la red"), e.get("dato", "presión para la red")))
                cx += ANCHO["bomba_periferica"]
            elif nombre == "filtro_cartucho":
                # como en el manual: VASO acanalado (ranuras verticales) con CABEZAL ancho de puertos laterales y un símbolo
                # (gota = sedimentos, puntos = carbón); la línea sube, ENTRA por el cabezal y SALE por el otro puerto
                fx_ = cx + 2.9
                yh_ = GY + 13.2
                fcc_ = PIPE if nv else OLD_W
                rbox(fx_, GY, 4.2, 12.2, 1.2, SOLIDO if nv else OLD_W, 5, ec=ec, lw=2.0)
                for k_ in range(1, 5):
                    ax.plot([fx_ + k_ * 0.84] * 2, [GY + 0.9, GY + 11.3], color=ec, lw=1.0, alpha=0.75, zorder=5.2)
                rbox(fx_ - 0.5, GY + 11.6, 5.2, 2.8, 0.7, fcc_, 5.4, ec=ec, lw=2.0)
                for nx_ in (fx_ - 1.3, fx_ + 4.7):
                    ax.add_patch(Rectangle((nx_, yh_ - 0.55), 0.8, 1.1, fc=fcc_, ec=ec, lw=1.3, zorder=5.3))
                ax.add_patch(Circle((fx_ + 2.1, yh_), 0.95, fc="white", ec=ec, lw=1.3, zorder=5.6))
                if e.get("simbolo") == "puntos":
                    for dx_, dy_ in ((-0.35, 0.2), (0.35, 0.2), (0.0, -0.35)):
                        ax.add_patch(Circle((fx_ + 2.1 + dx_, yh_ + dy_), 0.17, fc=ec, ec="none", zorder=5.7))
                else:
                    gota(fx_ + 2.1, yh_ - 0.45, 0.45, 5.7)
                if e.get("medida"):
                    ax.text(fx_ + 2.1, GY + 6.0, e["medida"], rotation=90, ha="center", va="center", fontsize=6.5, fontweight="bold",
                            color=ec if nv else OLD_T, zorder=5.8, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
                lifts.append((fx_ - 2.4, fx_ + 6.6, yh_))
                cx += ANCHO["filtro_cartucho"]
            elif nombre == "uv":
                # como en el manual: la lámpara UV se dibuja como un CÍRCULO de doble anillo con «UV», con sus puertos sobre la línea
                ucx = cx + 6.0
                ax.add_patch(Rectangle((ucx - 5.0, P - 0.8), 1.2, 1.6, fc=ec, ec=ec, lw=1.0, zorder=5.0))
                ax.add_patch(Rectangle((ucx + 3.8, P - 0.8), 1.2, 1.6, fc=ec, ec=ec, lw=1.0, zorder=5.0))
                ax.add_patch(Circle((ucx, P), 4.0, fc=SOLIDO if nv else OLD_W, ec=ec, lw=2.2, zorder=5.1))
                ax.add_patch(Circle((ucx, P), 3.0, fc="white" if nv else OLD_F, ec=ec, lw=1.6, zorder=5.2))
                ax.text(ucx, P, "UV", ha="center", va="center", fontsize=9, fontweight="bold", color=ec if nv else OLD_T, zorder=5.3)
                flecha(ucx + 5.6, P, 0)
                cx += ANCHO["uv"]
            elif nombre == "hidroneumatico":
                pildora(cx + 2, GY, 6.0, 13.8, nuevo=nv)                                 # redondo, sólido y a piso
                if b1 is not None:
                    flecha(cx - 0.2, P, 0)
                medida(cx + 5.0, GY + 9.9, e.get("medida"), nv, fs=8, z=5.5)
                num(cx + 5.0, GY + 17.6, etapa("Hidroneumático", e.get("dato", "regulación de presión")))
                hid_cx = cx + 5.0
                cx += ANCHO["hidroneumatico"]
        YH = GY + 18.75                                                               # altura de las boquillas del cabezal
        f_in = f_out = None
        if filtro_x is not None:
            f_in, f_out = filtro_x - 4.0, filtro_x + 4.0                              # columnas de entrada y salida del filtro
        y1 = GY + 24.6                                                                # recorrido superior hacia el estanque exterior
        if filtro_x is not None and not ext:
            pipe([abierta, f_in, f_in, filtro_x - 2.4], [P, P, YH, YH])                  # entra por el cabezal
            pipe([filtro_x + 2.4, f_out, f_out, fin(idx, x + w + GAP)], [YH, YH, P, P])  # sale por el cabezal
        elif not ext:
            _px, _py = [abierta], [P]
            for _xa, _xb, _yh in sorted(lifts):
                _px += [_xa, _xa, _xb, _xb]; _py += [P, _yh, _yh, P]
            _px.append(fin(idx, x + w + GAP)); _py.append(P)
            pipe(_px, _py)
        else:
            # Estanque afuera sin cruces: (1) filtro → estanque por arriba, (2) estanque → bombas por arriba, más bajo,
            # (3) hidroneumático → macromedidor por debajo de la tierra, pasando bajo el estanque y su radier.
            if filtro_x is not None:
                pipe([abierta, f_in, f_in, filtro_x - 2.4], [P, P, YH, YH])                    # entrada por el cabezal
                pipe([filtro_x + 2.4, f_out, f_out, ex + 12, ex + 12],
                     [YH, YH, y1, y1, GY + 21.4], z=7.5)                                  # salida por el cabezal; entra por la cúpula
                src_x = f_out
            elif ext_src is not None:
                _px, _py = [abierta], [P]
                for _xa, _xb, _yh in sorted(lifts):
                    _px += [_xa, _xa, _xb, _xb]; _py += [P, _yh, _yh, P]
                _px.append(ext_src); _py.append(P)
                pipe(_px, _py)                                                        # línea de la planta (con subidas por los cabezales)
                pipe([ext_src, ext_src, ex + 12, ex + 12], [P, y1, y1, GY + 21.4], z=7.5)
                src_x = ext_src
            else:
                src_x = x + 2.5
                pipe([x, src_x], [P, P])
                pipe([src_x, src_x, ex + 12, ex + 12], [P, y1, y1, GY + 21.4], z=7.5)
            flecha((src_x + ex + 12) / 2, y1, 0, z=7.7)
            caida(ex + 12, GY + 21.0, GY + RAD + (eh - ew / 2 * 0.3) * 0.66, nvx)        # se ve caer el agua adentro
            if b0 is not None:                                                        # (2) las bombas chupan a 20 cm del fondo del estanque
                ys0 = GY + RAD + 2.2                                                  # 20 cm sobre el fondo (1 unidad ≈ 9 cm)
                xv = ex - 2.5
                y2 = GY + (20 if ch >= 27 else ch - 4)                                # altura del tramo horizontal (dentro de la caseta)
                pipe([ex + 1, xv, xv, b0, b0], [ys0, ys0, y2, y2, b_top], z=6.5)
                flecha(xv, (ys0 + y2) / 2, 90)
                flecha((xv + b0) / 2, y2, 180)
            if hid_cx is not None and b1 is not None:
                xe = x + w + GAP                                                      # donde empieza el macromedidor
                pipe([b1, hid_cx], [P, P])
                xd = hid_cx + 4.6                                                     # baja justo después del hidroneumático
                pipe([hid_cx, xd, xd, xe], [P, P, YU, YU], z=3)                       # y va ENTERRADA casi todo el trayecto
                flecha((xd + xe) / 2, YU, 0)
                prev_buried = True
            elif b1 is not None:                                                      # sin hidroneumático: sale de la bomba y va enterrada
                xe = x + w + GAP
                xd = b1 + 3.0
                pipe([b1, xd, xd, xe], [P, P, YU, YU], z=3)
                flecha((xd + xe) / 2, YU, 0)
                prev_buried = True
        if _n_et is not None:                      # número de la caseta completa (p. ej. «Planta purificadora»)
            num(x + _et.get("dx", wc / 2), GY + _et.get("y", ch - 3.6), _n_et)
    elif k == "estanque" and d.get("forma") == "vertical":
        nv = d.get("nuevo", True)
        _l = litros(d.get("medida")) or 5000
        tw_ = 21 * (_l / 5000) ** (1 / 3); bh_ = 17.5 * (_l / 5000) ** (1 / 3)          # tamaño proporcional al volumen
        tx = x + 2
        radier(tx - 2, tw_ + 4, nv)
        cxt, ytop_, ya_ = tanque_vertical(tx, GY + RAD, tw_, bh_, nv, d.get("medida"))
        if d.get("entrada") == "techo":
            xin = cxt                                                                               # por la boca de arriba
            yg = SCH["yw"] - 0.6
            n_sep = etapa("Recolector de primeras lluvias", d.get("dato_separador", "descarte de las primeras aguas y retiro de hojas"))
            pipe([xin, xin], [yg, ytop_ - 0.3], z=7.5, lw=4.4)                                      # por la altura: de la canaleta directo al estanque
            rbox(xin - 1.9, yg - 5.6, 3.8, 3.8, 0.7, LIGHT, 7.7, ec=LINE, lw=1.9)                    # recolector chico, EN la misma tubería
            for ix_ in (xin - 1.2, xin + 0.1):
                rbox(ix_, yg - 5.0, 1.1, 2.6, 0.4, SOLIDO, 7.8, ec=LINE, lw=1.2)
            despiche(xin + 1.9, yg - 5.0, True, z=7.9, dir=1, baja=1.1)                              # descarga de las primeras lluvias
            num(xin - 5.6, yg - 3.8, n_sep)
            ys_in = ytop_
        else:
            xin = tx + tw_ * 0.25
            ys_in = sup_hombro(tx, GY + RAD, tw_, bh_, xin)
            pipe([x - 4, x - 4, xin, xin], [P, ytop_ + 1.5, ytop_ + 1.5, ys_in - 0.8], z=7.5)     # entra por el hombro del estanque
        caida(xin, (ytop_ - 1.0) if d.get("entrada") == "techo" else ys_in - 1.2, ya_, nv, z=7.6)   # se ve caer el agua adentro
        xpr = tx + tw_ * 0.66
        pera(tx, tw_, GY + RAD, bh_, xpr, ya_, nv, z=7.4, topy=sup_hombro(tx, GY + RAD, tw_, bh_, xpr))   # pera de nivel
        if d.get("entrada") != "techo":
            flecha(x + 1.5, ytop_ + 1.5, 0, z=7.7)
        despiche(tx + tw_, GY + RAD + 1.0, nv, z=7.2)
        if ft == "lluvia" and idx == len(tramos) - 1:                                           # agua lluvia sin planta: sale enterrada a la llave de la escuela
            xf_ = SCH["xf"]
            xdb_ = tx + tw_ + 5.0
            pipe([cxt, xdb_, xdb_, xf_, xf_], [P, P, YU, YU, GY + 6.5], z=4, lw=4.5)
            flecha(xdb_ - 3, P, 0, z=7.7)
            for fxx_ in ((xdb_ + xf_) / 2 - 8, (xdb_ + xf_) / 2 + 8):
                flecha(fxx_, YU, 0, s=0.5)
            grifo(xf_)
            num(cxt, GY + RAD + bh_ * 0.58, etapa(d.get("nombre", "Estanque de acumulación"), d.get("dato", "almacenamiento de agua")))
            num(xf_ + 1.6, GY + 17.5, etapa("Escuela", d.get("dato_destino", "cocina y baños")))
        else:
            pipe([cxt, fin(idx, x + w + GAP)], [P, P], z=2.4)                                   # salida por abajo
            flecha(x + w + 1, P, 0, z=4.3)
            num(cxt, GY + RAD + bh_ * 0.58, etapa(d.get("nombre", "Estanque de acumulación"), d.get("dato", "almacenamiento de agua")))
    elif k == "planta":
        # PLANTA PURIFICADORA como el gabinete real de la foto, COMPACTO: vidrio transparente, marco y repisa; arriba de la repisa la línea con
        # válvulas y cabezales de los filtros; abajo los vasos colgados, la UV, el bidón de cloro y las DOS bombas. Todo se dibuja a escala
        # y se reduce (s_pl) para que entre en una caseta chica sobre su radier.
        WP = d.get("ancho", 88)
        s_pl = d.get("escala", 0.62)
        X = lambda u: x + (u - x) * s_pl
        Y = lambda v: GY + (v - GY) * s_pl + RAD
        pot = d["estanque_potable"]; npot = nuevo_de(pot)
        _l = litros(pot.get("medida")) or 5000
        tw_ = 21 * (_l / 5000) ** (1 / 3); bh_ = 17.5 * (_l / 5000) ** (1 / 3)
        xw = x + WP * s_pl
        fy = GY + 2.0
        chh = 22
        LY = GY + 12.5
        n_pl = etapa(d["etapa"]["nombre"], d["etapa"].get("dato", ""))
        plata = "#D3ECFC"
        radier(x - 1.5, WP * s_pl + 3, True)                                                              # radier de la caseta
        _n0 = (len(ax.patches), len(ax.lines), len(ax.texts))
        for fxp in (x + 4, x + WP - 8):
            ax.add_patch(Rectangle((fxp, GY), 4, 2.0, fc=plata, ec=LINE, lw=1.6, zorder=2.3))               # patas
        ax.add_patch(Rectangle((x, fy), WP, chh, fc=(1, 1, 1, 0.13), ec=LINE, lw=LW, zorder=2))             # vidrio transparente
        for mx_ in (x + 22, x + 44, x + 66):
            ax.plot([mx_] * 2, [fy, fy + chh], color=LINE, lw=1.5, alpha=0.7, zorder=2.2)
        for gx_ in (x + 5, x + 27, x + 49, x + 71):
            ax.plot([gx_, gx_ + 4], [fy + chh - 2, fy + chh - 8], color="white", lw=1.7, alpha=0.4, zorder=2.2)
        ax.add_patch(Rectangle((x - 1.5, fy + chh), WP + 3, 2.0, fc=plata, ec=LINE, lw=LW, zorder=2.1))      # techo metálico
        ax.add_patch(Rectangle((x, GY + 11.0), WP, 0.9, fc=plata, ec=LINE, lw=1.6, zorder=2.4))             # repisa
        ax.add_patch(Rectangle((x - 0.4, fy), 0.8, chh, fc=plata, ec=LINE, lw=1.3, zorder=2.3))
        ax.add_patch(Rectangle((x + WP - 0.4, fy), 0.8, chh, fc=plata, ec=LINE, lw=1.3, zorder=2.3))
        pipe([x - 1.5, x + 4.0], [P, P], z=3)
        pc1, xc1 = bomba_per(x + 1.5, True, d.get("medida_bomba"))                                         # bomba 1 (lluvia → filtros)
        xr1, xr2 = xc1 + 3.0, x + 67.5
        pipe([xc1, xr1, xr1, xr2], [P, P, LY, LY], z=3)
        flecha(xc1 + 1.5, P, 0)
        ax.add_patch(Rectangle((x + 22.4, LY - 1.0), 1.8, 2.0, fc="white", ec=LINE, lw=1.4, zorder=5.0))   # válvula de bola
        ax.plot([x + 23.3, x + 25.0], [LY + 1.0, LY + 2.8], color=LINE, lw=2.2, solid_capstyle="round", zorder=5.1)
        filtro_manual(x + 30.5, LY, "gota", "Mecánico", nv=True)
        filtro_manual(x + 40.5, LY, "puntos", "Carbón", nv=True)
        flecha(x + 26.5, LY, 0)
        ux = x + 49.5
        ax.add_patch(Rectangle((ux - 4.4, LY - 0.8), 1.2, 1.6, fc=LINE, ec=LINE, lw=1.0, zorder=5.0))
        ax.add_patch(Rectangle((ux + 3.2, LY - 0.8), 1.2, 1.6, fc=LINE, ec=LINE, lw=1.0, zorder=5.0))
        ax.add_patch(Circle((ux, LY), 3.4, fc=SOLIDO, ec=LINE, lw=2.2, zorder=5.1))                          # UV (círculo del manual)
        ax.add_patch(Circle((ux, LY), 2.5, fc="white", ec=LINE, lw=1.6, zorder=5.2))
        ax.text(ux, LY, "UV", ha="center", va="center", fontsize=8, fontweight="bold", color=LINE, zorder=5.3)
        flecha(ux + 4.9, LY, 0)
        cloracion_std(x + 52.0, LY, fy, True)                                                              # cloración ESTÁNDAR (tanque + dosificadora mini)
        flecha(x + 58.0, LY, 0, z=5.3)
        pc2, xc2 = bomba_per(x + 69.0, True, None)                                                          # bomba 2 (estanque potable → red)
        b0, b1 = x + 69.4, pc2 + 9.9
        pipe([b0, b1], [P, P], z=3)
        # --- se reduce todo el gabinete (geometría, grosores y letras) ---
        T = Affine2D().translate(-x, -GY).scale(s_pl).translate(x, GY + RAD)
        for art in list(ax.patches[_n0[0]:]) + list(ax.lines[_n0[1]:]):
            art.set_transform(T + ax.transData)
            art.set_linewidth(max(0.7, art.get_linewidth() * 0.78))
        for tx_ in ax.texts[_n0[2]:]:
            tx_.set_transform(T + ax.transData)
            tx_.set_fontsize(max(3.4, tx_.get_fontsize() * s_pl * 1.1))
        # --- conexiones fuera del gabinete (ya en coordenadas reales) ---
        ex = xw + 7
        radier(ex - 2, tw_ + 4, npot)
        cxp, ytp, ya_p = tanque_vertical(ex, GY + RAD, tw_, bh_, npot, pot.get("medida"))
        x_e = ex - 1.0                                                                                       # la tubería va PEGADA al borde del estanque
        Yf = GY + 22.0
        pipe([X(xr2), X(xr2), x_e, x_e, cxp], [Y(LY), Yf, Yf, ytp - 0.3, ytp - 0.3], z=7.5)               # sube por su borde y entra por la boca de arriba
        flecha((X(xr2) + x_e) / 2, Yf, 0, z=7.7)
        caida(cxp, ytp - 1.0, ya_p, npot, z=7.6)
        xpr = ex + tw_ * 0.66
        pera(ex, tw_, GY + RAD, bh_, xpr, ya_p, npot, z=7.4, topy=sup_hombro(ex, GY + RAD, tw_, bh_, xpr))
        despiche(ex + tw_, GY + RAD + 1.0, npot, z=7.2)
        num(cxp, GY + RAD + bh_ * 0.58, etapa(pot.get("nombre", "Estanque de agua potable"), pot.get("dato", "")))
        ys0 = GY + RAD + 2.2                                                                               # el estanque entrega a 20 cm del fondo
        xv = xw + 4.5
        y2 = GY + 19.5
        pipe([ex + 1, xv, xv, X(b0), X(b0)], [ys0, ys0, y2, y2, Y(P)], z=6.5)
        flecha(xv, (ys0 + y2) / 2, 90)
        flecha((xv + X(b0)) / 2, y2, 180)
        xf = SCH["xf"]                                                                               # llave de la MISMA escuela que capta
        xdb = X(b1) + 2.0
        pipe([X(b1), xdb, xdb, xf, xf], [Y(P), Y(P), YU, YU, GY + 6.5], z=4, lw=4.5)                      # vuelve ENTERRADA a la escuela
        for fxx in (xdb + 14, (xdb + xf) / 2, xf - 10):
            flecha(fxx, YU, 0, s=0.5)
        num(X(x + 14), GY + 20.0, n_pl)
        num(X(pc2) , GY + 25.6, etapa(d.get("nombre_bomba", "Bombeo a la red"), d.get("dato_bomba", "presión para la escuela")))
        grifo(xf)
        num(xf + 1.6, GY + 17.5, etapa("Escuela", "cocina y baños"))
    elif k == "estanque":
        n = d.get("cantidad", 1); nv = d.get("nuevo", True); elev = d.get("elevado", False)
        ecx = LINE if nv else OLD_E
        base, eh = (GY + 14, 22) if elev else (GY + RAD, 25.5)     # a piso, sobre su radier
        xs = [x + 2 + i * 30 for i in range(n)]
        top = base + eh + 3
        pipe([x - 4, x - 4, xs[-1] + 7], [P, top, top], z=2.6)                       # entrada por arriba
        flecha(x + 2.5, top, 0)
        if n > 1:
            flecha(xs[0] + 27, P, 0, z=4.3)
        if not elev:
            flecha(x + w + 1, P, 0, z=4.3)
        for tx in xs:
            pipe([tx + 7, tx + 7], [top, base + eh - 1.5], z=2.6)
            if d.get("electrovalvula"):
                ax.add_patch(Circle((tx + 7, top), 1.6, fc=LIME, ec=LINE, lw=1.8, zorder=3))
        if elev:
            for tx in xs:
                for lx in (tx + 3, tx + 21):
                    ax.plot([lx, lx], [GY, base], color=ecx, lw=2.4, zorder=2.5)
                ax.plot([tx + 3, tx + 21], [GY, base], color=ecx, lw=1.5, zorder=2.5)
                ax.plot([tx + 21, tx + 3], [GY, base], color=ecx, lw=1.5, zorder=2.5)
            pipe([xs[-1] + 17, xs[-1] + 17, fin(idx, x + w + GAP)], [base + 1, P, P], z=2.4)
        else:
            for tx in xs:
                radier(tx - 2, 28, nv)
            pipe([xs[0] + 12, fin(idx, x + w + GAP)], [P, P], z=2.4)                     # salida por abajo
        for tx in xs:
            recipiente(tx, base, 24, eh, z=4, agua=0.66, nuevo=nv)
            despiche(tx + 24, base + 1.0, nv, z=4.2)                                     # despiche con llave (bajo la tubería)
            pipe([tx + 7, tx + 7], [base + eh + 0.5, base + 20.3], z=4.6)                # la tubería entra por la cúpula…
            caida(tx + 7, base + 20.0, base + (eh - 3.6) * 0.66, nv, z=4.7)              # …y se ve caer el agua
            if d.get("sonda_nivel"):
                pera(tx, 24, base, eh, tx + 17, base + (eh - 3.6) * 0.66, nv, z=4.4)
            medida(tx + 10, base + 5.5, d.get("medida"), nv, fs=10, z=4.6)
        nombre = d.get("nombre", "Estanques de acumulación" if n > 1 else "Estanque de acumulación")
        dato = d.get("dato", "almacenamiento de agua tratada")
        num((xs[0] + xs[-1]) / 2 + 10, base + eh * 0.5, etapa(nombre, dato))
    elif k == "contador":
        mx = x + w + 2                                                  # cerca de la escuela (al final del tramo)
        xr = mx - 5.4                                                   # la tubería sale a la superficie justo antes del macromedidor
        if prev_buried:
            pipe([x, xr, xr], [YU, YU, P], z=3)
        else:
            pipe([x, x, xr, xr], [P, YU, YU, P], z=3)
        pipe([xr, fin(idx, x + w + GAP)], [P, P])
        flecha((x + xr) / 2, YU, 0)
        prev_buried = False
        macromedidor(mx, d.get("nuevo", True))
        num(mx, P + 5.5, etapa(d.get("nombre", "Macromedidor"), d.get("dato", "medición del agua entregada")))
    x += w + GAP

# ---------- 5. destino ----------
dest = cfg.get("destino", {})
enterrada = dest.get("red_enterrada", True)
if ft == "lluvia":
    pass                                              # la escuela es la misma que capta la lluvia (ya está dibujada)
elif not ES_SSR:
    sx, sw, sh = x + 2, 32, 16
    if enterrada and prev_buried:
        pipe([x, sx + 4.2, sx + 4.2], [YU, YU, GY + 6.5], z=4, lw=4.5)               # la red ya viene enterrada desde la bomba
    elif enterrada:
        pipe([x, x, sx + 4.2, sx + 4.2], [P, YU, YU, GY + 6.5], z=4, lw=4.5)
    else:
        pipe([x, sx + 4.2, sx + 4.2], [P, P, GY + 6.5], z=4, lw=4.5)
    flecha((x + sx + 4.2) / 2, YU if enterrada else P, 0, s=0.42)
    escuela(sx, sw, sh)
    grifo(sx + 4.2)
    num(sx + sw / 2, GY + sh + 2.7, etapa("Escuela", dest.get("dato", "distribución por red nueva enterrada" if enterrada else "distribución por la red existente")))
else:
    casas = [(x + 3, 14, 13), (x + 22, 20, 18), (x + 46, 12, 11)]
    yr = YU
    pipe([x, x, casas[-1][0] + 3], [P, yr, yr], z=4, lw=4.5)
    flecha((x + casas[0][0] + 3) / 2, yr, 0, s=0.42)
    for hx2, hw, hh in casas:
        casa(hx2, hw, hh)
        pipe([hx2 + 3, hx2 + 3], [yr, GY + 5], z=4, lw=3.5)
    llave(casas[1][0] + 3)
    num(casas[1][0] + casas[1][1] / 2, GY + casas[1][2] + 5, etapa("Red de distribución", dest.get("dato", "distribución a los hogares")))

# ---------- 6. textos y logos ----------
if not SD:
    quien = ("SSR " if ES_SSR else "Escuela ") + cfg.get("nombre", "")
    ax.text(5, 102, "Proyecto de agua potable", fontsize=44, fontweight="bold", color=ARENA, va="center")
    ax.text(5, 92.5, quien, fontsize=30, fontweight="bold", color=WATER, va="center")
    # comuna y región (+ matrícula si se conoce) en una línea chica bajo el nombre; lo que no esté confirmado no se escribe
    _lugar = ", ".join(v for v in (cfg.get("comuna"), cfg.get("region")) if v)
    _partes = ([_lugar] if _lugar else []) + ([str(cfg["estudiantes"]) + " estudiantes"] if cfg.get("estudiantes") and not ES_SSR else [])
    if _partes:
        ax.text(5.2, 88.7, "  ·  ".join(_partes), fontsize=14.5, color="#C9D6EA", va="center")
    ax.plot([5, 90], [86.4, 86.4], color=DEST, lw=5, solid_capstyle="butt")
    ax.add_patch(Rectangle((5, 79.6), 2.1, 2.6, fc=LIGHT, ec=LINE, lw=1.6)); ax.add_patch(Rectangle((7.1, 79.6), 2.1, 2.6, fc=AMAR, ec=LINE, lw=1.6)); ax.text(10.6, 80.9, "Intervención Amulén", fontsize=13, color=ARENA, va="center")
    ax.add_patch(Rectangle((40, 79.6), 4.2, 2.6, fc=OLD_F, ec=OLD_E, lw=1.6))
    ax.text(45.6, 80.9, "Infraestructura existente", fontsize=13, color=ARENA, va="center")

ncol = cfg.get("ncol_etapas") or (4 if len(etapas) <= 8 else 5)
filas = math.ceil(len(etapas) / ncol)
paso_y = 6.6 if filas <= 2 else 5.2
for i, (t, dt) in enumerate(etapas):
    c, fila = i % ncol, i // ncol
    xx, yy = 5 + c * (W - 10) / ncol, 11.6 + (paso_y if filas > 2 else 0) - fila * paso_y
    num(xx + 1.8, yy, i + 1)
    ax.text(xx + 5, yy + 0.2, t, fontsize=14 * TE, fontweight="bold", color=ARENA, va="bottom", zorder=21)
    if dt:
        ax.text(xx + 5, yy - 0.3, dt, fontsize=11.5 * TE, color="#C9D6EA", va="top", zorder=21)

if not SD:
    def logo(ruta, x1, alto, yc):
        im = Image.open(ruta).convert("RGBA")
        im = im.crop(im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox())
        ancho = alto * im.width / im.height
        ax.imshow(np.asarray(im), extent=[x1 - ancho, x1, yc - alto / 2, yc + alto / 2], aspect="auto", zorder=30, interpolation="lanczos")
        return ancho
    if cfg.get("logo_proveedor") == "coihuin":
        _lg = Image.open(AQUI / "assets" / "logos" / "logo_coihuin.png").convert("RGBA")
        _ar = np.asarray(_lg).astype(float)
        _sat = _ar[..., :3].max(axis=2) - _ar[..., :3].min(axis=2)
        _gris = _sat < 40                                                   # el texto gris del logo se oscurece (se veía lavado)
        _ar[..., :3][_gris] = np.clip((_ar[..., :3][_gris] - 70) * 1.25, 0, 255)
        _lg = Image.fromarray(_ar.astype("uint8"))
        _al = 5.6; _an = _al * _lg.width / _lg.height
        rbox(W - 5 - _an - 4, 79.4, _an + 4, _al + 4.0, 1.4, "white", 29, ec="none", lw=0)
        ax.imshow(np.asarray(_lg), extent=[W - 5 - _an - 2, W - 7, 81.2, 81.2 + _al], aspect="auto", zorder=30, interpolation="lanczos")
        ax.text(W - 5, 91.6, "Planta purificadora de", fontsize=12, color=ARENA, ha="right", va="center", zorder=30)
    a = logo(LOGOS / "Amulén" / "Amulén horizontal blanco.png", W - 5, 7.5, 100)
    prog = LOGOS / ("Construye Redes/Construye Redes blanco.png" if ES_SSR else "Agua para Aprender/APA blanco (para fondos azules u oscuros).png")
    logo(prog, W - 5 - a - 7, 9.5, 100)

ax.set_xlim(0, W); ax.set_ylim(0, H)
from matplotlib.colors import to_rgba
_oe = tuple(round(v, 3) for v in to_rgba(OLD_E))
_ln = tuple(round(v, 3) for v in to_rgba(LINE))
for _pt in ax.patches:
    _ec = tuple(round(v, 3) for v in _pt.get_edgecolor())
    if _ec == _ln and _pt.get_linewidth() >= 1.8:                      # lo nuevo: contorno más grueso
        _pt.set_linewidth(_pt.get_linewidth() * 1.2)
if SD:                                                                  # recorta el hueco del título: solo el dibujo y las etapas
    _top = cfg.get("alto_dibujo", 100.0)
    ax.set_ylim(0, _top)
    fig.set_size_inches(W / 10, _top / 10)
fig.savefig(out, dpi=cfg.get("dpi", 200), facecolor=OSC)
print("ok", out, "· etapas:", ", ".join(n for n, _ in etapas))
