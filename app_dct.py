"""
Simulador interactivo de una caja de cambios de doble embrague (DCT) de 6 marchas.

Disposición (caja transversal de dos ejes, simplificada):
    · Doble embrague K1 / K2 unido al volante motor.
    · K1 mueve el eje de entrada interior (macizo), que lleva las marchas impares 1.ª, 3.ª y 5.ª.
    · K2 mueve el eje de entrada exterior (hueco, concéntrico), que lleva 2.ª, 4.ª, 6.ª y R.
    · Un eje de salida paralelo lleva los piñones locos y los sincronizadores
      S₁/₃, S₅, S₄/₆ y S₂/R. La salida va al grupo final y al diferencial.

Idea clave: mientras una semicaja transmite el par con su embrague cerrado, la otra tiene
ya engranada (preseleccionada) la marcha siguiente con su embrague abierto. El cambio
consiste en cruzar los embragues: uno se abre mientras el otro se cierra, sin interrumpir
el par en las ruedas.

Notación:
    z_e,j  piñón fijo del eje de entrada de la marcha j
    z_s,j  piñón loco del eje de salida de la marcha j (lo hace solidario el sincronizador)

    i_j = ω_e / ω_s = z_s,j / z_e,j          i_R = −z_s,R / z_e,R  (el piñón loco solo invierte)

Ejecutar con:   streamlit run app_dct.py
"""

import json
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Caja DCT de 6 marchas", layout="wide")

# ---------------------------------------------------------------------------
#  Datos por defecto
# ---------------------------------------------------------------------------
# Suma de dientes 60 en todas las parejas (mismo módulo y misma distancia entre ejes).
Z0 = {"e_1": 13, "s_1": 47, "e_2": 19, "s_2": 41, "e_3": 25, "s_3": 35,
      "e_4": 29, "s_4": 31, "e_5": 33, "s_5": 27, "e_6": 36, "s_6": 24,
      "e_R": 13, "loco": 21, "s_R": 43}

MARCHAS = ["R", "1", "2", "3", "4", "5", "6"]
ADELANTE = ["1", "2", "3", "4", "5", "6"]
IMPARES = ["1", "3", "5"]
PARES = ["2", "4", "6", "R"]
ETIQ = {"N": "N", "R": "R", "1": "1.ª", "2": "2.ª", "3": "3.ª", "4": "4.ª", "5": "5.ª", "6": "6.ª"}

# Sincronizador y sentido del desplazamiento de su manguito (−1 izquierda, +1 derecha)
SINCRO = {"R": ("SR2", -1), "2": ("SR2", +1), "6": ("S46", -1), "4": ("S46", +1),
          "3": ("S13", -1), "1": ("S13", +1), "5": ("S5", -1)}
NOMBRE_S = {"SR2": "S₂/R", "S46": "S₄/₆", "S13": "S₁/₃", "S5": "S₅"}

COL_MARCHA = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100",
              "5": "#e87ba4", "6": "#008300", "R": "#6250d6"}
COL = {"motor": "#555555", "imp": "#d9822b", "par": "#4f6d9a", "sec": "#1f9e89",
       "loco": "#b9b9b9", "loco_R": "#c9a0dc", "manguito": "#7a7a7a", "on": "#d21e1e",
       "pre": "#e69500"}

# Curva de par a plena carga (motor de gasolina 2,0 L orientativo)
N_CURVA = np.array([800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500])
M_CURVA = np.array([120, 140, 165, 185, 195, 200, 200, 198, 193, 185, 175, 160, 140])

G = 9.81
RPM = 2 * math.pi / 60          # rad/s por rpm


# ---------------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------------
def coma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def cl(x, d=3):
    """Número con coma decimal para LaTeX."""
    return coma(x, d).replace(",", "{,}")


def entero(x):
    x = 0.0 if abs(x) < 0.5 else x
    return f"{x:,.0f}".replace(",", " ")


def embrague_de(m):
    return None if m == "N" else ("K1" if m in IMPARES else "K2")


def eje_de(m):
    return None if m == "N" else ("imp" if m in IMPARES else "par")


def relaciones(z):
    """i_j = ω_entrada / ω_salida (negativa en R)."""
    i = {j: z[f"s_{j}"] / z[f"e_{j}"] for j in ADELANTE}
    i["R"] = -z["s_R"] / z["e_R"]
    return i


def par_motor(n_rpm, escala=1.0):
    """Par a plena carga (N·m) interpolado; cero por encima del régimen máximo de la curva."""
    n = np.asarray(n_rpm, dtype=float)
    m = np.interp(n, N_CURVA, M_CURVA, left=M_CURVA[0], right=0.0) * escala
    return np.where(n > N_CURVA[-1], 0.0, m)


def manguitos(engranadas):
    """Posición de cada manguito para un conjunto de marchas engranadas."""
    m = {s: 0 for s in NOMBRE_S}
    for g in engranadas:
        if g in SINCRO:
            m[SINCRO[g][0]] = SINCRO[g][1]
    return m


# ---------------------------------------------------------------------------
#  Estado
# ---------------------------------------------------------------------------
ss = st.session_state
if "marcha" not in ss:
    ss.marcha = "N"
    ss.anterior = "N"
    ss.presel_actual = ["1", "R"]
    ss.cambio = None
    ss.aviso = None


def pedir(p):
    ss.peticion = p


# ---------------------------------------------------------------------------
#  Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.header("Funcionamiento")
modo = st.sidebar.radio("Se fija", ["Velocidad del vehículo", "Régimen del motor"],
                        help="Con la velocidad fijada, al cambiar de marcha cambia el régimen "
                             "del motor (como en la carretera) y la TCU bloquea los cambios "
                             "imposibles. Con el régimen fijado, cambia la velocidad del vehículo.")
if modo == "Velocidad del vehículo":
    v_kmh = st.sidebar.slider("Velocidad del vehículo (km/h)", 0.0, 250.0, 50.0, 1.0)
    n_fijo = None
else:
    n_fijo = st.sidebar.slider("Régimen del motor (rpm)", 800.0, 7000.0, 2500.0, 50.0)
    v_kmh = None
acel = st.sidebar.slider("Pedal del acelerador (%)", 0, 100, 60, 5) / 100
pend = st.sidebar.slider("Pendiente (%)", -10.0, 30.0, 0.0, 0.5)
modo_presel = st.sidebar.radio(
    "Preselección", ["Automática (predictiva)", "Manual"],
    help="Automática: la TCU prepara la marcha siguiente si se acelera (pedal ≥ 15 %) y la "
         "anterior si se levanta el pie. Manual: eliges tú qué marcha queda preparada en el "
         "eje libre.")

with st.sidebar.expander("Vehículo y motor"):
    m_veh = st.number_input("Masa m (kg)", 600.0, 4000.0, 1450.0, 50.0)
    r_rueda = st.number_input("Radio de la rueda r (m)", 0.20, 0.50, 0.31, 0.005, format="%.3f")
    i_d = st.number_input("Relación del grupo i_d", 2.0, 6.0, 3.95, 0.01)
    eta = st.number_input("Rendimiento de la transmisión η_t", 0.70, 1.00, 0.93, 0.01)
    cd = st.number_input("C_d", 0.15, 0.60, 0.30, 0.01)
    area = st.number_input("Área frontal A (m²)", 1.0, 4.0, 2.2, 0.05)
    f_r = st.number_input("Coef. de rodadura f_r", 0.005, 0.040, 0.013, 0.001, format="%.3f")
    rho = 1.2
    esc_par = st.slider("Escala de la curva de par", 0.5, 2.0, 1.0, 0.05,
                        help="Par máximo de referencia: 200 N·m a 3000–3500 rpm.")
    n_ral = st.number_input("Ralentí (rpm)", 500.0, 1200.0, 800.0, 50.0)
    n_max = st.number_input("Régimen máximo (rpm)", 4000.0, 8000.0, 6500.0, 100.0)

with st.sidebar.expander("Cambio (doble embrague)"):
    J_m = st.number_input("Inercia del motor y del tambor J_m (kg·m²)", 0.05, 0.60, 0.20, 0.01,
                          help="Cigüeñal, volante bimasa y tambor del doble embrague.")
    t_par = st.number_input("Duración de la fase de par t_par (s)", 0.05, 0.60, 0.15, 0.01)
    t_in = st.number_input("Duración de la fase de inercia t_in (s)", 0.05, 0.80, 0.25, 0.01)
    t_sinc = st.number_input("Sincronización (preselección) t_sinc (s)", 0.05, 0.80, 0.25, 0.01)
    interv = st.checkbox("Intervención en el par del motor durante la fase de inercia", True,
                         help="Al subir, la ECU retrasa el encendido y reduce el par; al reducir, "
                              "da un golpe de gas. Así el embrague no tiene que absorber "
                              "la inercia del motor y no hay tirón.")
    t_man = st.number_input("Interrupción del par en una caja manual (s)", 0.2, 1.5, 0.6, 0.05,
                            help="Solo para comparar.")

with st.sidebar.expander("Números de dientes"):
    z = {}
    for j in ADELANTE:
        c1, c2 = st.columns(2)
        z[f"e_{j}"] = int(c1.number_input(f"z_e {ETIQ[j]} (entrada)", 8, 80, Z0[f"e_{j}"], 1,
                                          key=f"ze{j}"))
        z[f"s_{j}"] = int(c2.number_input(f"z_s {ETIQ[j]} (salida)", 8, 80, Z0[f"s_{j}"], 1,
                                          key=f"zs{j}"))
    c1, c2, c3 = st.columns(3)
    z["e_R"] = int(c1.number_input("z_e R", 8, 80, Z0["e_R"], 1, key="zeR"))
    z["loco"] = int(c2.number_input("z loco", 8, 80, Z0["loco"], 1, key="zloco"))
    z["s_R"] = int(c3.number_input("z_s R", 8, 80, Z0["s_R"], 1, key="zsR"))
    st.caption("Impares en el eje interior (K1); pares y R en el eje hueco (K2).")

st.sidebar.header("Animación")
vel_anim = st.sidebar.slider("Velocidad de la animación", 0.1, 3.0, 1.0, 0.1)

I = relaciones(z)
k_v = RPM * r_rueda * 3.6 / i_d          # km/h por rpm del eje de salida
vel_fija = modo == "Velocidad del vehículo"


def presel_auto(m, n_s_abs, v):
    """Marcha que la TCU deja preparada en el eje libre."""
    if m == "N":
        return ["1", "R"] if v < 5 else []
    if m == "R":
        return ["1"]
    k = int(m)
    arriba = str(k + 1) if k < 6 else None
    abajo = str(k - 1) if k > 1 else None
    for c in ([arriba, abajo] if acel >= 0.15 else [abajo, arriba]):
        if c and n_s_abs * I[c] <= n_max:
            return [c]
    return []


def elegir_puente(a, g):
    """Marcha del otro eje entre a y g, la más próxima a g."""
    lo, hi = sorted([int(a), int(g)])
    cands = [str(k) for k in range(lo + 1, hi) if eje_de(str(k)) != eje_de(a)]
    return min(cands, key=lambda c: abs(int(c) - int(g)))


# ---------------------------------------------------------------------------
#  Petición de cambio (la TCU decide si es posible y cómo se hace)
# ---------------------------------------------------------------------------
m0 = ss.marcha
if vel_fija:
    v_antes = v_kmh
else:
    v_antes = 0.0 if m0 == "N" else n_fijo / abs(I[m0]) * k_v
n_s_antes = v_antes / k_v


def procesar(p):
    a = ss.marcha
    if p == "+":
        g = "1" if a == "N" else (str(int(a) + 1) if a in ADELANTE and a != "6" else None)
    elif p == "-":
        g = str(int(a) - 1) if a in ADELANTE and a != "1" else None
    else:
        g = p
    if g is None or g == a:
        return
    if vel_fija:
        if g == "R" and a != "R" and v_antes > 5:
            ss.aviso = ("error", f"La TCU no engrana la marcha atrás con el vehículo a "
                                 f"{entero(v_antes)} km/h hacia delante.")
            return
        if a == "R" and g in ADELANTE and v_antes > 5:
            ss.aviso = ("error", "La TCU no engrana una marcha adelante con el vehículo "
                                 "retrocediendo: hay que pararlo antes.")
            return
        if g in ADELANTE:
            n_nuevo = n_s_antes * I[g]
            if n_nuevo > n_max:
                ss.aviso = ("error", f"Cambio rechazado por la TCU: en {ETIQ[g]} el motor iría a "
                                     f"{entero(n_nuevo)} rpm (> {entero(n_max)} rpm).")
                return
            if a in ADELANTE and int(g) > int(a) and n_nuevo < n_ral + 200 and v_antes > 5:
                ss.aviso = ("warning", f"Cambio rechazado por la TCU: en {ETIQ[g]} el motor "
                                       f"caería a {entero(n_nuevo)} rpm, demasiado cerca del "
                                       "ralentí.")
                return
    presel_antes = list(ss.presel_actual)
    puente = None
    if g == "N":
        tipo = "a_N"
    elif a == "N":
        tipo = "desde_N"
    elif "R" in (a, g) or (vel_fija and v_antes <= 5 and embrague_de(a) == embrague_de(g)):
        tipo = "parado"
    elif embrague_de(a) != embrague_de(g):
        tipo = "preseleccionada" if g in presel_antes else "sincronizar"
    else:
        tipo = "puente"
        puente = elegir_puente(a, g)
    ss.cambio = {"de": a, "a": g, "tipo": tipo, "puente": puente,
                 "presel_antes": presel_antes, "v": v_antes}
    ss.anterior = a
    ss.marcha = g
    ss.animar = True
    ss.reset_presel = True


pet = ss.pop("peticion", None)
if pet is not None:
    procesar(pet)

marcha = ss.marcha
anterior = ss.anterior

# ---------------------------------------------------------------------------
#  Cinemática
# ---------------------------------------------------------------------------
patina = False
if vel_fija:
    n_s = (-v_kmh if marcha == "R" else v_kmh) / k_v        # salida, sentido de marcha
    if marcha == "N":
        n_mot = n_ral
    else:
        n_rig = n_s * I[marcha]
        patina = n_rig < n_ral
        n_mot = max(n_rig, n_ral)
else:
    n_mot = n_fijo
    n_s = 0.0 if marcha == "N" else n_fijo / I[marcha]
    v_kmh = abs(n_s) * k_v

auto = presel_auto(marcha, abs(n_s), v_kmh)

# ---------------------------------------------------------------------------
#  Cabecera y selector
# ---------------------------------------------------------------------------
st.title("Caja de cambios de doble embrague (DCT) de 6 marchas")
st.caption("Dos semicajas: K1 con las marchas impares y K2 con las pares y R. Elige la marcha: "
           "si estaba preseleccionada, el cambio es un simple cruce de embragues sin "
           "interrupción del par.")

col_pal, col_res = st.columns([1, 2.2])
with col_pal:
    st.markdown("**Selector**")

    def boton(col, m, etiqueta=None):
        col.button(etiqueta or ETIQ[m], key="b_" + m, on_click=pedir, args=(m,),
                   type="primary" if marcha == m else "secondary", width="stretch")

    fila = st.columns(2)
    boton(fila[0], "R")
    boton(fila[1], "N")
    fila = st.columns(6)
    for c, m in zip(fila, ADELANTE):
        boton(c, m, m)
    fila = st.columns(2)
    fila[0].button("− Reducir", key="b_menos", on_click=pedir, args=("-",), width="stretch")
    fila[1].button("+ Subir", key="b_mas", on_click=pedir, args=("+",), width="stretch")

    if modo_presel == "Manual" and marcha != "N":
        otro = PARES if eje_de(marcha) == "imp" else IMPARES
        opciones = ["ninguna"] + [g for g in otro
                                  if abs(n_s) * abs(I[g]) <= n_max and (g != "R" or v_kmh < 5)]
        defecto = auto[0] if auto and auto[0] in opciones else "ninguna"
        if ss.pop("reset_presel", False) or ss.get("presel_manual") not in opciones:
            ss.presel_manual = defecto
        sel = st.selectbox(f"Marcha preparada en el eje de {'K2' if otro is PARES else 'K1'}",
                           opciones, key="presel_manual",
                           format_func=lambda g: "ninguna" if g == "ninguna" else ETIQ[g],
                           help="Solo aparecen las que no harían girar el eje por encima del "
                                "régimen máximo.")
        presel = [] if sel == "ninguna" else [sel]
    else:
        ss.pop("reset_presel", None)
        presel = auto
        if modo_presel == "Manual":
            st.caption("En punto muerto la TCU deja preparadas 1.ª y R.")
    st.caption("1.ª, 3.ª, 5.ª → K1 · 2.ª, 4.ª, 6.ª, R → K2")

ss.presel_actual = presel

# Velocidades físicas (signo +: sentido de giro del motor)
w = {"motor": n_mot, "sal": -n_s}
libre = {}
for e, gs in (("imp", IMPARES), ("par", PARES)):
    eng = [g for g in gs if g == marcha or g in presel]
    w[e] = n_s * I[eng[0]] if eng else 0.0
    libre[e] = not eng
for j in MARCHAS:
    w["l" + j] = -w[eje_de(j)] / I[j]
w["loco"] = -w["par"] * z["e_R"] / z["loco"]

# Fuerzas
v_ms = v_kmh / 3.6
theta = math.atan(pend / 100)
R_a = 0.5 * rho * cd * area * v_ms ** 2
R_rl = f_r * m_veh * G * math.cos(theta)
R_g = m_veh * G * math.sin(theta)
if marcha == "N":
    M_m, F_t, i_tot = 0.0, 0.0, 0.0
else:
    i_tot = abs(I[marcha]) * i_d
    M_m = float(par_motor(n_mot, esc_par)) * acel
    F_t = M_m * i_tot * eta / r_rueda
gamma = 1.04 + 0.0025 * i_tot ** 2
a_veh = (F_t - R_a - R_rl - R_g) / (gamma * m_veh) if marcha not in ("R", "N") else None

with col_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Marcha", ETIQ[marcha],
              None if anterior == marcha else f"antes {ETIQ[anterior]}", delta_color="off")
    c2.metric("Preparada", ", ".join(ETIQ[g] for g in presel) or "—")
    c3.metric("Embrague cerrado", "ninguno" if marcha == "N"
              else embrague_de(marcha) + (" (desliza)" if patina else ""))
    c4.metric("Relación i", "—" if marcha == "N" else coma(I[marcha]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Motor", f"{entero(n_mot)} rpm")
    c2.metric("Vehículo (marcha atrás)" if marcha == "R" else "Vehículo", f"{entero(v_kmh)} km/h")
    c3.metric("Fuerza de tracción F", f"{entero(F_t)} N")
    c4.metric("Aceleración", "—" if a_veh is None else f"{coma(a_veh, 2)} m/s²")

    aviso = ss.pop("aviso", None)
    if aviso:
        (st.error if aviso[0] == "error" else st.warning)(aviso[1])
    c = ss.cambio
    if c and c["a"] == marcha and c["de"] == anterior and not aviso:
        de, a_ = ETIQ[c["de"]], ETIQ[c["a"]]
        txt = {
            "preseleccionada": lambda: f"{de} → {a_}: la marcha ya estaba preseleccionada. Cambio por "
                               f"cruce de embragues {embrague_de(c['de'])} → "
                               f"{embrague_de(c['a'])} sin interrumpir el par.",
            "sincronizar": lambda: f"{de} → {a_}: la marcha no estaba preparada ("
                           + (f"lo estaba {', '.join(ETIQ[g] for g in c['presel_antes'])}). "
                              if c["presel_antes"] else "no había ninguna preparada). ")
                           + 
                           f"Primero el sincronizador {NOMBRE_S[SINCRO[c['a']][0]]} la engrana "
                           "en el eje libre y después se cruzan los embragues: sin interrupción "
                           "del par, pero con más retraso.",
            "puente": lambda: f"{de} → {a_}: las dos marchas están en el mismo eje "
                      f"({embrague_de(c['de'])}), no se pueden cruzar embragues directamente. "
                      f"La TCU pasa por {ETIQ[c['puente'] or '1']} (marcha puente del otro eje) "
                      f"mientras el sincronizador cambia de {de} a {a_}.",
            "desde_N": lambda: f"N → {a_}: se cierra {embrague_de(c['a'])} de forma progresiva"
                       + (" (arranque)." if c["v"] < 5 else "."),
            "a_N": lambda: f"{de} → N: se abren los dos embragues.",
            "parado": lambda: f"{de} → {a_}: con el vehículo parado se abre el embrague, "
                      "el sincronizador engrana la nueva marcha y se cierra "
                      f"{embrague_de(c['a'])} de forma progresiva.",
        }[c["tipo"]]()
        st.info(txt)
    if marcha != "N" and vel_fija:
        if n_mot > n_max:
            st.error(f"Motor pasado de vueltas ({entero(n_mot)} rpm): en modo automático la "
                     "TCU ya habría subido de marcha.")
        elif patina:
            st.warning(f"A esta velocidad el motor iría a {entero(n_s * I[marcha])} rpm, por "
                       f"debajo del ralentí: {embrague_de(marcha)} desliza y el motor se queda "
                       f"a {entero(n_ral)} rpm (no se cala, como en un arranque)."
                       + ("" if marcha in ("1", "2", "R") else " La TCU reduciría de marcha."))
    if marcha == "N":
        st.info("Punto muerto: los dos embragues abiertos. Parado, la TCU deja 1.ª (K1) y R (K2) "
                "engranadas a la vez: están en ejes distintos, así que no hay bloqueo.")

# ---------------------------------------------------------------------------
#  Esquema animado
# ---------------------------------------------------------------------------
cambio = ss.cambio
animar = bool(ss.pop("animar", False))
eng_fin = ([marcha] if marcha != "N" else []) + presel
m_fin = manguitos(eng_fin)
p_fin = {"K1": 1.0 if embrague_de(marcha) == "K1" else 0.0,
         "K2": 1.0 if embrague_de(marcha) == "K2" else 0.0}
if cambio and cambio["a"] == marcha:
    de_c, a_c = cambio["de"], cambio["a"]
    eng_ant = ([de_c] if de_c != "N" else []) + cambio["presel_antes"]
    m_ant = manguitos(eng_ant)
    if a_c == "N":
        m_mid = m_ant
    else:
        m_mid = manguitos([x for x in eng_ant if eje_de(x) != eje_de(a_c)] + [a_c])
    p_ant = {"K1": 1.0 if embrague_de(de_c) == "K1" else 0.0,
             "K2": 1.0 if embrague_de(de_c) == "K2" else 0.0}
    k_de, k_a = embrague_de(de_c) or "—", embrague_de(a_c) or "—"
    if cambio["tipo"] == "preseleccionada":
        f1 = f"1 · {ETIQ[a_c]} ya estaba preseleccionada"
    elif a_c == "N":
        f1 = "1 · Punto muerto"
    else:
        f1 = f"1 · Sincronizador {NOMBRE_S[SINCRO[a_c][0]]}: engrana {ETIQ[a_c]}"
    f2 = f"2 · Cruce de embragues {k_de} → {k_a}" + (
        f" (vía {ETIQ[cambio['puente']]})" if cambio["puente"] else "")
    f3 = "3 · Nueva preselección: " + (", ".join(ETIQ[g] for g in presel) or "ninguna")
    txt_fases = [f1, f2, f3]
    hay_cambio = True
else:
    de_c = marcha
    m_ant = m_mid = m_fin
    p_ant = p_fin
    txt_fases = ["", "", ""]
    hay_cambio = False
    animar = False

param = {
    "z": z, "w": w, "marcha": marcha, "de": de_c, "etiq": ETIQ[marcha],
    "i": "—" if marcha == "N" else coma(I[marcha]),
    "emb": embrague_de(marcha) or "ninguno", "patina": patina,
    "pre_txt": ", ".join(ETIQ[g] for g in presel) or "ninguna",
    "m_ant": m_ant, "m_mid": m_mid, "m_fin": m_fin, "p_ant": p_ant, "p_fin": p_fin,
    "txt": txt_fases, "animar": animar, "hay_cambio": hay_cambio, "anim": vel_anim,
    "libre": libre, "col": COL, "colm": COL_MARCHA,
}

HTML = r"""
<div style="font-family:sans-serif">
  <canvas id="c" width="1200" height="500" style="width:100%;max-width:1200px"></canvas>
  <div style="margin-top:2px">
    <button id="bp">Pausa</button> <button id="br">Repetir el cambio</button>
  </div>
</div>
<script>
const P = __PARAM__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, TAU = 2 * Math.PI;
const C = P.col, CM = P.colm, z = P.z, w = P.w;
const YP = 165, A = 150, YS = YP + A;            // ejes de entrada y de salida
const GW = 24;
const fmt = x => (Math.abs(x) < 0.5 ? 0 : x).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const X = {k1: 70, k2: 110, R: 205, SR2: 258, '2': 310, '6': 400, S46: 455, '4': 510, hueco: 555,
           '3': 615, S13: 670, '1': 725, '5': 815, S5: 870, finInt: 905};
const EJE = {'1': 'imp', '3': 'imp', '5': 'imp', '2': 'par', '4': 'par', '6': 'par', 'R': 'par'};
const SG = {SR2: {'-1': 'R', '1': '2'}, S46: {'-1': '6', '1': '4'},
            S13: {'-1': '3', '1': '1'}, S5: {'-1': '5'}};
const NOMS = {SR2: 'S2/R', S46: 'S4/6', S13: 'S1/3', S5: 'S5'};
const LADO = {R: +1, '2': -1, '6': +1, '4': -1, '3': +1, '1': -1, '5': +1};
const rad = (ze, zs) => [A * ze / (ze + zs), A * zs / (ze + zs)];
const R = {};
['1', '2', '3', '4', '5', '6'].forEach(j => R[j] = rad(z['e_' + j], z['s_' + j]));
const mod = 2 * A / (z.e_1 + z.s_1);
R.R = [mod * z.e_R / 2, mod * z.s_R / 2];
const T1 = 0.8, T2 = 1.6, T3 = 2.4;

let t = 0, run = true, last = null, anim = P.animar;
document.getElementById('bp').onclick = e => { run = !run; e.target.textContent = run ? 'Pausa' : 'Seguir'; };
document.getElementById('br').onclick = () => { t = 0; anim = P.hay_cambio; };
const K = TAU * 0.5 / 1000;                      // 1000 rpm → 0,5 vueltas/s en pantalla

function text(s, x, y, o = {}) {
  ctx.save(); ctx.font = (o.bold ? 'bold ' : '') + (o.size || 12) + 'px sans-serif';
  ctx.textAlign = o.align || 'center'; ctx.textBaseline = 'middle';
  ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,0.92)'; ctx.strokeText(s, x, y);
  ctx.fillStyle = o.color || '#222'; ctx.fillText(s, x, y); ctx.restore();
}
function rueda(x, yc, r, zz, n, color, o = {}) {
  const ww = o.w || GW;
  ctx.save(); ctx.globalAlpha = o.alpha || 1;
  ctx.fillStyle = color; ctx.fillRect(x - ww / 2, yc - r, ww, 2 * r);
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  if (o.dash) ctx.setLineDash([4, 3]);
  ctx.strokeRect(x - ww / 2, yc - r, ww, 2 * r); ctx.setLineDash([]);
  const ph = K * n * t;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.2;
  for (let k = 0; k < zz; k++) {
    const a = ph + k * TAU / zz;
    if (Math.cos(a) > 0.05) {
      const y = yc - r * Math.sin(a);
      ctx.beginPath(); ctx.moveTo(x - ww / 2 + 1, y); ctx.lineTo(x + ww / 2 - 1, y); ctx.stroke();
    }
  }
  ctx.restore();
  if (o.loco) {
    ctx.fillStyle = 'white'; ctx.fillRect(x - ww / 2 + 3, yc - 10, ww - 6, 20);
    ctx.fillStyle = '#333';
    [-1, 1].forEach(s => { ctx.beginPath(); ctx.arc(x, yc + s * 7, 2.2, 0, TAU); ctx.fill(); });
  }
}
function garras(x, yc, lado, color) {
  ctx.fillStyle = color; ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  const x0 = lado > 0 ? x + GW / 2 : x - GW / 2 - 8;
  ctx.fillRect(x0, yc - 22, 8, 44); ctx.strokeRect(x0, yc - 22, 8, 44);
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  for (let k = -3; k <= 3; k++) { ctx.beginPath(); ctx.moveTo(x0 + 1, yc + 6 * k); ctx.lineTo(x0 + 7, yc + 6 * k); ctx.stroke(); }
}
function eje(x0, x1, y, color, h) { ctx.fillStyle = color; ctx.fillRect(x0, y - h / 2, x1 - x0, h); }
function manguito(x, pos, nombre, estado) {
  const dx = pos * 16;
  const col = estado === 'on' ? C.on : (estado === 'pre' ? C.pre : C.manguito);
  ctx.fillStyle = '#888'; ctx.fillRect(x - 9, YS - 15, 18, 30);
  ctx.fillStyle = col; ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
  ctx.fillRect(x + dx - 11, YS - 28, 22, 56); ctx.strokeRect(x + dx - 11, YS - 28, 22, 56);
  ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fillRect(x + dx - 11, YS - 3, 22, 6);
  ctx.strokeStyle = '#555'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x + dx, YS + 28); ctx.lineTo(x + dx, H - 56); ctx.stroke();
  ctx.fillStyle = '#555'; ctx.fillRect(x + dx - 10, H - 58, 20, 8);
  text(nombre, x + dx, H - 40, {size: 12, bold: estado !== '',
       color: estado === 'on' ? C.on : (estado === 'pre' ? '#a06a00' : '#555')});
}
function disco(x, h, col, p, nom, arriba) {
  const g = (1 - p) * 5;
  ctx.fillStyle = '#9a9a9a';                       // platos de acero (unidos al tambor)
  ctx.fillRect(x - 12 - g, YP - h - 6, 5, 2 * h + 12); ctx.fillRect(x + 7 + g, YP - h - 6, 5, 2 * h + 12);
  ctx.save(); ctx.globalAlpha = 0.3 + 0.7 * p; ctx.fillStyle = col;
  ctx.fillRect(x - 5, YP - h, 10, 2 * h); ctx.restore();
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.strokeRect(x - 5, YP - h, 10, 2 * h);
  text(nom, x, arriba ? YP - h - 14 : YP + h + 15, {size: 12, bold: true, color: col});
}
function barra(y, nom, p, col) {
  text(nom, 10, y, {align: 'left', bold: true, color: col});
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.strokeRect(36, y - 7, 70, 14);
  ctx.fillStyle = col; ctx.fillRect(36, y - 7, 70 * p, 14);
  text(Math.round(100 * p) + ' %', 112, y, {align: 'left', size: 11, color: '#555'});
}
function linea(pts, color, o = {}) {
  ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = o.alpha || 0.55; ctx.lineWidth = o.lw || 7;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round'; if (o.dash) ctx.setLineDash(o.dash);
  ctx.beginPath(); pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
  ctx.restore();
  if (o.punta) {
    const [xa, ya] = pts[pts.length - 2], [xb, yb] = pts[pts.length - 1];
    const a = Math.atan2(yb - ya, xb - xa);
    ctx.save(); ctx.globalAlpha = 0.8; ctx.fillStyle = color; ctx.beginPath();
    ctx.moveTo(xb + 6 * Math.cos(a), yb + 6 * Math.sin(a));
    ctx.lineTo(xb - 12 * Math.cos(a) + 9 * Math.sin(a), yb - 12 * Math.sin(a) - 9 * Math.cos(a));
    ctx.lineTo(xb - 12 * Math.cos(a) - 9 * Math.sin(a), yb - 12 * Math.sin(a) + 9 * Math.cos(a));
    ctx.closePath(); ctx.fill(); ctx.restore();
  }
}
function camino(j) {
  const xk = EJE[j] === 'imp' ? X.k1 : X.k2, x = X[j];
  if (j === 'R') {
    const yi = (YP + R.R[0] + YS - R.R[1]) / 2;
    return [[xk, YP], [x, YP], [x - 30, yi], [x, YS], [W - 20, YS]];
  }
  return [[xk, YP], [x, YP], [x, YS], [W - 20, YS]];
}
function interp(Am, Bm, s) {
  const o = {};
  for (const k in Bm) {
    const p0 = Am[k], p1 = Bm[k];
    o[k] = p0 === p1 ? p1 : (s < 0.5 ? p0 * (1 - 2 * s) : p1 * (2 * s - 1));
  }
  return o;
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
  // estado de la animación
  let mang = P.m_fin, p = P.p_fin, fase = '', potencia = P.marcha;
  if (anim && t < T3) {
    if (t < T1) { mang = interp(P.m_ant, P.m_mid, t / T1); p = P.p_ant; fase = P.txt[0]; potencia = P.de; }
    else if (t < T2) {
      const s = (t - T1) / (T2 - T1);
      mang = P.m_mid; fase = P.txt[1]; potencia = s < 0.5 ? P.de : P.marcha;
      p = {K1: P.p_ant.K1 * (1 - s) + P.p_fin.K1 * s, K2: P.p_ant.K2 * (1 - s) + P.p_fin.K2 * s};
    } else { mang = interp(P.m_mid, P.m_fin, (t - T2) / (T3 - T2)); fase = P.txt[2]; }
  }
  const engr = {};                                  // marcha engranada por cada manguito
  for (const k in mang) { const g = SG[k][String(mang[k])]; if (g) engr[g] = k; }
  const estadoM = g => !(g in engr) ? '' : (g === potencia ? 'on' : 'pre');

  // carcasa
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 2; ctx.setLineDash([8, 5]);
  ctx.strokeRect(150, 52, 835, H - 120); ctx.setLineDash([]);

  // ejes
  eje(0, 38, YP, C.motor, 12);
  eje(X.k2, X.hueco, YP, C.par, 20);              // eje hueco (pares)
  eje(X.k1, X.finInt, YP, C.imp, 7);              // eje interior (impares)
  eje(180, W, YS, C.sec, 10);                     // eje de salida

  // doble embrague
  ctx.fillStyle = 'rgba(85,85,85,0.08)'; ctx.fillRect(38, YP - 98, 100, 196);
  ctx.strokeStyle = C.motor; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(138, YP - 98); ctx.lineTo(38, YP - 98); ctx.lineTo(38, YP + 98);
  ctx.lineTo(138, YP + 98); ctx.stroke();
  disco(X.k1, 72, C.imp, p.K1, 'K1', true);
  disco(X.k2, 58, C.par, p.K2, 'K2', false);
  barra(24, 'K1', p.K1, C.imp);
  barra(44, 'K2', p.K2, C.par);
  text('motor', 2, YP - 20, {size: 11, bold: true, color: C.motor, align: 'left'});

  // marchas: piñón fijo arriba (entrada), piñón loco abajo (salida)
  ['R', '2', '6', '4', '3', '1', '5'].forEach(j => {
    const x = X[j], e = EJE[j], st = estadoM(j);
    const colE = e === 'imp' ? C.imp : C.par;
    const colL = st === '' ? (j === 'R' ? C.loco_R : C.loco) : CM[j];
    const o = {loco: true, alpha: st === 'pre' ? 0.6 : 1, dash: st === 'pre'};
    if (j === 'R') {
      rueda(x, YP, R.R[0], z.e_R, w.par, colE);
      rueda(x, YS, R.R[1], z.s_R, w.lR, colL, o);
      const y0 = YP + R.R[0] - 6, y1 = YS - R.R[1] + 6;
      rueda(x - 30, (y0 + y1) / 2, (y1 - y0) / 2, z.loco, w.loco, '#d9c3e6', {dash: true, w: 18});
      text('loco', x - 58, (y0 + y1) / 2, {size: 11, color: '#555'});
    } else {
      rueda(x, YP, R[j][0], z['e_' + j], w[e], colE);
      rueda(x, YS, R[j][1], z['s_' + j], w['l' + j], colL, o);
    }
    garras(x, YS, LADO[j], st === '' ? '#9a9a9a' : CM[j]);
    const rr = j === 'R' ? R.R[0] : R[j][0];
    text(j === 'R' ? 'R' : j + '.ª', x, YP - rr - 12,
         {size: 13, bold: true, color: st === 'on' ? '#111' : (st === 'pre' ? '#a06a00' : '#666')});
  });

  // manguitos
  for (const k in mang) {
    const g = SG[k][String(mang[k])];
    manguito(X[k], mang[k], NOMS[k], g ? estadoM(g) : '');
  }

  // flujo de potencia (rojo) y marcha preparada (ámbar)
  for (const g in engr) {
    if (g !== potencia) linea(camino(g).slice(0, -1), C.pre, {dash: [10, 7], lw: 5, alpha: 0.75});
  }
  linea([[5, YP], [X.k1 - 14, YP]], potencia === 'N' ? '#999' : C.on, {punta: potencia === 'N'});
  if (potencia !== 'N') linea([[5, YP]].concat(camino(potencia)), C.on, {punta: true});

  // rótulos
  text('motor: ' + fmt(w.motor) + ' rpm', 8, YP + 118, {size: 12, bold: true, color: C.motor, align: 'left'});
  text('K1 · eje interior: ' + (P.libre.imp ? 'libre' : fmt(w.imp) + ' rpm'), 8, YP + 136,
       {size: 12, bold: true, color: C.imp, align: 'left'});
  text('K2 · eje hueco: ' + (P.libre.par ? 'libre' : fmt(w.par) + ' rpm'), 8, YP + 154,
       {size: 12, bold: true, color: C.par, align: 'left'});
  text('eje de salida ' + fmt(w.sal) + ' rpm', W - 8, YS - 20, {size: 12, bold: true, color: C.sec, align: 'right'});
  text('→ grupo y diferencial', W - 8, YS + 22, {size: 11, color: '#777', align: 'right'});
  text('Marcha ' + P.etiq, 1000, 70, {size: 22, bold: true, align: 'left'});
  text('i = ' + P.i, 1000, 100, {size: 16, bold: true, align: 'left', color: C.on});
  text('Embrague: ' + P.emb + (P.patina ? ' (desliza)' : ''), 1000, 126, {size: 13, align: 'left'});
  text('Preparada: ' + P.pre_txt, 1000, 148, {size: 13, bold: true, align: 'left', color: '#a06a00'});
  text('Rojo: flujo de potencia', 1000, 196, {size: 11, align: 'left', color: C.on});
  text('Ámbar: marcha preseleccionada', 1000, 212, {size: 11, align: 'left', color: '#a06a00'});
  text('(engranada, embrague abierto)', 1000, 228, {size: 11, align: 'left', color: '#777'});
  if (fase) text(fase, 570, 24, {size: 16, bold: true, color: '#333'});
}
function loop(ts) {
  if (last !== null && run) t += (ts - last) / 1000 * P.anim;
  last = ts; draw(); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script>
"""


def incrustar_html(html, alto):
    """st.iframe en versiones recientes de Streamlit; components.html en las antiguas."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=alto)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=alto)


incrustar_html(HTML.replace("__PARAM__", json.dumps(param)), 545)

# ---------------------------------------------------------------------------
#  Pestañas de resultados
# ---------------------------------------------------------------------------
tab_rel, tab_cambio, tab_trac, tab_sim = st.tabs(
    ["Relaciones y velocidades", "Cambio de marcha (cruce de embragues)",
     "Diagramas de tracción", "Aceleración: DCT frente a manual"])

TEMA = dict(labelExpr="replace(datum.label, ',', ' ')", grid=True, gridColor="#e6e6e6",
            domainColor="#999", tickColor="#999", labelColor="#52514e", titleColor="#52514e")

# --- Relaciones y velocidades ----------------------------------------------
with tab_rel:
    c_izq, c_der = st.columns([1, 1.25])
    with c_izq:
        st.subheader("Relación de la marcha")
        if marcha == "N":
            st.info("Punto muerto: no hay relación entre la entrada y la salida.")
        else:
            j = marcha
            sg = "-" if j == "R" else ""
            st.latex(rf"i_{{{j}}} = \frac{{\omega_{{e}}}}{{\omega_{{s}}}} = "
                     rf"{sg}\frac{{z_{{s,{j}}}}}{{z_{{e,{j}}}}} = "
                     rf"{sg}\frac{{{z['s_' + j]}}}{{{z['e_' + j]}}} = \mathbf{{{cl(I[j])}}}")
            if j == "R":
                st.caption(f"El piñón loco (z = {z['loco']}) invierte el giro pero no cambia "
                           "la relación.")
            st.latex(rf"i_{{total}} = |i_{{{j}}}|\, i_d = {cl(abs(I[j]))}\cdot{cl(i_d, 2)} = "
                     rf"{cl(i_tot)}")
            st.markdown(f"Transmite **{embrague_de(j)}** "
                        f"({'eje interior, marchas impares' if j in IMPARES else 'eje hueco, marchas pares y R'}).")
        if presel:
            st.markdown("Preparada: " + ", ".join(
                f"**{ETIQ[g]}** en el eje de {embrague_de(g)}" for g in presel)
                + ". Su eje gira arrastrado por la salida, con el embrague abierto.")
    with c_der:
        st.subheader("Velocidades de giro (rpm)")
        filas = [("Motor y tambor del doble embrague", "", w["motor"], "", ""),
                 ("Eje interior (K1)", "", w["imp"], "libre" if libre["imp"] else "", ""),
                 ("Eje hueco (K2)", "", w["par"], "libre" if libre["par"] else "", "")]
        for j in ["1", "3", "5", "2", "4", "6", "R"]:
            est = "transmite" if j == marcha else ("preparada" if j in presel else "loco")
            unido = est != "loco"
            dn = "0" if unido else (
                "" if libre[eje_de(j)] else entero(w["l" + j] - w["sal"]))
            filas.append((f"Piñón {ETIQ[j]} del eje de salida", str(z["s_" + j]),
                          0.0 if libre[eje_de(j)] and not unido else w["l" + j],
                          est, dn))
        filas.append(("Piñón loco de R", str(z["loco"]), w["loco"], "", ""))
        filas.append(("Eje de salida", "", w["sal"], "", ""))
        st.dataframe([{"Elemento": a, "z": b, "n (rpm)": entero(c), "Estado": d,
                       "Δn con la salida": e} for a, b, c, d, e in filas],
                     hide_index=True, width="stretch")
        d1 = n_mot - w["imp"]
        d2 = n_mot - w["par"]
        st.markdown(f"Deslizamiento en **K1**: {entero(d1)} rpm "
                    f"({'cerrado' if marcha in IMPARES and not patina else 'abierto' if marcha not in IMPARES else 'desliza'}) · "
                    f"en **K2**: {entero(d2)} rpm "
                    f"({'cerrado' if marcha in PARES and not patina else 'abierto' if marcha not in PARES else 'desliza'})")
        st.caption("Signo +: sentido de giro del motor. Con ejes de entrada y salida paralelos, "
                   "el eje de salida gira al revés que el motor en las marchas adelante; el "
                   "grupo final lo vuelve a invertir. El embrague abierto soporta una diferencia "
                   "de velocidad permanente (pérdidas por arrastre, sobre todo en baño de aceite).")

    st.subheader("Todas las marchas")
    tabla, prev = [], None
    for j in ["1", "2", "3", "4", "5", "6", "R"]:
        ij = I[j]
        tabla.append({
            "Marcha": ETIQ[j], "z_e / z_s": f"{z['e_' + j]} / {z['s_' + j]}", "i": coma(ij),
            "i·i_d": coma(abs(ij) * i_d),
            "Salto": coma(prev / ij, 2) if prev and j != "R" else "",
            "Embrague": embrague_de(j), "Sincronizador": NOMBRE_S[SINCRO[j][0]],
            "v a n_máx (km/h)": entero(n_max / abs(ij) * k_v),
            "F máx (N)": entero(M_CURVA.max() * esc_par * abs(ij) * i_d * eta / r_rueda),
            "": "◀" if j == marcha else ("preparada" if j in presel else "")})
        prev = ij
    st.dataframe(tabla, hide_index=True, width="stretch")
    sumas = {j: z["e_" + j] + z["s_" + j] for j in ADELANTE}
    distintas = [ETIQ[j] for j, s in sumas.items() if s != sumas["1"]]
    msg = (f"Apertura de la caja i₁/i₆ = {coma(I['1'] / I['6'], 2)}. Las marchas consecutivas "
           "están siempre en ejes distintos: por eso cualquier cambio de una marcha a la "
           "contigua se hace cruzando embragues. ")
    if distintas:
        msg += (f"Con el mismo módulo todas las parejas deberían sumar {sumas['1']} dientes; no lo "
                f"cumplen: {', '.join(distintas)} (se corrige con el ángulo de hélice o con "
                "desplazamiento de perfil).")
    st.caption(msg)


# --- Cambio de marcha --------------------------------------------------------
def simular_cambio(etapas, n_sal, previo):
    """Cruce de embragues a velocidad del vehículo constante.

    etapas: lista de (de, a, sincronizar). previo: rpm inicial del eje de la marcha
    de llegada de la primera etapa (antes de sincronizarla).
    """
    dt = 0.002
    kF = i_d * eta / r_rueda
    filas, fases = [], []
    E = {"K1": 0.0, "K2": 0.0}
    t = [0.0]
    de0 = etapas[0][0]
    k0 = embrague_de(de0)
    otro = "K2" if k0 == "K1" else "K1"
    n_eje = {k0: n_sal * I[de0], otro: previo}
    marcha_eje = {k0: de0, otro: None}

    def M_de(n):
        return float(par_motor(max(n, n_ral), esc_par)) * acel

    def anota(n_e, M_e, T, fase):
        for Kk in ("K1", "K2"):
            E[Kk] += T[Kk] * abs(n_e - n_eje[Kk]) * RPM * dt
        F = sum(T[Kk] * (I[marcha_eje[Kk]] if marcha_eje[Kk] else 0.0) for Kk in T) * kF
        filas.append({"t (s)": t[0], "motor": n_e, "eje interior (K1)": n_eje["K1"],
                      "eje hueco (K2)": n_eje["K2"], "K1": T["K1"], "K2": T["K2"],
                      "par del motor M_m": M_e, "DCT": F, "fase": fase})
        t[0] += dt

    def tramo(dur, f):
        n = max(1, int(round(dur / dt)))
        t0 = t[0]
        for k in range(n):
            f(k / n)
        return t0

    def fijo(g, Kg):
        n_e = n_sal * I[g]
        M = M_de(n_e)
        T = {"K1": 0.0, "K2": 0.0}
        T[Kg] = M
        anota(n_e, M, T, "")

    tramo(0.15, lambda s: fijo(de0, k0))
    for de, a, sinc in etapas:
        Kd, Ka = embrague_de(de), embrague_de(a)
        n_d, n_a = n_sal * I[de], n_sal * I[a]
        if sinc:
            prev = n_eje[Ka]

            def f_sinc(s, prev=prev, n_a=n_a, de=de, Kd=Kd, Ka=Ka):
                n_eje[Ka] = prev + (n_a - prev) * s
                fijo(de, Kd)
            t0 = tramo(t_sinc, f_sinc)
            fases.append((t0, t[0], "sincronización"))
        n_eje[Ka] = n_a
        marcha_eje[Ka] = a
        alfa = (n_a - n_d) * RPM / t_in            # aceleración angular del motor (rad/s²)
        sube = n_a < n_d

        def f_par(s, n_d=n_d, n_a=n_a, sube=sube, Kd=Kd, Ka=Ka):
            n_e = n_d if sube else n_a
            M = M_de(n_e)
            anota(n_e, M, {Kd: M * (1 - s), Ka: M * s}, "fase de par")

        def f_inercia(s, n_d=n_d, n_a=n_a, alfa=alfa, sube=sube, Kd=Kd, Ka=Ka):
            n_e = n_d + (n_a - n_d) * s
            M = M_de(n_e)
            Mm = max(M + J_m * alfa, 0.0) if interv else M
            T_emb = max(Mm - J_m * alfa, 0.0)
            T = {Kd: 0.0, Ka: T_emb} if sube else {Kd: T_emb, Ka: 0.0}
            anota(n_e, Mm, T, "fase de inercia")

        orden = [("fase de par", t_par, f_par), ("fase de inercia", t_in, f_inercia)]
        for nombre, dur, f in (orden if sube else orden[::-1]):
            t0 = tramo(dur, f)
            fases.append((t0, t[0], nombre))
        marcha_eje[Kd] = de                        # la marcha anterior sigue engranada
    g_fin = etapas[-1][1]
    k_fin = embrague_de(g_fin)
    t_post = max(0.25, 0.15 + t_man + 0.25 - t[0])
    tramo(t_post, lambda s: fijo(g_fin, k_fin))

    df = pd.DataFrame(filas)
    F0 = M_de(n_sal * I[de0]) * I[de0] * kF
    F1 = M_de(n_sal * I[g_fin]) * I[g_fin] * kF
    tt = df["t (s)"]
    df["manual"] = np.where(tt < 0.15, F0, np.where(tt < 0.15 + t_man, 0.0, F1))
    return df, fases, E


COL_FASE = {"sincronización": "#f2dfa7", "fase de par": "#cfe0f3", "fase de inercia": "#f4cccc"}


def grafica_t(df, series, colores, titulo_y, fases, leyenda_fases, discontinuas=()):
    largo = df.melt(id_vars="t (s)", value_vars=series, var_name="serie", value_name="valor")
    bandas = alt.Chart(pd.DataFrame(fases, columns=["t0", "t1", "fase"])).mark_rect(
        opacity=0.55).encode(
        x=alt.X("t0:Q", title="t (s)"), x2="t1:Q",
        color=alt.Color("fase:N", scale=alt.Scale(domain=list(COL_FASE),
                                                  range=list(COL_FASE.values())),
                        legend=alt.Legend(orient="top", title=None) if leyenda_fases else None))
    lineas = alt.Chart(largo).mark_line(strokeWidth=2.3).encode(
        x=alt.X("t (s):Q", title="t (s)"), y=alt.Y("valor:Q", title=titulo_y),
        color=alt.Color("serie:N", scale=alt.Scale(domain=series, range=colores),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.FieldOneOfPredicate("serie", list(discontinuas)),
                                 alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("t (s):Q", format=".3f"), "serie",
                 alt.Tooltip("valor:Q", format=".0f")])
    return (alt.layer(bandas, lineas).resolve_scale(color="independent")
            .properties(height=250).configure_axis(**TEMA).configure_view(stroke=None))


with tab_cambio:
    c = ss.cambio
    if not c or c["de"] not in ADELANTE or c["a"] not in ADELANTE or c["tipo"] == "parado":
        st.info("Haz un cambio entre dos marchas adelante con el vehículo en movimiento (por "
                "ejemplo 2.ª → 3.ª, 4.ª → 3.ª o 3.ª → 5.ª) para ver cómo se cruzan los "
                "embragues. El cambio se analiza a velocidad del vehículo constante: dura "
                "tan poco que el coche apenas cambia de velocidad.")
    else:
        a, g = c["de"], c["a"]
        n_sal_c = c["v"] / k_v
        ka, kg = embrague_de(a), embrague_de(g)
        previas = [x for x in c["presel_antes"] if embrague_de(x) == kg]
        previo = n_sal_c * I[previas[0]] if previas else 0.0
        if c["tipo"] == "puente":
            b = c["puente"]
            etapas = [(a, b, b not in c["presel_antes"]), (b, g, True)]
        else:
            etapas = [(a, g, c["tipo"] == "sincronizar")]
        df_c, fases, E = simular_cambio(etapas, n_sal_c, previo)
        sube = I[g] < I[a]
        st.subheader(f"Último cambio: {ETIQ[a]} → {ETIQ[g]} "
                     f"({'subida' if sube else 'reducción'}) a {entero(c['v'])} km/h")
        if c["tipo"] == "puente":
            st.markdown(f"{ETIQ[a]} y {ETIQ[g]} comparten embrague ({ka}). Se hace en dos "
                        f"cruces: {ETIQ[a]} → {ETIQ[c['puente']]} y, mientras el sincronizador "
                        f"cambia de marcha en el eje libre, {ETIQ[c['puente']]} → {ETIQ[g]}.")

        F_antes = df_c["DCT"].iloc[0]
        F_desp = df_c["DCT"].iloc[-1]
        dur_dct = sum(f[1] - f[0] for f in fases)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duración del cambio DCT", f"{coma(dur_dct, 2)} s")
        c2.metric("F mínima durante el cambio",
                  f"{entero(df_c['DCT'].min())} N"
                  + (f" ({coma(100 * df_c['DCT'].min() / F_antes, 0)} %)" if F_antes > 0 else ""),
                  help="Entre paréntesis, respecto de la fuerza antes del cambio. En la caja "
                       "manual es 0 durante la interrupción.")
        c3.metric("Energía disipada en K1", f"{coma(E['K1'] / 1000, 1)} kJ")
        c4.metric("Energía disipada en K2", f"{coma(E['K2'] / 1000, 1)} kJ")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Régimen del motor y de los ejes de entrada**")
            st.altair_chart(grafica_t(df_c, ["motor", "eje interior (K1)", "eje hueco (K2)"],
                                      [COL["motor"], COL["imp"], COL["par"]], "n (rpm)",
                                      fases, True), width="stretch")
        with g2:
            st.markdown("**Par transmitido por cada embrague**")
            st.altair_chart(grafica_t(df_c, ["K1", "K2", "par del motor M_m"],
                                      [COL["imp"], COL["par"], "#333333"], "M (N·m)",
                                      fases, True, discontinuas=["par del motor M_m"]),
                            width="stretch")
        st.markdown("**Fuerza de tracción en las ruedas: DCT frente a caja manual**")
        st.altair_chart(grafica_t(df_c, ["DCT", "manual"], [COL["on"], "#8a8a8a"], "F (N)",
                                  fases, False, discontinuas=["manual"]), width="stretch")

        st.markdown("##### Qué ocurre en cada fase")
        st.markdown(
            "- **Sincronización** (solo si la marcha no estaba preparada): el sincronizador "
            "iguala la velocidad del eje libre con la del piñón de la nueva marcha y la engrana. "
            "El par sigue pasando por el embrague cerrado.\n"
            "- **Fase de par**: el embrague saliente se abre mientras el entrante se cierra; la "
            "suma de sus pares es el par del motor y el régimen no cambia. La fuerza en la rueda "
            "pasa de la de la marcha anterior a la de la nueva.\n"
            "- **Fase de inercia**: el embrague que transmite desliza hasta que el motor alcanza "
            "el régimen de la nueva marcha. "
            + ("Al subir, el motor tiene que frenarse." if sube else
               "Al reducir, el motor tiene que acelerarse; por eso en las reducciones con gas "
               "esta fase va **antes** que la de par."))
        Kn = kg if sube else ka
        st.latex(r"\text{Fase de par:}\quad M_{K,ant} + M_{K,nue} = M_m,\qquad "
                 r"F = \frac{(M_{K,ant}\, i_{ant} + M_{K,nue}\, i_{nue})\, i_d\, \eta_t}{r}")
        st.latex(rf"\text{{Fase de inercia:}}\quad M_{{{Kn}}} = M_m - J_m\,\dot\omega_m,"
                 rf"\qquad \dot\omega_m = \frac{{\omega_{{nue}} - \omega_{{ant}}}}{{t_{{in}}}}"
                 rf" = {cl((n_sal_c * (I[g] - I[a])) * RPM / t_in, 0)}\ \text{{rad/s}}^2")
        if c["tipo"] == "puente":
            st.caption("En el cambio con marcha puente la última fórmula se aplica a cada uno de "
                       "los dos cruces.")
        st.caption(
            ("Con intervención en el par, la ECU varía M_m durante la fase de inercia para que el "
             "embrague no tenga que absorber J_m·ω̇_m: la fuerza en la rueda no presenta pico. "
             if interv else
             "Sin intervención en el par, el embrague tiene que absorber también J_m·ω̇_m: "
             + ("pico de fuerza al subir (tirón)." if sube else "hueco de fuerza al reducir.")
             + " ")
            + f"La caja manual interrumpe el par durante {coma(t_man, 2)} s.")

# --- Diagramas de tracción ---------------------------------------------------
dom_m = [ETIQ[j] for j in ADELANTE]
esc_col = alt.Scale(domain=dom_m, range=[COL_MARCHA[j] for j in ADELANTE])

with tab_trac:
    v_max_graf = max(60.0, min(280.0, n_max / I["6"] * k_v * 1.05))
    filas_f, filas_n = [], []
    for j in ADELANTE:
        vj_min, vj_max = n_ral / I[j] * k_v, n_max / I[j] * k_v
        for v in np.linspace(vj_min, vj_max, 60):
            n_m = min(v / k_v * I[j], n_max)
            F = float(par_motor(n_m, esc_par)) * acel * I[j] * i_d * eta / r_rueda
            filas_f.append({"v (km/h)": v, "F (N)": F, "Marcha": ETIQ[j], "n (rpm)": n_m})
        filas_n += [{"v (km/h)": vj_min, "n (rpm)": n_ral, "Marcha": ETIQ[j]},
                    {"v (km/h)": vj_max, "n (rpm)": n_max, "Marcha": ETIQ[j]}]
    df_f, df_n = pd.DataFrame(filas_f), pd.DataFrame(filas_n)
    vv = np.linspace(0, v_max_graf, 80)
    df_r = pd.DataFrame({"v (km/h)": vv,
                         "R (N)": 0.5 * rho * cd * area * (vv / 3.6) ** 2 + R_rl + R_g})

    sel_m = ETIQ[marcha] if marcha in ADELANTE else None
    opac = alt.condition(alt.datum.Marcha == sel_m, alt.value(1.0), alt.value(0.45)) if sel_m \
        else alt.value(1.0)
    ancho = alt.condition(alt.datum.Marcha == sel_m, alt.value(3.5), alt.value(2)) if sel_m \
        else alt.value(2)
    ley = alt.Legend(orient="bottom", direction="horizontal")

    lin_f = alt.Chart(df_f).mark_line().encode(
        x=alt.X("v (km/h):Q", scale=alt.Scale(domain=[0, v_max_graf])),
        y=alt.Y("F (N):Q", title="F, R (N)"),
        color=alt.Color("Marcha:N", scale=esc_col, legend=ley), opacity=opac, strokeWidth=ancho,
        tooltip=["Marcha", alt.Tooltip("v (km/h):Q", format=".0f"),
                 alt.Tooltip("F (N):Q", format=".0f"), alt.Tooltip("n (rpm):Q", format=".0f")])
    lin_r = alt.Chart(df_r).mark_line(color="#333", strokeDash=[6, 4], strokeWidth=2).encode(
        x="v (km/h):Q", y="R (N):Q",
        tooltip=[alt.Tooltip("v (km/h):Q", format=".0f"), alt.Tooltip("R (N):Q", format=".0f")])
    et_r = alt.Chart(df_r.tail(1)).mark_text(align="right", dx=-4, dy=-10, color="#333").encode(
        x="v (km/h):Q", y="R (N):Q", text=alt.value("resistencias"))
    capas = [lin_f, lin_r, et_r]
    capas_n = [alt.Chart(df_n).mark_line(strokeWidth=2).encode(
        x=alt.X("v (km/h):Q", scale=alt.Scale(domain=[0, v_max_graf])),
        y=alt.Y("n (rpm):Q", scale=alt.Scale(domain=[0, n_max * 1.08])),
        color=alt.Color("Marcha:N", scale=esc_col, legend=ley), opacity=opac, strokeWidth=ancho,
        tooltip=["Marcha", alt.Tooltip("v (km/h):Q", format=".0f"),
                 alt.Tooltip("n (rpm):Q", format=".0f")])]
    for nn, txt in [(n_max, "régimen máximo"), (n_ral, "ralentí")]:
        capas_n.append(alt.Chart(pd.DataFrame({"n (rpm)": [nn]})).mark_rule(
            color="#999", strokeDash=[4, 4]).encode(y="n (rpm):Q"))
        capas_n.append(alt.Chart(pd.DataFrame({"n (rpm)": [nn], "v (km/h)": [v_max_graf]}))
                       .mark_text(align="right", dy=-8, color="#52514e")
                       .encode(x="v (km/h):Q", y="n (rpm):Q", text=alt.value(txt)))
    if sel_m:
        pto = pd.DataFrame({"v (km/h)": [v_kmh], "F (N)": [F_t], "n (rpm)": [n_mot],
                            "Marcha": [sel_m]})
        for lista, ycol in ((capas, "F (N)"), (capas_n, "n (rpm)")):
            lista.append(alt.Chart(pto).mark_point(size=140, filled=True, color="#111",
                                                   stroke="white", strokeWidth=2).encode(
                x="v (km/h):Q", y=f"{ycol}:Q",
                tooltip=["Marcha", alt.Tooltip(f"{ycol}:Q", format=".0f")]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fuerza de tracción y resistencias al avance**")
        st.altair_chart(alt.layer(*capas).properties(height=380).configure_axis(**TEMA)
                        .configure_view(stroke=None), width="stretch")
        st.caption(f"Curvas con el {int(acel * 100)} % de acelerador y pendiente "
                   f"{coma(pend, 1)} %. Punto negro: estado actual.")
    with c2:
        st.markdown("**Diagrama de velocidades (diente de sierra)**")
        st.altair_chart(alt.layer(*capas_n).properties(height=380).configure_axis(**TEMA)
                        .configure_view(stroke=None), width="stretch")
        st.caption("n = v · i_j · i_d / r. En la DCT la bajada de régimen al subir de marcha "
                   "no es vertical: dura la fase de inercia.")
    if sel_m and a_veh is not None:
        st.latex(rf"F = \frac{{M_m\, i_{{{marcha}}}\, i_d\, \eta_t}}{{r}} = "
                 rf"\frac{{{cl(M_m, 0)}\cdot {cl(I[marcha])}\cdot {cl(i_d, 2)}\cdot {cl(eta, 2)}}}"
                 rf"{{{cl(r_rueda)}}} = {cl(F_t, 0)}\ \text{{N}}"
                 r"\qquad a = \frac{F - R_a - R_{rl} - R_g}{\gamma\, m} = "
                 rf"\frac{{{cl(F_t, 0)} - {cl(R_a, 0)} - {cl(R_rl, 0)} - {cl(R_g, 0)}}}"
                 rf"{{{cl(gamma)}\cdot {cl(m_veh, 0)}}} = {cl(a_veh, 2)}\ \text{{m/s}}^2")
        st.caption("γ = 1,04 + 0,0025·(i_j·i_d)²: factor de masas rotativas.")

# --- Aceleración: DCT frente a manual ---------------------------------------
with tab_sim:
    st.markdown("Arranque desde parado a plena carga con las dos cajas y las mismas relaciones. "
                "En la manual cada cambio interrumpe el par; en la DCT el par pasa de un "
                "embrague a otro (fase de par) y después el embrague entrante desliza hasta "
                "sincronizar el motor (fase de inercia).")
    c1, c2, c3, c4 = st.columns(4)
    usadas = c1.multiselect("Marchas que se usan", ADELANTE, default=ADELANTE,
                            format_func=lambda m: ETIQ[m])
    n_cambio = c2.slider("Régimen de cambio (rpm)", 2000.0, float(n_max), min(6000.0, n_max), 100.0)
    n_salida = c3.slider("Régimen de salida (rpm)", 1200.0, 5000.0, 2500.0, 100.0)
    t_fin = c4.slider("Duración de la simulación (s)", 10.0, 90.0, 30.0, 5.0)

    usadas = sorted(usadas, key=lambda m: -I[m])
    if not usadas:
        st.warning("Elige al menos una marcha.")
    else:
        def simular(dct):
            dt = 0.005
            v, t, k = 0.0, 0.0, 0
            fase, t_ini, n_ini = "salida", 0.0, 0.0
            reg = []
            while t <= t_fin:
                j = usadas[k]
                n_sal = v / k_v
                n_rig = n_sal * I[j]
                i_ef = I[j]
                if fase == "salida":
                    n_m = max(n_salida, n_rig)
                    if n_rig >= n_salida:
                        fase = "engranada"
                    M_t = float(par_motor(n_m, esc_par))
                elif fase == "cambio" and not dct:
                    s = (t - t_ini) / t_man
                    n_m = n_ini + (n_rig - n_ini) * min(1.0, s)
                    M_t = 0.0
                    if s >= 1.0:
                        fase = "engranada"
                elif fase == "cambio":
                    jp = usadas[k - 1]
                    tau = t - t_ini
                    if tau < t_par:                       # fase de par
                        s = tau / t_par
                        n_m = n_sal * I[jp]
                        M = float(par_motor(n_m, esc_par))
                        M_t, i_ef = M, (1 - s) * I[jp] + s * I[j]
                        n_ini = n_m
                    else:                                 # fase de inercia
                        s = (tau - t_par) / t_in
                        n_m = n_ini + (n_rig - n_ini) * min(1.0, s)
                        alfa = (n_rig - n_ini) * RPM / t_in
                        M = float(par_motor(n_m, esc_par))
                        M_t = M if interv else M - J_m * alfa
                        if s >= 1.0:
                            fase = "engranada"
                else:
                    n_m = n_rig
                    M_t = float(par_motor(n_m, esc_par)) if n_m < n_max else 0.0
                    if n_m >= n_cambio and k < len(usadas) - 1:
                        fase, t_ini, n_ini = "cambio", t, n_m
                        k += 1
                        continue
                i_t = i_ef * i_d
                F = M_t * i_t * eta / r_rueda
                vms = v / 3.6
                Rt = 0.5 * rho * cd * area * vms ** 2 + R_rl + R_g
                gam = 1.04 + 0.0025 * (I[usadas[k]] * i_d) ** 2
                a_ = (F - Rt) / (gam * m_veh)
                reg.append((t, v, n_m, F, "cambio" if fase == "cambio" else ETIQ[usadas[k]]))
                v = max(0.0, v + a_ * dt * 3.6)
                t += dt
            caja = "DCT" if dct else "manual"
            df = pd.DataFrame(reg, columns=["t (s)", "v (km/h)", "n (rpm)", "F (N)", "Marcha"])
            df["Caja"] = caja
            return df

        df_d, df_m = simular(True), simular(False)

        def t_a(df, vv):
            s = df.loc[df["v (km/h)"] >= vv, "t (s)"]
            return float(s.iloc[0]) if len(s) else None

        c1, c2, c3, c4 = st.columns(4)
        td, tm = t_a(df_d, 100), t_a(df_m, 100)
        c1.metric("0–100 km/h DCT", f"{coma(td, 2)} s" if td else "no llega")
        c2.metric("0–100 km/h manual", f"{coma(tm, 2)} s" if tm else "no llega")
        c3.metric("Ventaja de la DCT", f"{coma(tm - td, 2)} s" if td and tm else "—")
        c4.metric(f"Velocidad a los {entero(t_fin)} s (DCT / manual)",
                  f"{entero(df_d['v (km/h)'].iloc[-1])} / {entero(df_m['v (km/h)'].iloc[-1])} km/h")

        dfa = pd.concat([df_d.iloc[::4], df_m.iloc[::4]])
        esc_caja = alt.Scale(domain=["DCT", "manual"], range=[COL["on"], "#8a8a8a"])
        base = alt.Chart(dfa).encode(
            x=alt.X("t (s):Q"),
            color=alt.Color("Caja:N", scale=esc_caja, legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.condition(alt.datum.Caja == "manual", alt.value([6, 4]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("t (s):Q", format=".2f"), "Caja", "Marcha",
                     alt.Tooltip("v (km/h):Q", format=".1f"), alt.Tooltip("n (rpm):Q", format=".0f"),
                     alt.Tooltip("F (N):Q", format=".0f")])
        g_v = base.mark_line(strokeWidth=2.3).encode(y="v (km/h):Q").properties(height=280)
        g_n = base.mark_line(strokeWidth=2.3).encode(
            y=alt.Y("n (rpm):Q", scale=alt.Scale(domain=[0, n_max * 1.05]))).properties(height=280)
        t_zoom = min(t_fin, (td or t_fin) + 2)
        g_f = alt.Chart(dfa[dfa["t (s)"] <= t_zoom]).mark_line(strokeWidth=2.3).encode(
            x=alt.X("t (s):Q"), y="F (N):Q",
            color=alt.Color("Caja:N", scale=esc_caja, legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.condition(alt.datum.Caja == "manual", alt.value([6, 4]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("t (s):Q", format=".2f"), "Caja", "Marcha",
                     alt.Tooltip("F (N):Q", format=".0f")]).properties(height=280)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Velocidad del vehículo**")
            st.altair_chart(g_v.configure_axis(**TEMA).configure_view(stroke=None), width="stretch")
        with c2:
            st.markdown("**Régimen del motor**")
            st.altair_chart(g_n.configure_axis(**TEMA).configure_view(stroke=None), width="stretch")
        st.markdown("**Fuerza de tracción** (hasta poco después de los 100 km/h)")
        st.altair_chart(g_f.configure_axis(**TEMA).configure_view(stroke=None), width="stretch")
        st.caption(f"Manual: {coma(t_man, 2)} s sin par en cada cambio. DCT: fase de par "
                   f"{coma(t_par, 2)} s + fase de inercia {coma(t_in, 2)} s, siempre con par. "
                   "Prueba a quitar la intervención en el par del motor o a cambiar a 4500 rpm.")
