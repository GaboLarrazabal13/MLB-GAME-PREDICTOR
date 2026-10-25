"""
Vistas del dashboard para la salud del modelo y la simulación de playoffs.

Leen primero la API (/stats/salud, /simulacion/playoffs) y, si no responde, calculan o leen lo
mismo en local. Usan las clases y colores del tema (src/assets/theme.css).
"""

import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

NARANJA, AMBAR, GRIS, VERDE, ROJO, TEXTO_3 = "#ff6a00", "#ffb000", "#7a7a7a", "#3ddc84", "#ff3b3b", "#7a7a7a"
ESTADOS = {
    "ok": ("success", "Modelo sano"),
    "vigilar": ("warning", "Vigilar"),
    "alerta": ("warning", "Alerta: el modelo rinde por debajo de la referencia"),
    "sin_datos": ("info", "Sin datos suficientes"),
}


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def _tarjeta(titulo, valor, descripcion, destacado=False):
    clase = "value-blue" if destacado else "value-purple"
    return (f'<div class="stats-card-mini"><h4>{titulo}</h4><div class="stats-card-val {clase}">{valor}</div>'
            f'<p class="stats-card-desc">{descripcion}</p></div>')


def _layout(fig, alto=320, eje_y_pct=True):
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a3a3a3"), hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%" if eje_y_pct else None),
    )
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def obtener_salud(api_url, dias=30):
    try:
        r = requests.get(f"{api_url}/stats/salud", params={"dias": dias}, timeout=20)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    try:  # sin API: mismo cálculo en local
        from mlb_config import DB_PATH
        from mlb_salud import calcular_salud

        with sqlite3.connect(DB_PATH) as conn:
            return calcular_salud(conn, ventana_dias=dias)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def obtener_simulacion(api_url):
    try:
        r = requests.get(f"{api_url}/simulacion/playoffs", timeout=20)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    try:
        from mlb_simulacion import SIMULACION_PATH

        if os.path.exists(SIMULACION_PATH):
            with open(SIMULACION_PATH, encoding="utf-8") as f:
                return json.load(f)
    except Exception:  # noqa: BLE001
        pass
    return None


def render_salud_modelo(api_url, render_alerta):
    """Sección 'Salud del modelo': estado, acierto reciente frente a referencias y evolución semanal."""
    st.markdown("### Salud del modelo")
    dias = st.select_slider("Ventana", options=[7, 14, 30, 60, 90], value=30, format_func=lambda d: f"Últimos {d} días", key="salud_dias")
    s = obtener_salud(api_url, dias)
    if not s:
        render_alerta("info", "Salud del modelo no disponible", "No se pudo consultar la API ni la base de datos local.")
        return
    tipo, titulo = ESTADOS.get(s.get("estado"), ("info", s.get("estado", "")))
    render_alerta(tipo, titulo, f"{s.get('motivo', '')}. Solo cuentan predicciones publicadas antes de cada partido.")

    v, t = s.get("ventana") or {}, s.get("temporada") or {}
    cols = st.columns(4)
    tarjetas = [
        ("Acierto del modelo", _pct(v.get("accuracy")), f"{v.get('n', 0)} partidos en la ventana", True),
        ("Elegir al de mejor récord", _pct(v.get("mejor_record")), "referencia simple a superar", False),
        ("Favorito de las casas", _pct(v.get("mercado")),
         (f"en {v['n_con_mercado']} partidos con cuota (modelo: {_pct(v.get('modelo_en_partidos_con_mercado'))})"
          if v.get("n_con_mercado") else "aparece cuando haya predicciones con cuota capturada"), False),
        ("Temporada completa", _pct(t.get("accuracy")), f"{t.get('n', 0)} partidos · mejor récord {_pct(t.get('mejor_record'))}", False),
    ]
    for col, (tit, val, desc, dest) in zip(cols, tarjetas):
        col.markdown(_tarjeta(tit, val, desc, dest), unsafe_allow_html=True)

    semanas = pd.DataFrame(s.get("semanas") or [])
    if semanas.empty:
        return
    semanas = semanas[semanas["n"] >= 10]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semanas["semana"], y=semanas["accuracy"], name="Modelo", mode="lines+markers",
                             line=dict(color=NARANJA, width=3), marker=dict(size=7), customdata=semanas[["n"]],
                             hovertemplate="%{y:.1%} · %{customdata[0]} partidos<extra>Modelo</extra>"))
    fig.add_trace(go.Scatter(x=semanas["semana"], y=semanas["mejor_record"], name="Mejor récord", mode="lines",
                             line=dict(color=GRIS, width=2, dash="dot"), hovertemplate="%{y:.1%}<extra>Mejor récord</extra>"))
    if "mercado" in semanas and semanas["mercado"].notna().any():
        fig.add_trace(go.Scatter(x=semanas["semana"], y=semanas["mercado"], name="Casas de apuestas", mode="lines",
                                 line=dict(color=AMBAR, width=2), hovertemplate="%{y:.1%}<extra>Casas</extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
    st.plotly_chart(_layout(fig), width="stretch")
    st.caption("Acierto por semana (semanas con al menos 10 partidos). La alerta salta si el modelo queda por debajo de 'mejor récord' tres semanas seguidas.")


def render_simulacion_playoffs(api_url, render_alerta, logo_html=None):
    """Página 'Playoffs': probabilidad de cada equipo de clasificar, ganar división, liga y Serie Mundial."""
    st.markdown('<div class="main-title">Simulación de Playoffs</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">10,000 simulaciones de lo que queda de temporada y de la postemporada completa</div>', unsafe_allow_html=True)
    sim = obtener_simulacion(api_url)
    if not sim or not sim.get("equipos"):
        render_alerta("info", "Todavía no hay simulación", "Se genera cada mañana tras actualizar los resultados.")
        return
    eq = pd.DataFrame(sim["equipos"])
    pendientes = sim.get("partidos_pendientes", 0)
    estado = (f"Quedan {pendientes} partidos de temporada regular." if pendientes
              else "La temporada regular terminó: la simulación usa la clasificación final.")
    st.caption(f"Temporada {sim.get('temporada')} · generada {str(sim.get('generado', ''))[:16].replace('T', ' ')} UTC · {estado} "
               "La fuerza de cada equipo es su ELO (con ventaja de localía).")

    top = eq.sort_values("campeon", ascending=False).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["campeon"], y=top["equipo"], orientation="h", marker=dict(color=NARANJA, cornerradius=4),
        text=[f"{v:.1%}" for v in top["campeon"]], textposition="outside", textfont=dict(color="#f2f2f2"),
        customdata=top[["liga", "playoffs"]], hovertemplate="<b>%{y}</b><br>Campeón: %{x:.1%}<br>Gana su liga: %{customdata[0]:.1%}<br>Playoffs: %{customdata[1]:.1%}<extra></extra>",
    ))
    st.markdown("#### Favoritos a ganar la Serie Mundial")
    st.plotly_chart(_layout(fig, alto=380).update_layout(hovermode="closest", xaxis=dict(tickformat=".0%")), width="stretch")

    st.markdown("#### Por división")
    st.markdown(
        "<style>.sim-tabla{width:100%;border-collapse:collapse;font-size:0.8rem}"
        ".sim-tabla th,.sim-tabla td{padding:4px 6px!important;border:none!important;white-space:nowrap;background:transparent!important}"
        ".sim-tabla th{color:#7a7a7a;font-weight:600;text-align:right}.sim-tabla th:first-child,.sim-tabla td:first-child{text-align:left}"
        ".sim-tabla td{text-align:right;color:#a3a3a3}.sim-tabla tr+tr td{border-top:1px solid #1a1a1a!important}"
        ".sim-tabla img{width:18px!important;margin-right:6px!important}</style>",
        unsafe_allow_html=True,
    )
    divisiones = list(dict.fromkeys(eq.sort_values("division_nombre")["division_nombre"]))
    for i in range(0, len(divisiones), 2):
        cols = st.columns(2)
        for col, div in zip(cols, divisiones[i:i + 2]):
            d = eq[eq["division_nombre"] == div].sort_values(["division", "playoffs", "victorias"], ascending=False)
            filas = "".join(
                f"<tr><td>{logo_html(r.equipo, 18) if logo_html else ''}<b style='color:#f2f2f2'>{r.equipo}</b></td>"
                f"<td>{int(r.victorias)}</td><td>{r.victorias_esperadas:.0f}</td><td>{r.playoffs:.0%}</td>"
                f"<td>{r.division:.0%}</td><td style='color:{NARANJA}'>{r.campeon:.1%}</td></tr>"
                for r in d.itertuples()
            )
            col.markdown(
                f"<div class='stats-card-mini'><h4>{div}</h4><table class='sim-tabla'>"
                "<tr><th>Equipo</th><th>V</th><th>V esp.</th><th>Playoffs</th><th>División</th><th>Título</th></tr>"
                f"{filas}</table></div>",
                unsafe_allow_html=True,
            )
    st.caption("V: victorias actuales · V esp.: victorias esperadas al final de la temporada · Div.: probabilidad de ganar la división.")
