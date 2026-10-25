"""
Seguimiento de partidos con la MLB Stats API y explicación de las predicciones.

Este módulo no depende de Streamlit ni del motor de predicción: solo consulta la
API pública de la MLB (statsapi.mlb.com), normaliza el feed de un partido y
construye textos deterministas (sin modelos de lenguaje) que explican por qué el
modelo eligió a un equipo y, con el partido terminado, qué explica el resultado.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import requests

MLB_API = "https://statsapi.mlb.com/api"
ZONA_ET = ZoneInfo("America/New_York")

# Código interno del proyecto -> id de equipo en la MLB Stats API
TEAM_MLB_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CHW": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KCR": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "ATH": 133, "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136,
    "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120,
}
CODE_BY_MLB_ID = {v: k for k, v in TEAM_MLB_ID.items()}

ORDINALES = {1: "1ª", 2: "2ª", 3: "3ª", 4: "4ª", 5: "5ª", 6: "6ª", 7: "7ª", 8: "8ª", 9: "9ª"}


# ============================================================================
# CONSULTAS A LA MLB STATS API
# ============================================================================


def _get(url, params=None, timeout=20):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def cartelera_del_dia(fecha, timeout=20):
    """Partidos oficiales de la MLB para una fecha (YYYY-MM-DD), con estado y probables."""
    data = _get(
        f"{MLB_API}/v1/schedule",
        params={"sportId": 1, "date": fecha, "hydrate": "probablePitcher"},
        timeout=timeout,
    )
    partidos = []
    for dia in data.get("dates", []):
        for g in dia.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            partidos.append(
                {
                    "game_pk": g["gamePk"],
                    "fecha": g.get("officialDate", fecha),
                    "hora_et": _hora_et(g.get("gameDate")),
                    "estado": _estado(g.get("status", {})),
                    "estado_detalle": g.get("status", {}).get("detailedState", ""),
                    "serie": g.get("seriesDescription") or g.get("gameType", ""),
                    "home_team": CODE_BY_MLB_ID.get(home["team"]["id"], home["team"].get("abbreviation", "?")),
                    "away_team": CODE_BY_MLB_ID.get(away["team"]["id"], away["team"].get("abbreviation", "?")),
                    "home_pitcher": (home.get("probablePitcher") or {}).get("fullName", ""),
                    "away_pitcher": (away.get("probablePitcher") or {}).get("fullName", ""),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                }
            )
    return partidos


def buscar_game_pk(fecha, home_code, away_code, timeout=20):
    """Devuelve el gamePk de la MLB para un partido del proyecto (fecha + códigos de equipo)."""
    home_id = TEAM_MLB_ID.get(home_code)
    away_id = TEAM_MLB_ID.get(away_code)
    if not home_id or not away_id:
        return None
    data = _get(f"{MLB_API}/v1/schedule", params={"sportId": 1, "date": fecha}, timeout=timeout)
    for dia in data.get("dates", []):
        for g in dia.get("games", []):
            if g["teams"]["home"]["team"]["id"] == home_id and g["teams"]["away"]["team"]["id"] == away_id:
                return g["gamePk"]
    return None


def obtener_feed(game_pk, timeout=25):
    """Feed completo (gameData + liveData) de un partido."""
    return _get(f"{MLB_API}/v1.1/game/{game_pk}/feed/live", timeout=timeout)


# ============================================================================
# NORMALIZACIÓN DEL FEED
# ============================================================================


def _hora_et(iso):
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZONA_ET).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def _estado(status):
    abstracto = (status.get("abstractGameState") or "").lower()
    detalle = (status.get("detailedState") or "").lower()
    if abstracto == "live":
        return "en_vivo"
    if abstracto == "final" or "final" in detalle or "completed" in detalle:
        return "final"
    if abstracto == "preview" or "scheduled" in detalle or "pre-game" in detalle or "warmup" in detalle:
        return "previo"
    return "otro"


def _ip_a_float(ip):
    """'5.2' (5 entradas y 2 outs) -> 5.667."""
    try:
        entero, _, dec = str(ip).partition(".")
        return int(entero or 0) + int(dec or 0) / 3
    except ValueError:
        return 0.0


def _float_a_ip(valor):
    """5.667 -> '5.2' (entradas completas y outs), notación habitual del béisbol."""
    entero = int(valor)
    outs = int(round((valor - entero) * 3))
    if outs == 3:
        entero, outs = entero + 1, 0
    return f"{entero}.{outs}"


def _linea_lanzador(jugador):
    st = (jugador.get("stats") or {}).get("pitching") or {}
    return {
        "nombre": jugador.get("person", {}).get("fullName", ""),
        "ip": st.get("inningsPitched", "0.0"),
        "h": st.get("hits", 0),
        "r": st.get("runs", 0),
        "er": st.get("earnedRuns", 0),
        "bb": st.get("baseOnBalls", 0),
        "k": st.get("strikeOuts", 0),
        "hr": st.get("homeRuns", 0),
        "lanzamientos": st.get("pitchesThrown") or st.get("numberOfPitches", 0),
    }


def _box_equipo(box_team):
    jugadores = box_team.get("players", {})
    bateo = (box_team.get("teamStats") or {}).get("batting") or {}
    pitcheo = (box_team.get("teamStats") or {}).get("pitching") or {}
    ids_lanzadores = box_team.get("pitchers", [])
    lanzadores = [_linea_lanzador(jugadores.get(f"ID{pid}", {})) for pid in ids_lanzadores]
    abridor = lanzadores[0] if lanzadores else None
    relevistas = lanzadores[1:]

    destacados = []
    for bid in box_team.get("batters", []):
        j = jugadores.get(f"ID{bid}", {})
        st = (j.get("stats") or {}).get("batting") or {}
        if not st or st.get("atBats", 0) == 0 and st.get("baseOnBalls", 0) == 0:
            continue
        destacados.append(
            {
                "nombre": j.get("person", {}).get("fullName", ""),
                "h": st.get("hits", 0),
                "ab": st.get("atBats", 0),
                "rbi": st.get("rbi", 0),
                "hr": st.get("homeRuns", 0),
                "bb": st.get("baseOnBalls", 0),
                "r": st.get("runs", 0),
            }
        )
    destacados.sort(key=lambda d: (d["rbi"], d["hr"], d["h"]), reverse=True)

    return {
        "bateo": {
            "carreras": bateo.get("runs", 0),
            "hits": bateo.get("hits", 0),
            "hr": bateo.get("homeRuns", 0),
            "bb": bateo.get("baseOnBalls", 0),
            "k": bateo.get("strikeOuts", 0),
            "lob": bateo.get("leftOnBase", 0),
        },
        "pitcheo": {
            "ip": pitcheo.get("inningsPitched", "0.0"),
            "hits": pitcheo.get("hits", 0),
            "carreras": pitcheo.get("runs", 0),
            "er": pitcheo.get("earnedRuns", 0),
            "bb": pitcheo.get("baseOnBalls", 0),
            "k": pitcheo.get("strikeOuts", 0),
        },
        "abridor": abridor,
        "relevistas": relevistas,
        "destacados": destacados[:3],
    }


def resumir_feed(feed):
    """Convierte el feed de la MLB en un diccionario compacto y estable para la interfaz."""
    gd = feed.get("gameData", {})
    ld = feed.get("liveData", {})
    status = gd.get("status", {})
    equipos = gd.get("teams", {})
    linescore = ld.get("linescore", {}) or {}
    plays = ld.get("plays", {}) or {}
    box = (ld.get("boxscore") or {}).get("teams", {}) or {}

    def equipo(lado):
        t = equipos.get(lado, {})
        return {"id": t.get("id"), "code": CODE_BY_MLB_ID.get(t.get("id"), t.get("abbreviation", "?")), "nombre": t.get("name", "")}

    home, away = equipo("home"), equipo("away")
    totales = linescore.get("teams", {}) or {}
    marcador = {
        "home": (totales.get("home") or {}).get("runs"),
        "away": (totales.get("away") or {}).get("runs"),
        "hits": {"home": (totales.get("home") or {}).get("hits", 0), "away": (totales.get("away") or {}).get("hits", 0)},
        "errores": {"home": (totales.get("home") or {}).get("errors", 0), "away": (totales.get("away") or {}).get("errors", 0)},
    }

    entradas = [
        {"num": e.get("num"), "away": (e.get("away") or {}).get("runs"), "home": (e.get("home") or {}).get("runs")}
        for e in linescore.get("innings", [])
    ]

    ofensiva = linescore.get("offense", {}) or {}
    defensa = linescore.get("defense", {}) or {}
    current = plays.get("currentPlay") or {}
    all_plays = plays.get("allPlays") or []
    ultima = current.get("result", {}).get("description") or ""
    if not ultima:
        for p in reversed(all_plays):
            if p.get("about", {}).get("isComplete") and p.get("result", {}).get("description"):
                ultima = p["result"]["description"]
                break
    mitad = "alta" if linescore.get("isTopInning", True) else "baja"
    situacion = {
        "entrada": linescore.get("currentInning"),
        "mitad": mitad,
        "estado_entrada": linescore.get("inningState", ""),
        "outs": linescore.get("outs", 0),
        "bolas": linescore.get("balls", 0),
        "strikes": linescore.get("strikes", 0),
        "bases": {
            "primera": bool(ofensiva.get("first")),
            "segunda": bool(ofensiva.get("second")),
            "tercera": bool(ofensiva.get("third")),
        },
        "bateador": (ofensiva.get("batter") or {}).get("fullName", ""),
        "lanzador": (defensa.get("pitcher") or {}).get("fullName", ""),
        "ultima_jugada": ultima,
    }

    jugadas_por_entrada = {}
    jugadas_anotacion = []
    for idx, p in enumerate(all_plays):
        about = p.get("about", {})
        res = p.get("result", {})
        if not about.get("isComplete") or not res.get("description"):
            continue
        num = about.get("inning")
        half = "alta" if about.get("halfInning") == "top" else "baja"
        jugada = {
            "indice": idx,
            "entrada": num,
            "mitad": half,
            "descripcion": res.get("description", ""),
            "evento": res.get("event", ""),
            "rbi": res.get("rbi", 0),
            "away": res.get("awayScore", 0),
            "home": res.get("homeScore", 0),
            "out": bool(res.get("isOut")),
            "anota": bool(about.get("isScoringPlay")),
            "bateador": (p.get("matchup", {}).get("batter") or {}).get("fullName", ""),
            "lanzador": (p.get("matchup", {}).get("pitcher") or {}).get("fullName", ""),
        }
        jugadas_por_entrada.setdefault(num, {"alta": [], "baja": []})[half].append(jugada)
        if jugada["anota"]:
            jugada["equipo"] = "away" if half == "alta" else "home"
            jugadas_anotacion.append(jugada)

    decisiones = ld.get("decisions") or {}
    estado = _estado(status)
    ganador = None
    if estado == "final" and marcador["home"] is not None and marcador["away"] is not None:
        ganador = "home" if marcador["home"] > marcador["away"] else "away"

    box_resumen = {"home": _box_equipo(box.get("home", {})), "away": _box_equipo(box.get("away", {}))}
    for lado in ("home", "away"):
        lob = (totales.get(lado) or {}).get("leftOnBase")
        if lob is not None:
            box_resumen[lado]["bateo"]["lob"] = lob

    probables = gd.get("probablePitchers") or {}
    clima = gd.get("weather") or {}
    return {
        "game_pk": gd.get("game", {}).get("pk") or feed.get("gamePk"),
        "estado": estado,
        "estado_detalle": status.get("detailedState", ""),
        "hora_et": _hora_et((gd.get("datetime") or {}).get("dateTime")),
        "lugar": (gd.get("venue") or {}).get("name", ""),
        "clima": ", ".join(x for x in (clima.get("condition"), f"{clima['temp']}°F" if clima.get("temp") else None) if x),
        "home": home,
        "away": away,
        "probables": {
            "home": (probables.get("home") or {}).get("fullName", ""),
            "away": (probables.get("away") or {}).get("fullName", ""),
        },
        "marcador": marcador,
        "entradas": entradas,
        "entradas_programadas": linescore.get("scheduledInnings", 9),
        "situacion": situacion,
        "jugadas_anotacion": jugadas_anotacion,
        "jugadas_por_entrada": jugadas_por_entrada,
        "box": box_resumen,
        "decisiones": {
            "ganador": (decisiones.get("winner") or {}).get("fullName", ""),
            "perdedor": (decisiones.get("loser") or {}).get("fullName", ""),
            "salvamento": (decisiones.get("save") or {}).get("fullName", ""),
        },
        "ganador": ganador,
    }


# ============================================================================
# POR QUÉ EL MODELO ELIGIÓ A UN EQUIPO
# ============================================================================


def _f(features, clave, default=None):
    v = features.get(clave)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ultimos_10(rate):
    if rate is None:
        return None
    g = int(round(rate * 10))
    return f"{g}-{10 - g}"


def _magnitud(valor, umbrales):
    """Palabra de intensidad según |valor| y umbrales (ligera, clara, amplia)."""
    v = abs(valor)
    if v < umbrales[0]:
        return "ligera"
    if v < umbrales[1]:
        return "clara"
    return "amplia"


def explicar_prediccion(prediccion, home, away, features, home_pitcher="", away_pitcher=""):
    """
    Lista de factores que sustentan la predicción, ordenados por peso.

    Cada factor: {"factor", "texto", "favorece" (código de equipo o None), "peso" (float)}.
    El peso es positivo si favorece al equipo predicho y negativo si lo contradice.
    """
    features = features or {}
    factores = []

    def signo(fav):
        if fav is None:
            return 0
        return 1 if fav == prediccion else -1

    # Abridores
    h_era, a_era = _f(features, "home_starter_ERA"), _f(features, "away_starter_ERA")
    h_whip, a_whip = _f(features, "home_starter_WHIP"), _f(features, "away_starter_WHIP")
    if h_era is not None and a_era is not None:
        score = (a_era - h_era) + 3 * ((a_whip or 0) - (h_whip or 0))
        fav = home if score > 0.05 else away if score < -0.05 else None
        mag = _magnitud(score, (0.4, 1.2))
        hp = home_pitcher or "el abridor local"
        ap = away_pitcher or "el abridor visitante"
        texto = (
            f"{hp} (ERA {h_era:.2f}" + (f", WHIP {h_whip:.2f}" if h_whip is not None else "") + ") frente a "
            f"{ap} (ERA {a_era:.2f}" + (f", WHIP {a_whip:.2f}" if a_whip is not None else "") + "). "
            + (f"Ventaja {mag} para {fav} en el montículo." if fav else "Abridores de nivel parejo.")
        )
        factores.append({"factor": "Abridores", "texto": texto, "favorece": fav, "peso": signo(fav) * min(abs(score), 3) * 1.2})

    # Forma reciente
    h10, a10 = _f(features, "home_win_rate_10"), _f(features, "away_win_rate_10")
    h_racha, a_racha = _f(features, "home_racha", 0), _f(features, "away_racha", 0)
    if h10 is not None and a10 is not None:
        diff = h10 - a10
        fav = home if diff > 0.05 else away if diff < -0.05 else None

        def racha_txt(r):
            r = int(r or 0)
            if r > 0:
                return f"racha de {r} victoria{'s' if r > 1 else ''}"
            if r < 0:
                return f"racha de {abs(r)} derrota{'s' if r < -1 else ''}"
            return "sin racha activa"

        texto = (
            f"{home} llega {_ultimos_10(h10)} en sus últimos 10 ({racha_txt(h_racha)}); "
            f"{away} llega {_ultimos_10(a10)} ({racha_txt(a_racha)})."
        )
        factores.append({"factor": "Forma reciente", "texto": texto, "favorece": fav, "peso": signo(fav) * abs(diff) * 4})

    # Temporada
    h_rec, a_rec = features.get("home_season_record"), features.get("away_season_record")
    hs, as_ = _f(features, "home_win_rate_season"), _f(features, "away_win_rate_season")
    if hs is not None and as_ is not None:
        diff = hs - as_
        fav = home if diff > 0.02 else away if diff < -0.02 else None
        texto = f"Récord de temporada: {home} {h_rec or f'{hs:.3f}'} y {away} {a_rec or f'{as_:.3f}'}."
        factores.append({"factor": "Temporada", "texto": texto, "favorece": fav, "peso": signo(fav) * abs(diff) * 6})

    # Ofensiva
    h_ops, a_ops = _f(features, "home_team_OPS"), _f(features, "away_team_OPS")
    if h_ops is not None and a_ops is not None:
        diff = h_ops - a_ops
        fav = home if diff > 0.01 else away if diff < -0.01 else None
        texto = f"OPS de equipo: {home} {h_ops:.3f} y {away} {a_ops:.3f}."
        h_best, a_best = _f(features, "home_best_OPS"), _f(features, "away_best_OPS")
        if h_best is not None and a_best is not None:
            texto += f" Sus mejores bates: {h_best:.3f} contra {a_best:.3f}."
        factores.append({"factor": "Ofensiva", "texto": texto, "favorece": fav, "peso": signo(fav) * abs(diff) * 20})

    # Bullpen
    h_bp, a_bp = _f(features, "home_bullpen_ERA"), _f(features, "away_bullpen_ERA")
    if h_bp is not None and a_bp is not None:
        diff = a_bp - h_bp
        fav = home if diff > 0.15 else away if diff < -0.15 else None
        texto = f"Bullpen: ERA {h_bp:.2f} ({home}) contra {a_bp:.2f} ({away})."
        factores.append({"factor": "Bullpen", "texto": texto, "favorece": fav, "peso": signo(fav) * abs(diff) * 0.8})

    # ELO
    h_elo, a_elo = _f(features, "home_elo"), _f(features, "away_elo")
    if h_elo is not None and a_elo is not None and abs(h_elo - a_elo) >= 1:
        diff = h_elo - a_elo
        fav = home if diff > 0 else away
        texto = f"ELO: {home} {h_elo:.0f} y {away} {a_elo:.0f} (diferencia de {abs(diff):.0f} puntos)."
        factores.append({"factor": "ELO", "texto": texto, "favorece": fav, "peso": signo(fav) * abs(diff) / 40})

    # Localía (siempre presente)
    factores.append(
        {
            "factor": "Localía",
            "texto": f"{home} juega en casa; el modelo incorpora la ventaja histórica del local.",
            "favorece": home,
            "peso": signo(home) * 0.35,
        }
    )

    factores.sort(key=lambda f: abs(f["peso"]), reverse=True)
    return factores


def resumen_factores(factores, prediccion, rival):
    """Frase corta: cuántos factores apoyan la predicción y cuántos la contradicen."""
    a_favor = sum(1 for f in factores if f["favorece"] == prediccion)
    en_contra = sum(1 for f in factores if f["favorece"] == rival)
    return f"{a_favor} factor{'es' if a_favor != 1 else ''} a favor de {prediccion}, {en_contra} a favor de {rival}."


# ============================================================================
# QUÉ EXPLICA EL RESULTADO (PARTIDO TERMINADO)
# ============================================================================


def _momento_decisivo(resumen):
    """Última jugada con la que el ganador tomó la delantera definitiva."""
    ganador = resumen.get("ganador")
    if not ganador:
        return None
    decisiva = None
    for j in resumen.get("jugadas_anotacion", []):
        lidera = j["home"] > j["away"] if ganador == "home" else j["away"] > j["home"]
        if lidera and decisiva is None:
            decisiva = j
        elif not lidera:
            decisiva = None  # perdió la ventaja: la decisiva vendrá después
    return decisiva


def _forma_del_partido(resumen):
    m = resumen["marcador"]
    if m["home"] is None or m["away"] is None:
        return ""
    margen = abs(m["home"] - m["away"])
    ganador = resumen.get("ganador")
    etiquetas = []
    if margen >= 5:
        etiquetas.append("paliza")
    elif margen <= 1:
        etiquetas.append("partido cerrado")
    # remontada: el ganador llegó a ir perdiendo por 2 o más
    max_desventaja = 0
    for j in resumen.get("jugadas_anotacion", []):
        desventaja = (j["away"] - j["home"]) if ganador == "home" else (j["home"] - j["away"])
        max_desventaja = max(max_desventaja, desventaja)
    if max_desventaja >= 2:
        etiquetas.append(f"remontada desde {max_desventaja} abajo")
    ultima = resumen["jugadas_anotacion"][-1] if resumen.get("jugadas_anotacion") else None
    if ganador == "home" and ultima and ultima["mitad"] == "baja" and ultima["entrada"] >= resumen.get("entradas_programadas", 9):
        etiquetas.append("victoria en la última entrada")
    if resumen["entradas"] and len(resumen["entradas"]) > resumen.get("entradas_programadas", 9):
        etiquetas.append("entradas extra")
    return ", ".join(etiquetas)


def _texto_abridor(linea, era_esperada, nombre_equipo):
    if not linea:
        return None
    ip = _ip_a_float(linea["ip"])
    base = f"{linea['nombre']} ({nombre_equipo}) lanzó {linea['ip']} entradas con {linea['h']} hits, {linea['r']} carreras ({linea['er']} limpias), {linea['bb']} base{'s' if linea['bb'] != 1 else ''} por bolas y {linea['k']} ponche{'s' if linea['k'] != 1 else ''}"
    if era_esperada is not None and ip > 0:
        era_partido = linea["er"] * 9 / ip
        if era_partido > era_esperada + 2.5:
            base += f": muy por encima de su ERA de {era_esperada:.2f}"
        elif era_partido < max(era_esperada - 1.5, 0.5):
            base += f": mejor que su ERA de {era_esperada:.2f}"
        else:
            base += f", en línea con su ERA de {era_esperada:.2f}"
    return base + "."


def _bullpen(box_equipo):
    rel = box_equipo.get("relevistas") or []
    ip = sum(_ip_a_float(r["ip"]) for r in rel)
    carreras = sum(r["r"] for r in rel)
    return ip, carreras, len(rel)


def analizar_resultado(prediccion, home, away, resumen, features=None):
    """
    Explica el resultado de un partido terminado frente a lo que esperaba el modelo.

    Devuelve {"acierto", "titulo", "parrafos": [...], "claves": [{"etiqueta", "valor"}], "momento": jugada|None}.
    """
    features = features or {}
    m = resumen["marcador"]
    if resumen.get("estado") != "final" or m["home"] is None:
        return {"acierto": None, "titulo": "El partido aún no ha terminado", "parrafos": [], "claves": [], "momento": None}

    ganador_lado = resumen["ganador"]
    ganador = home if ganador_lado == "home" else away
    perdedor = away if ganador_lado == "home" else home
    acierto = prediccion == ganador
    box_g = resumen["box"][ganador_lado]
    box_p = resumen["box"]["away" if ganador_lado == "home" else "home"]
    lado_pred = "home" if prediccion == home else "away"
    lado_rival = "away" if lado_pred == "home" else "home"

    parrafos = []
    forma = _forma_del_partido(resumen)
    marcador_txt = f"{away} {m['away']} - {m['home']} {home}"
    if acierto:
        titulo = f"Se cumplió: ganó {ganador}"
        parrafos.append(f"Resultado final {marcador_txt}" + (f" ({forma})" if forma else "") + ". El modelo había dado la victoria a {ganador}.".replace("{ganador}", ganador))
    else:
        titulo = f"No se cumplió: ganó {ganador}"
        parrafos.append(f"Resultado final {marcador_txt}" + (f" ({forma})" if forma else "") + f". El modelo esperaba la victoria de {prediccion}.")

    # Abridores: lo que hizo cada uno frente a su ERA esperada
    era_pred = _f(features, f"{lado_pred}_starter_ERA")
    era_rival = _f(features, f"{lado_rival}_starter_ERA")
    abridor_pred = resumen["box"][lado_pred].get("abridor")
    abridor_rival = resumen["box"][lado_rival].get("abridor")
    t1 = _texto_abridor(abridor_pred, era_pred, prediccion)
    t2 = _texto_abridor(abridor_rival, era_rival, away if lado_pred == "home" else home)
    if t1 or t2:
        parrafos.append(" ".join(x for x in (t1, t2) if x))

    # Ofensiva y bullpen
    bg, bp = box_g["bateo"], box_p["bateo"]
    ofensiva = (
        f"{ganador} conectó {bg['hits']} hits" + (f" con {bg['hr']} jonrón{'es' if bg['hr'] != 1 else ''}" if bg["hr"] else "")
        + f" y recibió {bg['bb']} bases por bolas; {perdedor} se quedó en {bp['hits']} hits"
        + (f" y dejó {bp['lob']} corredores en base" if bp["lob"] >= 6 else "")
        + "."
    )
    parrafos.append(ofensiva)
    ip_g, car_g, n_g = _bullpen(box_g)
    ip_p, car_p, n_p = _bullpen(box_p)
    if n_g or n_p:
        parrafos.append(
            f"Bullpen: el de {ganador} cubrió {_float_a_ip(ip_g)} entradas y permitió {car_g} carrera{'s' if car_g != 1 else ''}; "
            f"el de {perdedor} cubrió {_float_a_ip(ip_p)} y permitió {car_p}."
        )

    momento = _momento_decisivo(resumen)
    if momento:
        parrafos.append(
            f"La carrera decisiva llegó en la {momento['mitad']} de la {ORDINALES.get(momento['entrada'], str(momento['entrada']) + 'ª')}: {momento['descripcion']}"
        )

    if not acierto:
        # Diagnóstico corto de qué falló respecto a la predicción
        razones = []
        if abridor_pred and era_pred is not None:
            ip = _ip_a_float(abridor_pred["ip"])
            if ip > 0 and abridor_pred["er"] * 9 / ip > era_pred + 2.0:
                razones.append(f"el abridor de {prediccion} rindió muy por debajo de su temporada")
            elif ip < 5:
                razones.append(f"el abridor de {prediccion} no pasó de la quinta entrada")
        bateo_pred = resumen["box"][lado_pred]["bateo"]
        if bateo_pred["hits"] <= 5:
            razones.append(f"la ofensiva de {prediccion} apenas produjo ({bateo_pred['hits']} hits)")
        if bateo_pred["lob"] >= 8:
            razones.append(f"{prediccion} dejó {bateo_pred['lob']} corredores en base")
        _, car_bp_pred, n_bp_pred = _bullpen(resumen["box"][lado_pred])
        if n_bp_pred and car_bp_pred >= 3:
            razones.append(f"el bullpen de {prediccion} permitió {car_bp_pred} carreras")
        if abridor_rival:
            ip_r = _ip_a_float(abridor_rival["ip"])
            if ip_r >= 6 and abridor_rival["er"] <= 2:
                razones.append(f"{abridor_rival['nombre']} dominó con {abridor_rival['ip']} entradas y {abridor_rival['er']} limpias")
        if razones:
            parrafos.append("Qué falló: " + "; ".join(razones) + ".")
        else:
            parrafos.append(f"Partido parejo que se inclinó por detalles: no hay un factor dominante que explique la derrota de {prediccion}.")

    claves = [
        {"etiqueta": "Hits", "valor": f"{resumen['box']['away']['bateo']['hits']} - {resumen['box']['home']['bateo']['hits']}"},
        {"etiqueta": "Jonrones", "valor": f"{resumen['box']['away']['bateo']['hr']} - {resumen['box']['home']['bateo']['hr']}"},
        {"etiqueta": "En base", "valor": f"{resumen['box']['away']['bateo']['lob']} - {resumen['box']['home']['bateo']['lob']}"},
        {"etiqueta": "Errores", "valor": f"{m['errores']['away']} - {m['errores']['home']}"},
    ]
    d = resumen.get("decisiones", {})
    if d.get("ganador"):
        claves.append({"etiqueta": "Ganador (P)", "valor": d["ganador"]})
    if d.get("salvamento"):
        claves.append({"etiqueta": "Salvamento", "valor": d["salvamento"]})

    return {"acierto": acierto, "titulo": titulo, "parrafos": parrafos, "claves": claves, "momento": momento}


def pulso_en_vivo(prediccion, home, away, resumen):
    """Frase de estado para un partido en curso: cómo va el equipo que eligió el modelo."""
    m = resumen["marcador"]
    s = resumen["situacion"]
    if m["home"] is None or m["away"] is None or not s.get("entrada"):
        return "Esperando el primer lanzamiento."
    lado = "home" if prediccion == home else "away"
    mio, rival = (m["home"], m["away"]) if lado == "home" else (m["away"], m["home"])
    entrada = f"{s['mitad']} de la {ORDINALES.get(s['entrada'], str(s['entrada']) + 'ª')}"
    if mio > rival:
        return f"{prediccion} va ganando {mio}-{rival} en la {entrada}."
    if mio < rival:
        return f"{prediccion} va perdiendo {rival}-{mio} en la {entrada}."
    return f"Empate {mio}-{rival} en la {entrada}."
