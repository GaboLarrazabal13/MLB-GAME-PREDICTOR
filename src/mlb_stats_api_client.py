"""
MLB Stats API Client - Core Data Hydrator
Provides high-performance, date-hydrated snapshots for teams and players.
Eliminates temporal data leakage (Hindsight Bias) by querying statistics cumulative up to the day before any past game.
"""

import requests
import time
from datetime import datetime, timedelta

# ============================================================================
# MAPEO OFICIAL DE EQUIPOS MLB A IDs DE LA API
# ============================================================================
TEAM_CODE_TO_ID = {
    "ARI": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHC": 112,
    "CHW": 145,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KCR": 118,
    "LAA": 108,
    "LAD": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYM": 121,
    "NYY": 147,
    "ATH": 133,
    "PHI": 143,
    "PIT": 134,
    "SDP": 135,
    "SEA": 136,
    "SFG": 137,
    "STL": 138,
    "TBR": 139,
    "TEX": 140,
    "TOR": 141,
    "WSN": 120,
}


def safe_float(val, default=0.0):
    try:
        if val is None or val == "-" or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        if val is None or val == "-" or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def obtener_fecha_ayer(fecha_str):
    """Calcula el día anterior a una fecha YYYY-MM-DD para evitar data leakage."""
    try:
        dt = datetime.strptime(fecha_str, "%Y-%m-%d")
        ayer = dt - timedelta(days=1)
        return ayer.strftime("%Y-%m-%d")
    except Exception:
        return None


def query_api_with_retry(url, max_retries=2, delay=1):
    """Realiza peticiones HTTP a la API oficial con reintentos."""
    for intento in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (403, 429):
                time.sleep(delay * 2)
            else:
                time.sleep(delay)
        except Exception:
            time.sleep(delay)
    return None


def obtener_stats_equipo_api(team_code, year, group="hitting", up_to_date=None):
    """
    Obtiene estadísticas colectivas del equipo (hitting o pitching) para el año indicado.
    Si se especifica up_to_date (YYYY-MM-DD), devuelve los acumulados hasta esa fecha (sin leakage).
    """
    team_id = TEAM_CODE_TO_ID.get(team_code.upper().strip())
    if not team_id:
        return {}

    if up_to_date:
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
            f"?stats=byDateRange&group={group}&startDate={year}-03-01&endDate={up_to_date}&season={year}"
        )
    else:
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
            f"?stats=season&group={group}&season={year}"
        )

    data = query_api_with_retry(url)
    if data and "stats" in data and len(data["stats"]) > 0:
        splits = data["stats"][0].get("splits", [])
        if splits:
            return splits[0].get("stat", {})
    return {}


def obtener_stats_jugadores_equipo_api(team_code, year, group="hitting", up_to_date=None):
    """
    Obtiene estadísticas a nivel de jugador para todos los peloteros de un equipo.
    Si se especifica up_to_date (YYYY-MM-DD), obtiene acumulados sin data leakage.
    """
    team_id = TEAM_CODE_TO_ID.get(team_code.upper().strip())
    if not team_id:
        return []

    if up_to_date:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=byDateRange&group={group}&teamId={team_id}&startDate={year}-03-01&endDate={up_to_date}&season={year}&playerPool=all"
        )
    else:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=season&group={group}&teamId={team_id}&season={year}&playerPool=all"
        )

    data = query_api_with_retry(url)
    if data and "stats" in data and len(data["stats"]) > 0:
        return data["stats"][0].get("splits", [])
    return []


def obtener_stats_pitcher_api(pitcher_id, year, up_to_date=None):
    """
    Obtiene estadísticas oficiales de pitcheo del lanzador directamente por su ID de MLB.
    Si se pasa up_to_date, utiliza byDateRange para neutralizar retrospectiva (Hindsight Bias).
    """
    if not pitcher_id:
        return None

    if up_to_date:
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
            f"?stats=byDateRange&group=pitching&startDate={year}-03-01&endDate={up_to_date}&season={year}"
        )
    else:
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
            f"?stats=statsSingleSeason&group=pitching&season={year}"
        )

    data = query_api_with_retry(url)
    if data and "stats" in data and len(data["stats"]) > 0:
        splits = data["stats"][0].get("splits", [])
        if splits:
            stat = splits[0].get("stat", {})
            return {
                "ERA": safe_float(stat.get("era", 0.0)),
                "WHIP": safe_float(stat.get("whip", 0.0)),
                "H9": safe_float(stat.get("hitsPer9Inn", 0.0)),
                "SO9": safe_float(stat.get("strikeoutsPer9Inn", 0.0)),
                "W": safe_int(stat.get("wins", 0)),
                "L": safe_int(stat.get("losses", 0)),
                "IP": safe_float(stat.get("inningsPitched", 0.0)),
                "G": safe_int(stat.get("gamesPitched", 0)),
                "GS": safe_int(stat.get("gamesStarted", 0)),
            }
    return None


def obtener_stats_pitcher_por_nombre(team_code, pitcher_name, year, up_to_date=None):
    """
    Busca un lanzador por nombre dentro de las estadísticas de pitcheo del equipo.
    Sirve como recuperador secundario si no contamos con el pitcher ID exacto.
    """
    splits = obtener_stats_jugadores_equipo_api(team_code, year, group="pitching", up_to_date=up_to_date)
    if not splits:
        return None

    from train_model_hybrid_actions import normalizar_texto
    nombre_busqueda = normalizar_texto(pitcher_name)

    for split in splits:
        player_info = split.get("player", {})
        full_name = player_info.get("fullName", "")
        if normalizar_texto(full_name) == nombre_busqueda or nombre_busqueda in normalizar_texto(full_name) or normalizar_texto(full_name) in nombre_busqueda:
            stat = split.get("stat", {})
            return {
                "ERA": safe_float(stat.get("era", 0.0)),
                "WHIP": safe_float(stat.get("whip", 0.0)),
                "H9": safe_float(stat.get("hitsPer9Inn", 0.0)),
                "SO9": safe_float(stat.get("strikeoutsPer9Inn", 0.0)),
                "W": safe_int(stat.get("wins", 0)),
                "L": safe_int(stat.get("losses", 0)),
                "IP": safe_float(stat.get("inningsPitched", 0.0)),
                "G": safe_int(stat.get("gamesPitched", 0)),
                "GS": safe_int(stat.get("gamesStarted", 0)),
                "nombre_real": full_name,
            }
    return None


def extraer_top_bateadores_api(splits):
    """Filtra y devuelve los 3 mejores bateadores del equipo según OPS acumulado."""
    if not splits:
        return None

    valid_batters = []
    for split in splits:
        stat = split.get("stat", {})
        player = split.get("player", {})
        at_bats = safe_int(stat.get("atBats", 0))
        ops = safe_float(stat.get("ops", 0.0))
        
        # Filtro de bateadores activos (mínimo 15 turnos al bate)
        if at_bats >= 15 and player.get("fullName"):
            valid_batters.append({
                "n": player["fullName"],
                "ba": safe_float(stat.get("avg", 0.0)),
                "obp": safe_float(stat.get("obp", 0.0)),
                "slg": safe_float(stat.get("slg", 0.0)),
                "ops": ops,
                "hr": safe_int(stat.get("homeRuns", 0)),
                "rbi": safe_int(stat.get("rbi", 0)),
                "ab": at_bats
            })

    if not valid_batters:
        return None

    # Ordenar por OPS descendente y tomar top 3
    valid_batters = sorted(valid_batters, key=lambda x: x["ops"], reverse=True)
    top_3 = valid_batters[:3]

    return {
        "best_bat_BA": sum(b["ba"] for b in top_3) / len(top_3),
        "best_bat_OBP": sum(b["obp"] for b in top_3) / len(top_3),
        "best_bat_OPS": sum(b["ops"] for b in top_3) / len(top_3),
        "best_bat_HR": sum(b["hr"] for b in top_3) / len(top_3),
        "best_bat_RBI": sum(b["rbi"] for b in top_3) / len(top_3),
        "detalles_visuales": top_3
    }


def extraer_top_relevistas_api(splits):
    """Filtra los lanzadores relevistas y obtiene las métricas medias de los 3 mejores."""
    if not splits:
        return None

    bullpen = []
    for split in splits:
        stat = split.get("stat", {})
        player = split.get("player", {})
        gs = safe_int(stat.get("gamesStarted", 0))
        g = safe_int(stat.get("gamesPitched", 0))
        
        # Filtro de bullpen: Games Started menor al 50% de Games Pitched
        if g > 2 and gs < (g * 0.5):
            bullpen.append({
                "nombre": player.get("fullName", "Desconocido"),
                "era": safe_float(stat.get("era", 0.0)),
                "whip": safe_float(stat.get("whip", 0.0)),
                "sv": safe_int(stat.get("saves", 0)),
                "gf": safe_int(stat.get("gamesFinished", 0)),
                "ip": safe_float(stat.get("inningsPitched", 0.0))
            })

    if not bullpen:
        return None

    # Ordenar por saves (Closer), juegos finalizados (Setup) e innings
    bullpen = sorted(bullpen, key=lambda x: (x["sv"], x["gf"], x["ip"]), reverse=True)
    top_3 = bullpen[:3]

    return {
        "bullpen_ERA_mean": sum(r["era"] for r in top_3) / len(top_3),
        "bullpen_WHIP_mean": sum(r["whip"] for r in top_3) / len(top_3),
    }


def obtener_stats_completas_api(home_team, away_team, home_pitcher, away_pitcher, year, up_to_date=None):
    """
    Ingesta y calcula todas las features del partido en un único payload,
    reemplazando al 100% el web scraping a Baseball-Reference.
    """
    # 1. Stats generales de equipo (hitting & pitching)
    team_hit_h = obtener_stats_equipo_api(home_team, year, "hitting", up_to_date)
    team_pit_h = obtener_stats_equipo_api(home_team, year, "pitching", up_to_date)
    team_hit_a = obtener_stats_equipo_api(away_team, year, "hitting", up_to_date)
    team_pit_a = obtener_stats_equipo_api(away_team, year, "pitching", up_to_date)

    # 2. Roster de bateadores y lanzadores
    roster_hit_h = obtener_stats_jugadores_equipo_api(home_team, year, "hitting", up_to_date)
    roster_pit_h = obtener_stats_jugadores_equipo_api(home_team, year, "pitching", up_to_date)
    roster_hit_a = obtener_stats_jugadores_equipo_api(away_team, year, "hitting", up_to_date)
    roster_pit_a = obtener_stats_jugadores_equipo_api(away_team, year, "pitching", up_to_date)

    # 3. Stats de los Abridores
    sp_h = obtener_stats_pitcher_por_nombre(home_team, home_pitcher, year, up_to_date)
    sp_a = obtener_stats_pitcher_por_nombre(away_team, away_pitcher, year, up_to_date)

    # Fallback por si la búsqueda por nombre falló pero tenemos stats básicas por defecto
    if not sp_h:
        sp_h = {"ERA": 4.5, "WHIP": 1.35, "H9": 8.5, "SO9": 7.5, "W": 0, "L": 0, "IP": 0.0, "nombre_real": home_pitcher}
    if not sp_a:
        sp_a = {"ERA": 4.5, "WHIP": 1.35, "H9": 8.5, "SO9": 7.5, "W": 0, "L": 0, "IP": 0.0, "nombre_real": away_pitcher}

    # 4. Top Bateadores
    hb_h = extraer_top_bateadores_api(roster_hit_h)
    hb_a = extraer_top_bateadores_api(roster_hit_a)

    # Fallbacks de bateadores
    if not hb_h:
        hb_h = {"best_bat_BA": 0.260, "best_bat_OBP": 0.330, "best_bat_OPS": 0.750, "best_bat_HR": 10.0, "best_bat_RBI": 35.0, "detalles_visuales": []}
    if not hb_a:
        hb_a = {"best_bat_BA": 0.260, "best_bat_OBP": 0.330, "best_bat_OPS": 0.750, "best_bat_HR": 10.0, "best_bat_RBI": 35.0, "detalles_visuales": []}

    # 5. Bullpen
    rel_h = extraer_top_relevistas_api(roster_pit_h)
    rel_a = extraer_top_relevistas_api(roster_pit_a)

    # Fallbacks de bullpen
    if not rel_h:
        rel_h = {"bullpen_ERA_mean": 4.20, "bullpen_WHIP_mean": 1.30}
    if not rel_a:
        rel_a = {"bullpen_ERA_mean": 4.20, "bullpen_WHIP_mean": 1.30}

    # 6. Compilar todas las features unificadas
    features = {
        "home_team_OPS": safe_float(team_hit_h.get("ops", 0.720)),
        "away_team_OPS": safe_float(team_hit_a.get("ops", 0.720)),
        "diff_team_BA": safe_float(team_hit_h.get("avg", 0.245)) - safe_float(team_hit_a.get("avg", 0.245)),
        "diff_team_OPS": safe_float(team_hit_h.get("ops", 0.720)) - safe_float(team_hit_a.get("ops", 0.720)),
        "diff_team_ERA": safe_float(team_pit_a.get("era", 4.20)) - safe_float(team_pit_h.get("era", 4.20)),
        
        "home_starter_WHIP": sp_h["WHIP"],
        "away_starter_WHIP": sp_a["WHIP"],
        "home_starter_ERA": sp_h["ERA"],
        "away_starter_ERA": sp_a["ERA"],
        "home_starter_SO9": sp_h["SO9"],
        "away_starter_SO9": sp_a["SO9"],
        "diff_starter_ERA": sp_a["ERA"] - sp_h["ERA"],
        "diff_starter_WHIP": sp_a["WHIP"] - sp_h["WHIP"],
        "diff_starter_SO9": sp_h["SO9"] - sp_a["SO9"],
        
        "home_best_OPS": hb_h["best_bat_OPS"],
        "away_best_OPS": hb_a["best_bat_OPS"],
        "diff_best_BA": hb_h["best_bat_BA"] - hb_a["best_bat_BA"],
        "diff_best_OPS": hb_h["best_bat_OPS"] - hb_a["best_bat_OPS"],
        "diff_best_HR": hb_h["best_bat_HR"] - hb_a["best_bat_HR"],
        
        "home_bullpen_ERA": rel_h["bullpen_ERA_mean"],
        "away_bullpen_ERA": rel_a["bullpen_ERA_mean"],
        "home_bullpen_WHIP": rel_h["bullpen_WHIP_mean"],
        "away_bullpen_WHIP": rel_a["bullpen_WHIP_mean"],
        "diff_bullpen_ERA": rel_a["bullpen_ERA_mean"] - rel_h["bullpen_ERA_mean"],
        "diff_bullpen_WHIP": rel_a["bullpen_WHIP_mean"] - rel_h["bullpen_WHIP_mean"],
        
        "anchor_pitching_level": (sp_h["ERA"] + rel_h["bullpen_ERA_mean"]) - (sp_a["ERA"] + rel_a["bullpen_ERA_mean"]),
        "anchor_offensive_level": safe_float(team_hit_h.get("ops", 0.720)) - safe_float(team_hit_a.get("ops", 0.720)),
        
        "home_pitcher_name_real": sp_h["nombre_real"],
        "away_pitcher_name_real": sp_a["nombre_real"],
        "home_top_3_batters_details": hb_h["detalles_visuales"],
        "away_top_3_batters_details": hb_a["detalles_visuales"],
        "ip_home": sp_h["IP"],
        "ip_away": sp_a["IP"]
    }

    return features
