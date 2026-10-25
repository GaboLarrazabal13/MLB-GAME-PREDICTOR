"""Tests offline de mlb_live: normalización del feed y textos explicativos (sin red)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mlb_live  # noqa: E402


def _jugador(pid, nombre, pitching=None, batting=None):
    stats = {}
    if pitching:
        stats["pitching"] = pitching
    if batting:
        stats["batting"] = batting
    return {"person": {"id": pid, "fullName": nombre}, "stats": stats}


def _play(idx, inning, half, desc, away, home, scoring=False, batter="B", pitcher="P"):
    return {
        "result": {"description": desc, "event": "Single", "rbi": 1 if scoring else 0, "awayScore": away, "homeScore": home, "isOut": not scoring},
        "about": {"inning": inning, "halfInning": half, "isComplete": True, "isScoringPlay": scoring},
        "matchup": {"batter": {"fullName": batter}, "pitcher": {"fullName": pitcher}},
        "atBatIndex": idx,
    }


def _feed_final():
    """Partido terminado 4-5 (gana el local tras ir 0-4 abajo)."""
    plays = [
        _play(0, 1, "top", "Visitante conecta jonrón de 4 carreras.", 4, 0, scoring=True),
        _play(1, 1, "bottom", "Local anota con sencillo.", 4, 2, scoring=True),
        _play(2, 4, "bottom", "Local acerca con doble.", 4, 3, scoring=True),
        _play(3, 6, "bottom", "Local empata con elevado de sacrificio.", 4, 4, scoring=True),
        _play(4, 7, "bottom", "Local toma la ventaja con sencillo al centro.", 4, 5, scoring=True),
        _play(5, 9, "top", "Visitante se poncha.", 4, 5),
    ]
    return {
        "gamePk": 1,
        "gameData": {
            "game": {"pk": 1},
            "status": {"abstractGameState": "Final", "detailedState": "Final"},
            "teams": {"home": {"id": 158, "name": "Milwaukee Brewers"}, "away": {"id": 139, "name": "Tampa Bay Rays"}},
            "datetime": {"dateTime": "2026-10-25T00:00:00Z"},
            "venue": {"name": "American Family Field"},
            "weather": {"condition": "Roof Closed", "temp": "71"},
            "probablePitchers": {"home": {"fullName": "Abridor Local"}, "away": {"fullName": "Abridor Visitante"}},
        },
        "liveData": {
            "linescore": {
                "currentInning": 9,
                "isTopInning": True,
                "inningState": "End",
                "scheduledInnings": 9,
                "outs": 3,
                "balls": 0,
                "strikes": 0,
                "innings": [
                    {"num": 1, "away": {"runs": 4}, "home": {"runs": 2}},
                    {"num": 2, "away": {"runs": 0}, "home": {"runs": 0}},
                    {"num": 3, "away": {"runs": 0}, "home": {"runs": 0}},
                    {"num": 4, "away": {"runs": 0}, "home": {"runs": 1}},
                    {"num": 5, "away": {"runs": 0}, "home": {"runs": 0}},
                    {"num": 6, "away": {"runs": 0}, "home": {"runs": 1}},
                    {"num": 7, "away": {"runs": 0}, "home": {"runs": 1}},
                    {"num": 8, "away": {"runs": 0}, "home": {"runs": 0}},
                    {"num": 9, "away": {"runs": 0}, "home": {}},
                ],
                "teams": {"home": {"runs": 5, "hits": 11, "errors": 1}, "away": {"runs": 4, "hits": 7, "errors": 1}},
                "offense": {"batter": {"fullName": "Bateador"}, "first": {"id": 1}},
                "defense": {"pitcher": {"fullName": "Cerrador"}},
            },
            "plays": {"allPlays": plays, "currentPlay": plays[-1], "scoringPlays": [0, 1, 2, 3, 4]},
            "boxscore": {
                "teams": {
                    "home": {
                        "teamStats": {
                            "batting": {"runs": 5, "hits": 11, "homeRuns": 1, "baseOnBalls": 5, "strikeOuts": 6, "leftOnBase": 12},
                            "pitching": {"inningsPitched": "9.0", "hits": 7, "runs": 4, "earnedRuns": 3, "baseOnBalls": 2, "strikeOuts": 10},
                        },
                        "pitchers": [10, 11],
                        "batters": [20],
                        "players": {
                            "ID10": _jugador(10, "Abridor Local", pitching={"inningsPitched": "3.0", "hits": 3, "runs": 4, "earnedRuns": 3, "baseOnBalls": 1, "strikeOuts": 4}),
                            "ID11": _jugador(11, "Relevo Local", pitching={"inningsPitched": "6.0", "hits": 4, "runs": 0, "earnedRuns": 0, "baseOnBalls": 1, "strikeOuts": 6}),
                            "ID20": _jugador(20, "Bate Local", batting={"hits": 2, "atBats": 4, "rbi": 2, "homeRuns": 1}),
                        },
                    },
                    "away": {
                        "teamStats": {
                            "batting": {"runs": 4, "hits": 7, "homeRuns": 1, "baseOnBalls": 2, "strikeOuts": 10, "leftOnBase": 6},
                            "pitching": {"inningsPitched": "8.0", "hits": 11, "runs": 5, "earnedRuns": 5, "baseOnBalls": 5, "strikeOuts": 6},
                        },
                        "pitchers": [30, 31],
                        "batters": [40],
                        "players": {
                            "ID30": _jugador(30, "Abridor Visitante", pitching={"inningsPitched": "3.2", "hits": 6, "runs": 3, "earnedRuns": 3, "baseOnBalls": 2, "strikeOuts": 4}),
                            "ID31": _jugador(31, "Relevo Visitante", pitching={"inningsPitched": "4.1", "hits": 5, "runs": 2, "earnedRuns": 2, "baseOnBalls": 3, "strikeOuts": 2}),
                            "ID40": _jugador(40, "Bate Visitante", batting={"hits": 1, "atBats": 2, "rbi": 4, "homeRuns": 1}),
                        },
                    },
                }
            },
            "decisions": {"winner": {"fullName": "Relevo Local"}, "loser": {"fullName": "Relevo Visitante"}, "save": {}},
        },
    }


FEATURES = {
    "home_elo": 1500.0,
    "away_elo": 1500.0,
    "home_win_rate_10": 0.9,
    "away_win_rate_10": 0.7,
    "home_racha": 6,
    "away_racha": -1,
    "home_win_rate_season": 0.649,
    "away_win_rate_season": 0.618,
    "home_season_record": "109-59",
    "away_season_record": "105-65",
    "home_team_OPS": 0.744,
    "away_team_OPS": 0.728,
    "home_starter_ERA": 2.47,
    "away_starter_ERA": 2.67,
    "home_starter_WHIP": 0.84,
    "away_starter_WHIP": 0.91,
    "home_bullpen_ERA": 2.67,
    "away_bullpen_ERA": 2.73,
}


def test_resumir_feed_partido_final():
    r = mlb_live.resumir_feed(_feed_final())
    assert r["estado"] == "final"
    assert r["home"]["code"] == "MIL" and r["away"]["code"] == "TBR"
    assert r["ganador"] == "home"
    assert r["marcador"]["home"] == 5 and r["marcador"]["away"] == 4
    assert len(r["entradas"]) == 9 and r["entradas"][0]["away"] == 4
    assert r["situacion"]["bases"]["primera"] is True and r["situacion"]["bases"]["segunda"] is False
    assert r["situacion"]["ultima_jugada"] == "Visitante se poncha."
    assert [j["entrada"] for j in r["jugadas_anotacion"]] == [1, 1, 4, 6, 7]
    assert r["box"]["home"]["abridor"]["nombre"] == "Abridor Local"
    assert len(r["box"]["home"]["relevistas"]) == 1
    assert r["box"]["away"]["destacados"][0]["rbi"] == 4
    assert r["decisiones"]["ganador"] == "Relevo Local"
    assert r["hora_et"] == "2026-10-24 20:00"
    assert r["clima"] == "Roof Closed, 71°F"


def test_estado_previo_sin_datos():
    feed = {"gameData": {"status": {"abstractGameState": "Preview", "detailedState": "Scheduled"}, "teams": {}}, "liveData": {}}
    r = mlb_live.resumir_feed(feed)
    assert r["estado"] == "previo"
    assert r["marcador"]["home"] is None
    assert r["entradas"] == [] and r["jugadas_anotacion"] == []
    assert mlb_live.pulso_en_vivo("MIL", "MIL", "TBR", r) == "Esperando el primer lanzamiento."


def test_explicar_prediccion_ordena_por_peso_y_marca_favorecido():
    factores = mlb_live.explicar_prediccion("MIL", "MIL", "TBR", FEATURES, "Logan Henderson", "Drew Rasmussen")
    nombres = [f["factor"] for f in factores]
    assert "Abridores" in nombres and "Localía" in nombres and "Forma reciente" in nombres
    assert "ELO" not in nombres  # 1500 vs 1500 no aporta
    pesos = [abs(f["peso"]) for f in factores]
    assert pesos == sorted(pesos, reverse=True)
    abridores = next(f for f in factores if f["factor"] == "Abridores")
    assert abridores["favorece"] == "MIL" and "Logan Henderson" in abridores["texto"]
    forma = next(f for f in factores if f["factor"] == "Forma reciente")
    assert "9-1" in forma["texto"] and "racha de 6 victorias" in forma["texto"]
    assert mlb_live.resumen_factores(factores, "MIL", "TBR").startswith(f"{sum(1 for f in factores if f['favorece'] == 'MIL')} factor")


def test_explicar_prediccion_sin_features():
    factores = mlb_live.explicar_prediccion("TBR", "MIL", "TBR", {})
    assert len(factores) == 1 and factores[0]["factor"] == "Localía"
    assert factores[0]["peso"] < 0  # la localía contradice la predicción del visitante


def test_analizar_resultado_acierto_y_momento_decisivo():
    r = mlb_live.resumir_feed(_feed_final())
    a = mlb_live.analizar_resultado("MIL", "MIL", "TBR", r, FEATURES)
    assert a["acierto"] is True
    assert a["titulo"].startswith("Se cumplió")
    assert "remontada desde 4 abajo" in a["parrafos"][0]
    assert a["momento"]["entrada"] == 7 and a["momento"]["mitad"] == "baja"
    assert any("Abridor Local" in p and "muy por encima de su ERA" in p for p in a["parrafos"])
    assert {"etiqueta": "Hits", "valor": "7 - 11"} in a["claves"]


def test_analizar_resultado_fallo_explica_que_fallo():
    r = mlb_live.resumir_feed(_feed_final())
    a = mlb_live.analizar_resultado("TBR", "MIL", "TBR", r, FEATURES)
    assert a["acierto"] is False
    assert a["titulo"].startswith("No se cumplió")
    que_fallo = next(p for p in a["parrafos"] if p.startswith("Qué falló"))
    assert "rindió muy por debajo de su temporada" in que_fallo
    assert "apenas produjo" not in que_fallo  # 7 hits no cuenta como ofensiva apagada


def test_analizar_resultado_partido_no_terminado():
    feed = _feed_final()
    feed["gameData"]["status"] = {"abstractGameState": "Live", "detailedState": "In Progress"}
    r = mlb_live.resumir_feed(feed)
    assert r["estado"] == "en_vivo" and r["ganador"] is None
    a = mlb_live.analizar_resultado("MIL", "MIL", "TBR", r, FEATURES)
    assert a["acierto"] is None and a["parrafos"] == []
    assert mlb_live.pulso_en_vivo("MIL", "MIL", "TBR", r) == "MIL va ganando 5-4 en la alta de la 9ª."


def test_ip_a_float():
    assert mlb_live._ip_a_float("5.2") == 5 + 2 / 3
    assert mlb_live._ip_a_float("0.0") == 0.0
    assert mlb_live._ip_a_float("mal") == 0.0
