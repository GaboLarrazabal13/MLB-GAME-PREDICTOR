"""Utilidades para interpretar el calendario diario de Baseball-Reference."""

import re

MESES = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def extraer_fecha_desde_schedule_href(href):
    """Extrae YYYY-MM-DD desde links /previews/... o /boxes/... del schedule."""
    if not href:
        return None

    match = re.search(r"(\d{4})(\d{2})(\d{2})\d\.shtml", href)
    if not match:
        return None

    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def parsear_fecha_schedule_header(header_text, target_date=None):
    """Convierte encabezados tipo 'Monday, March 30, 2026' o 'Today's Games' a YYYY-MM-DD."""
    if not header_text:
        return None

    # Limpiar espacios múltiples y saltos de línea
    texto = " ".join(str(header_text).split()).strip()

    # Caso especial: "Today's Games"
    if "today's games" in texto.lower() and target_date:
        return target_date

    # Regex flexible para manejar posibles variaciones de espacios
    # Formato esperado: Day, Month Day, Year (ej: Tuesday, April 28, 2026)
    match = re.match(r"^[A-Za-z]+,\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$", texto)
    if not match:
        return None

    month_name, day, year = match.groups()
    month = MESES.get(month_name)
    if not month:
        return None

    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def iterar_secciones_schedule(soup, target_date=None):
    """Itera las secciones del calendario con sus partidos y fecha inferida."""
    for h3 in soup.find_all("h3"):
        label = " ".join(h3.stripped_strings)
        header_date = parsear_fecha_schedule_header(label, target_date=target_date)

        # Identificar si es hoy: por span id="today" O si el label contiene "today"
        # Bref a veces cambia el span por una clase o ID dinámico
        is_today = bool(h3.find("span", id=re.compile(r"today", re.I))) or "today" in label.lower()

        games = []
        inferred_dates = []

        for sibling in h3.find_next_siblings():
            if sibling.name == "h3":
                break

            if sibling.name == "p" and "game" in (sibling.get("class") or []):
                team_links = sibling.find_all("a", href=re.compile(r"/teams/\w+/\d+\.shtml"))
                if len(team_links) < 2:
                    continue

                away_match = re.search(r"/teams/(\w+)/", team_links[0].get("href", ""))
                home_match = re.search(r"/teams/(\w+)/", team_links[1].get("href", ""))
                if not away_match or not home_match:
                    continue

                game_link = None
                for link in sibling.find_all("a", href=True):
                    href = link.get("href", "")
                    if href.startswith("/previews/") or href.startswith("/boxes/"):
                        game_link = href
                        break

                inferred_date = extraer_fecha_desde_schedule_href(game_link)
                if inferred_date:
                    inferred_dates.append(inferred_date)

                games.append(
                    {
                        "away_team": away_match.group(1),
                        "home_team": home_match.group(1),
                        "game_link": game_link,
                        "node": sibling,
                    }
                )

        if not games:
            continue

        unique_dates = sorted(set(inferred_dates))
        section_date = header_date
        if not section_date and len(unique_dates) == 1:
            section_date = unique_dates[0]

        yield {
            "label": label,
            "date": section_date,
            "is_today": is_today,
            "games": games,
        }


def seleccionar_seccion_schedule(soup, fecha_objetivo_db):
    """Selecciona la sección más adecuada para una fecha MLB concreta."""
    secciones = list(iterar_secciones_schedule(soup, target_date=fecha_objetivo_db))
    if not secciones:
        return None

    for seccion in secciones:
        if seccion.get("date") == fecha_objetivo_db:
            return seccion

    secciones_futuras = [
        seccion for seccion in secciones if seccion.get("date") and seccion["date"] >= fecha_objetivo_db
    ]
    if secciones_futuras:
        return min(secciones_futuras, key=lambda item: item["date"])

    secciones_con_fecha = [seccion for seccion in secciones if seccion.get("date")]
    if secciones_con_fecha:
        return max(secciones_con_fecha, key=lambda item: item["date"])

    return next((seccion for seccion in secciones if seccion.get("is_today")), secciones[0])
