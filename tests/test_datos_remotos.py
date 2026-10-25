"""
Sincronización de la base de datos con el dataset de Hugging Face (sin red: se simula la librería).
"""

import sys
import types

import pytest

import mlb_datos_remotos as dr


@pytest.fixture
def hub_falso(monkeypatch, tmp_path):
    origen = tmp_path / "remoto.db"
    origen.write_bytes(b"SQLite falso")
    llamadas = {}

    def hf_hub_download(**kw):
        llamadas["download"] = kw
        return str(origen)

    class HfApi:
        def __init__(self, token=None):
            llamadas["token"] = token

        def upload_file(self, **kw):
            llamadas["upload"] = kw

    modulo = types.SimpleNamespace(hf_hub_download=hf_hub_download, HfApi=HfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", modulo)
    return llamadas


def test_descarga_si_falta(hub_falso, tmp_path):
    destino = tmp_path / "data" / "mlb.db"
    assert dr.descargar(str(destino))
    assert destino.read_bytes() == b"SQLite falso"
    assert hub_falso["download"]["repo_type"] == "dataset"


def test_no_descarga_si_ya_existe(hub_falso, tmp_path):
    destino = tmp_path / "mlb.db"
    destino.write_bytes(b"local")
    assert dr.descargar(str(destino))
    assert "download" not in hub_falso and destino.read_bytes() == b"local"


def test_forzar_descarga(hub_falso, tmp_path):
    destino = tmp_path / "mlb.db"
    destino.write_bytes(b"local")
    dr.descargar(str(destino), forzar=True)
    assert destino.read_bytes() == b"SQLite falso"


def test_subir_requiere_token(hub_falso, monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        dr.subir("x", origen=str(tmp_path / "mlb.db"))
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setenv("HF_DATASET_REPO", "usuario/datos")
    dr.subir("actualización", origen=str(tmp_path / "mlb.db"))
    assert hub_falso["upload"]["repo_id"] == "usuario/datos" and hub_falso["upload"]["commit_message"] == "actualización"


def test_asegurar_db_no_lanza_sin_red(monkeypatch, tmp_path):
    def falla(*a, **k):
        raise OSError("sin red")

    monkeypatch.setattr(dr, "descargar", falla)
    assert dr.asegurar_db(str(tmp_path / "no.db")) is False
