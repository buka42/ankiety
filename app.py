"""Interfejs przeglądarkowy generatora raportu.

Uruchomienie:
    python app.py
Następnie otwórz http://127.0.0.1:5000 w przeglądarce.
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import generuj_raport

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

OUT_DIR = (Path(__file__).parent / "raport").resolve()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_pl=generuj_raport.PL_MODEL)


@app.route("/generuj", methods=["POST"])
def generuj():
    plik = request.files.get("csv")
    if plik is None or plik.filename == "":
        return render_template("index.html", model_pl=generuj_raport.PL_MODEL, blad="Nie wybrano pliku CSV."), 400
    nazwa = secure_filename(plik.filename)
    if not nazwa.lower().endswith(".csv"):
        return (
            render_template(
                "index.html", model_pl=generuj_raport.PL_MODEL, blad="Plik musi mieć rozszerzenie .csv."
            ),
            400,
        )

    with tempfile.TemporaryDirectory() as tmp:
        sciezka_csv = Path(tmp) / nazwa
        plik.save(sciezka_csv)
        try:
            wynik = generuj_raport.generuj(sciezka_csv, OUT_DIR)
        except Exception as exc:
            traceback.print_exc()
            return (
                render_template(
                    "index.html",
                    model_pl=generuj_raport.PL_MODEL,
                    blad=f"Błąd podczas generowania raportu: {exc}",
                ),
                500,
            )

    pliki = {nazwa: sciezka.name for nazwa, sciezka in wynik.items()}
    return render_template("wynik.html", pliki=pliki)


@app.route("/raport/<path:nazwa>")
def pobierz(nazwa: str):
    bezpieczna = Path(nazwa).name
    sciezka = OUT_DIR / bezpieczna
    if not sciezka.exists():
        abort(404)
    return send_from_directory(OUT_DIR, bezpieczna, as_attachment=False)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
