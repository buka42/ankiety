#!/usr/bin/env python3
"""Generator raportu z ankiety o stresie studentów.

Użycie:
    python generuj_raport.py dane.csv

Plik CSV powinien zawierać 15 kolumn liczbowych (skala 1-5) i 3 kolumny
z odpowiedziami otwartymi. Pierwszy wiersz to nagłówki.

Analiza sentymentu odpowiedzi otwartych uruchamia trzy modele lokalnie:
  - VADER (angielski),
  - TextBlob (angielski),
  - wielojęzyczny model HuggingFace rozumiejący polski (domyślnie
    cardiffnlp/twitter-xlm-roberta-base-sentiment, można zmienić zmienną
    środowiskową POLISH_SENTIMENT_MODEL, np. na "Voicelab/HerBERT-Sentiment"
    lub "nlptown/bert-base-multilingual-uncased-sentiment").

Wagi modelu pobierane są przy pierwszym uruchomieniu i cache'owane lokalnie
(domyślnie ~/.cache/huggingface). Inference działa lokalnie, bez API.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_DIR = Path("raport")
TODAY = date.today().isoformat()
BASENAME = f"ankieta-{TODAY}"
FALLBACK_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
PL_MODEL = os.environ.get("POLISH_SENTIMENT_MODEL", FALLBACK_MODEL)


def wykryj_kolumny(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeryczne, otwarte = [], []
    for col in df.columns:
        seria = pd.to_numeric(df[col], errors="coerce")
        if seria.notna().sum() >= max(10, 0.5 * len(df)):
            numeryczne.append(col)
        else:
            otwarte.append(col)
    return numeryczne, otwarte


def statystyki(df: pd.DataFrame, kolumny: list[str]) -> pd.DataFrame:
    dane = df[kolumny].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "średnia": dane.mean().round(3),
            "mediana": dane.median(),
            "odch. std.": dane.std().round(3),
            "min": dane.min(),
            "max": dane.max(),
            "n": dane.count(),
        }
    )


def histogramy(df: pd.DataFrame, kolumny: list[str], sciezka: Path) -> Path:
    n = len(kolumny)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    axes = np.array(axes).reshape(-1)
    bins = np.arange(0.5, 6.5, 1)
    for ax, kol in zip(axes, kolumny):
        seria = pd.to_numeric(df[kol], errors="coerce").dropna()
        ax.hist(seria, bins=bins, edgecolor="black", color="#4C72B0")
        ax.set_title(kol, fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("ocena")
        ax.set_ylabel("liczba")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Histogramy odpowiedzi (skala 1-5)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(sciezka, dpi=150)
    plt.close(fig)
    return sciezka


def macierz_korelacji(df: pd.DataFrame, kolumny: list[str], sciezka: Path) -> tuple[Path, pd.DataFrame]:
    dane = df[kolumny].apply(pd.to_numeric, errors="coerce")
    corr = dane.corr()
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(kolumny)), max(7, 0.6 * len(kolumny))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Macierz korelacji (Pearson)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(sciezka, dpi=150)
    plt.close(fig)
    return sciezka, corr


def _sprobuj_zaladowac(model_id: str):
    return pipeline(
        "sentiment-analysis", model=model_id, tokenizer=model_id, truncation=True, use_fast=True
    )


def _zaladuj_model_pl():
    print(f"Ładowanie modelu PL: {PL_MODEL} (pierwsze uruchomienie pobiera wagi)...")
    try:
        return _sprobuj_zaladowac(PL_MODEL)
    except Exception as exc:
        komunikat = str(exc)
        problem_z_tokenizerem = (
            "sentencepiece" in komunikat.lower()
            or "Error parsing" in komunikat
            or "protobuf" in komunikat.lower()
        )
        if problem_z_tokenizerem and PL_MODEL != FALLBACK_MODEL:
            print(
                f"Model {PL_MODEL} ma problem z tokenizerem ({exc.__class__.__name__}: {exc}).\n"
                f"Próbuję modelu zapasowego {FALLBACK_MODEL} (BERT/WordPiece, bez SentencePiece)..."
            )
            try:
                return _sprobuj_zaladowac(FALLBACK_MODEL)
            except Exception as exc2:
                raise RuntimeError(
                    f"Nie udało się załadować modelu {PL_MODEL} ani zapasowego "
                    f"{FALLBACK_MODEL}.\nPierwotny błąd: {exc}\nBłąd fallbacku: {exc2}"
                ) from exc2
        if problem_z_tokenizerem:
            raise RuntimeError(
                "Tokenizer modelu nie wczytał się poprawnie. Spróbuj kolejno:\n"
                "  1) pip install --upgrade sentencepiece protobuf transformers tokenizers\n"
                "  2) usuń cache modelu i pobierz ponownie:\n"
                "     - Windows:  rmdir /S /Q "
                f"%USERPROFILE%\\.cache\\huggingface\\hub\\models--{PL_MODEL.replace('/', '--')}\n"
                "     - Linux/macOS:  rm -rf "
                f"~/.cache/huggingface/hub/models--{PL_MODEL.replace('/', '--')}\n"
                f"  3) wymuś inny model, np.:  POLISH_SENTIMENT_MODEL={FALLBACK_MODEL}\n"
                f"Oryginalny błąd: {exc}"
            ) from exc
        raise


def _normalizuj_etykiete_pl(etykieta: str, score: float) -> tuple[str, float]:
    et = etykieta.lower()
    if et in {"positive", "pozytywny", "pos", "label_2"}:
        return "pozytywny", score
    if et in {"negative", "negatywny", "neg", "label_0"}:
        return "negatywny", -score
    if et in {"neutral", "neutralny", "label_1"}:
        return "neutralny", 0.0
    if et.endswith(" stars") or et.endswith(" star"):
        try:
            gwiazdki = int(et.split()[0])
        except ValueError:
            return "neutralny", 0.0
        skala = (gwiazdki - 3) / 2
        if gwiazdki >= 4:
            return "pozytywny", skala
        if gwiazdki <= 2:
            return "negatywny", skala
        return "neutralny", 0.0
    return "neutralny", 0.0


def analiza_sentymentu(df: pd.DataFrame, kolumny: list[str]) -> dict[str, pd.DataFrame]:
    vader = SentimentIntensityAnalyzer()
    pl = _zaladuj_model_pl()
    wyniki = {}
    for kol in kolumny:
        rekordy = []
        for tekst in df[kol].fillna("").astype(str):
            tekst = tekst.strip()
            if not tekst:
                rekordy.append(
                    {"vader": np.nan, "textblob": np.nan, "pl_model": np.nan, "etykieta": "brak"}
                )
                continue
            v = vader.polarity_scores(tekst)["compound"]
            tb = TextBlob(tekst).sentiment.polarity
            wynik_pl = pl(tekst)[0]
            etykieta_pl, score_pl = _normalizuj_etykiete_pl(wynik_pl["label"], wynik_pl["score"])
            rekordy.append(
                {
                    "vader": v,
                    "textblob": tb,
                    "pl_model": score_pl,
                    "etykieta_pl": etykieta_pl,
                    "etykieta": etykieta_pl,
                }
            )
        wyniki[kol] = pd.DataFrame(rekordy)
    return wyniki


def wykres_sentymentu(sentyment: dict[str, pd.DataFrame], sciezka: Path) -> Path:
    etykiety = ["pozytywny", "neutralny", "negatywny", "brak"]
    kolory = {"pozytywny": "#55A868", "neutralny": "#C5C5C5", "negatywny": "#C44E52", "brak": "#999999"}
    dane = {kol: df["etykieta"].value_counts().reindex(etykiety, fill_value=0) for kol, df in sentyment.items()}
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(dane)), 5))
    x = np.arange(len(dane))
    szer = 0.2
    for i, et in enumerate(etykiety):
        wartosci = [dane[k][et] for k in dane]
        ax.bar(x + (i - 1.5) * szer, wartosci, szer, label=et, color=kolory[et])
    ax.set_xticks(x)
    ax.set_xticklabels(list(dane.keys()), rotation=15, ha="right")
    ax.set_ylabel("liczba odpowiedzi")
    ax.set_title(f"Sentyment odpowiedzi otwartych — model PL ({PL_MODEL})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sciezka, dpi=150)
    plt.close(fig)
    return sciezka


def strona_tytulowa(pdf: PdfPages, df: pd.DataFrame, kol_num: list[str], kol_otw: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    tekst = [
        "Raport z ankiety: stres studentów",
        f"Data wygenerowania: {TODAY}",
        "",
        f"Liczba respondentów: {len(df)}",
        f"Pytania liczbowe (skala 1-5): {len(kol_num)}",
        f"Pytania otwarte: {len(kol_otw)}",
        "",
        "Zawartość raportu:",
        "  1. Statystyki opisowe (średnia, mediana, odch. std.)",
        "  2. Histogramy odpowiedzi liczbowych",
        "  3. Macierz korelacji Pearsona",
        "  4. Analiza sentymentu odpowiedzi otwartych",
        f"     (VADER + TextBlob + lokalny model PL: {PL_MODEL})",
    ]
    ax.text(0.08, 0.92, "\n".join(tekst), va="top", ha="left", fontsize=12, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def tabela_pdf(pdf: PdfPages, tabela: pd.DataFrame, tytul: str) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(tytul, fontsize=14, pad=12)
    t = ax.table(
        cellText=tabela.round(3).astype(str).values,
        rowLabels=tabela.index,
        colLabels=tabela.columns,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    pdf.savefig(fig)
    plt.close(fig)


def zlozenie_pdf(
    sciezka_pdf: Path,
    df: pd.DataFrame,
    kol_num: list[str],
    kol_otw: list[str],
    stats: pd.DataFrame,
    corr: pd.DataFrame,
    png_hist: Path,
    png_corr: Path,
    png_sent: Path | None,
    sentyment: dict[str, pd.DataFrame],
) -> None:
    with PdfPages(sciezka_pdf) as pdf:
        strona_tytulowa(pdf, df, kol_num, kol_otw)
        tabela_pdf(pdf, stats, "Statystyki opisowe — pytania liczbowe")
        for png in [png_hist, png_corr]:
            fig = plt.figure(figsize=(11.69, 8.27))
            img = plt.imread(png)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        tabela_pdf(pdf, corr, "Macierz korelacji (Pearson)")
        if png_sent is not None:
            fig = plt.figure(figsize=(11.69, 8.27))
            img = plt.imread(png_sent)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            podsumowanie = pd.DataFrame(
                {kol: df["etykieta"].value_counts() for kol, df in sentyment.items()}
            ).fillna(0).astype(int)
            tabela_pdf(pdf, podsumowanie, "Sentyment — liczba odpowiedzi wg kategorii")


def generuj(plik: Path, out_dir: Path = OUT_DIR) -> dict[str, Path]:
    df = pd.read_csv(plik)
    kol_num, kol_otw = wykryj_kolumny(df)
    print(f"Wczytano {len(df)} respondentów.")
    print(f"Pytania liczbowe ({len(kol_num)}): {kol_num}")
    print(f"Pytania otwarte ({len(kol_otw)}): {kol_otw}")

    out_dir.mkdir(exist_ok=True)
    png_hist = out_dir / f"{BASENAME}-histogramy.png"
    png_corr = out_dir / f"{BASENAME}-korelacja.png"
    png_sent = out_dir / f"{BASENAME}-sentyment.png"
    sciezka_pdf = out_dir / f"{BASENAME}.pdf"
    sciezka_stats = out_dir / f"{BASENAME}-statystyki.csv"

    stats = statystyki(df, kol_num)
    stats.to_csv(sciezka_stats, encoding="utf-8")
    histogramy(df, kol_num, png_hist)
    _, corr = macierz_korelacji(df, kol_num, png_corr)

    sentyment = analiza_sentymentu(df, kol_otw) if kol_otw else {}
    png_sent_final: Path | None = None
    if sentyment:
        wykres_sentymentu(sentyment, png_sent)
        png_sent_final = png_sent

    zlozenie_pdf(
        sciezka_pdf, df, kol_num, kol_otw, stats, corr, png_hist, png_corr, png_sent_final, sentyment
    )
    wynik = {
        "pdf": sciezka_pdf,
        "histogramy": png_hist,
        "korelacja": png_corr,
        "statystyki": sciezka_stats,
    }
    if png_sent_final is not None:
        wynik["sentyment"] = png_sent_final
    return wynik


def main() -> int:
    if len(sys.argv) < 2:
        print("Użycie: python generuj_raport.py <plik.csv>", file=sys.stderr)
        return 1
    plik = Path(sys.argv[1])
    if not plik.exists():
        print(f"Plik nie istnieje: {plik}", file=sys.stderr)
        return 1
    wynik = generuj(plik)
    for nazwa, sciezka in wynik.items():
        print(f"Zapisano ({nazwa}): {sciezka}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
