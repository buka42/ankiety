# Generator raportu z ankiety o stresie studentów

Skrypt w Pythonie, który wczytuje plik CSV z wynikami ankiety
(15 pytań liczbowych w skali 1-5 + 3 pytania otwarte) i generuje
raport w PDF wraz z osobnymi plikami PNG.

Analiza sentymentu odpowiedzi otwartych działa **w pełni lokalnie**
(bez API i bez kont):

- **VADER** (angielski),
- **TextBlob** (angielski),
- **lokalny model HuggingFace** rozumiejący polski (domyślnie
  `nlptown/bert-base-multilingual-uncased-sentiment` — wielojęzyczny BERT
  używający WordPiece, dzięki czemu nie wymaga SentencePiece ani
  konkretnej wersji protobuf; opcjonalnie można przełączyć na
  `cardiffnlp/twitter-xlm-roberta-base-sentiment`).

Wynik trafia do folderu `raport/` jako `ankieta-RRRR-MM-DD.pdf`
i komplet PNG-ów.

---

## Instrukcja uruchomienia — krok po kroku

### 1. Wymagania wstępne

- Python **3.10+** (sprawdź: `python --version`)
- `pip` i `venv` (zwykle są w komplecie z Pythonem)
- ~2 GB miejsca na dysku (model językowy waży ok. 1 GB)
- Połączenie z internetem **tylko przy pierwszym uruchomieniu**,
  żeby pobrać wagi modelu z HuggingFace Hub. Później wszystko
  działa offline.

### 2. Sklonuj repozytorium

```bash
git clone https://github.com/buka42/ankiety.git
cd ankiety
git checkout claude/survey-report-generator-XZYll
```

### 3. Utwórz środowisko wirtualne i zainstaluj zależności

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Instalacja `torch` + `transformers` może chwilę potrwać (kilkaset MB).
Jeśli pracujesz bez GPU, wystarczy wersja CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Przygotuj plik CSV z wynikami

CSV powinien mieć:

- pierwszy wiersz z nagłówkami (nazwy pytań),
- 15 kolumn liczbowych (odpowiedzi w skali **1-5**),
- 3 kolumny tekstowe z odpowiedziami otwartymi,
- jeden wiersz na respondenta.

Skrypt sam wykrywa, które kolumny są liczbowe, a które otwarte
(na podstawie udziału wartości dających się skonwertować na liczbę).

Przykład minimalnego CSV (`dane.csv`):

```csv
Q1,Q2,...,Q15,Otwarte_1,Otwarte_2,Otwarte_3
4,2,...,5,"Stresują mnie egzaminy","Brak czasu","Wszystko ok"
...
```

### 5. Uruchom skrypt

Masz dwie opcje: CLI lub interfejs przeglądarkowy.

**CLI:**

```bash
python generuj_raport.py dane.csv
```

**Interfejs przeglądarkowy (Flask):**

```bash
python app.py
```

Następnie otwórz w przeglądarce: <http://127.0.0.1:5000>

Wgraj plik CSV przez formularz, a po zakończeniu zobaczysz stronę
z linkami do pobrania PDF i PNG-ów oraz podglądem wykresów.
Cała inference odbywa się lokalnie — przeglądarka łączy się tylko z `127.0.0.1`.

Co się dzieje:

1. Wczytanie CSV i wykrycie kolumn.
2. Obliczenie statystyk (średnia, mediana, odch. std., min, max, n).
3. Wygenerowanie histogramów (PNG).
4. Wygenerowanie macierzy korelacji Pearsona (PNG).
5. **Pierwsze uruchomienie**: pobranie wag modelu PL z HuggingFace
   do `~/.cache/huggingface`. Kolejne uruchomienia pomijają ten krok.
6. Analiza sentymentu (VADER + TextBlob + model PL).
7. Złożenie raportu PDF.

### 6. Wynik

Pliki w folderze `raport/`:

```
raport/
├── ankieta-2026-05-14.pdf
├── ankieta-2026-05-14-histogramy.png
├── ankieta-2026-05-14-korelacja.png
├── ankieta-2026-05-14-sentyment.png
└── ankieta-2026-05-14-statystyki.csv
```

Data w nazwie pliku odpowiada dniowi uruchomienia skryptu.

---

## Zmiana modelu polskiego

Domyślny model można podmienić zmienną środowiskową
`POLISH_SENTIMENT_MODEL`:

```bash
# XLM-RoBERTa, etykiety positive/neutral/negative (~1 GB)
POLISH_SENTIMENT_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment \
    python generuj_raport.py dane.csv
```

Inne sensowne opcje:

- `nlptown/bert-base-multilingual-uncased-sentiment` (domyślny, ~700 MB,
  1-5 gwiazdek automatycznie mapowane na pozytywny/neutralny/negatywny,
  nie wymaga SentencePiece),
- `cardiffnlp/twitter-xlm-roberta-base-sentiment` (~1 GB, wymaga
  poprawnie zainstalowanego `sentencepiece` + `protobuf`),
- dowolny model z HuggingFace Hub kompatybilny z `pipeline("sentiment-analysis")`.

Jeśli wybrany model ma problem z tokenizerem, skrypt **automatycznie
przełączy się** na model zapasowy (`nlptown/bert-base-multilingual-uncased-sentiment`)
i kontynuuje pracę.

---

## Typowe problemy

**`OSError: We couldn't connect to 'https://huggingface.co'`**
Pierwsze uruchomienie wymaga internetu, żeby pobrać model. Po pobraniu
wagi siedzą w `~/.cache/huggingface` i można pracować offline.

**`Error parsing line b'\x0e' in ...sentencepiece.bpe.model`**
To **nie** jest błąd `sentencepiece`, tylko **protobuf** próbujący sparsować
binarny plik jako tekst. Występuje przy ładowaniu XLM-RoBERTa, gdy:
wersja `protobuf` w środowisku jest niekompatybilna z transformers, albo
pobrany plik tokenizera jest uszkodzony, albo brak `sentencepiece`.

Najprostsze obejście — **przełącz się na domyślny BERT** (nie wymaga
SentencePiece ani protobuf):

```bash
# CLI
python generuj_raport.py dane.csv

# Web UI
python app.py
```

(od najnowszej wersji domyślnym modelem jest BERT-base-multilingual.
Skrypt także sam fallbackuje na BERT, jeśli wybrany model padnie z tym
błędem.)

Jeśli chcesz mimo wszystko użyć XLM-RoBERTa:

```bash
pip install --upgrade sentencepiece protobuf transformers tokenizers
```

a następnie usuń cache modelu i pozwól pobrać go ponownie:

- Windows (PowerShell):
  ```powershell
  Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--cardiffnlp--twitter-xlm-roberta-base-sentiment"
  ```
- Linux/macOS:
  ```bash
  rm -rf ~/.cache/huggingface/hub/models--cardiffnlp--twitter-xlm-roberta-base-sentiment
  ```

**`ModuleNotFoundError: No module named 'pandas'`** (itd.)
Środowisko wirtualne nie jest aktywne albo `pip install -r requirements.txt`
nie został uruchomiony.

**Sentyment „neutralny" dla wszystkich odpowiedzi PL przy VADER/TextBlob**
To normalne — VADER i TextBlob są wytrenowane na angielskim. Etykietę
końcową wyznacza model HF rozumiejący polski, VADER i TextBlob są
zapisywane jako kolumny pomocnicze.

**Wolna inference**
XLM-RoBERTa potrafi pociągnąć kilkanaście sekund dla 900 wpisów na CPU.
Jeśli masz GPU z CUDA, `torch` to wykorzysta automatycznie.

---

## Pliki w repozytorium

| Plik | Opis |
|------|------|
| `generuj_raport.py` | Główny skrypt generujący raport (CLI + funkcja `generuj()`) |
| `app.py` | Aplikacja Flask z formularzem uploadu CSV |
| `templates/` | Szablony HTML dla interfejsu przeglądarkowego |
| `requirements.txt` | Zależności Pythona |
| `.gitignore` | Pomija `raport/`, cache i venv |
