# Bitcoin Price Prediction Flask App

This project is a Flask-wrapped Bitcoin price direction predictor. It collects live market, funding, macro news, AI/semiconductor stock, and commodity signals, combines them with configurable rule-based weights, and displays a short-horizon BTC forecast in a web dashboard.

The submitted version contains only the working web application and runtime files. Training scripts, model fine-tuning reports, generated model artifacts, prediction logs, caches, and monitoring logs are intentionally excluded from the GitHub repository.

## Features

- Flask web interface for running the predictor from a browser.
- Selectable 1-hour and 4-hour prediction horizons.
- Live BTC market data from CoinGecko and Fear & Greed Index data.
- Binance funding/open-interest collectors with graceful error handling if unavailable.
- Optional NewsAPI-powered U.S. policy and geopolitics sentiment signals.
- AI chip stock and oil/gold factor inputs through `yfinance`.
- Factor breakdown, confidence label, composite score, BTC spot price, projected range, warnings, and recent local prediction summary.

## Project Structure

```text
.
├── app.py                  # Flask web application
├── script.py               # Prediction engine, collectors, scoring, and CLI helpers
├── ml_feature_contract.py  # Shared feature schema used by optional shadow logic
├── config.json             # Runtime configuration and factor weights
├── requirements.txt        # Python dependencies
├── .env.example            # Optional environment variable template
├── templates/
│   └── index.html          # Flask page template
└── static/
    └── app.css             # Dashboard styling
```

## Requirements

- Python 3.10 or newer
- Internet access for live market/news data
- Optional: a NewsAPI key for policy and geopolitics news sentiment

The app still runs without `NEWS_API_KEY`; the affected news factors are marked unavailable and the remaining factors are used.

## Setup

1. Clone the repository and enter the project folder:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Optional: configure NewsAPI access:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
NEWS_API_KEY=your_newsapi_key_here
```

## Running the Flask App

Start the web server:

```bash
python app.py
```

Open the local app in a browser:

```text
http://127.0.0.1:5050
```

Choose either the 1h or 4h horizon and click **Run Current Engine** to generate a prediction.

## Optional CLI Usage

The same prediction engine can also be run directly:

```bash
python script.py --hours 4
```

For a 1-hour prediction:

```bash
python script.py --hours 1
```

The CLI may append entries to `predictions.log`; this file is ignored by Git and is not required for the Flask app submission.

## Configuration

`config.json` controls:

- Default prediction horizon
- Factor weights
- Binance futures symbol
- Stock and commodity tickers
- News keywords
- Accuracy and cache-related settings

The Flask app disables optional shadow model inference at runtime, so no `.pkl` model file is required to run the web application.

## Files Excluded From GitHub

The `.gitignore` excludes local/generated files such as:

- `.env` and API keys
- Virtual environments
- `predictions.log`
- `news_cache.json`
- `open_interest_cache.json`
- `runlogs/`
- model artifacts in `models/`
- tuning/report documents in `docs/`
- monitoring/automation files
- training and evaluation scripts

This keeps the repository focused on the runnable Flask application only excluding the fine tuning model stage.

## Notes

This project is for educational use. The prediction output is a rule-based analysis of public market and news signals, not financial advice.
