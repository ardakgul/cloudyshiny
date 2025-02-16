import os
import json
import math
import time
from datetime import datetime, timezone, timedelta
import yfinance as yf
import feedparser
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize sentiment analyzers
analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

# --- Normalization Functions ---
def logistic_score(current, ma, k=10):
    diff_value = (current - ma) / ma
    score = 100 / (1 + math.exp(-k * diff_value))
    return score, float(diff_value)

def logistic_score_reversed(current, ma, k=10):
    diff_value = (ma - current) / ma
    score = 100 / (1 + math.exp(-k * diff_value))
    return score, float(diff_value)

# --- Data Fetching Helpers ---
def fetch_current_and_ma(ticker, period=125):
    extra_days = period + 30
    try:
        data = yf.download(ticker, period=f'{extra_days}d', progress=False)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None, None
    if data.empty or len(data) < period:
        return None, None
    current = data['Close'].iloc[-1].item()
    ma = data['Close'].rolling(window=period).mean().iloc[-1].item()
    return current, ma

def analyze_sentiment(_text):
    return analyzer.polarity_scores(_text)['compound']

# --- Economics News with FinBERT ---
def fetch_economics_news_to_txt(txt_filename):
    rss_url = 'http://feeds.reuters.com/reuters/businessNews'
    feed = feedparser.parse(rss_url)
    now = datetime.now(timezone.utc)
    one_day = timedelta(days=1)
    news_items = []
    for entry in feed.entries:
        published = None
        if 'published_parsed' in entry:
            published = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
        if published and now - published <= one_day:
            text = entry.title
            if 'summary' in entry:
                text += ". " + entry.summary
            news_items.append(text)
    with open(txt_filename, "w", encoding="utf-8") as f:
        for item in news_items:
            f.write(item + "\n\n")
    print(f"Saved {len(news_items)} news articles to {txt_filename}")

def score_economics_news(txt_filename):
    try:
        with open(txt_filename, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"Error reading {txt_filename}: {e}")
        return 50
    if not content:
        return 50
    articles = [article.strip() for article in content.split("\n\n") if article.strip()]
    sentiments = []
    for article in articles:
        try:
            result = finbert(article)[0]
            label = result['label']
            if label.lower() == 'positive':
                sentiments.append(1)
            elif label.lower() == 'negative':
                sentiments.append(-1)
            else:
                sentiments.append(0)
        except Exception as e:
            print(f"Error scoring article: {e}")
            continue
    if sentiments:
        avg_sentiment = np.mean(sentiments)
    else:
        avg_sentiment = 0
    score = (avg_sentiment + 1) * 50
    return score

# --- Other Market Components ---
def compute_oil_gold_change_score():
    oil_ticker = 'CL=F'
    gold_ticker = 'GC=F'
    oil_current, oil_ma = fetch_current_and_ma(oil_ticker, period=125)
    gold_current, gold_ma = fetch_current_and_ma(gold_ticker, period=125)
    if oil_current is None or oil_ma is None or gold_current is None or gold_ma is None:
        return 50
    oil_score, _ = logistic_score_reversed(oil_current, oil_ma, k=10)
    gold_score, _ = logistic_score_reversed(gold_current, gold_ma, k=10)
    return (oil_score + gold_score) / 2

def compute_gold_ma_score():
    gold_ticker = 'GC=F'
    current, ma = fetch_current_and_ma(gold_ticker, period=125)
    if current is None or ma is None:
        return current, ma, 50, None
    score, diff = logistic_score_reversed(current, ma, k=10)
    return current, ma, score, diff

def compute_us10y_bonds_score():
    ticker = '^TNX'
    current, ma = fetch_current_and_ma(ticker, period=125)
    if current is None or ma is None:
        return current, ma, 50, None
    score, diff = logistic_score_reversed(current, ma, k=10)
    return current, ma, score, diff

# --- Composite CloudyShiny (Fear & Greed) Index Calculation ---
def compute_fear_greed_index():
    total_score = 0
    contributions = {}
    weights = {
        'sp500': 0.33,
        'shanghai': 0.05,
        'nikkei': 0.05,
        'hangseng': 0.05,
        'dax': 0.03,
        'cac40': 0.03,
        'bist': 0.02,
        'us10y': 0.21,
        'vix': 0.03,
        'rss': 0.03,
        'btc': 0.02,
        'oil_gold': 0.07,
        'gold_ma': 0.08,
        'economics_news': 0.03
    }
    
    # S&P500
    sp500_ticker = '^GSPC'
    current, ma = fetch_current_and_ma(sp500_ticker, period=120)
    if current is not None and ma is not None:
        sp500_score, diff = logistic_score(current, ma, k=10)
    else:
        sp500_score, diff = 50, None
    contrib = sp500_score * weights['sp500']
    contributions['S&P500'] = {'ticker': sp500_ticker, 'current': current, 'moving_average': ma,
                                 'diff_ratio': diff, 'score': sp500_score, 'weight': weights['sp500'],
                                 'contribution': contrib}
    total_score += contrib

    # Shanghai Composite
    shanghai_ticker = '000001.SS'
    current, ma = fetch_current_and_ma(shanghai_ticker, period=120)
    if current is not None and ma is not None:
        shanghai_score, diff = logistic_score(current, ma, k=10)
    else:
        shanghai_score, diff = 50, None
    contrib = shanghai_score * weights['shanghai']
    contributions['Shanghai'] = {'ticker': shanghai_ticker, 'current': current, 'moving_average': ma,
                                   'diff_ratio': diff, 'score': shanghai_score, 'weight': weights['shanghai'],
                                   'contribution': contrib}
    total_score += contrib

    # Nikkei 225
    nikkei_ticker = '^N225'
    current, ma = fetch_current_and_ma(nikkei_ticker, period=120)
    if current is not None and ma is not None:
        nikkei_score, diff = logistic_score(current, ma, k=10)
    else:
        nikkei_score, diff = 50, None
    contrib = nikkei_score * weights['nikkei']
    contributions['Nikkei'] = {'ticker': nikkei_ticker, 'current': current, 'moving_average': ma,
                                 'diff_ratio': diff, 'score': nikkei_score, 'weight': weights['nikkei'],
                                 'contribution': contrib}
    total_score += contrib

    # Hang Seng
    hangseng_ticker = '^HSI'
    current, ma = fetch_current_and_ma(hangseng_ticker, period=120)
    if current is not None and ma is not None:
        hangseng_score, diff = logistic_score(current, ma, k=10)
    else:
        hangseng_score, diff = 50, None
    contrib = hangseng_score * weights['hangseng']
    contributions['Hang Seng'] = {'ticker': hangseng_ticker, 'current': current, 'moving_average': ma,
                                   'diff_ratio': diff, 'score': hangseng_score, 'weight': weights['hangseng'],
                                   'contribution': contrib}
    total_score += contrib

    # DAX
    dax_ticker = '^GDAXI'
    current, ma = fetch_current_and_ma(dax_ticker, period=120)
    if current is not None and ma is not None:
        dax_score, diff = logistic_score(current, ma, k=10)
    else:
        dax_score, diff = 50, None
    contrib = dax_score * weights['dax']
    contributions['DAX'] = {'ticker': dax_ticker, 'current': current, 'moving_average': ma,
                              'diff_ratio': diff, 'score': dax_score, 'weight': weights['dax'],
                              'contribution': contrib}
    total_score += contrib

    # CAC40
    cac40_ticker = '^FCHI'
    current, ma = fetch_current_and_ma(cac40_ticker, period=120)
    if current is not None and ma is not None:
        cac40_score, diff = logistic_score(current, ma, k=10)
    else:
        cac40_score, diff = 50, None
    contrib = cac40_score * weights['cac40']
    contributions['CAC40'] = {'ticker': cac40_ticker, 'current': current, 'moving_average': ma,
                              'diff_ratio': diff, 'score': cac40_score, 'weight': weights['cac40'],
                              'contribution': contrib}
    total_score += contrib

    # BIST100 (125-day MA)
    bist_ticker = 'XU100.IS'
    current, ma = fetch_current_and_ma(bist_ticker, period=125)
    if current is not None and ma is not None:
        bist_score, diff = logistic_score(current, ma, k=10)
    else:
        bist_score, diff = 50, None
    contrib = bist_score * weights['bist']
    contributions['BIST100'] = {'ticker': bist_ticker, 'current': current, 'moving_average': ma,
                                 'diff_ratio': diff, 'score': bist_score, 'weight': weights['bist'],
                                 'contribution': contrib}
    total_score += contrib

    # US 10Y Bonds
    bonds_current, bonds_ma, bonds_score, diff = compute_us10y_bonds_score()
    contrib = bonds_score * weights['us10y']
    contributions['US 10Y Bonds'] = {'ticker': '^TNX', 'current': bonds_current, 'moving_average': bonds_ma,
                                     'diff_ratio': diff, 'score': bonds_score, 'weight': weights['us10y'],
                                     'contribution': contrib,
                                     'details': 'If yield > 125-day MA then fear (low score), else greed (high score)'}
    total_score += contrib

    # VIX (reversed scoring)
    vix_ticker = '^VIX'
    current, ma = fetch_current_and_ma(vix_ticker, period=120)
    if current is not None and ma is not None:
        vix_score, diff = logistic_score_reversed(current, ma, k=10)
    else:
        vix_score, diff = 50, None
    contrib = vix_score * weights['vix']
    contributions['VIX'] = {'ticker': vix_ticker, 'current': current, 'moving_average': ma,
                            'reversed_diff_ratio': diff, 'score': vix_score, 'weight': weights['vix'],
                            'contribution': contrib}
    total_score += contrib

    # RSS News Sentiment
    rss_urls = [
        'http://feeds.reuters.com/reuters/topNews',
        'http://feeds.reuters.com/reuters/businessNews'
    ]
    rss_score = fetch_rss_sentiment(rss_urls)
    contrib = rss_score * weights['rss']
    contributions['RSS News'] = {'score': rss_score, 'weight': weights['rss'],
                                  'contribution': contrib,
                                  'details': 'Aggregated sentiment from RSS feeds via VADER (last 24 hours)'}
    total_score += contrib

    # BTC (with k=3)
    btc_ticker = 'BTC-USD'
    current, ma = fetch_current_and_ma(btc_ticker, period=120)
    if current is not None and ma is not None:
        btc_score, diff = logistic_score(current, ma, k=3)
    else:
        btc_score, diff = 50, None
    contrib = btc_score * weights['btc']
    contributions['BTC'] = {'ticker': btc_ticker, 'current': current, 'moving_average': ma,
                              'diff_ratio': diff, 'score': btc_score, 'weight': weights['btc'],
                              'contribution': contrib}
    total_score += contrib

    # Oil & Gold Daily
    oil_gold_score = compute_oil_gold_change_score()
    contrib = oil_gold_score * weights['oil_gold']
    contributions['Oil & Gold Daily'] = {'score': oil_gold_score, 'weight': weights['oil_gold'],
                                          'contribution': contrib,
                                          'details': 'Average reversed score based on 125-day MA for oil & gold'}
    total_score += contrib

    # Gold MA
    gold_current, gold_ma, gold_score, diff = compute_gold_ma_score()
    contrib = gold_score * weights['gold_ma']
    contributions['Gold MA'] = {'ticker': 'GC=F', 'current': gold_current, 'moving_average': gold_ma,
                                 'diff_ratio': diff, 'score': gold_score, 'weight': weights['gold_ma'],
                                 'contribution': contrib,
                                 'details': 'Gold above MA contributes to fear (reversed score).'}
    total_score += contrib

    # Economics News via FinBERT
    news_txt_file = "economics_news.txt"
    fetch_economics_news_to_txt(news_txt_file)
    econ_news_score = score_economics_news(news_txt_file)
    contrib = econ_news_score * weights['economics_news']
    contributions['Economics News'] = {'score': econ_news_score, 'weight': weights['economics_news'],
                                        'contribution': contrib,
                                        'details': 'Sentiment score of economics news analyzed by FinBERT'}
    total_score += contrib

    return total_score, contributions

def fetch_rss_sentiment(rss_urls):
    sentiments = []
    now = datetime.now(timezone.utc)
    one_day = timedelta(days=1)
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published = None
                if 'published_parsed' in entry:
                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
                if published is not None and now - published > one_day:
                    continue
                text = entry.title
                if 'summary' in entry:
                    text += ". " + entry.summary
                sentiments.append(analyze_sentiment(text))
        except Exception as e:
            print(f"Error fetching RSS feed {url}: {e}")
    if sentiments:
        avg_sentiment = np.mean(sentiments)
    else:
        avg_sentiment = 0
    return (avg_sentiment + 1) * 50

# --- Historical Data Aggregation Functions ---
def load_history():
    history_file = "data/history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            history = []
    else:
        history = []
    return history

def save_history(history):
    with open("data/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def parse_timestamp(ts):
    return datetime.fromisoformat(ts.replace("Z", ""))

def aggregate_by_day(history, start_date):
    filtered = [record for record in history if parse_timestamp(record["timestamp"]) >= start_date]
    groups = {}
    for record in filtered:
        dt = parse_timestamp(record["timestamp"])
        day = dt.strftime("%Y-%m-%d")
        groups.setdefault(day, []).append(record["index_score"])
    labels = sorted(groups.keys())
    data = [sum(groups[day]) / len(groups[day]) for day in labels]
    return labels, data

def aggregate_by_week(history, start_date):
    filtered = [record for record in history if parse_timestamp(record["timestamp"]) >= start_date]
    groups = {}
    for record in filtered:
        dt = parse_timestamp(record["timestamp"])
        week = dt.strftime("%Y-W%U")
        groups.setdefault(week, []).append(record["index_score"])
    labels = sorted(groups.keys())
    data = [sum(groups[week]) / len(groups[week]) for week in labels]
    return labels, data

def aggregate_by_quarter(history, start_date):
    filtered = [record for record in history if parse_timestamp(record["timestamp"]) >= start_date]
    groups = {}
    for record in filtered:
        dt = parse_timestamp(record["timestamp"])
        quarter = (dt.month - 1) // 3 + 1
        key
