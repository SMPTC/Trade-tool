import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Trade Signal Tool", layout="wide")


# ---------- DATA ----------
@st.cache_data(ttl=600)
def load(sym, period="1y"):
    df = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=3600)
def fundamentals(sym):
    try:
        return yf.Ticker(sym).info
    except Exception:
        return {}


# ---------- INDICATORS ----------
def add_indicators(df):
    c = df["Close"]
    for n in (20, 50, 150, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    df["RSI"] = 100 - 100 / (1 + up / dn)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9).mean()
    tr = pd.concat([df.High - df.Low, (df.High - c.shift()).abs(), (df.Low - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["VolAvg"] = df["Volume"].rolling(20).mean()
    return df


def pivots(df, w=5):
    hi, lo = [], []
    for i in range(w, len(df) - w):
        seg = df.iloc[i - w : i + w + 1]
        if df.High.iloc[i] == seg.High.max():
            hi.append((i, df.High.iloc[i]))
        if df.Low.iloc[i] == seg.Low.min():
            lo.append((i, df.Low.iloc[i]))
    return hi, lo


def levels(df, hi, lo):
    price = df.Close.iloc[-1]
    sup = sorted([p for _, p in lo if p < price], reverse=True)
    res = sorted([p for _, p in hi if p > price])
    return (sup[0] if sup else df.Low.tail(60).min()), (res[0] if res else df.High.tail(60).max())


# ---------- STRATEGIES (each returns score, reasons) ----------
def minervini(df):
    r = df.iloc[-1]
    hi52, lo52 = df.High.tail(252).max(), df.Low.tail(252).min()
    ok = (r.Close > r.SMA150 > r.SMA200 and r.SMA50 > r.SMA150
          and df.SMA200.iloc[-1] > df.SMA200.iloc[-21]
          and r.Close >= 0.75 * hi52 and r.Close >= 1.3 * lo52)
    if ok:
        return 2, ["Minervini trend template pass: strong Stage-2 uptrend"]
    if r.Close < r.SMA50 < r.SMA200:
        return -2, ["Downtrend: price < 50 SMA < 200 SMA (Stage-4)"]
    return 0, []


def breakout(df):
    r = df.iloc[-1]
    h20, l20 = df.High.iloc[-21:-1].max(), df.Low.iloc[-21:-1].min()
    vol = r.Volume > 1.5 * r.VolAvg
    if r.Close > h20 and vol:
        return 2, [f"O'Neil/Darvas breakout above {h20:.2f} with high volume"]
    if r.Close < l20 and vol:
        return -2, [f"Breakdown below {l20:.2f} with high volume"]
    if r.Close > h20:
        return 0.5, ["Breakout but volume weak, confirmation needed"]
    return 0, []


def livermore(df):
    r = df.iloc[-1]
    h50 = df.High.iloc[-51:-1].max()
    if r.Close > h50:
        return 1, [f"Livermore pivotal point: new 50-day high above {h50:.2f}"]
    return 0, []


def pullback(df):
    r = df.iloc[-1]
    if r.SMA50 > r.SMA200 and abs(r.Close - r.SMA20) / r.Close < 0.02 and 40 <= r.RSI <= 55:
        return 1.5, ["Pullback to 20 SMA in an uptrend, RSI cooled to " + f"{r.RSI:.0f}"]
    if r.SMA50 < r.SMA200 and abs(r.Close - r.SMA20) / r.Close < 0.02 and 45 <= r.RSI <= 60:
        return -1.5, ["Weak rally into 20 SMA in a downtrend, sellers likely"]
    return 0, []


def patterns(df, hi, lo):
    tol = 0.02
    if len(hi) >= 2 and hi[-1][0] > len(df) - 60:
        a, b = hi[-2][1], hi[-1][1]
        if abs(a - b) / a < tol and df.Close.iloc[-1] < b * 0.98:
            return -1.5, ["Double top pattern, price turning down"]
    if len(lo) >= 2 and lo[-1][0] > len(df) - 60:
        a, b = lo[-2][1], lo[-1][1]
        if abs(a - b) / a < tol and df.Close.iloc[-1] > b * 1.02:
            return 1.5, ["Double bottom pattern, price turning up"]
    return 0, []


def momentum(df):
    r = df.iloc[-1]
    s, why = 0, []
    if r.MACD > r.MACDsig:
        s += 0.5; why.append("MACD bullish")
    else:
        s -= 0.5; why.append("MACD bearish")
    if r.RSI > 75:
        s -= 1; why.append(f"RSI {r.RSI:.0f} overbought, careful")
    elif r.RSI < 25:
        s += 1; why.append(f"RSI {r.RSI:.0f} oversold, bounce possible")
    return s, why


def fundamental_score(info):
    s, why = 0, []
    pe, g, de = info.get("trailingPE"), info.get("earningsGrowth"), info.get("debtToEquity")
    roe, ins = info.get("returnOnEquity"), info.get("heldPercentInsiders")
    if g is not None:
        s += 1 if g > 0.15 else (-1 if g < 0 else 0)
        why.append(f"Profit growth {g*100:.0f}%")
    if de is not None:
        s += 1 if de < 80 else (-1 if de > 200 else 0)
        why.append(f"Debt/Equity {de:.0f}")
    if roe is not None:
        s += 1 if roe > 0.15 else (-0.5 if roe < 0.08 else 0)
        why.append(f"ROE {roe*100:.0f}%")
    if pe is not None:
        s += -1 if pe > 60 else (0.5 if pe < 25 else 0)
        why.append(f"PE {pe:.0f}")
    if ins is not None:
        s += 0.5 if ins > 0.5 else 0
        why.append(f"Promoter/insider holding {ins*100:.0f}%")
    return s, why


# ---------- UI ----------
st.title("Trade Signal Tool")
sym = st.sidebar.text_input("Symbol (NSE: RELIANCE.NS, Nifty: ^NSEI, Crypto: BTC-USD, Forex: USDINR=X)", "RELIANCE.NS")
use_fund = st.sidebar.checkbox("Fundamentals include karo", True)
run = st.sidebar.button("Analyse", type="primary")

if run or sym:
    df = load(sym)
    if df.empty or len(df) < 220:
        st.error("Data nahi mila ya kam hai (200+ din chahiye). Symbol check karo.")
        st.stop()
    df = add_indicators(df)
    hi, lo = pivots(df)
    sup, res = levels(df, hi, lo)

    tech, reasons = 0, []
    for fn in (minervini, breakout, livermore, pullback, momentum):
        s, w = fn(df); tech += s; reasons += w
    s, w = patterns(df, hi, lo); tech += s; reasons += w

    fund, fwhy = (0, [])
    if use_fund:
        fund, fwhy = fundamental_score(fundamentals(sym))
    total = tech + fund * 0.7

    if total >= 3:
        signal = "BUY"
    elif total <= -3:
        signal = "SELL"
    else:
        signal = "WAIT"

    price, atr = df.Close.iloc[-1], df.ATR.iloc[-1]
    if signal == "BUY":
        sl = max(price - 1.5 * atr, sup * 0.99) if sup > price - 3 * atr else price - 1.5 * atr
        tgt = price + 2 * (price - sl)
    elif signal == "SELL":
        sl = min(price + 1.5 * atr, res * 1.01) if res < price + 3 * atr else price + 1.5 * atr
        tgt = price - 2 * (sl - price)
    else:
        sl = tgt = None

    left, right = st.columns([2, 1])
    with left:
        color = {"BUY": "green", "SELL": "red", "WAIT": "orange"}[signal]
        st.markdown(f"## Signal: :{color}[{signal}]  (score {total:.1f})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{price:.2f}")
        c2.metric("Entry", f"{price:.2f}" if sl else "-")
        c3.metric("Stoploss", f"{sl:.2f}" if sl else "-")
        c4.metric("Target (1:2)", f"{tgt:.2f}" if tgt else "-")

        d = df.tail(150)
        fig = go.Figure(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close))
        fig.add_trace(go.Scatter(x=d.index, y=d.SMA50, name="SMA50"))
        fig.add_trace(go.Scatter(x=d.index, y=d.SMA200, name="SMA200"))
        fig.add_hline(y=sup, line_dash="dash", line_color="green", annotation_text="Support")
        fig.add_hline(y=res, line_dash="dash", line_color="red", annotation_text="Resistance")
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Reason chat")
        with st.chat_message("assistant"):
            st.write(f"**Support:** {sup:.2f}  |  **Resistance:** {res:.2f}")
        with st.chat_message("assistant"):
            st.write("**Technical**")
            for x in reasons:
                st.write("- " + x)
        if fwhy:
            with st.chat_message("assistant"):
                st.write("**Fundamental**")
                for x in fwhy:
                    st.write("- " + x)
        with st.chat_message("assistant"):
            st.write(f"**Final:** {signal}. " + ("Confusing/mixed setup, isliye wait." if signal == "WAIT" else "Stoploss zaroor lagana."))

    st.caption("Educational tool hai, financial advice nahi. Pehle paper trading karo.")
