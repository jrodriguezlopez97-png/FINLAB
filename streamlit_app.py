import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="FINLAB | Intelligence Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background:radial-gradient(circle at 15% 0%,#142b49 0%,#08111f 48%,#070d17 100%);color:#e5edf7}
[data-testid="stSidebar"] {background:#0b1728;border-right:1px solid #1e3a5f}
.block-container {padding-top:1.4rem}
.metric-card {background:#142943;border:1px solid #24476b;border-radius:14px;padding:14px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:2.1rem;font-weight:850;letter-spacing:.14em;color:#eaf4ff">◈ FINLAB</div>
<div style="font-size:.78rem;letter-spacing:.22em;color:#5fc8ff">INTELLIGENCE LAB</div>
<p style="color:#91a6bd">Investigación cuantitativa · Analítica de riesgo · Laboratorio educativo</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ◈ FINLAB")
st.sidebar.caption("Configuración del análisis")
symbols_text = st.sidebar.text_input("Símbolos separados por comas", "SPY, QQQ, AAPL, MSFT, ^GSPC")
period = st.sidebar.selectbox("Historial", ["1y", "2y", "5y", "10y"], index=2)
rf_pct = st.sidebar.number_input("Supuesto anual libre de riesgo (%)", 0.0, 20.0, 0.0, 0.25)
benchmark = st.sidebar.text_input("Índice de referencia para beta", "^GSPC").strip().upper()
st.sidebar.caption("Herramienta educativa. No ejecuta operaciones ni ofrece recomendaciones.")

symbols = list(dict.fromkeys(x.strip().upper() for x in symbols_text.split(",") if x.strip()))
if benchmark and benchmark not in symbols:
    symbols.append(benchmark)

@st.cache_data(ttl=900, show_spinner=False)
def get_prices(tickers, period):
    series, errors = {}, {}
    for ticker in tickers:
        try:
            frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
            if frame.empty or "Close" not in frame:
                errors[ticker] = "Sin datos disponibles"
                continue
            s = frame["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if len(s):
                series[ticker] = s.rename(ticker)
        except Exception as exc:
            errors[ticker] = str(exc)[:160]
    if not series:
        return pd.DataFrame(), errors
    return pd.concat(series.values(), axis=1).sort_index(), errors

with st.spinner("Descargando datos históricos..."):
    prices, errors = get_prices(tuple(symbols), period)

if prices.empty:
    st.error("No se recibieron datos. Revisa los símbolos o inténtalo más tarde.")
    if errors:
        st.write(errors)
    st.stop()

prices = prices.dropna()
returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
if returns.empty:
    st.error("No hay suficientes rendimientos para calcular indicadores.")
    st.stop()

if benchmark not in returns.columns:
    benchmark = returns.columns[0]

rf_daily = (1 + rf_pct / 100) ** (1 / 252) - 1
rows, dd_series = [], {}
for ticker in returns.columns:
    r = returns[ticker]
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    dd_series[ticker] = dd * 100
    vol = r.std() * np.sqrt(252)
    ann_return = wealth.iloc[-1] ** (252 / len(r)) - 1
    excess = r - rf_daily
    sharpe = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    downside = np.minimum(excess, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    sortino = excess.mean() / downside_dev * np.sqrt(252) if downside_dev > 0 else np.nan
    q05 = r.quantile(.05)
    var95 = -q05
    cvar95 = -r[r <= q05].mean()
    market = returns[benchmark]
    beta = r.cov(market) / market.var() if market.var() > 0 else np.nan
    rows.append({
        "Activo": ticker,
        "Rentabilidad anualizada (%)": ann_return * 100,
        "Volatilidad anualizada (%)": vol * 100,
        "Máx. drawdown (%)": dd.min() * 100,
        "VaR diario 95% (%)": var95 * 100,
        "CVaR diario 95% (%)": cvar95 * 100,
        "Sharpe": sharpe, "Sortino": sortino,
        f"Beta vs {benchmark}": beta,
        "Peor día (%)": r.min() * 100
    })

metrics = pd.DataFrame(rows).set_index("Activo")
corr = returns.corr()

st.title("Panel de control")
st.caption(f"Datos ajustados · {prices.index.min().date()} a {prices.index.max().date()} · {len(prices):,} sesiones comunes · caché de 15 minutos")
a, b, c, d = st.columns(4)
a.metric("Activos analizados", len(prices.columns))
b.metric("Sesiones comunes", f"{len(prices):,}")
c.metric("Referencia beta", benchmark)
d.metric("Historial", period)

if errors:
    with st.expander("Avisos de descarga"):
        for ticker, message in errors.items():
            st.warning(f"{ticker}: {message}")

tab_market, tab_risk, tab_corr, tab_data, tab_guide = st.tabs(["◈ Mercado", "⬡ Riesgo", "⌘ Correlaciones", "▤ Datos", "ⓘ Guía"])

with tab_market:
    st.subheader("Evolución comparativa (base 100)")
    normalized = prices / prices.iloc[0] * 100
    fig = px.line(normalized, x=normalized.index, y=normalized.columns, template="plotly_dark",
                  labels={"value":"Índice base 100","index":"Fecha","variable":"Activo"})
    fig.update_layout(height=420, margin=dict(l=8,r=8,t=25,b=8))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Rentabilidad anualizada histórica")
    f = px.bar(metrics.reset_index(), x="Activo", y="Rentabilidad anualizada (%)",
               template="plotly_dark", color="Rentabilidad anualizada (%)", color_continuous_scale="Teal")
    f.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(f, use_container_width=True)

with tab_risk:
    st.subheader("Indicadores de riesgo")
    st.caption("VaR y CVaR son estimaciones históricas diarias, no límites garantizados de pérdida.")
    st.dataframe(metrics.style.format("{:.2f}"), use_container_width=True)
    f = px.bar(metrics.reset_index(), x="Activo", y="Volatilidad anualizada (%)",
               template="plotly_dark", color="Volatilidad anualizada (%)", color_continuous_scale="Sunset")
    f.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(f, use_container_width=True)
    st.subheader("Drawdown histórico")
    dd = pd.DataFrame(dd_series)
    f = px.line(dd, x=dd.index, y=dd.columns, template="plotly_dark",
                labels={"value":"Drawdown (%)","index":"Fecha","variable":"Activo"})
    f.update_layout(height=380)
    st.plotly_chart(f, use_container_width=True)

with tab_corr:
    st.subheader("Matriz de correlaciones")
    f = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
                  aspect="auto", template="plotly_dark")
    f.update_layout(height=460)
    st.plotly_chart(f, use_container_width=True)
    st.info("La correlación describe la relación lineal histórica de los rendimientos; no garantiza que se mantenga.")

with tab_data:
    st.subheader("Últimos precios")
    st.dataframe(prices.tail(30).sort_index(ascending=False), use_container_width=True)
    st.download_button("Descargar precios CSV", prices.to_csv().encode("utf-8"),
                       "finlab_precios.csv", "text/csv", use_container_width=True)
    st.download_button("Descargar métricas CSV", metrics.to_csv().encode("utf-8"),
                       "finlab_metricas.csv", "text/csv", use_container_width=True)

with tab_guide:
    st.subheader("Guía de indicadores")
    st.markdown("""
- **Rentabilidad anualizada:** tasa geométrica anual equivalente en la muestra.
- **Volatilidad:** desviación típica diaria anualizada con √252.
- **Drawdown:** mayor caída de la curva de riqueza desde un máximo anterior.
- **VaR 95% diario:** percentil histórico del 5% de rendimientos expresado como pérdida positiva.
- **CVaR 95% diario:** pérdida media en el 5% de días más adversos.
- **Sharpe:** exceso de rendimiento medio sobre el supuesto libre de riesgo dividido por volatilidad.
- **Sortino:** exceso de rendimiento medio dividido por desviación bajista.
- **Beta:** covarianza con el índice de referencia dividida por su varianza.

Los datos proceden de un proveedor externo y pueden contener huecos o revisiones. Las métricas históricas no predicen el futuro y no constituyen recomendaciones financieras.
""")

st.divider()
st.markdown("<p style='text-align:center;color:#7890aa;font-size:.78rem'>FINLAB · EDUCATIONAL QUANTITATIVE RESEARCH · NO TRADING EXECUTION</p>", unsafe_allow_html=True)
