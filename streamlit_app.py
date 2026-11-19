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

tab_market, tab_risk, tab_corr, tab_data, tab_guide, tab_ai, tab_portfolio, tab_risk_center = st.tabs([
    "◈ Mercado",
    "⬡ Riesgo",
    "⌘ Correlaciones",
    "▤ Datos",
    "ⓘ Guía",
    "🧠 FINLAB AI Analyst",
    "📊 Portfolio Simulator",
    "🛡️ Risk Intelligence"
])

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

with tab_ai:
    st.subheader("🧠 FINLAB AI Analyst")
    st.caption("Analista educativo basado en reglas cuantitativas transparentes.")

    st.info(
        "Este módulo interpreta datos históricos. "
        "No predice el mercado ni recomienda comprar o vender activos."
    )

    selected = st.selectbox(
        "Selecciona un activo para analizar",
        metrics.index.tolist()
    )

    m = metrics.loc[selected]

    st.markdown(f"### Análisis de {selected}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rentabilidad anualizada",
        f"{m['Rentabilidad anualizada (%)']:.2f}%"
    )

    col2.metric(
        "Volatilidad anualizada",
        f"{m['Volatilidad anualizada (%)']:.2f}%"
    )

    col3.metric(
        "Máximo drawdown",
        f"{m['Máx. drawdown (%)']:.2f}%"
    )

    st.markdown("### Interpretación de indicadores")

    median_vol = metrics["Volatilidad anualizada (%)"].median()
    asset_vol = m["Volatilidad anualizada (%)"]

    if asset_vol > median_vol:
        st.warning(
            "La volatilidad de este activo está por encima "
            "de la mediana de los activos analizados."
        )
    else:
        st.success(
            "La volatilidad de este activo está igual o por debajo "
            "de la mediana de los activos analizados."
        )

    drawdown = m["Máx. drawdown (%)"]

    st.write(
        f"**Drawdown:** el máximo descenso histórico calculado "
        f"desde un máximo anterior fue del {abs(drawdown):.2f}%."
    )

    var_value = m["VaR diario 95% (%)"]
    cvar_value = m["CVaR diario 95% (%)"]

    st.write(
        f"**VaR diario 95%:** {var_value:.2f}%. "
        "Es una estimación histórica del umbral de pérdida "
        "que no se superó en aproximadamente el 95% de las sesiones."
    )

    st.write(
        f"**CVaR diario 95%:** {cvar_value:.2f}%. "
        "Estima la pérdida media en las sesiones que quedaron "
        "en el 5% más desfavorable."
    )

    sharpe = m["Sharpe"]
    sortino = m["Sortino"]

    st.write(
        f"**Sharpe:** {sharpe:.2f}. "
        "Relaciona el exceso de rentabilidad histórica con la volatilidad."
        if pd.notna(sharpe)
        else "**Sharpe:** no disponible."
    )

    st.write(
        f"**Sortino:** {sortino:.2f}. "
        "Relaciona el exceso de rentabilidad con la volatilidad bajista."
        if pd.notna(sortino)
        else "**Sortino:** no disponible."
    )

    st.markdown("### Comparación con los demás activos")

    comparison = metrics[
        [
            "Rentabilidad anualizada (%)",
            "Volatilidad anualizada (%)",
            "Máx. drawdown (%)",
            "Sharpe",
            "Sortino"
        ]
    ].sort_values(
        "Volatilidad anualizada (%)",
        ascending=False
    )

    st.dataframe(
        comparison.style.format("{:.2f}"),
        use_container_width=True
    )

    st.markdown("### Conclusión educativa")

    st.write(
        f"{selected} presenta una volatilidad anualizada "
        f"del {asset_vol:.2f}%, frente a una mediana del "
        f"{median_vol:.2f}% en el conjunto analizado."
    )

    st.write(
        "Estos resultados describen el comportamiento pasado "
        "durante el periodo seleccionado. No permiten asegurar "
        "cómo se comportará el activo en el futuro."
    )

    st.caption(
        "FINLAB AI Analyst V1 · Análisis automatizado mediante reglas, "
        "no inteligencia artificial generativa."
    )


with tab_portfolio:
    st.subheader("📊 FINLAB Portfolio Simulator")
    st.caption("Simulación histórica de carteras hipotéticas.")

    st.info(
        "Simulador educativo con datos históricos. "
        "No utiliza dinero real ni ejecuta operaciones."
    )

    available_assets = returns.columns.tolist()

    selected_assets = st.multiselect(
        "Selecciona los activos de tu cartera",
        available_assets,
        default=available_assets
    )

    if not selected_assets:
        st.warning("Selecciona al menos un activo.")
    else:
        st.markdown("### Distribución hipotética")

        raw_weights = {}

        default_weight = round(100 / len(selected_assets))

        for ticker in selected_assets:
            raw_weights[ticker] = st.slider(
                f"Peso de {ticker} (%)",
                min_value=0,
                max_value=100,
                value=default_weight,
                key=f"portfolio_weight_{ticker}"
            )

        total_weight = sum(raw_weights.values())

        if total_weight == 0:
            st.warning(
                "Todos los pesos están a cero. "
                "Se utilizará una distribución igualitaria."
            )
            weights = pd.Series(
                1 / len(selected_assets),
                index=selected_assets
            )
        else:
            weights = pd.Series(raw_weights) / total_weight

        st.markdown("### Distribución normalizada")

        weight_table = pd.DataFrame({
            "Activo": weights.index,
            "Peso efectivo (%)": weights.values * 100
        })

        st.dataframe(
            weight_table.style.format({
                "Peso efectivo (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

        f = px.pie(
            weight_table,
            names="Activo",
            values="Peso efectivo (%)",
            hole=0.45,
            template="plotly_dark"
        )
        f.update_layout(height=350)
        st.plotly_chart(f, use_container_width=True)

        # Rendimientos diarios de la cartera
        portfolio_returns = (
            returns[selected_assets] @ weights
        ).dropna()

        # Evolución acumulada
        portfolio_wealth = (1 + portfolio_returns).cumprod()

        # Indicadores
        portfolio_vol = (
            portfolio_returns.std() * np.sqrt(252)
        )

        portfolio_ann_return = (
            portfolio_wealth.iloc[-1] **
            (252 / len(portfolio_returns)) - 1
        )

        portfolio_drawdown = (
            portfolio_wealth /
            portfolio_wealth.cummax() - 1
        )

        portfolio_max_dd = portfolio_drawdown.min()

        portfolio_sharpe = (
            (portfolio_returns.mean() - rf_daily) /
            portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0 else np.nan
        )

        portfolio_q05 = portfolio_returns.quantile(0.05)
        portfolio_var = -portfolio_q05 * 100

        portfolio_cvar = (
            -portfolio_returns[
                portfolio_returns <= portfolio_q05
            ].mean() * 100
        )

        st.divider()
        st.markdown("### Resultados de la cartera")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rentabilidad anualizada",
            f"{portfolio_ann_return * 100:.2f}%"
        )

        c2.metric(
            "Volatilidad anualizada",
            f"{portfolio_vol * 100:.2f}%"
        )

        c3.metric(
            "Máximo drawdown",
            f"{portfolio_max_dd * 100:.2f}%"
        )

        c4.metric(
            "Sharpe",
            f"{portfolio_sharpe:.2f}"
            if pd.notna(portfolio_sharpe)
            else "No disponible"
        )

        st.write(
            f"**VaR diario histórico 95%:** "
            f"{portfolio_var:.2f}%"
        )

        st.write(
            f"**CVaR diario histórico 95%:** "
            f"{portfolio_cvar:.2f}%"
        )

        # Comparación histórica
        st.markdown("### Evolución comparativa")

        comparison_prices = (
            prices[selected_assets] /
            prices[selected_assets].iloc[0] * 100
        )

        portfolio_index = (
            portfolio_wealth /
            portfolio_wealth.iloc[0] * 100
        )

        comparison_prices["Cartera hipotética"] = (
            portfolio_index
        )

        f = px.line(
            comparison_prices,
            template="plotly_dark",
            labels={
                "value": "Índice base 100",
                "index": "Fecha",
                "variable": "Activo"
            }
        )

        f.update_layout(height=450)
        st.plotly_chart(f, use_container_width=True)

        # Drawdown de la cartera
        st.markdown("### Drawdown de la cartera")

        dd_portfolio = portfolio_drawdown * 100

        f = px.area(
            x=dd_portfolio.index,
            y=dd_portfolio.values,
            labels={
                "x": "Fecha",
                "y": "Drawdown (%)"
            },
            template="plotly_dark"
        )

        f.update_layout(height=320)
        st.plotly_chart(f, use_container_width=True)

        st.caption(
            "Supuesto: ponderaciones objetivo constantes, "
            "con rebalanceo diario implícito. No se incluyen "
            "comisiones, impuestos ni deslizamiento. "
            "Los resultados históricos no predicen el futuro."
        )


with tab_risk_center:
    st.subheader("🛡️ FINLAB Risk Intelligence Center")
    st.caption("Pruebas históricas de resistencia y diversificación.")

    st.info(
        "Los escenarios se calculan con rendimientos históricos. "
        "No representan pérdidas máximas posibles ni predicen el futuro."
    )

    risk_assets = st.multiselect(
        "Activos para el análisis",
        returns.columns.tolist(),
        default=returns.columns.tolist(),
        key="risk_center_assets"
    )

    window = st.selectbox(
        "Duración del escenario histórico",
        [5, 10, 20, 60],
        index=2,
        format_func=lambda x: f"{x} sesiones",
        key="risk_center_window"
    )

    if not risk_assets:
        st.warning("Selecciona al menos un activo.")

    elif len(returns) <= window:
        st.warning("No hay suficientes datos para este periodo.")

    else:
        st.markdown("### 1. Pesos hipotéticos")

        raw_risk_weights = {}

        default_risk_weight = round(
            100 / len(risk_assets)
        )

        for ticker in risk_assets:
            raw_risk_weights[ticker] = st.slider(
                f"Peso de {ticker} (%)",
                0, 100,
                default_risk_weight,
                key=f"risk_weight_{ticker}"
            )

        total_risk_weight = sum(
            raw_risk_weights.values()
        )

        if total_risk_weight == 0:
            risk_weights = pd.Series(
                1 / len(risk_assets),
                index=risk_assets
            )
        else:
            risk_weights = (
                pd.Series(raw_risk_weights) /
                total_risk_weight
            )

        selected_returns = (
            returns[risk_assets].dropna()
        )

        portfolio_risk_returns = (
            selected_returns @ risk_weights
        )

        st.markdown("### 2. Concentración de cartera")

        hhi = float((risk_weights ** 2).sum())
        effective_assets = 1 / hhi if hhi > 0 else 0

        c1, c2 = st.columns(2)

        c1.metric(
            "Índice de concentración HHI",
            f"{hhi:.3f}"
        )

        c2.metric(
            "Número efectivo de posiciones",
            f"{effective_assets:.2f}"
        )

        st.caption(
            "El HHI suma los cuadrados de los pesos. "
            "Cuanto mayor es, más concentrada está la distribución. "
            "El número efectivo es su inverso."
        )

        st.markdown("### 3. Diversificación")

        individual_vols = (
            selected_returns.std() * np.sqrt(252)
        )

        portfolio_volatility = (
            portfolio_risk_returns.std() * np.sqrt(252)
        )

        weighted_volatility = float(
            (individual_vols * risk_weights).sum()
        )

        diversification_ratio = (
            weighted_volatility / portfolio_volatility
            if portfolio_volatility > 0 else np.nan
        )

        st.metric(
            "Ratio de diversificación",
            f"{diversification_ratio:.2f}"
            if pd.notna(diversification_ratio)
            else "No disponible"
        )

        st.caption(
            "Compara la volatilidad ponderada de los activos "
            "con la volatilidad de la cartera. Un valor superior "
            "a 1 puede reflejar beneficios de diversificación."
        )

        st.markdown("### 4. Stress testing histórico")

        def rolling_compound(series, period):
            return (
                (1 + series)
                .rolling(period)
                .apply(np.prod, raw=True) - 1
            ).dropna()

        portfolio_windows = rolling_compound(
            portfolio_risk_returns,
            window
        )

        worst_end = portfolio_windows.idxmin()
        worst_position = (
            portfolio_risk_returns.index.get_loc(worst_end)
        )
        worst_start = max(0, worst_position - window + 1)

        start_date = (
            portfolio_risk_returns.index[worst_start]
        )

        worst_loss = portfolio_windows.min() * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            f"Peor periodo de {window} sesiones",
            f"{worst_loss:.2f}%"
        )

        c2.metric(
            "Inicio del periodo",
            str(start_date.date())
        )

        c3.metric(
            "Final del periodo",
            str(worst_end.date())
        )

        st.markdown(
            "### 5. Comparación de escenarios por activo"
        )

        stress_rows = []

        for ticker in risk_assets:
            asset_windows = rolling_compound(
                selected_returns[ticker],
                window
            )

            stress_rows.append({
                "Activo": ticker,
                "Peso (%)": risk_weights[ticker] * 100,
                "Peor periodo (%)": asset_windows.min() * 100,
                "Volatilidad anualizada (%)":
                    selected_returns[ticker].std()
                    * np.sqrt(252) * 100
            })

        stress_table = pd.DataFrame(
            stress_rows
        ).set_index("Activo")

        st.dataframe(
            stress_table.style.format("{:.2f}"),
            use_container_width=True
        )

        f = px.bar(
            stress_table.reset_index(),
            x="Activo",
            y="Peor periodo (%)",
            color="Peor periodo (%)",
            template="plotly_dark",
            title=f"Peor rendimiento histórico en {window} sesiones"
        )

        f.update_layout(height=350)
        st.plotly_chart(f, use_container_width=True)

        st.markdown("### 6. Evolución de la volatilidad")

        rolling_volatility = (
            portfolio_risk_returns
            .rolling(20)
            .std() * np.sqrt(252) * 100
        ).dropna()

        f = px.line(
            x=rolling_volatility.index,
            y=rolling_volatility.values,
            labels={
                "x": "Fecha",
                "y": "Volatilidad anualizada (%)"
            },
            template="plotly_dark",
            title="Volatilidad móvil de 20 sesiones"
        )

        f.update_layout(height=350)
        st.plotly_chart(f, use_container_width=True)

        st.caption(
            "FINLAB Risk Intelligence V1 · Los resultados no "
            "incluyen costes, impuestos ni cambios futuros de "
            "correlación. Las ventanas históricas se solapan."
        )

st.divider()
st.markdown("<p style='text-align:center;color:#7890aa;font-size:.78rem'>FINLAB · EDUCATIONAL QUANTITATIVE RESEARCH · NO TRADING EXECUTION</p>", unsafe_allow_html=True)
