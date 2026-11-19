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


tab_market, tab_risk, tab_corr, tab_data, tab_guide, tab_ai, tab_portfolio, tab_risk_center, tab_market_intel, tab_research, tab_executive, tab_doctor, tab_stress, tab_v10, tab_v11 = st.tabs([
    "◈ Mercado",
    "⬡ Riesgo",
    "⌘ Correlaciones",
    "▤ Datos",
    "ⓘ Guía",
    "🧠 FINLAB AI Analyst",
    "📊 Portfolio Simulator",
    "🛡️ Risk Intelligence",
    "🌐 Market Intelligence",
    "🔬 AI Research Lab",
    "🚀 Executive Dashboard",
    "🩺 AI Portfolio Doctor",
    "⚡ Stress Testing Lab",
    "💎 Smart Intelligence Center",
    "🏛️ Institutional Intelligence"
        
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


with tab_market_intel:
    st.subheader("🌐 FINLAB Market Intelligence Center")
    st.caption("Indicadores técnicos y análisis histórico del mercado.")

    st.info(
        "Los indicadores técnicos describen precios pasados. "
        "No predicen con certeza los movimientos futuros."
    )

    technical_assets = st.multiselect(
        "Selecciona activos",
        prices.columns.tolist(),
        default=list(prices.columns),
        key="technical_assets"
    )

    if not technical_assets:
        st.warning("Selecciona al menos un activo.")

    else:
        selected_technical = st.selectbox(
            "Activo para el análisis detallado",
            technical_assets,
            key="selected_technical"
        )

        price_series = (
            prices[selected_technical]
            .dropna()
            .astype(float)
        )

        if len(price_series) < 30:
            st.warning(
                "Se necesitan al menos 30 sesiones "
                "para calcular los indicadores."
            )

        else:
            # Medias móviles
            sma20 = price_series.rolling(20).mean()
            sma50 = price_series.rolling(50).mean()
            sma200 = price_series.rolling(200).mean()

            # RSI de 14 sesiones
            delta = price_series.diff()

            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)

            avg_gain = gains.ewm(
                alpha=1/14,
                adjust=False,
                min_periods=14
            ).mean()

            avg_loss = losses.ewm(
                alpha=1/14,
                adjust=False,
                min_periods=14
            ).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # Si no hubo pérdidas en el periodo,
            # el RSI se aproxima a 100.
            rsi = rsi.mask(
                (avg_loss == 0) & (avg_gain > 0),
                100
            )

            # MACD
            ema12 = price_series.ewm(
                span=12,
                adjust=False
            ).mean()

            ema26 = price_series.ewm(
                span=26,
                adjust=False
            ).mean()

            macd = ema12 - ema26

            signal = macd.ewm(
                span=9,
                adjust=False
            ).mean()

            histogram = macd - signal

            # Volatilidad móvil de 20 sesiones
            asset_returns = (
                price_series.pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            rolling_vol = (
                asset_returns.rolling(20).std()
                * np.sqrt(252) * 100
            )

            st.markdown("### 1. Indicadores actuales")

            latest_rsi = rsi.iloc[-1]
            latest_macd = macd.iloc[-1]
            latest_signal = signal.iloc[-1]
            latest_vol = rolling_vol.iloc[-1]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "RSI (14)",
                f"{latest_rsi:.2f}"
                if pd.notna(latest_rsi)
                else "No disponible"
            )

            c2.metric(
                "MACD",
                f"{latest_macd:.4f}"
            )

            c3.metric(
                "Volatilidad móvil",
                f"{latest_vol:.2f}%"
                if pd.notna(latest_vol)
                else "No disponible"
            )

            st.markdown("### 2. Medias móviles")

            ma_data = pd.DataFrame({
                "Precio": price_series,
                "SMA 20": sma20,
                "SMA 50": sma50,
                "SMA 200": sma200
            })

            f = px.line(
                ma_data,
                template="plotly_dark",
                title=f"Precio y medias móviles · {selected_technical}"
            )

            f.update_layout(height=430)
            st.plotly_chart(f, use_container_width=True)

            st.markdown("### 3. RSI")

            rsi_data = pd.DataFrame({
                "RSI": rsi
            })

            f = px.line(
                rsi_data,
                template="plotly_dark",
                title="Índice de fuerza relativa · 14 sesiones"
            )

            f.add_hline(y=70, line_dash="dash")
            f.add_hline(y=30, line_dash="dash")

            f.update_yaxes(range=[0, 100])
            f.update_layout(height=320)

            st.plotly_chart(f, use_container_width=True)

            st.caption(
                "Como referencia convencional, RSI superior a 70 "
                "se considera elevado e inferior a 30 se considera bajo. "
                "No son señales automáticas."
            )

            st.markdown("### 4. MACD")

            macd_data = pd.DataFrame({
                "MACD": macd,
                "Señal": signal,
                "Histograma": histogram
            })

            f = px.line(
                macd_data[["MACD", "Señal"]],
                template="plotly_dark",
                title="MACD y línea de señal"
            )

            f.update_layout(height=330)
            st.plotly_chart(f, use_container_width=True)

            f = px.bar(
                macd_data.reset_index(),
                x="Date",
                y="Histograma",
                template="plotly_dark",
                title="Histograma MACD"
            )

            f.update_layout(height=280)
            st.plotly_chart(f, use_container_width=True)

            st.markdown("### 5. Resumen técnico")

            if pd.notna(latest_rsi):
                if latest_rsi >= 70:
                    st.warning(
                        "RSI elevado: el movimiento reciente "
                        "ha sido intenso en términos históricos."
                    )
                elif latest_rsi <= 30:
                    st.warning(
                        "RSI bajo: el movimiento reciente "
                        "ha sido débil en términos históricos."
                    )
                else:
                    st.info(
                        "RSI dentro del intervalo convencional "
                        "de 30 a 70."
                    )

            if latest_macd > latest_signal:
                st.write(
                    "MACD: está por encima de su línea de señal."
                )
            else:
                st.write(
                    "MACD: está por debajo de su línea de señal."
                )

            if (
                pd.notna(sma50.iloc[-1])
                and pd.notna(sma200.iloc[-1])
            ):
                if sma50.iloc[-1] > sma200.iloc[-1]:
                    st.write(
                        "Medias móviles: SMA 50 por encima "
                        "de SMA 200 en la última sesión."
                    )
                else:
                    st.write(
                        "Medias móviles: SMA 50 por debajo "
                        "de SMA 200 en la última sesión."
                    )
            else:
                st.write(
                    "Medias móviles: historial insuficiente "
                    "para comparar SMA 50 y SMA 200."
                )

            st.caption(
                "FINLAB Market Intelligence V1 · "
                "Indicadores descriptivos, sin recomendaciones "
                "de inversión."
            )

st.divider()

with tab_research:
    st.header("🔬 FINLAB AI Research Lab")
    st.caption("Laboratorio de investigación cuantitativa · Análisis educativo")

    st.info(
        "Los resultados son análisis históricos y simulaciones educativas. "
        "No son predicciones ni recomendaciones de inversión."
    )

    available_assets = list(prices.columns)

    selected_assets = st.multiselect(
        "Selecciona activos para investigar",
        available_assets,
        default=available_assets[:min(3, len(available_assets))]
    )

    if len(selected_assets) >= 1:
        research_prices = prices[selected_assets].dropna(how="all")
        research_returns = research_prices.pct_change().dropna()

        if not research_returns.empty:
            st.subheader("1. Comparación cuantitativa")

            summary = pd.DataFrame({
                "Rentabilidad anualizada (%)":
                    research_returns.mean() * 252 * 100,
                "Volatilidad anualizada (%)":
                    research_returns.std() * np.sqrt(252) * 100,
                "Mejor día (%)":
                    research_returns.max() * 100,
                "Peor día (%)":
                    research_returns.min() * 100
            })

            st.dataframe(
                summary.round(2),
                use_container_width=True
            )

            st.subheader("2. Detección de movimientos inusuales")

            anomalies = []

            for asset in selected_assets:
                series = research_returns[asset].dropna()

                if len(series) >= 20:
                    rolling_mean = series.rolling(20).mean()
                    rolling_std = series.rolling(20).std()
                    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)

                    unusual = z_score.abs() > 2

                    anomalies.append({
                        "Activo": asset,
                        "Movimientos inusuales": int(unusual.sum()),
                        "Mayor desviación": round(
                            float(z_score.abs().max()), 2
                        ) if z_score.notna().any() else 0
                    })

            if anomalies:
                st.dataframe(
                    pd.DataFrame(anomalies),
                    use_container_width=True
                )

            st.subheader("3. Evolución comparativa")

            normalized_research = (
                research_prices / research_prices.iloc[0] * 100
            )

            st.line_chart(normalized_research)

            st.subheader("4. Informe de investigación")

            report = summary.round(2).to_string()

            st.text_area(
                "Resumen del análisis",
                value=(
                    "FINLAB AI RESEARCH LAB\n"
                    "Informe cuantitativo educativo\n\n"
                    + report
                    + "\n\nLos datos históricos no garantizan "
                    "resultados futuros."
                ),
                height=250
            )

            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from io import BytesIO

            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(50, 800, "FINLAB AI Research Lab")

            pdf.setFont("Helvetica", 10)
            y = 765

            for line in (
                "Informe cuantitativo educativo",
                "",
                *report.splitlines(),
                "",
                "Los datos historicos no garantizan resultados futuros.",
                "Este documento no constituye asesoramiento financiero."
            ):
                if y < 60:
                    pdf.showPage()
                    y = 800
                    pdf.setFont("Helvetica", 10)

                pdf.drawString(50, y, line[:100])
                y -= 18

            pdf.save()
            pdf_buffer.seek(0)

            st.download_button(
                "📄 Descargar informe PDF",
                data=pdf_buffer,
                file_name="FINLAB_AI_Research_Report.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("No hay suficientes datos para realizar el análisis.")
    else:
        st.warning("Selecciona al menos un activo.")


with tab_executive:
    st.header("🚀 FINLAB Executive Dashboard")
    st.caption("Centro de inteligencia y análisis cuantitativo")

    st.info(
        "Panel educativo basado en datos históricos. "
        "No ofrece predicciones ni recomendaciones de inversión."
    )

    if not prices.empty and not returns.empty:
        assets = list(prices.columns)

        total_assets = len(assets)
        latest_date = prices.index[-1].strftime("%Y-%m-%d")

        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        avg_return = annual_returns.mean() * 100
        avg_volatility = annual_volatility.mean() * 100

        latest_values = prices.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Activos analizados", total_assets)
        col2.metric("Rentabilidad media anualizada", f"{avg_return:.2f}%")
        col3.metric("Volatilidad media", f"{avg_volatility:.2f}%")
        col4.metric("Últimos datos", latest_date)

        st.divider()

        st.subheader("📊 Indicadores por activo")

        executive_table = pd.DataFrame({
            "Rentabilidad anualizada (%)": annual_returns * 100,
            "Volatilidad anualizada (%)": annual_volatility * 100,
            "Último precio": latest_values
        })

        st.dataframe(
            executive_table.round(2),
            use_container_width=True
        )

        st.subheader("📈 Comparación de rentabilidad y riesgo")

        dashboard_chart = pd.DataFrame({
            "Rentabilidad (%)": annual_returns * 100,
            "Volatilidad (%)": annual_volatility * 100
        })

        st.scatter_chart(
            dashboard_chart,
            x="Volatilidad (%)",
            y="Rentabilidad (%)"
        )

        st.subheader("🏆 Clasificación histórica")

        ranking = annual_returns.sort_values(ascending=False)

        st.bar_chart(ranking * 100)

        st.caption(
            "La clasificación refleja exclusivamente el rendimiento "
            "histórico anualizado de los datos disponibles."
        )

    else:
        st.warning("No hay datos suficientes para construir el dashboard.")


with tab_doctor:
    st.header("🩺 FINLAB AI Portfolio Doctor")
    st.caption("Diagnóstico cuantitativo de carteras simuladas")

    st.info(
        "Herramienta educativa. Analiza datos históricos y no "
        "ofrece recomendaciones de inversión."
    )

    doctor_assets = list(returns.columns)

    selected_doctor_assets = st.multiselect(
        "Selecciona los activos de tu cartera simulada",
        doctor_assets,
        default=doctor_assets[:min(3, len(doctor_assets))],
        key="doctor_assets"
    )

    if selected_doctor_assets:
        doctor_returns = returns[selected_doctor_assets].dropna()

        st.subheader("1. Distribución simulada")

        weights = np.ones(len(selected_doctor_assets)) / len(selected_doctor_assets)

        weight_df = pd.DataFrame({
            "Activo": selected_doctor_assets,
            "Peso simulado (%)": weights * 100
        })

        st.dataframe(weight_df.round(2), use_container_width=True)

        st.subheader("2. Diagnóstico de concentración")

        hhi = float(np.sum(weights ** 2))
        effective_assets = 1 / hhi

        col1, col2 = st.columns(2)
        col1.metric("Índice HHI", f"{hhi:.3f}")
        col2.metric("Activos efectivos", f"{effective_assets:.2f}")

        if hhi > 0.5:
            st.warning("Concentración elevada en la distribución simulada.")
        elif hhi > 0.25:
            st.info("Concentración intermedia en la distribución simulada.")
        else:
            st.success("La distribución presenta menor concentración.")

        st.subheader("3. Riesgo histórico")

        portfolio_returns = doctor_returns.mean(axis=1)

        annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
        historical_var = portfolio_returns.quantile(0.05) * 100
        historical_cvar = portfolio_returns[
            portfolio_returns <= portfolio_returns.quantile(0.05)
        ].mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Volatilidad anualizada", f"{annual_vol:.2f}%")
        col2.metric("VaR histórico 95%", f"{historical_var:.2f}%")
        col3.metric("CVaR histórico 95%", f"{historical_cvar:.2f}%")

        st.subheader("4. Evolución de la cartera simulada")

        equity_curve = (1 + portfolio_returns).cumprod()
        st.line_chart(equity_curve)

        st.subheader("5. Informe del diagnóstico")

        if annual_vol > 30:
            diagnosis = "La volatilidad histórica observada es elevada."
        elif annual_vol > 15:
            diagnosis = "La volatilidad histórica observada es intermedia."
        else:
            diagnosis = "La volatilidad histórica observada es relativamente baja."

        st.write(diagnosis)
        st.write(
            f"La cartera contiene {len(selected_doctor_assets)} activos "
            f"con pesos iguales simulados. Su índice HHI es {hhi:.3f}."
        )

        st.caption(
            "Las métricas describen el pasado y no predicen resultados futuros."
        )

    else:
        st.warning("Selecciona al menos un activo.")


with tab_stress:
    st.header("⚡ FINLAB Stress Testing Lab")
    st.caption("Simulación de escenarios hipotéticos de mercado")

    st.info(
        "Los escenarios son hipotéticos y educativos. "
        "No representan predicciones del mercado."
    )

    stress_assets = list(returns.columns)

    selected_stress_assets = st.multiselect(
        "Selecciona los activos",
        stress_assets,
        default=stress_assets[:min(3, len(stress_assets))],
        key="stress_assets"
    )

    if selected_stress_assets:
        stress_weights = np.ones(len(selected_stress_assets))
        stress_weights = stress_weights / stress_weights.sum()

        scenarios = [-5, -10, -20, -30]

        st.subheader("1. Configuración de escenarios")

        scenario_choice = st.multiselect(
            "Escenarios de caída hipotética (%)",
            scenarios,
            default=scenarios,
            key="stress_scenarios"
        )

        if scenario_choice:
            st.subheader("2. Impacto estimado")

            scenario_results = []

            for scenario in scenario_choice:
                impact = scenario / 100

                scenario_results.append({
                    "Escenario (%)": scenario,
                    "Impacto cartera (%)": impact * 100,
                    "Valor final por cada 10.000 unidades": 10000 * (1 + impact)
                })

            scenario_df = pd.DataFrame(scenario_results)

            st.dataframe(
                scenario_df.round(2),
                use_container_width=True
            )

            st.subheader("3. Comparación visual")

            chart_df = scenario_df.set_index("Escenario (%)")[
                ["Impacto cartera (%)"]
            ]

            st.bar_chart(chart_df)

            st.subheader("4. Impacto por activo")

            asset_impacts = pd.DataFrame({
                "Activo": selected_stress_assets,
                "Peso simulado (%)": stress_weights * 100,
                "Impacto individual (%)": [
                    scenario_choice[0] for _ in selected_stress_assets
                ]
            })

            st.dataframe(
                asset_impacts.round(2),
                use_container_width=True
            )

            st.subheader("5. Resumen del laboratorio")

            st.write(
                f"Se han analizado {len(selected_stress_assets)} activos "
                f"con pesos iguales simulados."
            )

            st.write(
                f"Se han evaluado {len(scenario_choice)} escenarios "
                "hipotéticos."
            )

            st.caption(
                "Este modelo aplica una caída uniforme a todos los activos. "
                "No incorpora correlaciones cambiantes, liquidez ni "
                "reacciones reales del mercado."
            )

            csv_data = scenario_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Descargar escenarios CSV",
                data=csv_data,
                file_name="FINLAB_Stress_Testing.csv",
                mime="text/csv"
            )

        else:
            st.warning("Selecciona al menos un escenario.")

    else:
        st.warning("Selecciona al menos un activo.")


with tab_v10:
    st.header("💎 FINLAB Smart Intelligence Center")
    st.caption("Alertas, rendimiento histórico e informes avanzados")

    st.info(
        "Herramienta educativa basada en datos históricos. "
        "Las alertas no son predicciones ni recomendaciones."
    )

    if not returns.empty and not prices.empty:

        st.subheader("1. 📈 Performance Lab")

        performance_assets = list(returns.columns)

        selected_performance = st.multiselect(
            "Selecciona activos para analizar",
            performance_assets,
            default=performance_assets[:min(3, len(performance_assets))],
            key="v10_performance"
        )

        if selected_performance:
            performance_returns = returns[selected_performance].dropna()

            performance_table = pd.DataFrame({
                "Rentabilidad anualizada (%)":
                    performance_returns.mean() * 252 * 100,
                "Volatilidad anualizada (%)":
                    performance_returns.std() * np.sqrt(252) * 100,
                "Peor día (%)":
                    performance_returns.min() * 100
            })

            st.dataframe(
                performance_table.round(2),
                use_container_width=True
            )

            equity = (1 + performance_returns).cumprod()
            st.line_chart(equity)

            st.subheader("2. 🚨 Smart Alerts")

            volatility_limit = st.slider(
                "Umbral de volatilidad anualizada (%)",
                min_value=5,
                max_value=80,
                value=30,
                key="v10_vol_limit"
            )

            daily_limit = st.slider(
                "Umbral de movimiento diario (%)",
                min_value=1,
                max_value=15,
                value=5,
                key="v10_daily_limit"
            )

            for asset in selected_performance:
                asset_returns = performance_returns[asset]

                annual_vol = asset_returns.std() * np.sqrt(252) * 100
                worst_day = asset_returns.min() * 100

                if annual_vol > volatility_limit:
                    st.warning(
                        f"{asset}: volatilidad histórica de "
                        f"{annual_vol:.2f}%, superior al umbral."
                    )

                if abs(worst_day) > daily_limit:
                    st.warning(
                        f"{asset}: movimiento diario histórico de "
                        f"{worst_day:.2f}%, superior al umbral."
                    )

            st.subheader("3. 📄 PDF Pro")

            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from io import BytesIO

            pdf_data = BytesIO()
            pdf = canvas.Canvas(pdf_data, pagesize=A4)

            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(45, 800, "FINLAB V10 - Intelligence Report")

            pdf.setFont("Helvetica", 10)
            y = 765

            for line in [
                "Educational quantitative research report",
                "",
                *performance_table.round(2).to_string().splitlines(),
                "",
                "Historical data does not guarantee future results.",
                "This report is not financial advice."
            ]:
                if y < 60:
                    pdf.showPage()
                    y = 800
                    pdf.setFont("Helvetica", 10)

                pdf.drawString(45, y, line[:100])
                y -= 18

            pdf.save()
            pdf_data.seek(0)

            st.download_button(
                "📄 Descargar informe PDF Pro",
                data=pdf_data,
                file_name="FINLAB_V10_Intelligence_Report.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("Selecciona al menos un activo.")

    else:
        st.warning("No hay datos suficientes para el análisis.")


with tab_v11:
    st.header("🏛️ FINLAB Institutional Intelligence")
    st.caption("Advanced quantitative research laboratory")

    st.info(
        "Herramientas estadísticas educativas. "
        "Los resultados históricos no predicen el futuro."
    )

    institutional_assets = list(returns.columns)

    selected_institutional = st.multiselect(
        "Selecciona activos para el análisis",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v11_assets"
    )

    if len(selected_institutional) >= 2:
        institutional_returns = returns[selected_institutional].dropna()

        st.subheader("1. Correlation Matrix Pro")

        correlation_matrix = institutional_returns.corr()

        st.dataframe(
            correlation_matrix.round(3),
            use_container_width=True
        )

        st.subheader("2. Mapa visual de correlaciones")

        st.dataframe(
    correlation_matrix.round(2),
    use_container_width=True
)

        st.subheader("3. Relaciones más destacadas")

        correlation_pairs = []

        for i in range(len(selected_institutional)):
            for j in range(i + 1, len(selected_institutional)):
                asset_a = selected_institutional[i]
                asset_b = selected_institutional[j]

                value = correlation_matrix.loc[asset_a, asset_b]

                correlation_pairs.append({
                    "Activo A": asset_a,
                    "Activo B": asset_b,
                    "Correlación": round(float(value), 3)
                })

        correlation_ranking = pd.DataFrame(correlation_pairs)

        if not correlation_ranking.empty:
            correlation_ranking["Correlación absoluta"] = (
                correlation_ranking["Correlación"].abs()
            )

            correlation_ranking = correlation_ranking.sort_values(
                "Correlación absoluta",
                ascending=False
            ).drop(columns=["Correlación absoluta"])

            st.dataframe(
                correlation_ranking,
                use_container_width=True
            )

        st.subheader("4. Interpretación")

        st.write(
            "Una correlación cercana a +1 indica que dos series "
            "han tendido a moverse en la misma dirección."
        )

        st.write(
            "Una correlación cercana a -1 indica movimientos "
            "históricamente opuestos."
        )

        st.write(
            "Una correlación cercana a 0 indica una relación lineal "
            "histórica débil."
        )

        csv_corr = correlation_matrix.to_csv().encode("utf-8")

        st.download_button(
            "📥 Descargar matriz CSV",
            data=csv_corr,
            file_name="FINLAB_Correlation_Matrix.csv",
            mime="text/csv"
        )

    else:
        st.warning("Selecciona al menos dos activos.")
    st.divider()
    st.subheader("5. Rolling Correlation")
    st.caption("Evolución temporal de la correlación entre dos activos")

    rolling_assets = st.multiselect(
        "Selecciona dos activos",
        institutional_assets,
        default=institutional_assets[:2],
        key="v12_rolling_assets"
    )

    rolling_window = st.slider(
        "Ventana de análisis (días)",
        min_value=20,
        max_value=120,
        value=60,
        key="v12_rolling_window"
    )

    if len(rolling_assets) == 2:
        rolling_corr = (
            returns[rolling_assets[0]]
            .rolling(rolling_window)
            .corr(returns[rolling_assets[1]])
            .dropna()
        )

        if not rolling_corr.empty:
            st.line_chart(rolling_corr)
            st.metric(
                "Correlación más reciente",
                f"{rolling_corr.iloc[-1]:.3f}"
            )
        else:
            st.warning("No hay suficientes datos para este análisis.")
    else:
        st.info("Selecciona exactamente dos activos.")
    st.divider()
    st.subheader("6. Covariance Matrix Pro")
    st.caption("Matriz de covarianzas entre los activos seleccionados")

    covariance_matrix = institutional_returns.cov() * 252

    st.dataframe(
        covariance_matrix.round(5),
        use_container_width=True
    )

    st.caption(
        "Covarianza anualizada estimada a partir de rendimientos diarios. "
        "Es una medida estadística histórica, no una predicción."
    )

    csv_covariance = covariance_matrix.to_csv().encode("utf-8")

    st.download_button(
        "📥 Descargar matriz de covarianzas",
        data=csv_covariance,
        file_name="FINLAB_Covariance_Matrix.csv",
        mime="text/csv",
        key="v12_covariance_download"
    )
    st.divider()
    st.subheader("7. Diversification Score")

    st.caption("Análisis de diversificación con pesos iguales o personalizados")

    diversification_mode = st.radio(
        "Método de ponderación",
        ["Equal Weight", "Custom Weight"],
        horizontal=True,
        key="v12_diversification_mode"
    )

    n_assets = len(selected_institutional)

    if n_assets >= 2:

        if diversification_mode == "Equal Weight":
            portfolio_weights = pd.Series(
                1 / n_assets,
                index=selected_institutional
            )

        else:
            st.write("Asigna un peso a cada activo (%)")

            custom_weights = {}

            for asset in selected_institutional:
                custom_weights[asset] = st.number_input(
                    f"Peso de {asset} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(round(100 / n_assets, 2)),
                    step=1.0,
                    key=f"v12_weight_{asset}"
                )

            total_weight = sum(custom_weights.values())

            if total_weight > 0:
                portfolio_weights = pd.Series(custom_weights) / total_weight
            else:
                portfolio_weights = None

        if portfolio_weights is not None:

            hhi = float((portfolio_weights ** 2).sum())
            effective_assets = 1 / hhi
            diversification_score = (
                1 - (hhi - 1/n_assets) / (1 - 1/n_assets)
            ) * 100

            st.metric(
                "Diversification Score",
                f"{diversification_score:.1f}/100"
            )

            st.metric(
                "Número efectivo de activos",
                f"{effective_assets:.2f}"
            )

            st.metric(
                "Índice HHI",
                f"{hhi:.4f}"
            )

            st.subheader("Distribución de pesos")

            weights_display = (
                portfolio_weights * 100
            ).round(2)

            st.bar_chart(weights_display)

            st.caption(
                "El indicador mide concentración de pesos, "
                "no garantiza una reducción del riesgo."
            )
    st.divider()
    st.subheader("8. Asset Clustering")
    st.caption("Agrupación de activos según sus rendimientos históricos")

    clustering_assets = st.multiselect(
        "Selecciona activos para agrupar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v12_clustering_assets"
    )

    n_clusters = st.slider(
        "Número de grupos",
        min_value=2,
        max_value=min(5, len(clustering_assets)),
        value=2,
        key="v12_n_clusters"
    ) if len(clustering_assets) >= 2 else 2

    if len(clustering_assets) >= 2:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        clustering_data = (
            returns[clustering_assets]
            .dropna()
            .T
        )

        if clustering_data.shape[1] >= 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)

            model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(scaled_data)

            clustering_results = pd.DataFrame({
                "Activo": clustering_assets,
                "Grupo": labels + 1
            })

            st.dataframe(
                clustering_results,
                use_container_width=True
            )

            st.subheader("Distribución de grupos")

            st.bar_chart(
                clustering_results["Grupo"].value_counts().sort_index()
            )
        else:
            st.warning("No hay suficientes observaciones.")
    else:
        st.info("Selecciona al menos dos activos.")
    st.divider()
    st.subheader("9. FINLAB Quantitative Report")
    st.caption("Generador de informes cuantitativos en PDF")

    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    if len(selected_institutional) >= 2:

        pdf_buffer = BytesIO()

        document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4
        )

        styles = getSampleStyleSheet()
        report_content = []

        report_content.append(
            Paragraph(
                "FINLAB - Quantitative Intelligence Report",
                styles["Title"]
            )
        )

        report_content.append(Spacer(1, 15))

        report_content.append(
            Paragraph(
                "Institutional Quantitative Analysis",
                styles["Heading2"]
            )
        )

        report_content.append(Spacer(1, 10))

        report_content.append(
            Paragraph(
                f"Assets analyzed: {len(selected_institutional)}",
                styles["Normal"]
            )
        )

        report_content.append(Spacer(1, 15))

        report_content.append(
            Paragraph(
                "Correlation Matrix",
                styles["Heading2"]
            )
        )

        corr_table_data = [
            ["Asset"] + list(correlation_matrix.columns)
        ]

        for asset in correlation_matrix.index:
            corr_table_data.append(
                [asset] +
                [
                    round(float(value), 3)
                    for value in correlation_matrix.loc[asset]
                ]
            )

        corr_table = Table(
            corr_table_data,
            repeatRows=1
        )

        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        report_content.append(corr_table)
        report_content.append(Spacer(1, 20))

        report_content.append(
            Paragraph(
                "Covariance Matrix (Annualized)",
                styles["Heading2"]
            )
        )

        covariance_data = [
            ["Asset"] + list(covariance_matrix.columns)
        ]

        for asset in covariance_matrix.index:
            covariance_data.append(
                [asset] +
                [
                    round(float(value), 5)
                    for value in covariance_matrix.loc[asset]
                ]
            )

        covariance_table = Table(
            covariance_data,
            repeatRows=1
        )

        covariance_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        report_content.append(covariance_table)
        report_content.append(Spacer(1, 20))

        report_content.append(
            Paragraph(
                "Diversification Analysis",
                styles["Heading2"]
            )
        )

        report_content.append(
            Paragraph(
                "Diversification Score: "
                + f"{diversification_score:.2f}/100",
                styles["Normal"]
            )
        )

        report_content.append(
            Paragraph(
                "Effective Assets: "
                + f"{effective_assets:.2f}",
                styles["Normal"]
            )
        )

        report_content.append(
            Paragraph(
                "HHI: " + f"{hhi:.4f}",
                styles["Normal"]
            )
        )

        report_content.append(Spacer(1, 20))

        report_content.append(
            Paragraph(
                "Historical statistical analysis only. "
                "This report is educational and does not predict "
                "future performance.",
                styles["Italic"]
            )
        )

        document.build(report_content)

        st.download_button(
            "📄 Descargar FINLAB Quantitative Report",
            data=pdf_buffer.getvalue(),
            file_name="FINLAB_Quantitative_Report.pdf",
            mime="application/pdf",
            key="v12_quantitative_pdf"
        )

    else:
        st.info("Selecciona al menos dos activos para generar el informe.")
    st.divider()
    st.subheader("10. Trend Forecasting")
    st.caption("Análisis estadístico de tendencias históricas")

    forecast_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_forecast_asset"
    )

    forecast_days = st.slider(
        "Horizonte de proyección (días)",
        min_value=5,
        max_value=60,
        value=20,
        key="v13_forecast_days"
    )

    forecast_history = returns[forecast_asset].dropna()

    if len(forecast_history) >= 30:

        import numpy as np

        historical_prices = (
            (1 + forecast_history).cumprod() * 100
        )

        x = np.arange(len(historical_prices))
        y = historical_prices.values

        slope, intercept = np.polyfit(x, y, 1)

        future_x = np.arange(
            len(historical_prices),
            len(historical_prices) + forecast_days
        )

        projected_values = slope * future_x + intercept

        historical_series = pd.Series(
            y[-120:],
            index=historical_prices.index[-120:],
            name="Histórico"
        )

        future_dates = pd.bdate_range(
            start=historical_prices.index[-1],
            periods=forecast_days + 1
        )[1:]

        projected_series = pd.Series(
            projected_values,
            index=future_dates,
            name="Proyección experimental"
        )

        forecast_chart = pd.concat(
            [historical_series, projected_series]
        )

        st.line_chart(forecast_chart)

        st.metric(
            "Valor final estimado por el modelo",
            f"{projected_values[-1]:.2f}"
        )

        st.caption(
            "Proyección lineal ilustrativa basada en datos históricos. "
            "No incorpora noticias, cambios de mercado ni eventos futuros."
        )

    else:
        st.warning("Se necesitan al menos 30 observaciones.")
    st.divider()
    st.subheader("11. Moving Average Analysis")
    st.caption("Comparación de medias móviles históricas")

    ma_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_ma_asset"
    )

    ma_returns = returns[ma_asset].dropna()

    ma_prices = (1 + ma_returns).cumprod() * 100

    ma_data = pd.DataFrame({
        "Precio normalizado": ma_prices,
        "Media móvil 20": ma_prices.rolling(20).mean(),
        "Media móvil 50": ma_prices.rolling(50).mean(),
        "Media móvil 200": ma_prices.rolling(200).mean()
    })

    st.line_chart(ma_data)

    latest = ma_data.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MA 20",
        f"{latest['Media móvil 20']:.2f}"
    )

    col2.metric(
        "MA 50",
        f"{latest['Media móvil 50']:.2f}"
    )

    col3.metric(
        "MA 200",
        f"{latest['Media móvil 200']:.2f}"
    )

    if pd.notna(latest["Media móvil 20"]) and pd.notna(latest["Media móvil 50"]):

        if latest["Media móvil 20"] > latest["Media móvil 50"]:
            st.success("MA 20 está por encima de MA 50.")
        else:
            st.info("MA 20 está por debajo de MA 50.")

    st.caption(
        "Las medias móviles suavizan datos históricos. "
        "No garantizan la dirección futura del mercado."
    )
    st.divider()
    st.subheader("12. Volatility Forecast")
    st.caption("Análisis de volatilidad histórica y estimaciones")

    vol_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_vol_asset"
    )

    vol_returns = returns[vol_asset].dropna()

    if len(vol_returns) >= 60:

        volatility_data = pd.DataFrame(index=vol_returns.index)

        volatility_data["Volatilidad 20 días"] = (
            vol_returns.rolling(20).std() * (252 ** 0.5) * 100
        )

        volatility_data["Volatilidad 60 días"] = (
            vol_returns.rolling(60).std() * (252 ** 0.5) * 100
        )

        st.line_chart(volatility_data)

        vol_20 = volatility_data["Volatilidad 20 días"].iloc[-1]
        vol_60 = volatility_data["Volatilidad 60 días"].iloc[-1]

        col1, col2 = st.columns(2)

        col1.metric(
            "Volatilidad anualizada (20 días)",
            f"{vol_20:.2f}%"
        )

        col2.metric(
            "Volatilidad anualizada (60 días)",
            f"{vol_60:.2f}%"
        )

        st.subheader("Interpretación estadística")

        if vol_20 > vol_60:
            st.warning(
                "La volatilidad reciente supera la referencia "
                "de 60 días."
            )
        else:
            st.info(
                "La volatilidad reciente está por debajo "
                "de la referencia de 60 días."
            )

        st.caption(
            "La volatilidad mide la variabilidad histórica de los "
            "rendimientos. No indica por sí sola la dirección del mercado."
        )

    else:
        st.warning("Se necesitan al menos 60 observaciones.")
    st.divider()
    st.subheader("13. Scenario Simulator")
    st.caption("Simulación educativa de escenarios hipotéticos")

    scenario_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_scenario_asset"
    )

    st.write("Define tres escenarios hipotéticos (%)")

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario_1 = st.number_input(
            "Escenario negativo",
            min_value=-100.0,
            max_value=100.0,
            value=-10.0,
            step=1.0,
            key="v13_scenario_1"
        )

    with col2:
        scenario_2 = st.number_input(
            "Escenario neutral",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="v13_scenario_2"
        )

    with col3:
        scenario_3 = st.number_input(
            "Escenario positivo",
            min_value=-100.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="v13_scenario_3"
        )

    initial_value = 100.0

    scenario_results = pd.DataFrame({
        "Escenario": [
            "Negativo",
            "Neutral",
            "Positivo"
        ],
        "Cambio hipotético (%)": [
            scenario_1,
            scenario_2,
            scenario_3
        ]
    })

    scenario_results["Valor simulado"] = (
        initial_value *
        (1 + scenario_results["Cambio hipotético (%)"] / 100)
    )

    scenario_results["Variación"] = (
        scenario_results["Valor simulado"] - initial_value
    )

    st.subheader("Resultados de simulación")

    st.dataframe(
        scenario_results.round(2),
        use_container_width=True
    )

    st.bar_chart(
        scenario_results.set_index("Escenario")["Valor simulado"]
    )

    st.caption(
        "Se utiliza un valor inicial ilustrativo de 100 unidades. "
        "Los escenarios son hipotéticos y no representan predicciones "
        "ni recomendaciones de inversión."
    )
    st.divider()
    st.subheader("14. Model Accuracy Lab")
    st.caption("Evaluación histórica de un modelo de referencia")

    accuracy_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_accuracy_asset"
    )

    accuracy_returns = returns[accuracy_asset].dropna()

    if len(accuracy_returns) >= 30:

        actual = accuracy_returns.iloc[1:]
        predicted = accuracy_returns.shift(1).dropna()

        comparison = pd.DataFrame({
            "Real (%)": actual * 100,
            "Estimado (%)": predicted * 100
        }).dropna()

        comparison["Error absoluto (%)"] = (
            comparison["Real (%)"] -
            comparison["Estimado (%)"]
        ).abs()

        mae = comparison["Error absoluto (%)"].mean()

        direction_accuracy = (
            (
                (comparison["Real (%)"] >= 0) ==
                (comparison["Estimado (%)"] >= 0)
            ).mean() * 100
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Error absoluto medio",
                f"{mae:.3f}%"
            )

        with col2:
            st.metric(
                "Coincidencia de dirección",
                f"{direction_accuracy:.1f}%"
            )

        st.subheader("Comparación de resultados")

        st.line_chart(
            comparison[["Real (%)", "Estimado (%)"]].tail(100)
        )

        st.subheader("Últimos registros")

        st.dataframe(
            comparison.tail(10).round(3),
            use_container_width=True
        )

        st.caption(
            "Modelo de referencia: utiliza el rendimiento del día anterior "
            "como estimación del siguiente. La coincidencia histórica "
            "no garantiza resultados futuros."
        )

    else:
        st.warning(
            "Se necesitan al menos 30 observaciones para realizar "
            "esta evaluación."
        )
    st.divider()
    st.subheader("15. Portfolio Optimization Lab")
    st.caption("Comparación educativa de carteras hipotéticas")

    optimization_assets = st.multiselect(
        "Selecciona los activos",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_optimization_assets"
    )

    if len(optimization_assets) >= 2:

        optimization_returns = returns[optimization_assets].dropna()

        if len(optimization_returns) >= 20:

            st.write("Distribución hipotética de pesos (%)")

            optimization_weights = {}

            for asset in optimization_assets:
                optimization_weights[asset] = st.slider(
                    f"Peso de {asset}",
                    min_value=0,
                    max_value=100,
                    value=100 // len(optimization_assets),
                    key=f"v13_weight_{asset}"
                )

            total_weight = sum(optimization_weights.values())

            if total_weight > 0:

                normalized_weights = {
                    asset: weight / total_weight
                    for asset, weight in optimization_weights.items()
                }

                weights_series = pd.Series(normalized_weights)

                portfolio_returns = (
                    optimization_returns * weights_series
                ).sum(axis=1)

                annual_return = (
                    (1 + portfolio_returns).prod()
                    ** (252 / len(portfolio_returns)) - 1
                ) * 100

                annual_volatility = (
                    portfolio_returns.std() * (252 ** 0.5) * 100
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Rentabilidad anualizada histórica",
                        f"{annual_return:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Volatilidad anualizada",
                        f"{annual_volatility:.2f}%"
                    )

                st.subheader("Distribución de la cartera")

                weights_df = pd.DataFrame({
                    "Activo": list(normalized_weights.keys()),
                    "Peso (%)": [
                        value * 100
                        for value in normalized_weights.values()
                    ]
                })

                st.dataframe(
                    weights_df.round(2),
                    use_container_width=True
                )

                st.subheader("Evolución histórica simulada")

                cumulative_portfolio = (
                    (1 + portfolio_returns).cumprod()
                )

                st.line_chart(cumulative_portfolio)

            else:
                st.warning("Asigna un peso superior a cero.")

        else:
            st.warning("No hay suficientes datos históricos.")

    else:
        st.info("Selecciona al menos dos activos.")

    st.caption(
        "Los resultados utilizan datos históricos y pesos hipotéticos. "
        "No constituyen predicciones ni recomendaciones financieras."
    )
    st.divider()
    st.subheader("16. FINLAB Performance Analytics")
    st.caption("Análisis histórico de rendimiento de una cartera hipotética")

    performance_assets = st.multiselect(
        "Selecciona activos para analizar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_performance_assets"
    )

    if len(performance_assets) >= 1:

        performance_data = returns[performance_assets].dropna()

        if len(performance_data) >= 2:

            performance_portfolio = performance_data.mean(axis=1)

            cumulative_performance = (
                (1 + performance_portfolio).cumprod()
            )

            total_return = (
                (cumulative_performance.iloc[-1] - 1) * 100
            )

            annualized_return = (
                (cumulative_performance.iloc[-1]
                 ** (252 / len(performance_portfolio)) - 1) * 100
            )

            best_day = performance_portfolio.max() * 100
            worst_day = performance_portfolio.min() * 100

            daily_volatility = (
                performance_portfolio.std() * (252 ** 0.5) * 100
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Rentabilidad acumulada",
                    f"{total_return:.2f}%"
                )

                st.metric(
                    "Mejor día",
                    f"{best_day:.2f}%"
                )

            with col2:
                st.metric(
                    "Rentabilidad anualizada",
                    f"{annualized_return:.2f}%"
                )

                st.metric(
                    "Peor día",
                    f"{worst_day:.2f}%"
                )

            st.metric(
                "Volatilidad anualizada",
                f"{daily_volatility:.2f}%"
            )

            st.subheader("Evolución de la cartera")

            st.line_chart(cumulative_performance)

            st.subheader("Distribución de rendimientos diarios")

            st.bar_chart(
                performance_portfolio.tail(60) * 100
            )

            st.caption(
                "Cartera hipotética con ponderaciones iguales. "
                "Las métricas utilizan datos históricos y no garantizan "
                "resultados futuros."
            )

        else:
            st.warning("No hay suficientes observaciones para el análisis.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("17. FINLAB Risk Monitor")
    st.caption("Monitorización de riesgos basada en datos históricos")

    risk_monitor_assets = st.multiselect(
        "Selecciona activos para monitorizar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_risk_monitor_assets"
    )

    if len(risk_monitor_assets) >= 1:

        monitor_data = returns[risk_monitor_assets].dropna()

        if len(monitor_data) >= 20:

            monitor_returns = monitor_data.mean(axis=1)

            annual_vol = monitor_returns.std() * (252 ** 0.5) * 100

            worst_daily = monitor_returns.min() * 100

            cumulative = (1 + monitor_returns).cumprod()

            running_max = cumulative.cummax()

            drawdown = (
                (cumulative / running_max) - 1
            ) * 100

            max_drawdown = drawdown.min()

            st.subheader("Indicadores de riesgo")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Volatilidad anualizada",
                    f"{annual_vol:.2f}%"
                )

            with col2:
                st.metric(
                    "Peor día",
                    f"{worst_daily:.2f}%"
                )

            with col3:
                st.metric(
                    "Caída máxima",
                    f"{max_drawdown:.2f}%"
                )

            st.subheader("Estado de monitorización")

            if annual_vol >= 30:
                st.warning("Volatilidad histórica elevada.")
            else:
                st.success("Volatilidad histórica por debajo del umbral.")

            if worst_daily <= -5:
                st.warning("Se ha observado una caída diaria importante.")
            else:
                st.success("No se ha superado el umbral de caída diaria.")

            if max_drawdown <= -20:
                st.warning("La caída máxima histórica ha superado el 20%.")
            else:
                st.success("La caída máxima histórica no ha superado el 20%.")

            st.subheader("Evolución del drawdown")

            st.line_chart(drawdown)

            st.caption(
                "Los umbrales son ilustrativos. Las alertas describen "
                "datos históricos y no constituyen recomendaciones "
                "financieras ni predicciones."
            )

        else:
            st.warning("Se necesitan al menos 20 observaciones.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("18. FINLAB AI Insights Engine")
    st.caption("Generador automático de observaciones cuantitativas")

    insights_assets = st.multiselect(
        "Selecciona activos para analizar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_insights_assets"
    )

    if len(insights_assets) >= 1:

        insights_data = returns[insights_assets].dropna()

        if len(insights_data) >= 20:

            insights_portfolio = insights_data.mean(axis=1)

            annual_volatility = (
                insights_portfolio.std() * (252 ** 0.5) * 100
            )

            cumulative = (1 + insights_portfolio).cumprod()

            total_return = (cumulative.iloc[-1] - 1) * 100

            running_max = cumulative.cummax()

            drawdown = (
                (cumulative / running_max) - 1
            ) * 100

            max_drawdown = drawdown.min()

            positive_days = (
                insights_portfolio > 0
            ).mean() * 100

            st.subheader("FINLAB generated insights")

            if total_return > 0:
                st.success(
                    f"Rendimiento acumulado positivo: {total_return:.2f}%."
                )
            elif total_return < 0:
                st.warning(
                    f"Rendimiento acumulado negativo: {total_return:.2f}%."
                )
            else:
                st.info("El rendimiento acumulado se encuentra en torno a cero.")

            if annual_volatility > 30:
                st.warning(
                    "La volatilidad histórica es elevada."
                )
            else:
                st.info(
                    "La volatilidad histórica está por debajo del 30%."
                )

            if max_drawdown < -20:
                st.warning(
                    f"La caída máxima histórica ha sido del {max_drawdown:.2f}%."
                )
            else:
                st.success(
                    "La caída máxima histórica no ha superado el 20%."
                )

            st.info(
                f"Porcentaje de días con rendimiento positivo: "
                f"{positive_days:.1f}%."
            )

            st.subheader("Resumen cuantitativo")

            insights_summary = pd.DataFrame({
                "Indicador": [
                    "Rendimiento acumulado",
                    "Volatilidad anualizada",
                    "Caída máxima",
                    "Días positivos"
                ],
                "Resultado": [
                    f"{total_return:.2f}%",
                    f"{annual_volatility:.2f}%",
                    f"{max_drawdown:.2f}%",
                    f"{positive_days:.1f}%"
                ]
            })

            st.dataframe(
                insights_summary,
                use_container_width=True
            )

            st.caption(
                "Este motor utiliza reglas estadísticas programadas. "
                "No es un modelo de inteligencia artificial generativa "
                "ni ofrece predicciones financieras."
            )

        else:
            st.warning("No hay suficientes datos históricos.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("19. FINLAB Global Market Dashboard")
    st.caption("Comparación global de mercados y activos")

    global_assets = st.multiselect(
        "Selecciona mercados o activos",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_global_assets"
    )

    if len(global_assets) >= 1:

        global_data = returns[global_assets].dropna()

        if len(global_data) >= 2:

            st.subheader("Rendimiento acumulado")

            global_cumulative = (
                (1 + global_data).cumprod() - 1
            ) * 100

            st.line_chart(global_cumulative)

            st.subheader("Comparación de indicadores")

            global_metrics = []

            for asset in global_assets:

                asset_returns = global_data[asset]

                total_return = (
                    ((1 + asset_returns).prod() - 1) * 100
                )

                annual_volatility = (
                    asset_returns.std() * (252 ** 0.5) * 100
                )

                positive_days = (
                    (asset_returns > 0).mean() * 100
                )

                global_metrics.append({
                    "Activo": asset,
                    "Rendimiento acumulado (%)": total_return,
                    "Volatilidad anualizada (%)": annual_volatility,
                    "Días positivos (%)": positive_days
                })

            global_metrics_df = pd.DataFrame(global_metrics)

            st.dataframe(
                global_metrics_df.round(2),
                use_container_width=True
            )

            st.subheader("Rendimiento acumulado por activo")

            st.bar_chart(
                global_metrics_df.set_index("Activo")[
                    "Rendimiento acumulado (%)"
                ]
            )

            st.caption(
                "Los resultados comparan exclusivamente los activos "
                "seleccionados y disponibles en FINLAB. No representan "
                "una visión completa de todos los mercados mundiales."
            )

        else:
            st.warning("No hay suficientes datos para comparar los activos.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("20. FINLAB Quantitative Intelligence Center")
    st.caption("Centro avanzado de análisis cuantitativo")

    quant_assets = st.multiselect(
        "Selecciona activos para el análisis",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_quant_assets"
    )

    if len(quant_assets) >= 1:

        quant_data = returns[quant_assets].dropna()

        if len(quant_data) >= 20:

            quant_portfolio = quant_data.mean(axis=1)

            cumulative_quant = (1 + quant_portfolio).cumprod()

            total_return = (
                (cumulative_quant.iloc[-1] - 1) * 100
            )

            annual_return = (
                cumulative_quant.iloc[-1]
                ** (252 / len(quant_portfolio)) - 1
            ) * 100

            annual_volatility = (
                quant_portfolio.std() * (252 ** 0.5) * 100
            )

            downside_returns = quant_portfolio[
                quant_portfolio < 0
            ]

            downside_volatility = (
                downside_returns.std() * (252 ** 0.5) * 100
                if len(downside_returns) > 1 else 0
            )

            sharpe_ratio = (
                annual_return / annual_volatility
                if annual_volatility > 0 else 0
            )

            sortino_ratio = (
                annual_return / downside_volatility
                if downside_volatility > 0 else 0
            )

            running_max = cumulative_quant.cummax()

            drawdown = (
                cumulative_quant / running_max - 1
            )

            max_drawdown = drawdown.min() * 100

            positive_days = (
                (quant_portfolio > 0).mean() * 100
            )

            st.subheader("Quantitative Scorecard")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rendimiento total", f"{total_return:.2f}%")
                st.metric("Rendimiento anualizado", f"{annual_return:.2f}%")

            with col2:
                st.metric("Volatilidad anualizada", f"{annual_volatility:.2f}%")
                st.metric("Sharpe simplificado", f"{sharpe_ratio:.2f}")

            with col3:
                st.metric("Sortino simplificado", f"{sortino_ratio:.2f}")
                st.metric("Drawdown máximo", f"{max_drawdown:.2f}%")

            st.metric("Días positivos", f"{positive_days:.1f}%")

            st.subheader("Evolución del rendimiento")

            st.line_chart(cumulative_quant)

            st.subheader("Evolución del riesgo")

            st.line_chart(drawdown * 100)

            quant_summary = pd.DataFrame({
                "Indicador": [
                    "Rendimiento total",
                    "Rendimiento anualizado",
                    "Volatilidad anualizada",
                    "Sharpe simplificado",
                    "Sortino simplificado",
                    "Drawdown máximo",
                    "Días positivos"
                ],
                "Resultado": [
                    f"{total_return:.2f}%",
                    f"{annual_return:.2f}%",
                    f"{annual_volatility:.2f}%",
                    f"{sharpe_ratio:.2f}",
                    f"{sortino_ratio:.2f}",
                    f"{max_drawdown:.2f}%",
                    f"{positive_days:.1f}%"
                ]
            })

            st.subheader("Informe cuantitativo")

            st.dataframe(
                quant_summary,
                use_container_width=True
            )

            st.caption(
                "Cartera hipotética con ponderaciones iguales. "
                "Los ratios son simplificados y no descuentan una tasa "
                "libre de riesgo. Los datos históricos no garantizan "
                "resultados futuros."
            )

        else:
            st.warning("Se necesitan al menos 20 observaciones.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("21. FINLAB Institutional Report Generator")
    st.caption("Generador de informes cuantitativos en PDF")

    report_assets = st.multiselect(
        "Selecciona activos para el informe",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_report_assets"
    )

    if len(report_assets) >= 1:

        report_data = returns[report_assets].dropna()

        if len(report_data) >= 20:

            report_portfolio = report_data.mean(axis=1)

            report_cumulative = (1 + report_portfolio).cumprod()

            report_total_return = (
                (report_cumulative.iloc[-1] - 1) * 100
            )

            report_volatility = (
                report_portfolio.std() * (252 ** 0.5) * 100
            )

            report_running_max = report_cumulative.cummax()

            report_drawdown = (
                report_cumulative / report_running_max - 1
            )

            report_max_drawdown = report_drawdown.min() * 100

            report_summary = pd.DataFrame({
                "Indicador": [
                    "Rendimiento acumulado",
                    "Volatilidad anualizada",
                    "Drawdown máximo",
                    "Días positivos"
                ],
                "Resultado": [
                    f"{report_total_return:.2f}%",
                    f"{report_volatility:.2f}%",
                    f"{report_max_drawdown:.2f}%",
                    f"{(report_portfolio > 0).mean() * 100:.2f}%"
                ]
            })

            st.subheader("Vista previa del informe")

            st.dataframe(
                report_summary,
                use_container_width=True
            )

            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle
            )
            from reportlab.lib.styles import getSampleStyleSheet
            from io import BytesIO

            pdf_buffer = BytesIO()

            pdf_document = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4
            )

            styles = getSampleStyleSheet()

            report_elements = [
                Paragraph(
                    "FINLAB - Institutional Quantitative Report",
                    styles["Title"]
                ),
                Spacer(1, 20),
                Paragraph(
                    "Educational historical market analysis",
                    styles["Normal"]
                ),
                Spacer(1, 15),
                Paragraph(
                    "Selected assets: " + ", ".join(report_assets),
                    styles["Normal"]
                ),
                Spacer(1, 20)
            ]

            table_data = [
                ["Indicator", "Result"]
            ] + report_summary.values.tolist()

            report_table = Table(
                table_data,
                colWidths=[250, 180]
            )

            report_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 8)
            ]))

            report_elements.append(report_table)

            report_elements.extend([
                Spacer(1, 25),
                Paragraph(
                    "This report is educational and based on historical "
                    "data. It is not investment advice or a prediction "
                    "of future performance.",
                    styles["Normal"]
                )
            ])

            pdf_document.build(report_elements)

            pdf_buffer.seek(0)

            st.download_button(
                label="📄 Descargar informe FINLAB PDF",
                data=pdf_buffer.getvalue(),
                file_name="FINLAB_Institutional_Report.pdf",
                mime="application/pdf",
                key="v13_institutional_pdf"
            )

        else:
            st.warning("No hay suficientes datos para generar el informe.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("22. FINLAB Portfolio Intelligence Dashboard")
    st.caption("Análisis comparativo de activos y comportamiento histórico")

    intelligence_assets = st.multiselect(
        "Selecciona activos",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_intelligence_assets"
    )

    if len(intelligence_assets) >= 1:

        intelligence_data = returns[intelligence_assets].dropna()

        if len(intelligence_data) >= 20:

            intelligence_metrics = []

            for asset in intelligence_assets:

                asset_returns = intelligence_data[asset]

                cumulative = (1 + asset_returns).cumprod()

                total_return = (cumulative.iloc[-1] - 1) * 100

                annual_volatility = (
                    asset_returns.std() * (252 ** 0.5) * 100
                )

                positive_days = (
                    (asset_returns > 0).mean() * 100
                )

                running_max = cumulative.cummax()

                drawdown = (
                    cumulative / running_max - 1
                )

                max_drawdown = drawdown.min() * 100

                intelligence_metrics.append({
                    "Activo": asset,
                    "Rendimiento (%)": total_return,
                    "Volatilidad (%)": annual_volatility,
                    "Días positivos (%)": positive_days,
                    "Drawdown máximo (%)": max_drawdown
                })

            intelligence_df = pd.DataFrame(intelligence_metrics)

            st.subheader("Comparación de activos")

            st.dataframe(
                intelligence_df.round(2),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                best_performance = intelligence_df.loc[
                    intelligence_df["Rendimiento (%)"].idxmax()
                ]

                st.metric(
                    "Mayor rendimiento histórico",
                    best_performance["Activo"],
                    f'{best_performance["Rendimiento (%)"]:.2f}%'
                )

            with col2:
                lowest_volatility = intelligence_df.loc[
                    intelligence_df["Volatilidad (%)"].idxmin()
                ]

                st.metric(
                    "Menor volatilidad histórica",
                    lowest_volatility["Activo"],
                    f'{lowest_volatility["Volatilidad (%)"]:.2f}%'
                )

            st.subheader("Rendimiento histórico")

            st.bar_chart(
                intelligence_df.set_index("Activo")["Rendimiento (%)"]
            )

            st.subheader("Volatilidad histórica")

            st.bar_chart(
                intelligence_df.set_index("Activo")["Volatilidad (%)"]
            )

            st.caption(
                "Las clasificaciones describen únicamente el periodo "
                "histórico analizado. Un mejor resultado pasado no "
                "garantiza un mejor resultado futuro."
            )

        else:
            st.warning("No hay suficientes observaciones.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("23. FINLAB Advanced Risk Analytics")
    st.caption("Análisis avanzado de riesgo histórico")

    advanced_risk_assets = st.multiselect(
        "Selecciona activos para analizar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_advanced_risk_assets"
    )

    if len(advanced_risk_assets) >= 1:

        advanced_data = returns[advanced_risk_assets].dropna()

        if len(advanced_data) >= 30:

            advanced_portfolio = advanced_data.mean(axis=1)

            var_95 = advanced_portfolio.quantile(0.05) * 100
            var_99 = advanced_portfolio.quantile(0.01) * 100

            cvar_95 = (
                advanced_portfolio[
                    advanced_portfolio <= advanced_portfolio.quantile(0.05)
                ].mean() * 100
            )

            cvar_99 = (
                advanced_portfolio[
                    advanced_portfolio <= advanced_portfolio.quantile(0.01)
                ].mean() * 100
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("VaR histórico 95%", f"{var_95:.2f}%")
                st.metric("VaR histórico 99%", f"{var_99:.2f}%")

            with col2:
                st.metric("CVaR histórico 95%", f"{cvar_95:.2f}%")
                st.metric("CVaR histórico 99%", f"{cvar_99:.2f}%")

            st.subheader("Distribución de rendimientos")

            st.bar_chart(
                advanced_portfolio.tail(100) * 100
            )

            risk_comparison = pd.DataFrame({
                "Indicador": [
                    "VaR 95%",
                    "VaR 99%",
                    "CVaR 95%",
                    "CVaR 99%"
                ],
                "Resultado (%)": [
                    var_95,
                    var_99,
                    cvar_95,
                    cvar_99
                ]
            })

            st.subheader("Resumen de riesgo")

            st.dataframe(
                risk_comparison.round(3),
                use_container_width=True
            )

            st.caption(
                "VaR y CVaR se estiman mediante rendimientos históricos "
                "con ponderaciones iguales. Son medidas estadísticas "
                "y no límites máximos de pérdidas posibles."
            )

        else:
            st.warning("Se necesitan al menos 30 observaciones.")

    else:
        st.info("Selecciona al menos un activo.")
    st.divider()
    st.subheader("24. FINLAB Market Regime Detector")
    st.caption("Clasificación estadística del comportamiento del mercado")

    regime_asset = st.selectbox(
        "Selecciona un activo",
        institutional_assets,
        key="v13_regime_asset"
    )

    regime_returns = returns[regime_asset].dropna()

    if len(regime_returns) >= 50:

        regime_index = (1 + regime_returns).cumprod()

        regime_short = regime_index.rolling(20).mean()
        regime_long = regime_index.rolling(50).mean()

        regime_data = pd.DataFrame({
            "Índice acumulado": regime_index,
            "Media móvil 20": regime_short,
            "Media móvil 50": regime_long
        }).dropna()

        regime_data["Estado"] = "Lateral"

        regime_data.loc[
            regime_data["Media móvil 20"] >
            regime_data["Media móvil 50"],
            "Estado"
        ] = "Tendencia positiva"

        regime_data.loc[
            regime_data["Media móvil 20"] <
            regime_data["Media móvil 50"],
            "Estado"
        ] = "Tendencia negativa"

        current_regime = regime_data["Estado"].iloc[-1]

        st.subheader("Estado histórico más reciente")

        if current_regime == "Tendencia positiva":
            st.success(f"Estado detectado: {current_regime}")
        elif current_regime == "Tendencia negativa":
            st.warning(f"Estado detectado: {current_regime}")
        else:
            st.info(f"Estado detectado: {current_regime}")

        st.subheader("Indicadores de tendencia")

        st.line_chart(
            regime_data[
                [
                    "Índice acumulado",
                    "Media móvil 20",
                    "Media móvil 50"
                ]
            ]
        )

        st.subheader("Distribución de estados históricos")

        regime_distribution = (
            regime_data["Estado"]
            .value_counts(normalize=True) * 100
        )

        st.bar_chart(regime_distribution)

        st.caption(
            "La clasificación utiliza medias móviles de 20 y 50 "
            "observaciones sobre un índice construido con rendimientos. "
            "Es una aproximación educativa, no una señal de compra "
            "o venta ni una predicción."
        )

    else:
        st.warning("Se necesitan al menos 50 observaciones.")
    st.divider()
    st.subheader("25. FINLAB Quantitative Research Lab")
    st.caption("Laboratorio de investigación cuantitativa")

    research_assets = st.multiselect(
        "Selecciona activos para investigar",
        institutional_assets,
        default=institutional_assets[:min(5, len(institutional_assets))],
        key="v13_research_assets"
    )

    if len(research_assets) >= 2:

        research_data = returns[research_assets].dropna()

        if len(research_data) >= 30:

            research_metrics = []

            for asset in research_assets:

                asset_returns = research_data[asset]

                annual_return = (
                    (1 + asset_returns).prod()
                    ** (252 / len(asset_returns)) - 1
                ) * 100

                annual_volatility = (
                    asset_returns.std() * (252 ** 0.5) * 100
                )

                research_metrics.append({
                    "Activo": asset,
                    "Rentabilidad anualizada (%)": annual_return,
                    "Volatilidad anualizada (%)": annual_volatility
                })

            research_df = pd.DataFrame(research_metrics)

            st.subheader("Comparación cuantitativa")

            st.dataframe(
                research_df.round(2),
                use_container_width=True
            )

            st.subheader("Relación entre rentabilidad y riesgo")

            st.scatter_chart(
                research_df,
                x="Volatilidad anualizada (%)",
                y="Rentabilidad anualizada (%)",
                color="Activo"
            )

            st.subheader("Matriz de correlaciones")

            research_correlation = research_data.corr()

            st.dataframe(
                research_correlation.round(3),
                use_container_width=True
            )

            st.subheader("Resumen estadístico")

            st.write(
                "Activo con mayor rentabilidad histórica:",
                research_df.loc[
                    research_df["Rentabilidad anualizada (%)"].idxmax(),
                    "Activo"
                ]
            )

            st.write(
                "Activo con menor volatilidad histórica:",
                research_df.loc[
                    research_df["Volatilidad anualizada (%)"].idxmin(),
                    "Activo"
                ]
            )

            st.caption(
                "Análisis estadístico basado en datos históricos. "
                "Los resultados no constituyen predicciones ni "
                "recomendaciones de inversión."
            )

        else:
            st.warning("Se necesitan al menos 30 observaciones.")

    else:
        st.info("Selecciona al menos dos activos.")

st.markdown("<p style='text-align:center;color:#7890aa;font-size:.78rem'>FINLAB · EDUCATIONAL QUANTITATIVE RESEARCH · NO TRADING EXECUTION</p>", unsafe_allow_html=True)
