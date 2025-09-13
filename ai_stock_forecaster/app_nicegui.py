import json
from datetime import datetime
import plotly.graph_objects as go
from nicegui import ui
from src.agent.graph import build_graph


def convert_timestamps_to_strings(data):
    """Convert any pandas Timestamps to strings for JSON serialization"""
    if hasattr(data, "strftime"):
        return data.strftime("%Y-%m-%d")
    elif hasattr(data, "astype"):
        return data.astype(str)
    elif isinstance(data, (list, tuple)):
        return [str(x) for x in data]
    else:
        return [str(x) for x in data] if hasattr(data, "__iter__") else str(data)


def plot_prices_and_signals(df_feat, sig_rule, sig_ml):
    fig = go.Figure()

    # Convert index to JSON-serializable format
    x_dates = convert_timestamps_to_strings(df_feat.index)

    fig.add_trace(go.Scatter(x=x_dates, y=df_feat["adj_close"], name="Adj Close"))
    fig.add_trace(go.Scatter(x=x_dates, y=df_feat["sma_10"], name="SMA10"))
    fig.add_trace(go.Scatter(x=x_dates, y=df_feat["sma_50"], name="SMA50"))

    # Get signal dates and convert to strings
    rule_signal_mask = sig_rule.astype(bool)
    ml_signal_mask = sig_ml.astype(bool)

    on_dates_rule = df_feat.index[rule_signal_mask]
    on_dates_ml = df_feat.index[ml_signal_mask]

    # Convert to JSON-serializable format
    on_dates_rule_str = convert_timestamps_to_strings(on_dates_rule)
    on_dates_ml_str = convert_timestamps_to_strings(on_dates_ml)

    fig.add_trace(
        go.Scatter(
            x=on_dates_rule_str,
            y=df_feat.loc[on_dates_rule, "adj_close"],
            mode="markers",
            name="Rule Long",
            marker_symbol="triangle-up",
            marker_size=6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=on_dates_ml_str,
            y=df_feat.loc[on_dates_ml, "adj_close"],
            mode="markers",
            name="ML Long",
            marker_symbol="diamond",
            marker_size=6,
        )
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))

    return fig


def plot_equity(eq_rule, eq_ml):
    fig = go.Figure()

    # Convert index to datetime strings to avoid JSON serialization issues
    if hasattr(eq_rule.index, "strftime"):
        x_dates_rule = eq_rule.index.strftime("%Y-%m-%d")
    elif hasattr(eq_rule.index, "astype"):
        x_dates_rule = eq_rule.index.astype(str)
    else:
        x_dates_rule = [str(x) for x in eq_rule.index]

    if hasattr(eq_ml.index, "strftime"):
        x_dates_ml = eq_ml.index.strftime("%Y-%m-%d")
    elif hasattr(eq_ml.index, "astype"):
        x_dates_ml = eq_ml.index.astype(str)
    else:
        x_dates_ml = [str(x) for x in eq_ml.index]

    fig.add_trace(go.Scatter(x=x_dates_rule, y=eq_rule, name="Equity Rule"))
    fig.add_trace(go.Scatter(x=x_dates_ml, y=eq_ml, name="Equity ML"))

    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))

    return fig


ui.colors(primary="#0ea5e9")

with ui.header().classes("items-center justify-between"):
    ui.label("LLM Stock Forecaster (NiceGUI, ta)").classes("text-xl font-medium")
    ui.label(str(datetime.now().strftime("%Y-%m-%d")))

with ui.row().classes("w-full"):
    with ui.card().classes("min-w-[320px] max-w-[420px]"):
        ui.label("Inputs").classes("text-lg font-medium")
        symbol_in = ui.input("Ticker", value="AAPL")
        provider_sel = ui.select(["yf"], value="yf", label="Provider")
        outputsize_sel = ui.select(
            ["compact", "full"], value="compact", label="AlphaVantage outputsize"
        )
        fee_in = ui.number(label="Fee (bps per trade)", value=1.0, step=0.5, min=0)
        run_btn = ui.button("Run", color="primary")
        status = ui.label().classes("text-sm text-gray-500")

    with ui.column().classes("w-full"):
        # Create empty figures for initial display
        empty_price_fig = go.Figure()
        empty_price_fig.update_layout(
            title="Stock Prices and Signals",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        empty_equity_fig = go.Figure()
        empty_equity_fig.update_layout(
            title="Equity Curves", height=320, margin=dict(l=20, r=20, t=30, b=20)
        )

        price_plot = ui.plotly(empty_price_fig).classes("w-full")
        equity_plot = ui.plotly(empty_equity_fig).classes("w-full")
        stats_table = ui.table(
            columns=[
                {"name": "metric", "label": "Metric", "field": "metric"},
                {"name": "rule", "label": "Rule", "field": "rule"},
                {"name": "ml", "label": "ML", "field": "ml"},
            ],
            rows=[],
        ).classes("w-full")
        meta_json = ui.code(language="json").classes("w-full")


async def _run():
    try:
        status.text = "Processing... Building graph"
        from src.core.backtest import run_backtest

        app = build_graph()

        status.text = "Processing... Fetching data"
        state = {"question": "", "symbol": symbol_in.value.strip().upper()}
        if provider_sel.value == "alpha":
            state["provider"] = "alpha"
        out = app.invoke(state)

        status.text = "Processing... Analyzing data"
        feat = out["features"]
        sig_r = out["signals"]["rule"]
        sig_m = out["signals"]["ml"]

        status.text = "Processing... Updating charts"
        price_plot.update_figure(plot_prices_and_signals(feat, sig_r, sig_m))

        # Use features directly since it already contains adj_close
        px = feat
        r_rule = run_backtest(px, sig_r, fee_bps=fee_in.value)
        r_ml = run_backtest(px, sig_m, fee_bps=fee_in.value)

        equity_plot.update_figure(plot_equity(r_rule["equity"], r_ml["equity"]))

        status.text = "Processing... Calculating statistics"
        rows = []
        for m in ["CAGR", "Sharpe", "MaxDD", "Trades"]:
            rows.append(
                {
                    "metric": m,
                    "rule": (
                        f'{r_rule["stats"][m]:.4f}'
                        if isinstance(r_rule["stats"][m], float)
                        else r_rule["stats"][m]
                    ),
                    "ml": (
                        f'{r_ml["stats"][m]:.4f}'
                        if isinstance(r_ml["stats"][m], float)
                        else r_ml["stats"][m]
                    ),
                }
            )
        stats_table.rows = rows
        stats_table.update()

        meta = {
            "provider": out["provider"],
            "symbol": out["symbol"],
            "rows": len(out["data"]),
        }
        meta_json.set_content(json.dumps(meta, indent=2))
        status.text = "Done."

    except Exception as e:
        status.text = f"Error: {str(e)}"
        # Optionally log the full error for debugging
        print(f"Error in _run: {e}")
        import traceback

        traceback.print_exc()  # This will print the full stack trace to console


run_btn.on_click(_run)

ui.run(title="LLM Stock Forecaster", reload=False)
