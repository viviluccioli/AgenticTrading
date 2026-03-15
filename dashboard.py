"""
FinAgent Orchestration — Live Trading Dashboard
================================================
Run with:  streamlit run dashboard.py
"""

import sys
import os
import time
import random
import uuid
import json
import statistics
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─── project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─── optional real-agent imports ──────────────────────────────────────────────
AUTONOMOUS_AVAILABLE = False
MOMENTUM_AVAILABLE   = False

try:
    from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent
    from FinAgents.agent_pools.alpha_agent_pool.schema.theory_driven_schema import (
        AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
    )
    AUTONOMOUS_AVAILABLE = True
except Exception:
    pass

try:
    from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent import MomentumAgent
    MOMENTUM_AVAILABLE = True
except Exception:
    pass

try:
    from neo4j import GraphDatabase
    NEO4J_DRIVER = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "password"))
    NEO4J_DRIVER.verify_connectivity()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE  = False
    NEO4J_DRIVER     = None

# ─── constants ────────────────────────────────────────────────────────────────
SYMBOLS      = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "SPY", "AMZN"]
AGENTS       = ["AutonomousAgent", "MomentumAgent", "RiskAgent", "PortfolioAgent", "MemoryAgent"]
REGIME_TAGS  = ["bullish_trend", "bearish_trend", "neutral_range", "volatile"]
SIGNAL_COLORS = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffab00"}
CAPITAL      = 100_000.0

DAG_NODES = ["Orchestrator", "DataAgent", "AlphaAgent", "RiskAgent", "PortfolioAgent"]
DAG_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]   # simplified topology

EVENT_TYPES = [
    "signal_received", "strategy_shared", "performance_updated",
    "agent_connected", "memory_indexed", "pattern_detected"
]
EVENT_COLORS = {
    "signal_received":      "#00e676",
    "strategy_shared":      "#40c4ff",
    "performance_updated":  "#ffab00",
    "agent_connected":      "#b388ff",
    "memory_indexed":       "#80cbc4",
    "pattern_detected":     "#ff80ab",
    "alert_triggered":      "#ff1744",
    "agent_disconnected":   "#757575",
}

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAgent Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }

/* cards */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 6px;
}

/* signal badges */
.badge-buy  { background:#0d2818; color:#00e676; border:1px solid #00e676;
              border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }
.badge-sell { background:#2b0a0a; color:#ff1744; border:1px solid #ff1744;
              border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }
.badge-hold { background:#2a1f00; color:#ffab00; border:1px solid #ffab00;
              border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }

/* agent status dot */
.dot-green  { height:9px; width:9px; background:#00e676; border-radius:50%;
              display:inline-block; margin-right:6px; animation: pulse 1.4s infinite; }
.dot-red    { height:9px; width:9px; background:#ff1744; border-radius:50%;
              display:inline-block; margin-right:6px; }
.dot-yellow { height:9px; width:9px; background:#ffab00; border-radius:50%;
              display:inline-block; margin-right:6px; animation: pulse 2s infinite; }

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,230,118,.5); }
    70%  { box-shadow: 0 0 0 7px rgba(0,230,118,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}

/* metric overrides */
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }

/* hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─── session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "signals":          [],
        "portfolio_history":[],
        "task_log":         [],
        "memory_events":    [],
        "dag_step":         0,
        "last_gen_time":    0.0,
        "auto_refresh":     True,
        "refresh_rate":     3,
        "selected_symbols": ["AAPL", "TSLA", "NVDA"],
        "signal_counts":    {"BUY": 0, "SELL": 0, "HOLD": 0},
        "session_start":    datetime.now().isoformat(),
        "portfolio_value":  CAPITAL,
        "agent_obj":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── lazy agent singleton ─────────────────────────────────────────────────────
def get_agent():
    if st.session_state.agent_obj is None and AUTONOMOUS_AVAILABLE:
        try:
            st.session_state.agent_obj = AutonomousAgent(agent_id="dashboard_agent")
        except Exception:
            pass
    return st.session_state.agent_obj


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gbm_prices(symbol: str, n: int = 20) -> list[float]:
    """Geometric Brownian Motion price series seeded per symbol."""
    seed = sum(ord(c) for c in symbol) + int(time.time() // 10)
    rng  = random.Random(seed)
    base = {"AAPL": 175, "TSLA": 250, "NVDA": 480, "MSFT": 380,
            "GOOGL": 160, "SPY": 520, "AMZN": 185}.get(symbol, 150)
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.gauss(0.0003, 0.012)))
    return prices


def _simulate_flow(symbol: str) -> dict:
    prices   = _gbm_prices(symbol)
    cur      = prices[-1]
    prev     = prices[-2]
    momentum = (cur - prev) / prev

    if momentum > 0.008:
        signal, conf = "BUY",  round(random.uniform(0.70, 0.95), 2)
    elif momentum < -0.008:
        signal, conf = "SELL", round(random.uniform(0.65, 0.92), 2)
    else:
        signal, conf = "HOLD", round(random.uniform(0.50, 0.75), 2)

    regime = (
        "bullish_trend"  if signal == "BUY"  and conf > 0.75 else
        "bearish_trend"  if signal == "SELL" and conf > 0.75 else
        "volatile"       if abs(momentum) > 0.015 else
        "neutral_range"
    )

    agent_name = random.choice([
        "autonomous_alpha_agent", "momentum_agent", "empirical_agent"
    ])
    alpha_id   = f"sim_{uuid.uuid4().hex[:8]}"
    now        = datetime.now().isoformat()

    return {
        "alpha_id":        alpha_id,
        "version":         "1.0",
        "timestamp":       now,
        "market_context":  {
            "symbol":         symbol,
            "regime_tag":     regime,
            "input_features": {
                "current_price":       round(cur, 2),
                "momentum":            round(momentum, 5),
                "sma_10":              round(statistics.mean(prices[-10:]), 2),
                "volatility":          round(statistics.stdev(prices) / cur, 4),
                "price_change_pct":    round((cur - prices[0]) / prices[0] * 100, 2),
            },
        },
        "decision": {
            "signal":           signal,
            "confidence":       conf,
            "reasoning":        _reasoning(signal, momentum, conf, symbol),
            "predicted_return": round(momentum * conf, 4),
            "risk_estimate":    round(random.uniform(0.01, 0.06), 4),
            "signal_type":      "directional",
            "asset_scope":      [symbol],
        },
        "action": {
            "execution_weight": round(conf * 0.5, 3),
            "order_type":       "market",
            "order_price":      round(cur, 2),
            "execution_delay":  "T+0",
        },
        "performance_feedback": {"status": "pending", "evaluation_link": None},
        "metadata": {
            "generator_agent":  agent_name,
            "strategy_prompt":  f"Dashboard analysis for {symbol}",
            "code_hash":        f"sha256:{uuid.uuid4().hex[:16]}",
            "context_id":       f"dash_{now[:10].replace('-','')}_{now[11:13]}",
        },
    }


def _reasoning(signal: str, momentum: float, conf: float, symbol: str) -> str:
    if signal == "BUY":
        return (f"{symbol} shows positive momentum ({momentum:+.3%}). "
                f"Moving averages trending upward with {conf:.0%} confidence.")
    elif signal == "SELL":
        return (f"{symbol} shows bearish momentum ({momentum:+.3%}). "
                f"Price below key moving averages — {conf:.0%} confidence in reversal.")
    else:
        return (f"{symbol} in consolidation phase (momentum: {momentum:+.3%}). "
                "Waiting for clearer directional signal.")


def _real_flow(symbol: str) -> dict | None:
    agent = get_agent()
    if agent is None:
        return None
    try:
        prices  = _gbm_prices(symbol, 15)
        result  = agent._perform_autonomous_analysis(prices, f"Analyze {symbol}")
        result.update({"current_price": prices[-1], "symbol": symbol,
                       "analysis_timestamp": datetime.now().isoformat()})
        flow = agent._generate_strategy_flow(symbol, result, f"Dashboard: {symbol}")
        return flow if isinstance(flow, dict) else flow.dict()
    except Exception:
        return None


def generate_signal(symbol: str) -> dict:
    flow = _real_flow(symbol)
    return flow if flow else _simulate_flow(symbol)


def simulate_memory_event(agent_name: str, signal: str) -> dict:
    etype = random.choice(EVENT_TYPES)
    return {
        "event_id":   uuid.uuid4().hex[:10],
        "event_type": etype,
        "timestamp":  datetime.now().strftime("%H:%M:%S"),
        "agent_id":   agent_name,
        "payload":    {"signal": signal, "detail": f"{agent_name} processed {etype}"},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL CYCLE  (runs once per rerun when enough time elapsed)
# ══════════════════════════════════════════════════════════════════════════════

def run_signal_cycle():
    now = time.time()
    if now - st.session_state.last_gen_time < st.session_state.refresh_rate:
        return
    st.session_state.last_gen_time = now

    for sym in st.session_state.selected_symbols:
        flow = generate_signal(sym)
        st.session_state.signals.append(flow)
        if len(st.session_state.signals) > 120:
            st.session_state.signals.pop(0)

        sig = flow["decision"]["signal"]
        st.session_state.signal_counts[sig] += 1

        # portfolio update
        delta = {"BUY": 0.003, "SELL": -0.002, "HOLD": 0.0003}[sig]
        noise = random.gauss(0, 0.002)
        st.session_state.portfolio_value *= (1 + delta + noise)
        st.session_state.portfolio_history.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "value":  round(st.session_state.portfolio_value, 2),
            "signal": sig,
            "symbol": sym,
        })
        if len(st.session_state.portfolio_history) > 200:
            st.session_state.portfolio_history.pop(0)

        # task log entry (show decomposed tasks)
        tasks = [
            f"Query memory for {sym} context",
            f"Generate analysis tool for {sym}",
            f"Execute technical analysis on {sym}",
            f"Generate strategy flow for {sym}",
            f"Validate results for {sym}",
        ]
        st.session_state.task_log.append({
            "instruction": f"Analyze {sym}",
            "tasks":       tasks,
            "timestamp":   datetime.now().strftime("%H:%M:%S"),
            "signal":      sig,
        })
        if len(st.session_state.task_log) > 30:
            st.session_state.task_log.pop(0)

        # memory event
        agent_name = flow["metadata"]["generator_agent"]
        st.session_state.memory_events.append(simulate_memory_event(agent_name, sig))
        if len(st.session_state.memory_events) > 50:
            st.session_state.memory_events.pop(0)

    # advance DAG step
    st.session_state.dag_step = (st.session_state.dag_step + 1) % len(DAG_NODES)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_chart() -> go.Figure:
    hist = st.session_state.portfolio_history
    if not hist:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                          font_color="#8b949e", height=280,
                          margin=dict(l=10, r=10, t=10, b=10))
        return fig

    df     = pd.DataFrame(hist)
    colors = df["signal"].map(SIGNAL_COLORS).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["value"],
        mode="lines", name="Portfolio",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["value"],
        mode="markers", name="Signals",
        marker=dict(color=colors, size=7, line=dict(width=1, color="#161b22")),
        hovertemplate="<b>%{text}</b><br>$%{y:,.0f}<extra></extra>",
        text=df["signal"] + " " + df["symbol"],
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9", height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(showgrid=False, color="#8b949e",
                   tickmode="auto", nticks=8, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#21262d",
                   color="#8b949e", tickprefix="$", tickformat=",.0f"),
        hovermode="x unified",
    )
    return fig


def signal_gauge(confidence: float, signal: str) -> go.Figure:
    color = SIGNAL_COLORS.get(signal, "#8b949e")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#21262d",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  40], "color": "#161b22"},
                {"range": [40, 70], "color": "#1c2128"},
                {"range": [70, 100],"color": "#1c2a20"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": confidence * 100,
            },
        },
        title={"text": f"Confidence — <b>{signal}</b>",
               "font": {"color": color, "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", font_color="#c9d1d9",
        height=200, margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def signal_dist_chart() -> go.Figure:
    counts = st.session_state.signal_counts
    fig = go.Figure(go.Bar(
        x=list(counts.values()),
        y=list(counts.keys()),
        orientation="h",
        marker_color=[SIGNAL_COLORS[k] for k in counts],
        text=list(counts.values()),
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9", height=150,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(showgrid=False, color="#8b949e"),
        yaxis=dict(showgrid=False, color="#c9d1d9", tickfont={"size": 12}),
    )
    return fig


def dag_chart(active_node_idx: int) -> go.Figure:
    xs = [0.05, 0.27, 0.50, 0.73, 0.95, 0.50]
    ys = [0.50, 0.78, 0.50, 0.78, 0.50, 0.15]
    node_labels = DAG_NODES + ["MemoryAgent"]

    node_colors = []
    for i, _ in enumerate(node_labels):
        if i == active_node_idx:
            node_colors.append("#58a6ff")
        elif i < active_node_idx:
            node_colors.append("#00e676")
        else:
            node_colors.append("#21262d")

    edges_x, edges_y = [], []
    all_edges = DAG_EDGES + [(i, 5) for i in range(5)]  # memory connects to all
    for src, dst in all_edges:
        edges_x += [xs[src], xs[dst], None]
        edges_y += [ys[src], ys[dst], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode="lines",
        line=dict(color="#30363d", width=1.5),
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(color=node_colors, size=34,
                    line=dict(color="#0d1117", width=2)),
        text=node_labels,
        textfont=dict(color="#c9d1d9", size=9),
        textposition="middle center",
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        height=260, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.0]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 FinAgent")
    st.caption("Real-time Agentic Trading Dashboard")
    st.divider()

    st.markdown("**Symbol Selection**")
    selected = st.multiselect(
        "Active symbols", SYMBOLS,
        default=st.session_state.selected_symbols, label_visibility="collapsed"
    )
    st.session_state.selected_symbols = selected or ["AAPL"]

    st.divider()
    st.markdown("**Refresh Controls**")
    rate = st.slider("Refresh interval (s)", 1, 10,
                     st.session_state.refresh_rate, label_visibility="collapsed")
    st.session_state.refresh_rate = rate

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Resume" if not st.session_state.auto_refresh else "⏸ Pause",
                     use_container_width=True):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
    with col_b:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["signals","portfolio_history","task_log","memory_events",
                      "signal_counts","portfolio_value","dag_step","last_gen_time"]:
                del st.session_state[k]
            _init_state()
            st.rerun()

    st.divider()
    st.markdown("**Agent Status**")

    agent_status = [
        ("AutonomousAgent",   AUTONOMOUS_AVAILABLE, "live" if AUTONOMOUS_AVAILABLE else "sim"),
        ("MomentumAgent",     MOMENTUM_AVAILABLE,   "live" if MOMENTUM_AVAILABLE   else "sim"),
        ("RiskAgent",         False,                "sim"),
        ("PortfolioAgent",    False,                "sim"),
        ("MemoryAgent / Neo4j", NEO4J_AVAILABLE,    "live" if NEO4J_AVAILABLE else "offline"),
    ]
    for name, ok, label in agent_status:
        dot   = "dot-green" if ok else "dot-yellow"
        color = "#00e676" if ok else "#ffab00"
        st.markdown(
            f'<span class="{dot}"></span>'
            f'<span style="font-size:12px;color:#c9d1d9;">{name}</span> '
            f'<span style="font-size:10px;color:{color};">[{label}]</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("**Manually Trigger Analysis**")
    trigger_sym = st.selectbox("Symbol", SYMBOLS, label_visibility="collapsed")
    if st.button("Run Analysis Now", use_container_width=True):
        flow = generate_signal(trigger_sym)
        st.session_state.signals.append(flow)
        sig = flow["decision"]["signal"]
        st.session_state.signal_counts[sig] += 1
        delta = {"BUY": 0.003, "SELL": -0.002, "HOLD": 0.0003}[sig]
        st.session_state.portfolio_value *= (1 + delta)
        st.session_state.portfolio_history.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "value":  round(st.session_state.portfolio_value, 2),
            "signal": sig, "symbol": trigger_sym,
        })
        st.rerun()

    st.divider()
    elapsed = str(timedelta(seconds=int(
        (datetime.now() - datetime.fromisoformat(st.session_state.session_start)).total_seconds()
    )))
    st.caption(f"Session: {elapsed}")
    st.caption(f"Total signals: {len(st.session_state.signals)}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL CYCLE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.auto_refresh:
    run_signal_cycle()


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 1 — Title + KPI strip
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📈 FinAgent Orchestration — Live Dashboard")
mode_str = "🟢 Live agents active" if AUTONOMOUS_AVAILABLE else "🟡 Simulation mode"
st.caption(mode_str + f" · Auto-refresh every {st.session_state.refresh_rate}s")

pnl      = st.session_state.portfolio_value - CAPITAL
pnl_pct  = pnl / CAPITAL * 100
total_sig = len(st.session_state.signals)
counts    = st.session_state.signal_counts
accuracy  = (counts["BUY"] / max(total_sig, 1)) * 100   # proxy metric

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Portfolio Value",  f"${st.session_state.portfolio_value:,.0f}",
           f"{pnl_pct:+.2f}%")
k2.metric("P&L",              f"${pnl:+,.0f}")
k3.metric("Total Signals",    total_sig)
k4.metric("BUY / SELL / HOLD",
           f"{counts['BUY']} / {counts['SELL']} / {counts['HOLD']}")
k5.metric("Active Symbols",   len(st.session_state.selected_symbols))

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2 — Portfolio chart  |  Signal feed
# ══════════════════════════════════════════════════════════════════════════════
col_chart, col_feed = st.columns([3, 2])

with col_chart:
    st.markdown('<div class="card-title">Portfolio Equity Curve</div>',
                unsafe_allow_html=True)
    st.plotly_chart(portfolio_chart(), use_container_width=True, key="equity")

with col_feed:
    st.markdown('<div class="card-title">Live Signal Feed</div>',
                unsafe_allow_html=True)

    if st.session_state.signals:
        rows = []
        for s in reversed(st.session_state.signals[-18:]):
            sig = s["decision"]["signal"]
            badge = f'<span class="badge-{sig.lower()}">{sig}</span>'
            rows.append({
                "Time":       s["timestamp"][11:19],
                "Symbol":     s["market_context"]["symbol"],
                "Signal":     sig,
                "Conf":       f"{s['decision']['confidence']:.0%}",
                "Regime":     s["market_context"]["regime_tag"].replace("_"," "),
                "Agent":      s["metadata"]["generator_agent"].replace("_agent",""),
            })
        feed_df = pd.DataFrame(rows)

        def color_signal(val):
            c = {"BUY": "#0d2818", "SELL": "#2b0a0a", "HOLD": "#2a1f00"}.get(val, "")
            t = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffab00"}.get(val, "")
            return f"background-color:{c}; color:{t}; font-weight:bold" if c else ""

        styled = feed_df.style.map(color_signal, subset=["Signal"])
        st.dataframe(styled, use_container_width=True, height=290, hide_index=True)
    else:
        st.info("Waiting for first signals…")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 3 — Gauge  |  DAG viz  |  Memory events
# ══════════════════════════════════════════════════════════════════════════════
col_gauge, col_dag, col_mem = st.columns(3)

with col_gauge:
    st.markdown('<div class="card-title">Latest Signal Confidence</div>',
                unsafe_allow_html=True)
    if st.session_state.signals:
        latest = st.session_state.signals[-1]
        conf   = latest["decision"]["confidence"]
        sig    = latest["decision"]["signal"]
        sym    = latest["market_context"]["symbol"]
        st.plotly_chart(signal_gauge(conf, sig), use_container_width=True, key="gauge")
        st.markdown(
            f'**{sym}** · {latest["market_context"]["regime_tag"].replace("_"," ").title()}<br>'
            f'<span style="font-size:12px;color:#8b949e;">{latest["decision"]["reasoning"]}</span>',
            unsafe_allow_html=True
        )
        st.plotly_chart(signal_dist_chart(), use_container_width=True, key="dist")
    else:
        st.info("No signals yet.")

with col_dag:
    st.markdown('<div class="card-title">DAG Execution Flow</div>',
                unsafe_allow_html=True)
    active = st.session_state.dag_step % len(DAG_NODES)
    st.plotly_chart(dag_chart(active), use_container_width=True, key="dag")

    # task decomposition log
    st.markdown('<div class="card-title" style="margin-top:8px;">Task Decomposition</div>',
                unsafe_allow_html=True)
    if st.session_state.task_log:
        latest_task = st.session_state.task_log[-1]
        with st.expander(
            f"[{latest_task['timestamp']}] {latest_task['instruction']} → "
            f"{latest_task['signal']}", expanded=True
        ):
            for i, t in enumerate(latest_task["tasks"]):
                icon = "✅" if i < active else ("🔄" if i == active else "⏳")
                st.markdown(f"{icon} {t}")

        if len(st.session_state.task_log) > 1:
            with st.expander(f"Previous ({len(st.session_state.task_log)-1} runs)"):
                for entry in reversed(st.session_state.task_log[:-1][-5:]):
                    st.markdown(
                        f"`{entry['timestamp']}` **{entry['instruction']}** "
                        f"→ {entry['signal']}"
                    )
    else:
        st.info("No tasks yet.")

with col_mem:
    st.markdown('<div class="card-title">Memory / Event Stream</div>',
                unsafe_allow_html=True)

    # Try Neo4j query
    neo_rows = []
    if NEO4J_AVAILABLE and NEO4J_DRIVER:
        try:
            with NEO4J_DRIVER.session(database="neo4j") as session:
                result = session.run(
                    "MATCH (m:Memory) RETURN m.agent_id AS agent, "
                    "m.event_type AS etype, m.timestamp AS ts "
                    "ORDER BY m.timestamp DESC LIMIT 8"
                )
                neo_rows = [dict(r) for r in result]
        except Exception:
            pass

    events_to_show = list(reversed(st.session_state.memory_events[-20:]))

    if neo_rows:
        st.caption("📡 Live Neo4j events:")
        for row in neo_rows[:5]:
            col = EVENT_COLORS.get(str(row.get("etype","")),"#8b949e")
            st.markdown(
                f'<div style="border-left:3px solid {col};padding:4px 8px;'
                f'margin:3px 0;font-size:11px;color:#c9d1d9;">'
                f'<b style="color:{col};">{row.get("etype","")}</b> · '
                f'{row.get("agent","?")} · {str(row.get("ts",""))[:19]}</div>',
                unsafe_allow_html=True
            )
        st.divider()

    st.caption("🔄 Simulated event stream:")
    for ev in events_to_show[:15]:
        col = EVENT_COLORS.get(ev["event_type"], "#8b949e")
        st.markdown(
            f'<div style="border-left:3px solid {col};padding:4px 8px;'
            f'margin:3px 0;font-size:11px;color:#c9d1d9;">'
            f'<b style="color:{col};">{ev["event_type"]}</b> '
            f'<span style="color:#8b949e;">[{ev["timestamp"]}]</span><br>'
            f'<span style="color:#8b949e;">{ev["agent_id"]}</span></div>',
            unsafe_allow_html=True
        )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 4 — Per-symbol breakdown
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card-title">Per-Symbol Signal Breakdown</div>',
            unsafe_allow_html=True)

if st.session_state.signals:
    sym_data: dict[str, dict] = {}
    for s in st.session_state.signals:
        sym  = s["market_context"]["symbol"]
        sig  = s["decision"]["signal"]
        conf = s["decision"]["confidence"]
        if sym not in sym_data:
            sym_data[sym] = {"signals": [], "confs": [], "last": sig, "last_conf": conf,
                             "last_regime": s["market_context"]["regime_tag"],
                             "last_price": s["action"]["order_price"]}
        sym_data[sym]["signals"].append(sig)
        sym_data[sym]["confs"].append(conf)
        sym_data[sym]["last"] = sig
        sym_data[sym]["last_conf"] = conf
        sym_data[sym]["last_regime"] = s["market_context"]["regime_tag"]
        sym_data[sym]["last_price"] = s["action"]["order_price"]

    cols = st.columns(min(len(sym_data), 7))
    for i, (sym, d) in enumerate(sym_data.items()):
        with cols[i % len(cols)]:
            sig = d["last"]
            c   = SIGNAL_COLORS[sig]
            avg_conf = statistics.mean(d["confs"]) if d["confs"] else 0
            st.markdown(
                f'<div style="background:#161b22;border:1px solid {c};border-radius:8px;'
                f'padding:12px;text-align:center;">'
                f'<div style="font-size:16px;font-weight:700;color:#c9d1d9;">{sym}</div>'
                f'<div style="font-size:22px;font-weight:800;color:{c};margin:4px 0;">{sig}</div>'
                f'<div style="font-size:12px;color:#8b949e;">'
                f'${d["last_price"]:,.2f} · {avg_conf:.0%} avg conf<br>'
                f'{d["last_regime"].replace("_"," ")}</div>'
                f'<div style="font-size:11px;color:#8b949e;margin-top:4px;">'
                f'{len(d["signals"])} signals</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH TAIL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_rate)
    st.rerun()
