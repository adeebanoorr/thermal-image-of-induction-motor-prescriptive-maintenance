# src/visualization/dashboard.py
# Prescriptive Maintenance Dashboard — Premium Industrial UI

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import base64

# === File Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "prescriptive_outputs", "prescriptive_predictions.csv")
IMAGE_DIR   = os.path.join(PROJECT_ROOT, "reports", "prescriptive_outputs")
HIGH_SEV_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "high_severity_log.csv")

# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_data():
    if os.path.exists(REPORT_PATH):
        df = pd.read_csv(REPORT_PATH)
        df.rename(columns={"Image": "Image_Name"}, inplace=True)
        return df
    return pd.DataFrame()


def encode_image(image_name):
    path = os.path.join(IMAGE_DIR, image_name)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    return None


def get_severity_class(severity):
    """Returns CSS class depending on severity level."""
    if severity >= 50:
        return "critical"
    elif severity >= 30:
        return "warning"
    return "healthy"


def get_fault_badge_class(fault_type):
    ft = fault_type.lower()
    if "stator" in ft: return "stator"
    if "rotor"  in ft: return "rotor"
    if "cooling" in ft or "fan" in ft: return "cooling"
    if "healthy" in ft: return "healthy"
    return "unknown"


# ── Card Renderer ──────────────────────────────────────────────────────────────

def render_card(c):
    try:
        sev = float(c["Severity"])
    except Exception:
        sev = 0.0

    sev_cls      = get_severity_class(sev)
    fault_bg     = get_fault_badge_class(c["Fault_Type"])
    is_critical  = sev >= 30

    # Severity colour labels for inline style
    bar_colour_map = {
        "critical": "fill-critical",
        "warning":  "fill-warning",
        "healthy":  "fill-healthy",
    }
    bar_class      = bar_colour_map[sev_cls]
    sev_val_class  = f"severity-value-{sev_cls}"
    card_extra_cls = "critical-card" if sev >= 50 else ("" if sev >= 30 else "healthy-card" if sev < 15 else "")

    # ── Alert Banner ──
    alert_banner = html.Div([
        html.Span("⚠", className="alert-icon"),
        html.Span("HIGH SEVERITY — Immediate attention required!", className="alert-text"),
    ], className="alert-critical") if is_critical else html.Div()

    # ── Recommendation block ──
    rec_section = html.Div([
        html.Div([
            html.Span("📌", className="rec-icon"),
            html.Div([
                html.Div("Reason",         className="rec-label"),
                html.Div(c.get("Reason","—"), className="rec-value"),
            ])
        ], className="rec-row"),
        html.Div([
            html.Span("🔩", className="rec-icon"),
            html.Div([
                html.Div("Recommendation",          className="rec-label"),
                html.Div(c.get("Recommendation","—"), className="rec-value"),
            ])
        ], className="rec-row"),
        html.Div([
            html.Span("🚨", className="rec-icon"),
            html.Div([
                html.Div("Next Step",             className="rec-label"),
                html.Div(c.get("Next_Step","—"),  className="rec-value"),
            ])
        ], className="rec-row"),
    ], className="recommendation-section") if is_critical else html.Div()

    return html.Div([
        # Colour-coded top strip
        html.Div(className=f"card-header-strip severity-{sev_cls}"),

        html.Div([

            # Top: thumbnail + fault badge + filename
            html.Div([
                html.Div([
                    html.Img(src=c.get("Image_Encoded"), style={"width":"100%","height":"100%","objectFit":"cover"})
                    if c.get("Image_Encoded") else html.Div("No Image", style={"color":"#475569","fontSize":"11px","display":"flex","alignItems":"center","justifyContent":"center","height":"100%"})
                ], className="card-image-wrap"),

                html.Div([
                    html.Span(c["Fault_Type"], className=f"fault-badge {fault_bg}"),
                    html.Div(c["Image_Name"], className="image-filename", title=c["Image_Name"]),
                ], className="card-badges-col"),
            ], className="card-top-row"),

            # Alert banner (critical only)
            alert_banner,

            # Severity bar
            html.Div([
                html.Div([
                    html.Span("Severity", className="severity-label"),
                    html.Span(f"{sev:.1f}%", className=f"severity-label {sev_val_class}"),
                ], className="severity-header"),
                html.Div([
                    html.Div(className=f"severity-bar-fill {bar_class}",
                             style={"width": f"{min(sev, 100):.1f}%"}),
                ], className="severity-bar-track"),
            ], className="severity-section"),

            # Info grid
            html.Div([
                html.Div([
                    html.Div("Action",               className="info-item-label"),
                    html.Div(c["ActionTaken"],        className="info-item-value"),
                ], className="info-item"),
                html.Div([
                    html.Div("Fault Type",            className="info-item-label"),
                    html.Div(c["Fault_Type"],         className="info-item-value"),
                ], className="info-item"),
                html.Div([
                    html.Div("Est. Cost",             className="info-item-label"),
                    html.Div(f"₹{c['Estimated_Cost']:,.2f}", className="info-item-value mono"),
                ], className="info-item"),
                html.Div([
                    html.Div("Downtime",              className="info-item-label"),
                    html.Div(f"{c['Estimated_Downtime_Days']:.1f} days", className="info-item-value mono"),
                ], className="info-item"),
            ], className="info-grid"),

            # Recommendation (critical only)
            rec_section,

        ], className="card-body"),

    ], className=f"maint-card {card_extra_cls}")


# ── Dash App ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    assets_folder=os.path.join(BASE_DIR, "assets"),
    title="Prescriptive Maintenance Dashboard",
)
server = app.server


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("⚙", className="header-icon"),
                html.Div([
                    html.Div("Prescriptive Maintenance", className="header-title"),
                    html.Div("Induction Motor Thermal Fault Analysis System", className="header-subtitle"),
                ]),
            ], className="header-logo"),
            html.Div([
                html.Div(className="live-dot"),
                html.Span("Live"),
            ], className="live-badge"),
        ], className="dashboard-header"),

        # ── Stats Bar ────────────────────────────────────────────────────────
        html.Div(id="stats-bar", className="stats-bar"),

        # ── Controls ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Filter: ", className="controls-label"),
                dcc.RadioItems(
                    id="filter-option",
                    options=[
                        {"label": "All Records", "value": "all"},
                        {"label": "High Severity Only", "value": "high"},
                    ],
                    value="all",
                    labelStyle={"display": "inline-block", "marginRight": "18px",
                                "cursor": "pointer", "fontSize": "13px",
                                "color": "#94a3b8", "fontWeight": "500"},
                    inputStyle={"marginRight": "6px", "accentColor": "#4f8ef7"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="record-counter", style={
                "fontSize": "12px", "color": "#475569",
                "fontFamily": "'JetBrains Mono', monospace",
            }),
        ], className="controls-bar"),

        # ── Card Feed ────────────────────────────────────────────────────────
        dcc.Store(id="current-index", data=0),
        dcc.Store(id="shown-cards",   data=[]),
        dcc.Interval(id="interval-update", interval=4000, n_intervals=0),

        html.Div(id="dashboard-content", className="cards-grid",
                 style={"maxHeight": "68vh", "overflowY": "auto", "paddingRight": "4px"}),

    ], id="page-wrapper"),

], style={"background": "#f0f4f8", "minHeight": "100vh"})


# ── Stats Bar Callback ────────────────────────────────────────────────────────

@app.callback(
    Output("stats-bar", "children"),
    Input("interval-update", "n_intervals"),
)
def update_stats(_):
    df = load_data()
    if df.empty:
        total, high_sev, total_cost, healthy_pct = 0, 0, 0.0, 0
    else:
        total      = len(df)
        high_sev   = int((df["Severity"] >= 30).sum())
        total_cost = float(df["Estimated_Cost"].sum())
        healthy    = int((df["Severity"] < 15).sum())
        healthy_pct = round(healthy / total * 100) if total else 0

    def stat(label, value, cls, icon):
        return html.Div([
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(icon,  className="stat-icon"),
        ], className=f"stat-card {cls}")

    return [
        stat("Total Processed",   str(total),          "blue",     "📸"),
        stat("High Severity",     str(high_sev),        "critical", "🔴"),
        stat("Est. Total Cost",   f"₹{total_cost:,.0f}", "warning",  "💰"),
        stat("Healthy %",         f"{healthy_pct}%",    "healthy",  "✅"),
    ]


# ── Main Feed Callback ────────────────────────────────────────────────────────

@app.callback(
    Output("dashboard-content", "children"),
    Output("current-index",     "data"),
    Output("shown-cards",       "data"),
    Output("record-counter",    "children"),
    Input("interval-update",    "n_intervals"),
    Input("filter-option",      "value"),
    State("current-index",      "data"),
    State("shown-cards",        "data"),
)
def update_dashboard(n, filter_value, current_index, shown_cards_data):
    df = load_data()

    if df.empty:
        empty = html.Div([
            html.Div("🔍", className="empty-icon"),
            html.Div("No prediction data available. Run predict.py first.", className="empty-text"),
        ], className="empty-state")
        return [empty], current_index, shown_cards_data, ""

    # All images shown — show final view
    if current_index >= len(df):
        done_banner = html.Div([
            html.Span("✅"),
            html.Span(f"All {len(df)} records have been processed."),
        ], className="done-banner")

        visible_cards = [render_card(c) for c in shown_cards_data[::-1]
                         if filter_value == "all" or float(c["Severity"]) >= 30]

        counter = f"{len(visible_cards)} / {len(df)} shown"
        return [done_banner] + visible_cards, current_index, shown_cards_data, counter

    # Load next row
    row = df.iloc[current_index]
    img_src = encode_image(row["Image_Name"])

    # Log high-severity
    try:
        sev = float(row["Severity"])
        if sev >= 30:
            if not os.path.exists(HIGH_SEV_LOG_PATH):
                row.to_frame().T.to_csv(HIGH_SEV_LOG_PATH, index=False)
            else:
                logged = pd.read_csv(HIGH_SEV_LOG_PATH)
                if row["Image_Name"] not in logged["Image_Name"].values:
                    row.to_frame().T.to_csv(HIGH_SEV_LOG_PATH, mode="a", header=False, index=False)
    except Exception:
        pass

    new_card = {
        "Image_Name":            row["Image_Name"],
        "Fault_Type":            row["Fault_Type"],
        "Severity":              row["Severity"],
        "ActionTaken":           row["ActionTaken"],
        "Estimated_Cost":        row["Estimated_Cost"],
        "Estimated_Downtime_Days": row["Estimated_Downtime_Days"],
        "Reason":                row.get("Reason", ""),
        "Recommendation":        row.get("Recommendation", ""),
        "Next_Step":             row.get("Next_Step", ""),
        "Image_Encoded":         img_src,
    }
    shown_cards_data.append(new_card)

    filtered = [render_card(c) for c in shown_cards_data[::-1]
                if filter_value == "all" or float(c["Severity"]) >= 30]

    counter = f"{current_index + 1} / {len(df)} loading..."
    return filtered, current_index + 1, shown_cards_data, counter


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
