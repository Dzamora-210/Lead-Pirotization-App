import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================


def load_data():
    if os.path.exists("lead_cleaning.pkl"):
        df = pd.read_pickle("lead_cleaning.pkl")
    else:
        # Synthetic dataset matching full schema for error-free execution
        np.random.seed(42)
        companies = [
            "Northstar Analytics",
            "Evergreen Pharma",
            "Atlas Media Co",
            "Oakline Telecom",
            "Clearwater Foods",
            "Summit Retail Group",
            "Greenfield Energy",
            "Redwood Consulting",
            "Westbridge Legal",
            "Pioneer Logistics",
            "Lighthouse Biotech",
            "Ironwood Construction",
            "Silverstone Automotive",
            "Harbor Insurance",
            "Vertex Software",
            "Acme Health Systems",
            "Brightpath Education",
            "Cedar Manufacturing",
            "Blue Ridge Financial",
            "Metro Hospitality",
        ]
        statuses = ["New", "Contacted", "Qualified", "Nurturing", "Converted", "Lost"]
        campaigns = [
            "Abm - Enterprise",
            "Content Download",
            "Fall Webinar",
            "Free Trial",
            "Google Search",
            "Healthcare Outreach",
            "Linkedin Prospecting",
            "Partner Referral",
            "Product Demo",
            "Q3 Demand Gen",
        ]
        reps = [
            "Alex Kim",
            "Chris Brown",
            "Jordan Lee",
            "Morgan Patel",
            "Taylor Smith",
            "Unassigned",
        ]
        job_titles = [
            "CEO",
            "CMO",
            "VP of Sales",
            "Marketing Director",
            "Senior Manager",
            "Product Lead",
            "Manager",
        ]
        activities = [
            "Demo Request",
            "Contact Form",
            "Phone Call",
            "Form Fill",
            "Content Download",
            "Email Click",
            "Website Visit",
        ]
        industries = [
            "Technology",
            "Healthcare",
            "Finance",
            "Retail",
            "Manufacturing",
            "Logistics",
        ]
        notes_list = [
            "Demo scheduled",
            "Follow up requested",
            "Interested in pricing",
            "Downloaded whitepaper",
            "No response yet",
        ]
        sources = ["Organic", "Paid Search", "Social", "Referral", "Email"]

        records = []
        for i in range(123):
            comp = np.random.choice(companies)
            records.append(
                {
                    "lead_id": f"L-10{i:03d}",
                    "company_name": comp,
                    "lead_status": np.random.choice(
                        statuses, p=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1]
                    ),
                    "campaign": np.random.choice(campaigns),
                    "sales_rep": np.random.choice(reps),
                    "is_assigned": "Assigned"
                    if np.random.choice([True, False])
                    else "Unassigned",
                    "created_date": pd.to_datetime("2024-01-01")
                    + pd.to_timedelta(np.random.randint(0, 180), unit="D"),
                    "first_contact_date": pd.to_datetime("2024-01-01")
                    + pd.to_timedelta(np.random.randint(0, 180), unit="D")
                    if np.random.rand() > 0.3
                    else pd.NaT,
                    "estimated_deal_value_usd": np.random.choice(
                        [8000, 15000, 35000, 55000, 80000, 120000]
                    ),
                    "lead_full_name": f"Lead_{i}",
                    "job_title": np.random.choice(job_titles),
                    "last_activity": np.random.choice(activities),
                    "lead_score": np.random.randint(30, 100),
                    "industry": np.random.choice(industries),
                    "phone": f"+1-555-01{i:02d}",
                    "email": f"user{i}@example.com",
                    "lead_source": np.random.choice(sources),
                    "round_robin_assignment": np.random.choice(
                        ["Round Robin A", "Round Robin B", "Manual"]
                    ),
                    "notes": np.random.choice(notes_list),
                }
            )
        df = pd.DataFrame(records)

    # Ensure datetimes
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["first_contact_date"] = pd.to_datetime(df["first_contact_date"])
    return df


df_raw = load_data()

# ==========================================
# HELPER FUNCTIONS & LEAD SCORING LOGIC
# ==========================================


def calculate_lead_score(
    last_activity, lead_score, job_title, lead_status, notes, deal_value
):
    score = 0

    activity = str(last_activity).lower()
    if "demo" in activity:
        score += 20
    elif "contact" in activity:
        score += 18
    elif "phone" in activity:
        score += 15
    elif "form" in activity:
        score += 12
    elif "content" in activity:
        score += 10
    elif "email" in activity:
        score += 8
    elif "website" in activity or "web" in activity:
        score += 5

    if lead_score >= 90:
        score += 20
    elif lead_score >= 80:
        score += 15
    elif lead_score >= 70:
        score += 10
    elif lead_score >= 60:
        score += 8
    elif lead_score >= 50:
        score += 5
    elif lead_score >= 40:
        score += 2

    title = str(job_title).lower()
    if any(
        x in title for x in ["ceo", "cmo", "cto", "cio", "chief", "founder", "owner"]
    ):
        score += 20
    elif any(x in title for x in ["vp", "vice president"]):
        score += 18
    elif any(x in title for x in ["director", "head"]):
        score += 15
    elif any(x in title for x in ["senior manager", "senior"]):
        score += 12
    elif "manager" in title:
        score += 10
    elif any(x in title for x in ["lead", "principal"]):
        score += 8

    status = str(lead_status).lower()
    if any(x in status for x in ["lost", "unqualified", "converted"]):
        score += 0
    elif any(x in status for x in ["qualified", "new"]):
        score += 15
    elif any(x in status for x in ["contacted"]):
        score += 10
    elif any(x in status for x in ["nurturing"]):
        score += 5

    notes_text = str(notes).lower()
    if any(x in notes_text for x in ["demo", "pricing"]):
        score += 15
    elif any(x in notes_text for x in ["follow", "referral"]):
        score += 10
    elif any(x in notes_text for x in ["interested", "downloaded"]):
        score += 5
    elif any(x in notes_text for x in ["wrong", "no response", "duplicate"]):
        score += 0

    if deal_value >= 100000:
        score += 15
    elif deal_value >= 75000:
        score += 12
    elif deal_value >= 50000:
        score += 10
    elif deal_value >= 25000:
        score += 7
    elif deal_value >= 10000:
        score += 5

    return score


def process_lead_prioritization(df):
    df_calc = df.copy()

    df_calc["lead_priority_score"] = df_calc.apply(
        lambda row: calculate_lead_score(
            row.get("last_activity", ""),
            row.get("lead_score", 0),
            row.get("job_title", ""),
            row.get("lead_status", ""),
            row.get("notes", ""),
            row.get("estimated_deal_value_usd", 0),
        ),
        axis=1,
    )

    df_calc["score_percentile"] = df_calc["lead_priority_score"].rank(
        pct=True, method="average"
    )

    df_calc["priority"] = pd.cut(
        df_calc["score_percentile"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        include_lowest=True,
    ).astype(int)

    columns = [
        "priority",
        "lead_full_name",
        "job_title",
        "company_name",
        "industry",
        "phone",
        "email",
        "lead_status",
        "lead_source",
        "last_activity",
        "estimated_deal_value_usd",
        "lead_score",
        "round_robin_assignment",
        "notes",
    ]

    existing_cols = [c for c in columns if c in df_calc.columns]

    priority_view = (
        df_calc[existing_cols]
        .loc[df_calc["lead_status"] != "Lost"]
        .sort_values("priority", ascending=True)
    )

    priority_view.columns = (
        priority_view.columns.str.replace("_", " ", regex=False)
        .str.title()
        .str.replace("id", "ID", regex=False)
        .str.replace("Usd", "USD", regex=False)
    )

    return priority_view


def get_most_recent_funnel_status(
    df, group_col="company_name", date_col="created_date", status_col="lead_status"
):
    recent_df = df.sort_values(by=date_col).drop_duplicates(
        subset=[group_col], keep="last"
    )
    return recent_df[[group_col, status_col]].reset_index(drop=True)


def process_campaign_performance(df):
    table_view_df = (
        df.groupby("campaign")
        .agg(
            total_leads=("lead_id", "count"),
            contacted=("lead_status", lambda x: (x == "Contacted").sum()),
            sql=("lead_status", lambda x: (x == "Qualified").sum()),
            closed_won=("lead_status", lambda x: (x == "Converted").sum()),
            new=("lead_status", lambda x: (x == "New").sum()),
        )
        .reset_index()
    )

    total_leads_sum = table_view_df["total_leads"].sum()
    total_contacted_sum = table_view_df["contacted"].sum()
    total_sql_sum = table_view_df["sql"].sum()
    total_closed_won_sum = table_view_df["closed_won"].sum()
    total_new_sum = table_view_df["new"].sum()

    rate_cols = [
        ("contact_rate", "contacted", "total_leads"),
        ("sql_rate", "sql", "total_leads"),
        ("win_rate", "closed_won", "total_leads"),
    ]

    for new_col, num, denom in rate_cols:
        table_view_df[new_col] = np.where(
            table_view_df[denom] > 0,
            ((table_view_df[num] / table_view_df[denom]) * 100).round(2).astype(str)
            + "%",
            "0.0%",
        )

    if total_leads_sum > 0:
        total_row = pd.DataFrame(
            [
                {
                    "campaign": "Total",
                    "total_leads": total_leads_sum,
                    "new": total_new_sum,
                    "contacted": total_contacted_sum,
                    "sql": total_sql_sum,
                    "closed_won": total_closed_won_sum,
                    "contact_rate": f"{((total_contacted_sum / total_leads_sum) * 100):.2f}%",
                    "sql_rate": f"{((total_sql_sum / total_leads_sum) * 100):.2f}%",
                    "win_rate": f"{((total_closed_won_sum / total_leads_sum) * 100):.2f}%",
                }
            ]
        )
        table_view_df = pd.concat([table_view_df, total_row], ignore_index=True)

    return table_view_df


SHARED_DARK_TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "borderRadius": "6px"},
    "style_cell": {
        "textAlign": "left",
        "padding": "12px 16px",
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "fontSize": "13px",
        "color": "#f1f5f9",
        "backgroundColor": "#1e293b",
        "border": "1px solid #334155",
    },
    "style_header": {
        "backgroundColor": "#0f172a",
        "fontWeight": "600",
        "color": "#38bdf8",
        "border": "1px solid #334155",
        "fontSize": "12px",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
    },
    "style_data": {"backgroundColor": "#1e293b", "color": "#f1f5f9"},
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "#182232"},
        {
            "if": {"state": "active"},
            "backgroundColor": "#334155",
            "border": "1px solid #64748b",
        },
    ],
}

# ==========================================
# 2. DASH APPLICATION INITIALIZATION
# ==========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Lead Intelligence Dashboard"

campaign_options = [
    {"label": c, "value": c} for c in sorted(df_raw["campaign"].dropna().unique())
]
status_options = [
    {"label": s, "value": s} for s in sorted(df_raw["lead_status"].dropna().unique())
]
rep_options = [
    {"label": r, "value": r} for r in sorted(df_raw["sales_rep"].dropna().unique())
]
company_options = [
    {"label": c, "value": c} for c in sorted(df_raw["company_name"].dropna().unique())
]
industry_options = (
    [{"label": i, "value": i} for i in sorted(df_raw["industry"].dropna().unique())]
    if "industry" in df_raw.columns
    else []
)

pinned_filters = dbc.Card(
    dbc.CardBody(
        [
            html.H6(
                "Global Dashboard Filters",
                className="card-title text-uppercase font-weight-bold mb-3",
                style={
                    "letterSpacing": "0.5px",
                    "fontSize": "12px",
                    "color": "#38bdf8",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Company", className="fw-bold fs-7 mb-1 text-light"
                            ),
                            dcc.Dropdown(
                                id="filter-company",
                                options=company_options,
                                multi=True,
                                placeholder="All Companies",
                                className="dash-dropdown-dark",
                            ),
                        ],
                        lg=2,
                        md=4,
                        sm=6,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Industry", className="fw-bold fs-7 mb-1 text-light"
                            ),
                            dcc.Dropdown(
                                id="filter-industry",
                                options=industry_options,
                                multi=True,
                                placeholder="All Industries",
                                className="dash-dropdown-dark",
                            ),
                        ],
                        lg=2,
                        md=4,
                        sm=6,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Campaign", className="fw-bold fs-7 mb-1 text-light"
                            ),
                            dcc.Dropdown(
                                id="filter-campaign",
                                options=campaign_options,
                                multi=True,
                                placeholder="All Campaigns",
                                className="dash-dropdown-dark",
                            ),
                        ],
                        lg=3,
                        md=4,
                        sm=6,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Lead Status", className="fw-bold fs-7 mb-1 text-light"
                            ),
                            dcc.Dropdown(
                                id="filter-status",
                                options=status_options,
                                multi=True,
                                placeholder="All Statuses",
                                className="dash-dropdown-dark",
                            ),
                        ],
                        lg=2,
                        md=6,
                        sm=6,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Sales Rep", className="fw-bold fs-7 mb-1 text-light"
                            ),
                            dcc.Dropdown(
                                id="filter-rep",
                                options=rep_options,
                                multi=True,
                                placeholder="All Sales Reps",
                                className="dash-dropdown-dark",
                            ),
                        ],
                        lg=3,
                        md=6,
                        sm=12,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
        ]
    ),
    style={
        "position": "sticky",
        "top": "0",
        "zIndex": "1020",
        "backgroundColor": "#1e293b",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.5)",
        "marginBottom": "20px",
        "borderRadius": "0px 0px 8px 8px",
        "border": "1px solid #334155",
    },
)

app.layout = dbc.Container(
    [
        pinned_filters,
        # Section 1: High Level Metrics (Wider, Centered, Scaled-up Fonts, Single Line Values)
        html.Div(id="metrics-container", className="mb-4"),
        # Section 2: Lead Prioritization Table
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(
                        "Lead Prioritization",
                        className="mb-0",
                        style={"fontWeight": "600", "color": "#38bdf8"},
                    )
                ),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id="lead-prioritization-table",
                            page_size=10,
                            sort_action="native",
                            **SHARED_DARK_TABLE_STYLE,
                        )
                    ]
                ),
            ],
            className="mb-4",
            style={"backgroundColor": "#1e293b", "border": "1px solid #334155"},
        ),
        # Section 3: Campaign Performance Table
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(
                        "Campaign Performance Breakdown",
                        className="mb-0",
                        style={"fontWeight": "600", "color": "#38bdf8"},
                    )
                ),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id="campaign-performance-table",
                            columns=[
                                {"name": "Campaign", "id": "campaign"},
                                {"name": "Total Leads", "id": "total_leads"},
                                {"name": "New", "id": "new"},
                                {"name": "Contacted", "id": "contacted"},
                                {"name": "SQL", "id": "sql"},
                                {"name": "Closed Won", "id": "closed_won"},
                                {"name": "Contact Rate", "id": "contact_rate"},
                                {"name": "SQL Rate", "id": "sql_rate"},
                                {"name": "Win Rate", "id": "win_rate"},
                            ],
                            page_size=100,
                            sort_action="native",
                            style_table=SHARED_DARK_TABLE_STYLE["style_table"],
                            style_cell=SHARED_DARK_TABLE_STYLE["style_cell"],
                            style_header=SHARED_DARK_TABLE_STYLE["style_header"],
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#182232",
                                },
                                {
                                    "if": {"filter_query": '{campaign} = "Total"'},
                                    "fontWeight": "bold",
                                    "backgroundColor": "#0f172a",
                                    "color": "#38bdf8",
                                },
                            ],
                        )
                    ]
                ),
            ],
            className="mb-4",
            style={"backgroundColor": "#1e293b", "border": "1px solid #334155"},
        ),
        # Section 4: Company Activity Matrix Scatter Plot
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(
                        "Company Activity Matrix by Funnel Status",
                        className="mb-0",
                        style={"fontWeight": "600", "color": "#38bdf8"},
                    )
                ),
                dbc.CardBody([dcc.Graph(id="company-activity-matrix")]),
            ],
            className="mb-4",
            style={"backgroundColor": "#1e293b", "border": "1px solid #334155"},
        ),
        # Section 5: Customer Journey Funnel Analysis
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(
                        "Customer Journey Funnel Analysis",
                        className="mb-0",
                        style={"fontWeight": "600", "color": "#38bdf8"},
                    )
                ),
                dbc.CardBody([dcc.Graph(id="sankey-funnel-analysis")]),
            ],
            className="mb-4",
            style={"backgroundColor": "#1e293b", "border": "1px solid #334155"},
        ),
    ],
    fluid=True,
    className="px-4 py-2",
    style={"backgroundColor": "#0f172a", "minHeight": "100vh"},
)

# ==========================================
# 3. INTERACTIVE CALLBACKS
# ==========================================


@app.callback(
    [
        Output("metrics-container", "children"),
        Output("lead-prioritization-table", "columns"),
        Output("lead-prioritization-table", "data"),
        Output("campaign-performance-table", "data"),
        Output("company-activity-matrix", "figure"),
        Output("sankey-funnel-analysis", "figure"),
    ],
    [
        Input("filter-company", "value"),
        Input("filter-industry", "value"),
        Input("filter-campaign", "value"),
        Input("filter-status", "value"),
        Input("filter-rep", "value"),
    ],
)
def update_dashboard(
    selected_companies,
    selected_industries,
    selected_campaigns,
    selected_statuses,
    selected_reps,
):
    filtered_df = df_raw.copy()

    if selected_companies:
        filtered_df = filtered_df[filtered_df["company_name"].isin(selected_companies)]
    if selected_industries and "industry" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["industry"].isin(selected_industries)]
    if selected_campaigns:
        filtered_df = filtered_df[filtered_df["campaign"].isin(selected_campaigns)]
    if selected_statuses:
        filtered_df = filtered_df[filtered_df["lead_status"].isin(selected_statuses)]
    if selected_reps:
        filtered_df = filtered_df[filtered_df["sales_rep"].isin(selected_reps)]

    # 1. High Level Metrics (Full-width 5-card layout, scaled fonts, strictly single-line text)
    total_leads = len(filtered_df)
    contacted_count = (filtered_df["lead_status"] == "Contacted").sum()
    qualified_count = (filtered_df["lead_status"] == "Qualified").sum()
    converted_count = (filtered_df["lead_status"] == "Converted").sum()
    total_pipeline = (
        filtered_df["estimated_deal_value_usd"].sum()
        if "estimated_deal_value_usd" in filtered_df.columns
        else 0
    )

    contact_rate = (
        f"{(contacted_count / total_leads * 100):.1f}%" if total_leads > 0 else "0.0%"
    )
    sql_rate = (
        f"{(qualified_count / total_leads * 100):.1f}%" if total_leads > 0 else "0.0%"
    )
    win_rate = (
        f"{(converted_count / total_leads * 100):.1f}%" if total_leads > 0 else "0.0%"
    )

    card_col_style = {
        "flex": "0 0 20%",
        "maxWidth": "20%",
        "paddingLeft": "6px",
        "paddingRight": "6px",
    }

    title_style = {
        "fontSize": "13px",
        "letterSpacing": "0.5px",
        "opacity": "0.85",
        "whiteSpace": "nowrap",
        "fontWeight": "600",
    }

    value_style = {
        "fontSize": "clamp(1.5rem, 2.3vw, 2.2rem)",
        "fontWeight": "700",
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    card_data = [
        ("Total Leads", f"{total_leads:,}", "#38bdf8"),
        ("Contact Rate", contact_rate, "#818cf8"),
        ("SQL Rate", sql_rate, "#fbbf24"),
        ("Win Rate", win_rate, "#34d399"),
        ("Est. Pipeline Value", f"${total_pipeline:,.0f}", "#f43f5e"),
    ]

    metric_cols = []
    for title, val, color in card_data:
        metric_cols.append(
            html.Div(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        title,
                                        className="text-light text-uppercase mb-2",
                                        style=title_style,
                                    ),
                                    html.H3(
                                        val,
                                        className="mb-0",
                                        style={"color": color, **value_style},
                                    ),
                                ],
                                className="py-3 px-2",
                            )
                        ],
                        className="h-100 text-center shadow",
                        style={
                            "backgroundColor": "#1e293b",
                            "border": "1px solid #334155",
                        },
                    )
                ],
                style=card_col_style,
                className="mb-2",
            )
        )

    metrics_cards = dbc.Row(
        metric_cols, className="g-0 justify-content-center align-items-stretch"
    )

    # 2. Lead Prioritization Data
    if not filtered_df.empty:
        priority_view_df = process_lead_prioritization(filtered_df)
        lead_cols = [{"name": c, "id": c} for c in priority_view_df.columns]
        lead_data = priority_view_df.to_dict("records")
    else:
        lead_cols = []
        lead_data = []

    # 3. Campaign Performance Data
    if not filtered_df.empty:
        campaign_df = process_campaign_performance(filtered_df)
        campaign_data = campaign_df.to_dict("records")
    else:
        campaign_data = []

    # 4. Interactive Company Activity Matrix Scatter Plot
    if not filtered_df.empty:
        latest_status_df = get_most_recent_funnel_status(filtered_df)

        viz_df = (
            filtered_df.groupby("company_name")
            .agg(
                marketing_activity=("created_date", "count"),
                sales_activity=("first_contact_date", "count"),
            )
            .reset_index()
        )

        joined_df = pd.merge(viz_df, latest_status_df, on="company_name", how="inner")
        numeric_columns = joined_df.select_dtypes(include=["number"]).columns

        if len(numeric_columns) > 0:
            scaler = MinMaxScaler()
            joined_df[numeric_columns] = scaler.fit_transform(
                joined_df[numeric_columns]
            )

        fig_matrix = px.scatter(
            joined_df,
            x="marketing_activity",
            y="sales_activity",
            color="lead_status",
            symbol="lead_status",
            hover_name="company_name",
            title="Company Activity Matrix by Funnel Status",
            labels={
                "marketing_activity": "Normalized Marketing Activity",
                "sales_activity": "Normalized Sales Activity",
                "lead_status": "Last Funnel Status",
            },
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )

        fig_matrix.update_traces(
            marker=dict(size=12, opacity=0.9, line=dict(width=1, color="#0f172a")),
            hovertemplate="<b>%{hovertext}</b><br><br>"
            + "Marketing Activity: %{x:.2f}<br>"
            + "Sales Activity: %{y:.2f}<br>"
            + "Funnel Status: %{customdata[0]}<extra></extra>",
            customdata=joined_df[["lead_status"]],
        )

        fig_matrix.update_layout(
            template="plotly_dark",
            font=dict(family="'Inter', sans-serif", size=13, color="#f1f5f9"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor="#334155",
                zeroline=True,
                zerolinecolor="#475569",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#334155",
                zeroline=True,
                zerolinecolor="#475569",
            ),
            legend=dict(
                title=dict(font=dict(weight="bold")),
                bordercolor="#334155",
                borderwidth=1,
                bgcolor="#1e293b",
            ),
        )
    else:
        fig_matrix = go.Figure()
        fig_matrix.update_layout(
            template="plotly_dark",
            title="No data available for selected filters",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )

    # 5. Customer Journey Funnel Analysis (Sankey Diagram)
    if not filtered_df.empty:
        all_labels = (
            list(filtered_df["campaign"].unique())
            + list(filtered_df["is_assigned"].unique())
            + list(filtered_df["lead_status"].unique())
        )
        label_indices = {label: idx for idx, label in enumerate(all_labels)}

        flow_1 = (
            filtered_df.groupby(["campaign", "is_assigned"])
            .size()
            .reset_index(name="value")
        )
        flow_1.columns = ["source_label", "target_label", "value"]

        flow_2 = (
            filtered_df.groupby(["is_assigned", "lead_status"])
            .size()
            .reset_index(name="value")
        )
        flow_2.columns = ["source_label", "target_label", "value"]

        combined_flows = pd.concat([flow_1, flow_2], ignore_index=True)

        combined_flows["source"] = combined_flows["source_label"].map(label_indices)
        combined_flows["target"] = combined_flows["target_label"].map(label_indices)

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=18,
                        thickness=20,
                        line=dict(color="#0f172a", width=0.5),
                        label=all_labels,
                        color="#0ea5e9",
                    ),
                    link=dict(
                        source=combined_flows["source"],
                        target=combined_flows["target"],
                        value=combined_flows["value"],
                        color="rgba(56, 189, 248, 0.25)",
                        hovertemplate="Flow: %{source.label} ➔ %{target.label}<br>Volume: %{value}<extra></extra>",
                    ),
                )
            ]
        )

        fig_sankey.update_layout(
            template="plotly_dark",
            title_text="Customer Journey Funnel Analysis",
            font=dict(family="'Inter', sans-serif", size=12, color="#f1f5f9"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            margin=dict(l=40, r=40, t=50, b=40),
        )
    else:
        fig_sankey = go.Figure()
        fig_sankey.update_layout(
            template="plotly_dark",
            title="No data available for selected filters",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )

    return metrics_cards, lead_cols, lead_data, campaign_data, fig_matrix, fig_sankey


# ==========================================
# 4. RUN SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
