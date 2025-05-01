import json
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os

# -----------------------------
# Load dataset
# -----------------------------
with open('dataset.json') as f:
    data = json.load(f)
years = sorted(data.keys(), key=lambda y: int(y))
start_year, end_year = years[0], years[-1]

# Prepare time-series DataFrames
aum_df = pd.DataFrame([
    {'Year': y, 'AUM': data[y]['fund_overview']['total_fund_value_million']}
    for y in years
])

# Calculate AUM Growth and Net Flows
aum_df['AUM_Growth'] = aum_df['AUM'].pct_change() * 100
aum_df['Net_Flows'] = aum_df['AUM'].diff()

returns_df = pd.DataFrame([
    {
        'Year': y,
        'Fund_1Y': data[y]['returns'].get('1_year', 
            data[y]['returns'].get('6_month', np.nan)),  # Use 6-month if 1-year not available
        'Bench_1Y': data[y]['benchmark_returns'].get('1_year', 
            data[y]['benchmark_returns'].get('6_month', np.nan)),  # Use 6-month if 1-year not available
        'Fund_3Y': data[y]['returns'].get('3_year_annualized', np.nan),
        'Bench_3Y': data[y]['benchmark_returns'].get('3_year_annualized', np.nan),
        'Fund_Since_Inception': data[y]['returns'].get('since_inception_annualized', np.nan)
    }
    for y in years
])

perf_df = pd.DataFrame([
    {
        'Year': y,
        'Alpha': data[y]['performance_vs_benchmark'].get('alpha', np.nan),
        'Value_Fund_Return': data[y]['returns'].get('value_fund_return', np.nan),
        'Small_Cap_Return': data[y]['returns'].get('small_cap_fund_return', np.nan),
        'Growth_Plus_Income_Return': data[y]['returns'].get('growth_plus_income_fund_return', np.nan)
    }
    for y in years
])

aoa = pd.DataFrame([
    {'Year': y, **data[y]['asset_allocation']}
    for y in years
]).set_index('Year')

# Get all unique sectors across all years
all_sectors = set()
for y in years:
    all_sectors.update(data[y]['sector_allocation'].keys())

# Create sector allocation DataFrame with all sectors
soa_data = []
for y in years:
    sector_data = {'Year': y}
    for sector in all_sectors:
        sector_data[sector] = data[y]['sector_allocation'].get(sector, 0)
    soa_data.append(sector_data)

soa = pd.DataFrame(soa_data).set_index('Year')

div_df = pd.DataFrame([
    {
        'Year': y,
        'Dividend_Paid': data[y]['dividend_metrics'].get('dividend_paid_thousands', 0) / 1000,  # Convert to millions
        'Cumulative_Dividends': data[y]['dividend_metrics'].get('cumulative_dividends_million', 0),
        'Dividends_to_OU': data[y]['dividend_metrics'].get('dividends_paid_to_ou_million', 0)
    }
    for y in years
])

# Calculate Dividend Yield
div_df['Dividend_Yield'] = (div_df['Dividend_Paid'] / aum_df['AUM']) * 100


# Safe construction of the ESG DataFrame
esg_records = []
for y in years:
    metrics = data[y].get('esg_metrics', {})
    fund_perf = metrics.get('esg_fund_performance', {})
    esg_focus = metrics.get('esg_focus') or 'NA'
    esg_records.append({
        'Year': y,
        'ESG_6M_Performance': fund_perf.get('6_month', np.nan),
        'ESG_Focus': esg_focus
    })
esg_df = pd.DataFrame(esg_records)


# Calculate Beta and Tracking Error with proper handling of missing values
returns_df['Fund_Returns'] = returns_df['Fund_1Y'].fillna(method='ffill')
returns_df['Bench_Returns'] = returns_df['Bench_1Y'].fillna(method='ffill')

# Calculate rolling covariance and variance
returns_df['Covariance'] = returns_df['Fund_Returns'].rolling(window=3, min_periods=1).cov(returns_df['Bench_Returns'])
returns_df['Bench_Variance'] = returns_df['Bench_Returns'].rolling(window=3, min_periods=1).var()

# Calculate Beta and Tracking Error
perf_df['Beta'] = returns_df['Covariance'] / returns_df['Bench_Variance']
perf_df['Tracking_Error'] = np.sqrt(np.square(returns_df['Fund_Returns'] - returns_df['Bench_Returns']).rolling(window=3, min_periods=1).mean())

style_counts = pd.Series(
    np.concatenate([data[y]['fund_overview']['investment_style'] for y in years])
).value_counts()

# Projection helper
def compute_projection(series, extend=2):
    x = np.arange(len(series))
    m, b = np.polyfit(x, series, 1)
    xs = np.arange(len(series) + extend)
    ys = m * xs + b
    yrs = [str(int(start_year) + i) for i in xs]
    return yrs, ys

proj_years, proj_vals = compute_projection(aum_df['AUM'])

# Insights computation
def compute_insights():
    n = 5
    if len(aum_df) >= n + 1:
        cagr = (aum_df['AUM'].iloc[-1] / aum_df['AUM'].iloc[-n-1]) ** (1/n) - 1
    else:
        cagr = np.nan
    peak_year = aum_df.loc[aum_df['AUM'].idxmax(), 'Year']
    peak_val = aum_df['AUM'].max()
    cum_max = aum_df['AUM'].cummax()
    drawdowns = (cum_max - aum_df['AUM']) / cum_max
    max_dd = drawdowns.max()
    # Calculate latest net flows using the most recent AUM change
    latest_flows = aum_df['Net_Flows'].iloc[-1] if not pd.isna(aum_df['Net_Flows'].iloc[-1]) else 0
    return cagr, peak_year, peak_val, max_dd, latest_flows

cagr, peak_year, peak_val, max_dd, latest_flows = compute_insights()

# Initialize Dash app with custom theme
app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.LUX,
        'https://www.stern.nyu.edu/themes/custom/nyustern_theme/dist/css/style.min.css'
    ],
    suppress_callback_exceptions=True  # Add this to handle dynamic components
)
server = app.server
app.title = 'NYU Stern MPSIF Dashboard'

# Custom colors
NYU_PURPLE = '#6B378E'
NYU_LIGHT_GRAY = '#EEEEEE'
NYU_VIOLET = '#8B4DB8'
NYU_LIGHT_PURPLE = '#A66DD4'
NYU_GRAY = '#343a40'

# Custom styles
CARD_STYLE = {
    'borderRadius': '8px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'marginBottom': '20px',
    'transition': 'transform 0.2s',
    'backgroundColor': 'white',
    'border': f'1px solid {NYU_LIGHT_GRAY}'
}

CARD_HOVER_STYLE = {
    'transform': 'translateY(-5px)',
    'boxShadow': '0 6px 12px rgba(0, 0, 0, 0.15)'
}

# Navbar with improved styling
navbar = dbc.NavbarSimple(
    brand=html.Span(
        'NYU Stern MPSIF Dashboard',
        style={
            'color': NYU_PURPLE,
            'fontSize': '1.5rem',
            'fontWeight': 'bold',
            'fontFamily': '"Helvetica Neue", Helvetica, Arial, sans-serif'
        }
    ),
    color='white',
    dark=False,
    className='mb-4',
    style={
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'borderBottom': f'3px solid {NYU_PURPLE}',
        'padding': '1rem'
    }
)

# Tabs with improved styling
tabs = dbc.Tabs([
    dbc.Tab(label='Overview', tab_id='overview', className='px-3', 
            label_style={'color': NYU_PURPLE, 'fontWeight': 'bold'}),
    dbc.Tab(label='Performance', tab_id='performance', className='px-3',
            label_style={'color': NYU_PURPLE, 'fontWeight': 'bold'}),
    dbc.Tab(label='Trends', tab_id='trends', className='px-3',
            label_style={'color': NYU_PURPLE, 'fontWeight': 'bold'}),
    dbc.Tab(label='Projections & Insights', tab_id='projections', className='px-3',
            label_style={'color': NYU_PURPLE, 'fontWeight': 'bold'})
], id='tabs', active_tab='overview', className='mb-4')

# Layout with improved styling
dashboard_layout = dbc.Container([
    navbar,
    tabs,
    dcc.Loading(
        id='loading-tab-content',
        type='default',          # or 'circle', 'dot', etc.
        children=html.Div(
            id='tab-content',
            className='p-4'
        )
    )
], fluid=True, style={'backgroundColor': NYU_LIGHT_GRAY})

app.layout = dashboard_layout

# Callbacks for tab content
def overview_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Select Year', className='bg-primary text-white'),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='year-dropdown',
                            options=[{'label': y, 'value': y} for y in years],
                            value=end_year,
                            clearable=False,
                            style={'marginBottom': '20px'}  # Replace className with style
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=3)
        ]),
        html.Div(id='overview-content')
    ], fluid=True)

def performance_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('1Y Returns: Fund vs Benchmark', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(returns_df, x='Year', y=['Fund_1Y', 'Bench_1Y'],
                                         title='', markers=True)
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Return (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('3Y Annualized Returns', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(returns_df, x='Year', y=['Fund_3Y', 'Bench_3Y'],
                                         title='', markers=True)
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Return (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Alpha Over Time', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(perf_df, x='Year', y='Alpha',
                                         title='', markers=True)
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=False,
                                yaxis_title='Alpha (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                line=dict(width=3, color=NYU_PURPLE),
                                marker=dict(size=8, color=NYU_PURPLE),
                                hovertemplate='<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Total Fund Returns', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(returns_df, x='Year', y=['Fund_1Y', 'Fund_3Y', 'Fund_Since_Inception'],
                                         title='', markers=True)
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Return (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6)
        ])
    ], fluid=True)

def trends_tab():
    # Prepare data for visualization
    aoa_melted = aoa.reset_index().melt(id_vars=['Year'], var_name='Asset', value_name='Allocation')
    soa_melted = soa.reset_index().melt(id_vars=['Year'], var_name='Sector', value_name='Allocation')
    
    # Filter out zero allocations for better visualization
    soa_melted = soa_melted[soa_melted['Allocation'] > 0]
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Asset Allocation Over Time', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.area(aoa_melted, x='Year', y='Allocation', color='Asset',
                                         title='')
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Allocation (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Sector Allocation Over Time', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.area(soa_melted, x='Year', y='Allocation', color='Sector',
                                         title='')
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Allocation (%)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Dividend Metrics', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(div_df, x='Year', y=['Dividend_Paid', 'Cumulative_Dividends', 'Dividends_to_OU'],
                                         title='', markers=True)
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                yaxis_title='Amount (Million $)',
                                transition_duration=500,
                                transition_easing='cubic-in-out'
                            )
                            .update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}M<extra></extra>'
                            )
                        )
                    ])
                ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
            ], md=6),
            # trends_tab() — replace the ESG “vs Benchmark” card with this:
dbc.Col([
    dbc.Card([
        dbc.CardHeader('ESG Fund 6-Month Performance', 
                       style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
        dbc.CardBody([
            dcc.Graph(
                figure=px.line(
                    esg_df,
                    x='Year',
                    y='ESG_6M_Performance',
                    title='ESG Fund 6-Month Performance',
                    markers=True
                )
                .update_layout(
                    yaxis_title='6M Return (%)',
                    xaxis_title='Year',
                    showlegend=False
                )
                .update_traces(
                    line=dict(width=3, color=NYU_PURPLE),
                    marker=dict(size=8, color=NYU_PURPLE),
                    hovertemplate='<b>%{x}</b><br>6M Return: %{y:.2f}%<extra></extra>'
                )
            )
        ])
    ], style={**CARD_STYLE, 'transition': 'all 0.3s ease-in-out'})
], md=6)
        ])
    ], fluid=True)

def projections_tab():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aum_df['Year'], 
        y=aum_df['AUM'], 
        mode='lines+markers', 
        name='Actual',
        line=dict(width=3, color=NYU_PURPLE),
        marker=dict(size=8, color=NYU_PURPLE)
    ))
    fig.add_trace(go.Scatter(
        x=proj_years, 
        y=proj_vals, 
        mode='lines', 
        name='Projected',
        line=dict(dash='dash', width=3, color=NYU_LIGHT_PURPLE)
    ))
    fig.update_layout(
        title='AUM Projection',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('5Y CAGR (%)', className='bg-primary text-white'),
            dbc.CardBody(html.H4(f"{cagr*100:.2f}%", className='text-center'))
        ], style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader('Peak AUM Year', className='bg-primary text-white'),
            dbc.CardBody(html.H4(f"{peak_year} ({peak_val:.2f}M)", className='text-center'))
        ], style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader('Max Drawdown (%)', className='bg-primary text-white'),
            dbc.CardBody(html.H4(f"{max_dd*100:.2f}%", className='text-center'))
        ], style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader('Latest Net Flows (M)', className='bg-primary text-white'),
            dbc.CardBody(html.H4(f"{latest_flows:.2f}M", className='text-center'))
        ], style=CARD_STYLE), md=3)
    ], className='mb-4')

    future_plans = dbc.Card([
        dbc.CardHeader('Future Investment Plans (2025-2026)', className='bg-primary text-white'),
        dbc.CardBody([
            html.H5("Strategic Focus Areas", className='mb-3'),
            html.Ul([
                html.Li("Expansion of ESG-focused investments to 40% of portfolio"),
                html.Li("Increased allocation to emerging markets (target: 25%)"),
                html.Li("Focus on technology and healthcare sectors (target: 35% combined)"),
                html.Li("Implementation of AI-driven investment strategies"),
                html.Li("Enhanced risk management through advanced analytics")
            ]),
            html.H5("Growth Targets", className='mt-4 mb-3'),
            html.Ul([
                html.Li("Target AUM growth: 15% annually"),
                html.Li("Reduce tracking error to below 2%"),
                html.Li("Maintain dividend yield above 2.5%"),
                html.Li("Achieve top quartile performance in ESG metrics")
            ])
        ])
    ], style=CARD_STYLE)

    return dbc.Container([
        dbc.Card([
            dbc.CardHeader('AUM Projection', className='bg-primary text-white'),
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ], style=CARD_STYLE),
        cards,
        future_plans
    ], fluid=True)

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab(tab):
    if tab == 'overview':
        return overview_tab()
    elif tab == 'performance':
        return performance_tab()
    elif tab == 'trends':
        return trends_tab()
    elif tab == 'projections':
        return projections_tab()
    else:
        return html.P('Tab not found')

@app.callback(
    Output('overview-content', 'children'),
    Input('year-dropdown', 'value')
)
def update_overview(year):
    if not year:
        return html.Div("Please select a year", style={'textAlign': 'center', 'padding': '20px'})
        
    try:
        d = data[str(year)]
        year_idx = years.index(str(year))
        
        # Get AUM metrics
        aum = aum_df.loc[aum_df['Year'] == year, 'AUM'].iloc[0]
        aum_growth = aum_df.loc[aum_df['Year'] == year, 'AUM_Growth'].iloc[0]
        net_flows = aum_df.loc[aum_df['Year'] == year, 'Net_Flows'].iloc[0]
        
        # Get dividend metrics
        div_paid = div_df.loc[div_df['Year'] == year, 'Dividend_Paid'].iloc[0]
        div_yield = div_df.loc[div_df['Year'] == year, 'Dividend_Yield'].iloc[0]
        
        # Get performance metrics
        tracking_error = perf_df.loc[perf_df['Year'] == year, 'Tracking_Error'].iloc[0]
        beta = perf_df.loc[perf_df['Year'] == year, 'Beta'].iloc[0]
        
        # Get ESG performance
        esg_perf = d.get('esg_metrics', {}).get('esg_fund_performance', {})
        year_val = esg_perf.get('1_year')
        if year_val is None or year_val == 0:
          year_val = esg_perf.get('6_month', np.nan)
          
        esg_performance = year_val
        
        # Format metrics with proper handling of NaN and 0 values
        def format_metric(value, is_percentage=False):
            if pd.isna(value) or value == 0:
                return "N/A"
            return f"{value:.2f}{'%' if is_percentage else ''}"
        
        # KPI Cards Row 1
        cards1 = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader('AUM (M)', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(aum), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Net Flows (M)', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(net_flows), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('AUM Growth %', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(aum_growth, True), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('1Y Return %', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(d['returns'].get('1_year', d['returns'].get('6_month', 0)), True), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Dividend Yield %', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(div_yield, True), className='text-center'))
            ], style=CARD_STYLE), width=2)
        ], className='mb-4 g-2')

        # KPI Cards Row 2
        cards2 = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader('Dividends Paid (M)', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(div_paid), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('ESG Performance %', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(esg_performance, True), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Alpha', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(d['performance_vs_benchmark'].get('alpha', np.nan), True), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Beta', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(beta), className='text-center'))
            ], style=CARD_STYLE), width=2),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Tracking Error', style={'backgroundColor': NYU_PURPLE, 'color': 'white'}),
                dbc.CardBody(html.H4(format_metric(tracking_error), className='text-center'))
            ], style=CARD_STYLE), width=2)
        ], className='mb-4')

        # Detailed Summary
        summary = dbc.Card([
            dbc.CardHeader('Summary', className='bg-primary text-white'),
            dbc.CardBody(html.P(d.get('summary','')))
        ], style=CARD_STYLE, className='mb-4')

        # Styles Badges
        styles = dbc.Card([
            dbc.CardHeader('Investment Styles', className='bg-primary text-white'),
            dbc.CardBody([
                html.Div([
                    dbc.Badge(s, color='primary', className='me-2 mb-2 p-2')
                    for s in d['fund_overview']['investment_style']
                ])
            ])
        ], style=CARD_STYLE, className='mb-4')

        # ESG Focus
        esg_focus = dbc.Card([
            dbc.CardHeader('ESG Focus', className='bg-primary text-white'),
            dbc.CardBody(html.P(d.get('esg_metrics', {}).get('esg_focus','')))
        ], style=CARD_STYLE, className='mb-4')

        # Allocation Pies
        alloc1 = px.pie(
            names=list(d['asset_allocation'].keys()), 
            values=list(d['asset_allocation'].values()), 
            title='Asset Allocation',
            color_discrete_sequence=[NYU_PURPLE, NYU_VIOLET, NYU_LIGHT_PURPLE]
        ).update_layout(showlegend=True)

        alloc2 = px.pie(
            names=list(d['sector_allocation'].keys()), 
            values=list(d['sector_allocation'].values()), 
            title='Sector Allocation',
            color_discrete_sequence=px.colors.qualitative.Set3
        ).update_layout(showlegend=True)

        pies = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader('Asset Allocation', className='bg-primary text-white'),
                dbc.CardBody(dcc.Graph(figure=alloc1))
            ], style=CARD_STYLE), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader('Sector Allocation', className='bg-primary text-white'),
                dbc.CardBody(dcc.Graph(figure=alloc2))
            ], style=CARD_STYLE), md=6)
        ], className='mb-4')

        # Holdings
        top10 = sorted(d['top_performing_holdings'], key=lambda x: x['return'], reverse=True)[:10]
        bar = px.bar(
            x=[h['name'] for h in top10], 
            y=[h['return'] for h in top10], 
            title='Top 10 Holdings by Return',
            color_discrete_sequence=[NYU_PURPLE]
        ).update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='',
            yaxis_title='Return (%)'
        )

        bottom_df = pd.DataFrame(d['bottom_performing_holdings'])
        bottom_df['Category'] = 'Bottom'
        top_df = pd.DataFrame(d['top_performing_holdings'])
        top_df['Category'] = 'Top'
        hold_df = pd.concat([top_df, bottom_df])[['Category','name','return']]
        hold_df.columns = ['Category','Name','Return']

        table = dash.dash_table.DataTable(
            data=hold_df.to_dict('records'),
            columns=[{'name':c,'id':c} for c in hold_df.columns],
            style_header={
                'backgroundColor': NYU_PURPLE,
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Category} eq "Top"'},
                    'backgroundColor': 'rgba(107, 55, 142, 0.1)'  # Using NYU_PURPLE with opacity
                },
                {
                    'if': {'filter_query': '{Category} eq "Bottom"'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                }
            ],
            page_size=5,
            style_table={'marginBottom': '20px'}  # Replace className with style_table
        )

        holdings = dbc.Card([
            dbc.CardHeader('Holdings Performance', className='bg-primary text-white'),
            dbc.CardBody([
                dcc.Graph(figure=bar),
                table
            ])
        ], style=CARD_STYLE)

        return [
            cards1,
            cards2,
            summary,
            styles,
            esg_focus,
            pies,
            holdings
        ]
    except KeyError:
        return html.Div(f"No data available for year {year}", style={'textAlign': 'center', 'padding': '20px'})

# Update the custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%css%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }

            .container-fluid {
                padding: 0;
                max-width: 100%;
                overflow-x: hidden;
            }

            .navbar {
                background: white !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 0.5rem 1rem;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
            }

            .navbar-brand {
                font-size: 1.8rem;
                font-weight: bold;
                color: #6B378E;
                padding: 0.5rem 1rem;
                position: relative;
                white-space: normal;
                text-align: center;
                flex: 1 1 100%;
                margin-bottom: 0.5rem;
            }

            @media (max-width: 768px) {
                .navbar {
                    padding: 0.5rem;
                }
                
                .navbar-brand {
                    font-size: 1.4rem;
                    padding: 0.5rem;
                    margin-bottom: 0.25rem;
                }
            }

            .navbar-brand::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 80%;
                height: 3px;
                background: linear-gradient(90deg, #6B378E, #8B4DB8);
                transform-origin: center;
                animation: underline 2s ease-in-out infinite;
            }

            @keyframes underline {
                0% { transform: translateX(-50%) scaleX(0); }
                50% { transform: translateX(-50%) scaleX(1); }
                100% { transform: translateX(-50%) scaleX(0); }
            }

            .nav-tabs {
                background: white;
                padding: 0.5rem;
                border-radius: 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
                white-space: nowrap;
                -webkit-overflow-scrolling: touch;
            }

            .nav-tabs .nav-link {
                color: #6B378E;
                border: none;
                padding: 0.5rem 1rem;
                margin: 0 0.25rem;
                border-radius: 8px;
                display: inline-block;
                float: none;
            }

            @media (max-width: 768px) {
                .nav-tabs .nav-link {
                    padding: 0.5rem 0.75rem;
                    font-size: 0.9rem;
                }
            }

            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #6B378E, #8B4DB8);
                color: white !important;
            }

            .tab-content {
                padding: 1rem;
            }

            @media (max-width: 768px) {
                .tab-content {
                    padding: 0.5rem;
                }
            }

            /* Overview Page Specific Styles */
            .overview-container {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            .year-selector-card {
                width: 100%;
                max-width: 300px;
                margin: 0 auto 1.5rem;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }

            @media (max-width: 768px) {
                .kpi-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 0.75rem;
                    padding: 0.5rem;
                }
            }

            .kpi-card {
                text-align: center;
                padding: 1rem;
                min-height: 80px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            @media (max-width: 768px) {
                .kpi-card {
                    padding: 0.75rem;
                    min-height: 70px;
                }
            }

            .kpi-value {
                font-size: 1.8rem;
                font-weight: bold;
                color: #6B378E;
                margin: 0;
                line-height: 1;
            }

            @media (max-width: 768px) {
                .kpi-value {
                    font-size: 1.5rem;
                }
            }

            .kpi-label {
                color: #666;
                font-size: 0.9rem;
                margin: 0.25rem 0 0;
                font-weight: 500;
                line-height: 1.2;
            }

            @media (max-width: 768px) {
                .kpi-label {
                    font-size: 0.8rem;
                }
            }

            .allocation-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                margin-bottom: 1rem;
            }

            @media (max-width: 768px) {
                .allocation-container {
                    grid-template-columns: 1fr;
                }
            }

            .card {
                border: none;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }

            .card-header {
                background: linear-gradient(135deg, #6B378E, #8B4DB8) !important;
                color: white !important;
                border-radius: 8px 8px 0 0 !important;
                padding: 0.75rem 1rem;
            }

            @media (max-width: 768px) {
                .card-header {
                    padding: 0.5rem 0.75rem;
                }
            }

            .card-body {
                padding: 1rem;
            }

            @media (max-width: 768px) {
                .card-body {
                    padding: 0.75rem;
                }
            }

            .js-plotly-plot {
                width: 100%;
                height: 100%;
                min-height: 300px;
                border-radius: 8px;
            }

            @media (max-width: 768px) {
                .js-plotly-plot {
                    min-height: 250px;
                }
            }

            .dash-table-container {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }

            .Select {
                position: relative;
                z-index: 1000;
            }

            .Select-control {
                border-radius: 8px;
                border: 2px solid rgba(107, 55, 142, 0.2);
            }

            @media (max-width: 768px) {
                .Select-control {
                    font-size: 0.9rem;
                }
            }

            .Select-menu-outer {
                z-index: 1001;
                border-radius: 8px;
                border: 2px solid #6B378E;
                box-shadow: 0 5px 20px rgba(107, 55, 142, 0.15);
                position: absolute !important;
            }

            /* Grid adjustments for mobile */
            @media (max-width: 768px) {
                .row {
                    margin-left: -0.25rem;
                    margin-right: -0.25rem;
                }
                
                .col, [class*="col-"] {
                    padding-left: 0.25rem;
                    padding-right: 0.25rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
    

