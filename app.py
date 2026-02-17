"""
Multi-page Dash app combining Season Explorer, Player Explorer,
and Rate vs Efficiency Explorer.

Data is loaded from pre-computed parquet files (see export_data.py).

Usage (local):
    python vercel_app/app.py
"""

import os
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

df_season = pd.read_parquet(os.path.join(DATA_DIR, 'season_stats.parquet'))
df_players = pd.read_parquet(os.path.join(DATA_DIR, 'player_stats.parquet'))

# ---------------------------------------------------------------------------
# Season Explorer helpers
# ---------------------------------------------------------------------------
SEASON_EXCLUDE = {
    'team_abbr', 'team_name', 'games', 'home_games', 'away_games',
    'game_id', 'game_date', 'is_home',
}
season_numeric_cols = sorted([
    c for c in df_season.columns
    if c not in SEASON_EXCLUDE and pd.api.types.is_numeric_dtype(df_season[c])
])


def friendly_season(col):
    label = col.replace('_', ' ').replace('pct', '%').replace('per100', '/100')
    if label.startswith('opp '):
        label = 'Opp ' + label[4:]
    return label.title()


season_col_options = [{'label': friendly_season(c), 'value': c} for c in season_numeric_cols]

# ---------------------------------------------------------------------------
# Player Explorer helpers
# ---------------------------------------------------------------------------
PLAYER_EXCLUDE = {
    'player_id', 'player_name', 'games', 'game_id', 'game_date',
    'height_inches', 'team_abbr',
    'close_fg', 'mid_fg', 'three_fg', 'ft_fg',
}
player_numeric_cols = sorted([
    c for c in df_players.columns
    if c not in PLAYER_EXCLUDE and pd.api.types.is_numeric_dtype(df_players[c])
])


def friendly_player(col):
    label = col.replace('_', ' ').replace('pct', '%').replace('per100', '/100').replace('per_game', '/G')
    return label.title()


player_col_options = [{'label': friendly_player(c), 'value': c} for c in player_numeric_cols]
player_teams = sorted(df_players['team_abbr'].unique())
player_team_options = [{'label': 'All Teams', 'value': 'ALL'}] + [
    {'label': t, 'value': t} for t in player_teams
]
DEFAULT_MIN_POSS = 200

# ---------------------------------------------------------------------------
# Rate vs Efficiency helpers
# ---------------------------------------------------------------------------
SHOT_CATEGORIES = [
    {'label': 'Close (≤5ft)', 'rate': 'close_freq', 'eff': 'close_pps',
     'eff_num': None, 'eff_den': None, 'color': 'green', 'symbol': 'circle'},
    {'label': 'Mid-Range', 'rate': 'mid_freq', 'eff': 'mid_pps',
     'eff_num': None, 'eff_den': None, 'color': 'blue', 'symbol': 'square'},
    {'label': '3-Pointer', 'rate': 'three_freq', 'eff': 'three_pps',
     'eff_num': None, 'eff_den': None, 'color': 'red', 'symbol': 'triangle-up'},
    {'label': 'FT Trip', 'rate': 'ft_trip_freq', 'eff': 'ft_ppt',
     'eff_num': None, 'eff_den': None, 'color': 'purple', 'symbol': 'diamond'},
]

EXTRA_CATEGORIES = [
    {'label': 'Steals', 'rate': 'stl_per100', 'eff': None,
     'eff_num': 'stl_pts', 'eff_den': 'stl', 'color': 'orange', 'symbol': 'star'},
    {'label': 'Turnovers', 'rate': 'tov_per100', 'eff': None,
     'eff_num': 'tov_pts', 'eff_den': 'tov', 'color': 'brown', 'symbol': 'pentagon'},
    {'label': 'Off. Rebounds', 'rate': 'oreb_per100', 'eff': None,
     'eff_num': 'orb_pts', 'eff_den': 'oreb', 'color': 'teal', 'symbol': 'hexagon'},
    {'label': 'Assists', 'rate': 'ast_per100', 'eff': None,
     'eff_num': 'ast_pts', 'eff_den': 'ast', 'color': 'gold', 'symbol': 'cross'},
    {'label': 'Blocks', 'rate': 'blk_per100', 'eff': None,
     'eff_num': None, 'eff_den': None, 'color': 'gray', 'symbol': 'bowtie'},
    {'label': 'Fouls', 'rate': 'pf_per100', 'eff': None,
     'eff_num': 'pf_pts', 'eff_den': 'pf', 'color': 'hotpink', 'symbol': 'hourglass'},
    {'label': 'Fast Break', 'rate': 'fb_poss_per100', 'eff': None,
     'eff_num': 'fb_pts', 'eff_den': 'fb_poss', 'color': 'cyan', 'symbol': 'triangle-down'},
    {'label': 'Clutch', 'rate': 'clutch_poss_per100', 'eff': None,
     'eff_num': 'clutch_pts', 'eff_den': 'clutch_poss', 'color': 'darkred', 'symbol': 'triangle-left'},
]

ALL_CATEGORIES = SHOT_CATEGORIES + EXTRA_CATEGORIES
CAT_BY_LABEL = {c['label']: c for c in ALL_CATEGORIES}


def _get_rate_col(cat, prefix):
    return f"{prefix}{cat['rate']}"


def _get_eff_series(df, cat, prefix):
    if cat['eff'] is not None:
        col = f"{prefix}{cat['eff']}"
        return df[col], col
    if cat['eff_num'] is None:
        return None, None
    num_col = f"{prefix}{cat['eff_num']}"
    den_col = f"{prefix}{cat['eff_den']}"
    series = (df[num_col] / df[den_col]).replace([np.inf, -np.inf], np.nan)
    return series, f"{cat['label']} eff"


def build_unfaceted(df, categories, prefix):
    fig = go.Figure()
    side = 'Opponent' if prefix == 'opp_' else 'Team'
    for cat in categories:
        rate_col = _get_rate_col(cat, prefix)
        eff_series, _ = _get_eff_series(df, cat, prefix)
        if eff_series is None:
            continue
        mask = eff_series.notna() & df[rate_col].notna()
        df_plot = df[mask]
        eff_vals = eff_series[mask]
        fig.add_trace(go.Scatter(
            x=df_plot[rate_col], y=eff_vals,
            mode='markers+text', name=cat['label'],
            text=df_plot['team_abbr'], textposition='top center',
            textfont=dict(size=8),
            marker=dict(symbol=cat['symbol'], size=12, color=cat['color'],
                        line=dict(width=1, color='black')),
            hovertemplate=(
                '<b>%{text}</b><br>' + f'{cat["label"]}<br>'
                + 'Rate: %{x:.1f}/100<br>Efficiency: %{y:.2f}<br><extra></extra>'
            ),
        ))
    fig.update_layout(
        title=f'Rate vs Efficiency ({side})',
        xaxis_title='Per 100 Possessions', yaxis_title='Points Per Event',
        height=700, width=1100, showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
    )
    return fig


def build_faceted(df, prefix):
    side = 'Opponent' if prefix == 'opp_' else 'Team'
    cats = SHOT_CATEGORIES
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[c['label'] for c in cats],
        horizontal_spacing=0.1, vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for cat, (row, col) in zip(cats, positions):
        rate_col = _get_rate_col(cat, prefix)
        eff_series, _ = _get_eff_series(df, cat, prefix)
        mask = eff_series.notna() & df[rate_col].notna()
        df_plot = df[mask]
        eff_vals = eff_series[mask]
        fig.add_trace(go.Scatter(
            x=df_plot[rate_col], y=eff_vals,
            mode='markers+text', name=cat['label'],
            text=df_plot['team_abbr'], textposition='top center',
            textfont=dict(size=8),
            marker=dict(size=10, color=cat['color'],
                        line=dict(width=1, color='black')),
            hovertemplate=(
                '<b>%{text}</b><br>Frequency: %{x:.1f}/100<br>'
                + 'PPS: %{y:.2f}<br><extra></extra>'
            ),
            showlegend=False,
        ), row=row, col=col)
        avg_rate = df_plot[rate_col].mean()
        avg_eff = eff_vals.mean()
        fig.add_hline(y=avg_eff, line_dash='dash', line_color='gray',
                      line_width=1, row=row, col=col)
        fig.add_vline(x=avg_rate, line_dash='dash', line_color='gray',
                      line_width=1, row=row, col=col)
    fig.update_xaxes(title_text='Freq/100', row=2, col=1)
    fig.update_xaxes(title_text='Freq/100', row=2, col=2)
    fig.update_yaxes(title_text='PPS', row=1, col=1)
    fig.update_yaxes(title_text='PPS', row=2, col=1)
    fig.update_layout(
        title=f'Shot Frequency vs Points Per Shot ({side})',
        height=800, width=1100,
        plot_bgcolor='white', paper_bgcolor='#fafafa',
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    requests_pathname_prefix='/explore/',
    suppress_callback_exceptions=True,
)

# -- Tab layouts -----------------------------------------------------------

season_tab = html.Div(
    style={'padding': '24px 0'},
    children=[
        html.P(
            f'{int(df_season["games"].iloc[0])} games loaded · '
            f'{len(df_season)} teams · {len(season_numeric_cols)} variables',
            style={'color': '#888', 'fontSize': '14px', 'marginBottom': '24px'},
        ),
        html.Div(
            style={'display': 'flex', 'gap': '24px', 'marginBottom': '16px'},
            children=[
                html.Div([
                    html.Label('X-axis', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Dropdown(id='season-x-col', options=season_col_options,
                                 value='pts_per100', clearable=False,
                                 style={'width': '340px'}),
                ]),
                html.Div([
                    html.Label('Y-axis', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Dropdown(id='season-y-col', options=season_col_options,
                                 value='opp_pts_per100', clearable=False,
                                 style={'width': '340px'}),
                ]),
            ],
        ),
        dcc.Graph(id='season-scatter', config={'displayModeBar': False}),
    ],
)

player_tab = html.Div(
    style={'padding': '24px 0'},
    children=[
        html.P(
            f'{len(df_players[df_players["poss_played"] >= DEFAULT_MIN_POSS])} players · '
            f'{len(player_numeric_cols)} variables',
            style={'color': '#888', 'fontSize': '14px', 'marginBottom': '24px'},
        ),
        html.Div(
            style={'display': 'flex', 'gap': '24px', 'marginBottom': '16px',
                   'flexWrap': 'wrap', 'alignItems': 'flex-end'},
            children=[
                html.Div([
                    html.Label('X-axis', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Dropdown(id='player-x-col', options=player_col_options,
                                 value='usage_pct', clearable=False,
                                 style={'width': '300px'}),
                ]),
                html.Div([
                    html.Label('Y-axis', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Dropdown(id='player-y-col', options=player_col_options,
                                 value='ts_pct', clearable=False,
                                 style={'width': '300px'}),
                ]),
                html.Div([
                    html.Label('Team', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Dropdown(id='player-team-filter', options=player_team_options,
                                 value='ALL', clearable=False,
                                 style={'width': '140px'}),
                ]),
                html.Div([
                    html.Label('Min possessions', style={'fontWeight': '600', 'fontSize': '13px'}),
                    dcc.Input(id='player-min-poss', type='number',
                              value=DEFAULT_MIN_POSS, min=0, step=50,
                              style={'width': '100px', 'height': '36px',
                                     'borderRadius': '4px', 'border': '1px solid #ccc',
                                     'paddingLeft': '8px'}),
                ]),
            ],
        ),
        dcc.Graph(id='player-scatter', config={'displayModeBar': False}),
    ],
)

rate_tab = html.Div(
    style={'padding': '24px 0'},
    children=[
        html.Div(
            style={'display': 'flex', 'gap': '48px', 'alignItems': 'flex-start',
                   'marginBottom': '16px', 'flexWrap': 'wrap'},
            children=[
                html.Div([
                    html.Label('View', style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.RadioItems(
                        id='rate-view-toggle',
                        options=[
                            {'label': 'Faceted (2x2 Shot Types)', 'value': 'faceted'},
                            {'label': 'Unfaceted (Overlay)', 'value': 'unfaceted'},
                        ],
                        value='faceted',
                        labelStyle={'display': 'block'},
                    ),
                ]),
                html.Div([
                    html.Label('Side', style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.RadioItems(
                        id='rate-side-toggle',
                        options=[
                            {'label': 'Offense', 'value': ''},
                            {'label': 'Defense (opponent)', 'value': 'opp_'},
                        ],
                        value='',
                        labelStyle={'display': 'block'},
                    ),
                ]),
                html.Div(
                    id='rate-category-container',
                    children=[
                        html.Label('Categories', style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                        dcc.Checklist(
                            id='rate-category-checklist',
                            options=[{'label': c['label'], 'value': c['label']}
                                     for c in ALL_CATEGORIES if c['label'] != 'Blocks'],
                            value=[c['label'] for c in SHOT_CATEGORIES],
                            labelStyle={'display': 'block'},
                        ),
                    ],
                ),
            ],
        ),
        dcc.Graph(id='rate-chart'),
    ],
)

# -- Main layout -----------------------------------------------------------

app.layout = html.Div(
    style={
        'fontFamily': '"Helvetica Neue", Helvetica, Arial, sans-serif',
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '24px',
        'backgroundColor': '#fafafa',
    },
    children=[
        html.H1('NBA Stats Explorers',
                 style={'fontSize': '28px', 'fontWeight': '600', 'marginBottom': '16px'}),
        dcc.Tabs(
            id='main-tabs',
            value='season',
            children=[
                dcc.Tab(label='Season Explorer', value='season'),
                dcc.Tab(label='Player Explorer', value='player'),
                dcc.Tab(label='Rate vs Efficiency', value='rate'),
            ],
        ),
        html.Div(id='tab-content'),
    ],
)


# -- Tab routing callback --------------------------------------------------

@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'season':
        return season_tab
    if tab == 'player':
        return player_tab
    return rate_tab


# ---------------------------------------------------------------------------
# Season Explorer callback
# ---------------------------------------------------------------------------

@app.callback(
    Output('season-scatter', 'figure'),
    Input('season-x-col', 'value'),
    Input('season-y-col', 'value'),
)
def update_season_chart(x_col, y_col):
    x = df_season[x_col]
    y = df_season[y_col]
    labels = df_season['team_abbr']

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.82, 0.18], row_heights=[0.18, 0.82],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )

    fig.add_trace(go.Histogram(
        x=x, nbinsx=12,
        marker_color='rgba(44, 120, 180, 0.5)',
        marker_line_color='rgba(44, 120, 180, 0.8)',
        marker_line_width=1, showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        y=y, nbinsx=12,
        marker_color='rgba(44, 120, 180, 0.5)',
        marker_line_color='rgba(44, 120, 180, 0.8)',
        marker_line_width=1, showlegend=False,
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        text=labels, textposition='top center',
        textfont=dict(size=10, color='#444'),
        marker=dict(size=9, color='rgba(44, 120, 180, 0.75)',
                    line=dict(width=1, color='white')),
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'{friendly_season(x_col)}: %{{x:.2f}}<br>'
            f'{friendly_season(y_col)}: %{{y:.2f}}<extra></extra>'
        ),
        showlegend=False,
    ), row=2, col=1)

    x_mean, y_mean = x.mean(), y.mean()
    fig.add_hline(y=y_mean, line_dash='dot', line_color='#bbb', line_width=1, row=2, col=1)
    fig.add_vline(x=x_mean, line_dash='dot', line_color='#bbb', line_width=1, row=2, col=1)

    fig.update_layout(
        height=700, width=900,
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=60, r=20, t=30, b=60), bargap=0.06,
    )
    fig.update_xaxes(title_text=friendly_season(x_col), row=2, col=1,
                     showgrid=True, gridcolor='#eee', zeroline=False,
                     title_font=dict(size=13))
    fig.update_yaxes(title_text=friendly_season(y_col), row=2, col=1,
                     showgrid=True, gridcolor='#eee', zeroline=False,
                     title_font=dict(size=13))
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# Player Explorer callback
# ---------------------------------------------------------------------------

@app.callback(
    Output('player-scatter', 'figure'),
    Input('player-x-col', 'value'),
    Input('player-y-col', 'value'),
    Input('player-team-filter', 'value'),
    Input('player-min-poss', 'value'),
)
def update_player_chart(x_col, y_col, team, min_poss):
    if min_poss is None or min_poss < 0:
        min_poss = 0

    df_all = df_players[df_players['poss_played'] >= min_poss].copy()

    x_league = df_all[x_col]
    y_league = df_all[y_col]
    x_league_mean = x_league.mean()
    y_league_mean = y_league.mean()

    league_mask = x_league.notna() & y_league.notna()
    x_pad = (x_league[league_mask].max() - x_league[league_mask].min()) * 0.05
    y_pad = (y_league[league_mask].max() - y_league[league_mask].min()) * 0.05
    x_range = [x_league[league_mask].min() - x_pad, x_league[league_mask].max() + x_pad]
    y_range = [y_league[league_mask].min() - y_pad, y_league[league_mask].max() + y_pad]

    league_trend = {}
    if league_mask.sum() >= 2:
        xf_l, yf_l = x_league[league_mask], y_league[league_mask]
        m, b = np.polyfit(xf_l, yf_l, 1)
        ss_res = ((yf_l - (m * xf_l + b)) ** 2).sum()
        ss_tot = ((yf_l - yf_l.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        league_trend = {'m': m, 'b': b, 'r2': r2,
                        'x_min': xf_l.min(), 'x_max': xf_l.max(), 'y_max': yf_l.max()}

    df = df_all if team == 'ALL' else df_all[df_all['team_abbr'] == team]
    x = df[x_col]
    y = df[y_col]
    labels = df['player_name']

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.82, 0.18], row_heights=[0.18, 0.82],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )

    fig.add_trace(go.Histogram(
        x=x_league, nbinsx=20,
        marker_color='rgba(44, 120, 180, 0.5)',
        marker_line_color='rgba(44, 120, 180, 0.8)',
        marker_line_width=1, showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        y=y_league, nbinsx=20,
        marker_color='rgba(44, 120, 180, 0.5)',
        marker_line_color='rgba(44, 120, 180, 0.8)',
        marker_line_width=1, showlegend=False,
    ), row=2, col=2)

    show_text = team != 'ALL'
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text' if show_text else 'markers',
        text=labels,
        textposition='top center' if show_text else None,
        textfont=dict(size=9, color='#444') if show_text else None,
        marker=dict(size=7, color='rgba(44, 120, 180, 0.6)',
                    line=dict(width=0.5, color='white')),
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'{friendly_player(x_col)}: %{{x:.2f}}<br>'
            f'{friendly_player(y_col)}: %{{y:.2f}}<extra></extra>'
        ),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=y_league_mean, line_dash='dot', line_color='#bbb', line_width=1, row=2, col=1)
    fig.add_vline(x=x_league_mean, line_dash='dot', line_color='#bbb', line_width=1, row=2, col=1)

    if league_trend:
        m, b, r2 = league_trend['m'], league_trend['b'], league_trend['r2']
        x_line = np.linspace(league_trend['x_min'], league_trend['x_max'], 100)
        y_line = m * x_line + b
        sign = '+' if b >= 0 else '-'
        eq_text = f'y = {m:.3f}x {sign} {abs(b):.3f}   R² = {r2:.3f}'
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color='rgba(220, 60, 60, 0.7)', width=2, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ), row=2, col=1)
        fig.add_annotation(
            text=eq_text,
            xref='x2', yref='y3',
            x=league_trend['x_min'] + (league_trend['x_max'] - league_trend['x_min']) * 0.02,
            y=league_trend['y_max'],
            showarrow=False,
            font=dict(size=12, color='rgba(220, 60, 60, 0.9)'),
            bgcolor='rgba(255,255,255,0.8)',
            row=2, col=1,
        )

    fig.update_layout(
        height=700, width=960,
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=60, r=20, t=30, b=60), bargap=0.06,
    )
    fig.update_xaxes(title_text=friendly_player(x_col), row=2, col=1,
                     showgrid=True, gridcolor='#eee', zeroline=False,
                     title_font=dict(size=13), range=x_range)
    fig.update_yaxes(title_text=friendly_player(y_col), row=2, col=1,
                     showgrid=True, gridcolor='#eee', zeroline=False,
                     title_font=dict(size=13), range=y_range)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# Rate vs Efficiency callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output('rate-category-container', 'style'),
    Input('rate-view-toggle', 'value'),
)
def toggle_rate_categories(view):
    if view == 'faceted':
        return {'display': 'none'}
    return {}


@app.callback(
    Output('rate-chart', 'figure'),
    Input('rate-view-toggle', 'value'),
    Input('rate-side-toggle', 'value'),
    Input('rate-category-checklist', 'value'),
)
def update_rate_chart(view, prefix, selected_labels):
    if view == 'faceted':
        return build_faceted(df_season, prefix)
    cats = [CAT_BY_LABEL[lbl] for lbl in selected_labels if lbl in CAT_BY_LABEL]
    if not cats:
        return go.Figure().update_layout(
            title='Select at least one category',
            height=700, plot_bgcolor='white', paper_bgcolor='#fafafa',
        )
    return build_unfaceted(df_season, cats, prefix)


# ---------------------------------------------------------------------------
# WSGI server object (used by api/index.py)
# ---------------------------------------------------------------------------
server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8053)
