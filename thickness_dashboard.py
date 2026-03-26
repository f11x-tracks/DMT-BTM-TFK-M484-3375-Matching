#!/usr/bin/env python3
"""
Thickness Comparison Dashboard
Interactive Dash app for DMT vs TFK thickness analysis
"""

import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Load data
try:
    df = pd.read_csv('matched_thickness_data.csv')
    print(f"Loaded {len(df)} matched measurement pairs")
except FileNotFoundError:
    print("Error: matched_thickness_data.csv not found. Please run the thickness comparison app first.")
    exit(1)

# Create site coordinates for trend analysis (combining X,Y into site identifier)
df['Site'] = df['TFK_X_mm'].round(0).astype(str) + ',' + df['TFK_Y_mm'].round(0).astype(str)
df['Site_ID'] = df.groupby(['TFK_X_mm', 'TFK_Y_mm']).ngroup()

# Calculate radius from center (0,0) for radial analysis
df['Radius_mm'] = np.sqrt(df['TFK_X_mm']**2 + df['TFK_Y_mm']**2)

# Calculate mean delta and adjust DMT thickness
mean_delta = df['Thickness_Delta'].mean()
mean_abs_delta = abs(mean_delta)
df['DMT_Adjusted'] = df['DMT_Thickness'] + mean_abs_delta

# Calculate radius from center (0,0) for radial analysis
df['Radius_mm'] = np.sqrt(df['TFK_X_mm']**2 + df['TFK_Y_mm']**2)

# Calculate mean delta and adjust DMT thickness
mean_delta = df['Thickness_Delta'].mean()
mean_abs_delta = abs(mean_delta)
df['DMT_Adjusted'] = df['DMT_Thickness'] + mean_abs_delta

# Calculate summary statistics
wafer_stats = df.groupby('WaferID').agg({
    'DMT_Thickness': ['mean', 'std', 'count'],
    'TFK_Thickness': ['mean', 'std', 'count'],
    'Thickness_Delta': ['mean', 'std', 'min', 'max'],
    'Distance_mm': ['mean', 'max']
}).round(2)

wafer_stats.columns = ['DMT_Mean', 'DMT_Std', 'DMT_Count', 
                      'TFK_Mean', 'TFK_Std', 'TFK_Count',
                      'Delta_Mean', 'Delta_Std', 'Delta_Min', 'Delta_Max',
                      'Match_Dist_Mean', 'Match_Dist_Max']

wafer_stats = wafer_stats.reset_index()

# Site statistics
site_stats = df.groupby(['Site', 'Site_ID', 'TFK_X_mm', 'TFK_Y_mm']).agg({
    'DMT_Thickness': ['mean', 'std'],
    'TFK_Thickness': ['mean', 'std'], 
    'Thickness_Delta': ['mean', 'std']
}).round(2)

site_stats.columns = ['DMT_Mean', 'DMT_Std', 'TFK_Mean', 'TFK_Std', 'Delta_Mean', 'Delta_Std']
site_stats = site_stats.reset_index()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "DMT vs TFK Thickness Analysis Dashboard"

# Define the layout
app.layout = html.Div([
    html.H1("DMT vs TFK Thickness Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Summary statistics cards
    html.Div([
        html.Div([
            html.H3(f"{len(df)}", style={'color': '#3498db', 'fontSize': '2em', 'margin': 0}),
            html.P("Total Matched Pairs", style={'margin': 0})
        ], className='summary-card', style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{len(df['WaferID'].unique())}", style={'color': '#e74c3c', 'fontSize': '2em', 'margin': 0}),
            html.P("Unique Wafers", style={'margin': 0})
        ], className='summary-card', style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{df['Thickness_Delta'].mean():.1f}Å", style={'color': '#27ae60', 'fontSize': '2em', 'margin': 0}),
            html.P("Mean Delta (DMT-TFK)", style={'margin': 0})
        ], className='summary-card', style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{df['Thickness_Delta'].std():.1f}Å", style={'color': '#f39c12', 'fontSize': '2em', 'margin': 0}),
            html.P("Delta Std Dev", style={'margin': 0})
        ], className='summary-card', style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'textAlign': 'center'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30, 'gap': '20px'}),
    
    # Chart 1: Site-by-site thickness trends with delta box plot
    html.H2("Site-by-Site Thickness Analysis", style={'color': '#2c3e50'}),
    html.P("Raw thickness data points at each measurement site (left) with DMT vs TFK raw data comparison (right)"),
    dcc.Graph(id='site-thickness-chart'),
    
    # Chart 2: Wafer average trends with box plot
    html.H2("Wafer Average Thickness Trends", style={'color': '#2c3e50', 'marginTop': 40}),
    html.P("Average thickness per wafer trends (left) with distribution analysis (right)"),
    dcc.Graph(id='wafer-average-chart'),
    
    # Chart 3: Wafer standard deviation trends with box plot
    html.H2("Wafer Standard Deviation Analysis", style={'color': '#2c3e50', 'marginTop': 40}),
    html.P("Standard deviation trends by wafer (left) with distribution analysis (right)"),
    dcc.Graph(id='wafer-std-chart'),
    
    # Chart 4: Radial thickness analysis with adjusted DMT
    html.H2("Radial Thickness Analysis (Adjusted DMT)", style={'color': '#2c3e50', 'marginTop': 40}),
    html.P(f"TFK vs Adjusted DMT thickness by radius from wafer center (Mean Delta: {mean_delta:.1f}Å, Adjustment: +{mean_abs_delta:.1f}Å to DMT)"),
    dcc.Graph(id='radial-thickness-chart'),
    
    # Summary Table
    html.H2("Detailed Summary Table", style={'color': '#2c3e50', 'marginTop': 40}),
    html.P("Statistical summary for each wafer"),
    dash_table.DataTable(
        id='summary-table',
        columns=[
            {'name': 'Wafer ID', 'id': 'WaferID'},
            {'name': 'DMT Mean (Å)', 'id': 'DMT_Mean', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'DMT Std (Å)', 'id': 'DMT_Std', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'TFK Mean (Å)', 'id': 'TFK_Mean', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'TFK Std (Å)', 'id': 'TFK_Std', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'Delta Mean (Å)', 'id': 'Delta_Mean', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'Delta Std (Å)', 'id': 'Delta_Std', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=1)},
            {'name': 'Points', 'id': 'DMT_Count', 'type': 'numeric'},
            {'name': 'Max Match Dist (mm)', 'id': 'Match_Dist_Max', 'type': 'numeric', 'format': dash_table.FormatTemplate.Format(precision=2)}
        ],
        data=wafer_stats.to_dict('records'),
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }
        ],
        sort_action='native',
        page_size=20
    )
], style={'margin': '20px', 'fontFamily': 'Arial, sans-serif'})

# Callback for site thickness chart
@app.callback(
    dash.dependencies.Output('site-thickness-chart', 'figure'),
    [dash.dependencies.Input('site-thickness-chart', 'id')]
)
def update_site_chart(_):
    # Create subplot with side-by-side layout
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=("Raw Thickness by Site", "DMT vs TFK Raw Data Comparison"),
        horizontal_spacing=0.05
    )
    
    # Sort sites by X, then Y coordinate for logical ordering
    site_order = site_stats.sort_values(['TFK_X_mm', 'TFK_Y_mm'])['Site_ID'].tolist()
    site_labels = site_stats.sort_values(['TFK_X_mm', 'TFK_Y_mm'])['Site'].tolist()
    
    # Left plot: Raw thickness points by site
    # DMT thickness points
    fig.add_trace(
        go.Scatter(
            x=df['Site_ID'],
            y=df['DMT_Thickness'],
            mode='markers',
            name='DMT Raw Data',
            marker=dict(color='#3498db', size=6, opacity=0.7),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # TFK thickness points  
    fig.add_trace(
        go.Scatter(
            x=df['Site_ID'],
            y=df['TFK_Thickness'],
            mode='markers',
            name='TFK Raw Data',
            marker=dict(color='#e74c3c', size=6, opacity=0.7),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Right plot: Box plot comparing DMT vs TFK raw data
    fig.add_trace(
        go.Box(
            y=df['DMT_Thickness'],
            name='DMT Raw',
            boxpoints='outliers',
            fillcolor='rgba(52, 152, 219, 0.7)',
            line=dict(color='rgba(52, 152, 219, 1.0)')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=df['TFK_Thickness'],
            name='TFK Raw',
            boxpoints='outliers',
            fillcolor='rgba(231, 76, 60, 0.7)',
            line=dict(color='rgba(231, 76, 60, 1.0)')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(len(site_labels))),
        ticktext=[f"Site {i}" for i in range(len(site_labels))],
        title="Measurement Site",
        row=1, col=1
    )
    fig.update_yaxes(title="Thickness (Å)", row=1, col=1)
    fig.update_yaxes(title="Raw Thickness (Å)", row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    
    fig.update_layout(
        height=800,
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback for wafer average chart
@app.callback(
    dash.dependencies.Output('wafer-average-chart', 'figure'),
    [dash.dependencies.Input('wafer-average-chart', 'id')]
)
def update_wafer_average_chart(_):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=("Average Thickness by Wafer", "Thickness Distribution"),
        horizontal_spacing=0.05
    )
    
    wafer_list = wafer_stats['WaferID'].tolist()
    x_pos = list(range(len(wafer_list)))
    
    # Top plot: Trend lines
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=wafer_stats['DMT_Mean'],
            mode='lines+markers',
            name='DMT Average',
            line=dict(color='#3498db', width=3),
            error_y=dict(
                type='data',
                array=wafer_stats['DMT_Std'],
                visible=True,
                color='#3498db'
            )
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=wafer_stats['TFK_Mean'],
            mode='lines+markers',
            name='TFK Average',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            error_y=dict(
                type='data',
                array=wafer_stats['TFK_Std'],
                visible=True,
                color='#e74c3c'
            )
        ),
        row=1, col=1
    )
    
    # Right plot: Box plots
    fig.add_trace(
        go.Box(
            y=wafer_stats['DMT_Mean'],
            name='DMT Average',
            boxpoints='all',
            fillcolor='rgba(52, 152, 219, 0.7)'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=wafer_stats['TFK_Mean'],
            name='TFK Average', 
            boxpoints='all',
            fillcolor='rgba(231, 76, 60, 0.7)'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_pos,
        ticktext=wafer_list,
        title="Wafer ID",
        row=1, col=1
    )
    fig.update_yaxes(title="Average Thickness (Å)", row=1, col=1)
    fig.update_yaxes(title="Thickness (Å)", row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    
    fig.update_layout(
        height=700,
        title_x=0.5,
        showlegend=True
    )
    
    return fig

# Callback for wafer std dev chart
@app.callback(
    dash.dependencies.Output('wafer-std-chart', 'figure'),
    [dash.dependencies.Input('wafer-std-chart', 'id')]
)
def update_wafer_std_chart(_):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=("Standard Deviation by Wafer", "Std Dev Distribution"),
        horizontal_spacing=0.05
    )
    
    wafer_list = wafer_stats['WaferID'].tolist()
    x_pos = list(range(len(wafer_list)))
    
    # Top plot: Std dev trends
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=wafer_stats['DMT_Std'],
            mode='lines+markers',
            name='DMT Std Dev',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=wafer_stats['TFK_Std'],
            mode='lines+markers',
            name='TFK Std Dev',
            line=dict(color='#f39c12', width=3, dash='dash'),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Right plot: Box plots
    fig.add_trace(
        go.Box(
            y=wafer_stats['DMT_Std'],
            name='DMT Std Dev',
            boxpoints='all',
            fillcolor='rgba(155, 89, 182, 0.7)'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=wafer_stats['TFK_Std'],
            name='TFK Std Dev',
            boxpoints='all',
            fillcolor='rgba(243, 156, 18, 0.7)'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_pos,
        ticktext=wafer_list,
        title="Wafer ID",
        row=1, col=1
    )
    fig.update_yaxes(title="Standard Deviation (Å)", range=[0, 10], row=1, col=1)
    fig.update_yaxes(title="Std Dev (Å)", range=[0, 10], row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    
    fig.update_layout(
        height=700,
        title_x=0.5,
        showlegend=True
    )
    
    return fig

# Callback for radial thickness chart
@app.callback(
    dash.dependencies.Output('radial-thickness-chart', 'figure'),
    [dash.dependencies.Input('radial-thickness-chart', 'id')]
)
def update_radial_chart(_):
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Thickness vs Radius (with Adjusted DMT)",)
    )
    
    # Sort data by radius for spline fitting
    df_sorted = df.sort_values('Radius_mm')
    
    # GROUP BY RADIUS TO HANDLE DUPLICATES - This is the key fix!
    # Many measurement points have the same radius, which breaks spline fitting
    df_grouped = df_sorted.groupby('Radius_mm').agg({
        'TFK_Thickness': 'mean',
        'DMT_Adjusted': 'mean'
    }).reset_index()
    
    print(f"Radial analysis: {len(df_sorted)} points grouped to {len(df_grouped)} unique radius values")
    
    # Create radius range for smooth spline using grouped data range
    radius_range = np.linspace(df_grouped['Radius_mm'].min(), df_grouped['Radius_mm'].max(), 300)
    
    # Fit splines using the grouped data (no more duplicate radius issues!)
    try:       
        # TFK spline with grouped data
        for s_param in [100, 200, 500]:  # More conservative smoothing parameters
            try:
                tfk_spline = UnivariateSpline(df_grouped['Radius_mm'], df_grouped['TFK_Thickness'], s=s_param)
                tfk_smooth = tfk_spline(radius_range)
                
                if np.all(np.isfinite(tfk_smooth)):
                    # Add TFK spline curve
                    fig.add_trace(
                        go.Scatter(
                            x=radius_range,
                            y=tfk_smooth,
                            mode='lines',
                            name='TFK Spline (Smoothed)',
                            line=dict(color='#e74c3c', width=3),
                            showlegend=True
                        )
                    )
                    print(f"TFK spline added successfully with s={s_param}")
                    break
            except Exception as e:
                continue
        else:
            print("All TFK spline fitting attempts failed")
        
    except Exception as e:
        print(f"TFK spline fitting error: {e}")
    
    try:
        # DMT Adjusted spline with grouped data
        for s_param in [100, 200, 500]:  # More conservative smoothing parameters
            try:
                dmt_spline = UnivariateSpline(df_grouped['Radius_mm'], df_grouped['DMT_Adjusted'], s=s_param)
                dmt_smooth = dmt_spline(radius_range)
                
                # Check if spline results are valid
                if np.all(np.isfinite(dmt_smooth)):
                    # Add DMT Adjusted spline curve
                    fig.add_trace(
                        go.Scatter(
                            x=radius_range,
                            y=dmt_smooth,
                            mode='lines',
                            name='DMT Adjusted Spline (Smoothed)',
                            line=dict(color='#3498db', width=3),
                            showlegend=True
                        )
                    )
                    print(f"DMT Adjusted spline added successfully with s={s_param}")
                    break
            except Exception as e:
                continue
        else:
            print("All DMT spline fitting attempts failed")
            
    except Exception as e:
        print(f"DMT Adjusted spline fitting error: {e}")
    
    # Add raw data points (show all original measurements)
    fig.add_trace(
        go.Scatter(
            x=df['Radius_mm'],
            y=df['TFK_Thickness'],
            mode='markers',
            name='TFK Raw Data',
            marker=dict(color='#e74c3c', size=4, opacity=0.6),
            showlegend=True
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Radius_mm'],
            y=df['DMT_Adjusted'],
            mode='markers',
            name='DMT Adjusted Raw Data',
            marker=dict(color='#3498db', size=4, opacity=0.6),
            showlegend=True
        )
    )
    
    # Update layout
    fig.update_xaxes(title="Radius from Center (mm)")
    fig.update_yaxes(title="Thickness (Å)")
    
    fig.update_layout(
        height=600,
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

if __name__ == '__main__':
    print("Starting Thickness Analysis Dashboard...")
    print("Open your web browser to: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=8050)