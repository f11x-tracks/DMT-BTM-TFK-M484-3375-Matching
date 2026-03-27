#!/usr/bin/env python3
"""
Comprehensive Wafer Map Analysis - Contour Plot Version
Creates wafer maps with contour plots, location summaries, and tool comparison tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

class WaferMapAnalysis:
    def __init__(self):
        self.dmt_tfk_data = None
        self.btm_dmt_data = None
        self.btm_tfk_data = None
        self.location_summary = None
        self.tool_wafer_summary = None
        
    def load_all_data(self):
        """Load all comparison datasets"""
        print("Loading comparison datasets...")
        
        # Load DMT-TFK comparison (main matched data)
        try:
            self.dmt_tfk_data = pd.read_csv('output/matched_thickness_data.csv')
            print(f"Loaded {len(self.dmt_tfk_data)} DMT-TFK matched pairs")
        except FileNotFoundError:
            print("Warning: output/matched_thickness_data.csv not found")
            
        # Load BTM-DMT comparison
        try:
            self.btm_dmt_data = pd.read_csv('btm_dmt_comparison_results/btm_dmt_matched_data.csv')
            print(f"Loaded {len(self.btm_dmt_data)} BTM-DMT matched pairs")
        except FileNotFoundError:
            print("Warning: btm_dmt_matched_data.csv not found")
            
        # Load BTM-TFK comparison
        try:
            self.btm_tfk_data = pd.read_csv('btm_tfk_comparison_results/btm_tfk_matched_data.csv')
            print(f"Loaded {len(self.btm_tfk_data)} BTM-TFK matched pairs")
        except FileNotFoundError:
            print("Warning: btm_tfk_matched_data.csv not found")
    
    def create_location_summary_table(self):
        """Create comprehensive location-based summary with all tool comparisons"""
        location_data = []
        
        # Process DMT-TFK data
        if self.dmt_tfk_data is not None:
            dmt_tfk_summary = self.dmt_tfk_data.groupby(['DMT_X_mm', 'DMT_Y_mm']).agg({
                'Thickness_Delta': ['mean', 'std', 'count'],
                'DMT_Thickness': 'mean',
                'TFK_Thickness': 'mean'
            }).round(2)
            
            dmt_tfk_summary.columns = ['Delta_Mean', 'Delta_Std', 'Count', 'DMT_Mean', 'TFK_Mean']
            dmt_tfk_summary = dmt_tfk_summary.reset_index()
            dmt_tfk_summary['Comparison'] = 'DMT-TFK'
            dmt_tfk_summary['Abs_Delta'] = abs(dmt_tfk_summary['Delta_Mean'])
            
            for _, row in dmt_tfk_summary.iterrows():
                location_data.append({
                    'X_mm': row['DMT_X_mm'],
                    'Y_mm': row['DMT_Y_mm'],
                    'Comparison_Type': 'DMT-TFK',
                    'Delta_Mean': row['Delta_Mean'],
                    'Delta_Std': row['Delta_Std'],
                    'Abs_Delta': row['Abs_Delta'],
                    'Count': row['Count'],
                    'Tool1_Mean': row['DMT_Mean'],
                    'Tool2_Mean': row['TFK_Mean']
                })
        
        # Process BTM-DMT data
        if self.btm_dmt_data is not None:
            btm_dmt_summary = self.btm_dmt_data.groupby(['BTM_X_mm', 'BTM_Y_mm']).agg({
                'Thickness_Delta': ['mean', 'std', 'count'],
                'BTM_Thickness': 'mean',
                'DMT_Thickness': 'mean'
            }).round(2)
            
            btm_dmt_summary.columns = ['Delta_Mean', 'Delta_Std', 'Count', 'BTM_Mean', 'DMT_Mean']
            btm_dmt_summary = btm_dmt_summary.reset_index()
            btm_dmt_summary['Abs_Delta'] = abs(btm_dmt_summary['Delta_Mean'])
            
            for _, row in btm_dmt_summary.iterrows():
                location_data.append({
                    'X_mm': row['BTM_X_mm'],
                    'Y_mm': row['BTM_Y_mm'], 
                    'Comparison_Type': 'BTM-DMT',
                    'Delta_Mean': row['Delta_Mean'],
                    'Delta_Std': row['Delta_Std'],
                    'Abs_Delta': row['Abs_Delta'],
                    'Count': row['Count'],
                    'Tool1_Mean': row['BTM_Mean'],
                    'Tool2_Mean': row['DMT_Mean']
                })
        
        # Process BTM-TFK data
        if self.btm_tfk_data is not None:
            btm_tfk_summary = self.btm_tfk_data.groupby(['BTM_X_mm', 'BTM_Y_mm']).agg({
                'Thickness_Delta': ['mean', 'std', 'count'],
                'BTM_Thickness': 'mean',
                'TFK_Thickness': 'mean'
            }).round(2)
            
            btm_tfk_summary.columns = ['Delta_Mean', 'Delta_Std', 'Count', 'BTM_Mean', 'TFK_Mean']
            btm_tfk_summary = btm_tfk_summary.reset_index()
            btm_tfk_summary['Abs_Delta'] = abs(btm_tfk_summary['Delta_Mean'])
            
            for _, row in btm_tfk_summary.iterrows():
                location_data.append({
                    'X_mm': row['BTM_X_mm'],
                    'Y_mm': row['BTM_Y_mm'],
                    'Comparison_Type': 'BTM-TFK',
                    'Delta_Mean': row['Delta_Mean'],
                    'Delta_Std': row['Delta_Std'],
                    'Abs_Delta': row['Abs_Delta'],
                    'Count': row['Count'],
                    'Tool1_Mean': row['BTM_Mean'],
                    'Tool2_Mean': row['TFK_Mean']
                })
        
        self.location_summary = pd.DataFrame(location_data)
        
        # Save to CSV
        Path('output').mkdir(exist_ok=True)
        self.location_summary.to_csv('output/location_delta_summary.csv', index=False)
        print(f"Location summary saved with {len(self.location_summary)} entries")
        
        return self.location_summary
    
    def create_tool_wafer_summary(self):
        """Create summary table with average and std dev for each tool by wafer"""
        tool_data = []
        
        # Process DMT-TFK data for tool statistics
        if self.dmt_tfk_data is not None:
            # DMT statistics by wafer
            dmt_stats = self.dmt_tfk_data.groupby('WaferID').agg({
                'DMT_Thickness': ['mean', 'std', 'count']
            }).round(2)
            dmt_stats.columns = ['Mean_Thickness', 'Std_Thickness', 'Count']
            dmt_stats = dmt_stats.reset_index()
            
            for _, row in dmt_stats.iterrows():
                tool_data.append({
                    'WaferID': row['WaferID'],
                    'Tool': 'DMT',
                    'Mean_Thickness': row['Mean_Thickness'],
                    'Std_Thickness': row['Std_Thickness'],
                    'Count': row['Count']
                })
            
            # TFK statistics by wafer
            tfk_stats = self.dmt_tfk_data.groupby('WaferID').agg({
                'TFK_Thickness': ['mean', 'std', 'count']
            }).round(2)
            tfk_stats.columns = ['Mean_Thickness', 'Std_Thickness', 'Count']
            tfk_stats = tfk_stats.reset_index()
            
            for _, row in tfk_stats.iterrows():
                tool_data.append({
                    'WaferID': row['WaferID'],
                    'Tool': 'TFK',
                    'Mean_Thickness': row['Mean_Thickness'],
                    'Std_Thickness': row['Std_Thickness'],
                    'Count': row['Count']
                })
        
        # Process BTM-DMT data for BTM statistics
        if self.btm_dmt_data is not None:
            btm_stats = self.btm_dmt_data.groupby('WaferID').agg({
                'BTM_Thickness': ['mean', 'std', 'count']
            }).round(2)
            btm_stats.columns = ['Mean_Thickness', 'Std_Thickness', 'Count']
            btm_stats = btm_stats.reset_index()
            
            for _, row in btm_stats.iterrows():
                tool_data.append({
                    'WaferID': row['WaferID'],
                    'Tool': 'BTM',
                    'Mean_Thickness': row['Mean_Thickness'],
                    'Std_Thickness': row['Std_Thickness'],
                    'Count': row['Count']
                })
        
        self.tool_wafer_summary = pd.DataFrame(tool_data)
        
        # Pivot to have tools as columns for easier reading
        pivot_summary = self.tool_wafer_summary.pivot_table(
            index='WaferID',
            columns='Tool',
            values=['Mean_Thickness', 'Std_Thickness', 'Count'],
            fill_value=0
        ).round(2)
        
        # Flatten column names
        pivot_summary.columns = [f'{col[1]}_{col[0]}' for col in pivot_summary.columns]
        pivot_summary = pivot_summary.reset_index()
        
        # Save to CSV
        Path('output').mkdir(exist_ok=True)
        self.tool_wafer_summary.to_csv('output/tool_wafer_summary.csv', index=False)
        pivot_summary.to_csv('output/tool_wafer_summary_pivot.csv', index=False)
        
        print(f"Tool-wafer summary saved with {len(self.tool_wafer_summary)} entries")
        
        return self.tool_wafer_summary, pivot_summary
    
    def create_contour_grid(self, x, y, z, grid_size=50):
        """Create a regular grid for contouring from scattered data"""
        # Define grid bounds with some padding
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.1  # 10% padding
        
        x_min -= x_range * padding
        x_max += x_range * padding  
        y_min -= y_range * padding
        y_max += y_range * padding
        
        # Create regular grid
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate data onto grid
        zi_grid = griddata(
            points=(x, y),
            values=z,
            xi=(xi_grid, yi_grid),
            method='cubic',
            fill_value=np.nan
        )
        
        return xi, yi, zi_grid
    
    def create_wafer_map_plot(self, comparison_type='DMT-TFK', save_plots=True):
        """Create wafer map showing contour plots instead of scatter plots"""
        
        # Filter data for the requested comparison
        if comparison_type == 'DMT-TFK' and self.dmt_tfk_data is not None:
            data = self.dmt_tfk_data.copy()
            x_col, y_col = 'DMT_X_mm', 'DMT_Y_mm'
            title_prefix = "DMT-TFK"
            tool1_col, tool2_col = 'DMT_Thickness', 'TFK_Thickness'
            tool1_name, tool2_name = 'DMT', 'TFK'
        elif comparison_type == 'BTM-DMT' and self.btm_dmt_data is not None:
            data = self.btm_dmt_data.copy()
            x_col, y_col = 'BTM_X_mm', 'BTM_Y_mm'
            title_prefix = "BTM-DMT"
            tool1_col, tool2_col = 'BTM_Thickness', 'DMT_Thickness'
            tool1_name, tool2_name = 'BTM', 'DMT'
        elif comparison_type == 'BTM-TFK' and self.btm_tfk_data is not None:
            data = self.btm_tfk_data.copy()
            x_col, y_col = 'BTM_X_mm', 'BTM_Y_mm'
            title_prefix = "BTM-TFK"
            tool1_col, tool2_col = 'BTM_Thickness', 'TFK_Thickness'
            tool1_name, tool2_name = 'BTM', 'TFK'
        else:
            print(f"No data available for {comparison_type} comparison")
            return None
        
        # Calculate absolute delta for highlighting
        data['Abs_Delta'] = abs(data['Thickness_Delta'])
        
        # Calculate average offset between tools
        avg_offset = data['Thickness_Delta'].mean()
        data['Corrected_Delta'] = data['Thickness_Delta'] - avg_offset
        data['Abs_Corrected_Delta'] = abs(data['Corrected_Delta'])
        
        # Find locations with highest deltas (top 10%)
        threshold = data['Abs_Delta'].quantile(0.9)
        data['High_Delta'] = data['Abs_Delta'] >= threshold
        
        print(f"\n{title_prefix} Average Offset: {avg_offset:.2f} Å ({tool1_name} - {tool2_name})")
        if avg_offset > 0:
            print(f"  → {tool1_name} measures {avg_offset:.2f} Å higher on average")
        else:
            print(f"  → {tool2_name} measures {abs(avg_offset):.2f} Å higher on average")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'{tool1_name} Mean Thickness Contour Map ({tool1_name} coordinates)',
                f'{tool2_name} Mean Thickness Contour Map ({tool2_name} coordinates)',
                f'{title_prefix} Delta Distribution',
                f'{title_prefix} High Delta Locations',
                f'{title_prefix} Offset-Corrected Delta Contour Map',
                f'Offset: {avg_offset:.2f} Å'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Create grids for contour plots
        x_data, y_data = data[x_col], data[y_col]
        
        # Plot 1: Tool 1 mean thickness contour map
        # Aggregate duplicate coordinates for all tools to avoid interpolation artifacts
        if tool1_name == 'TFK':
            tool1_x_data = data['TFK_X_mm'] if 'TFK_X_mm' in data.columns else x_data
            tool1_y_data = data['TFK_Y_mm'] if 'TFK_Y_mm' in data.columns else y_data
        elif tool1_name == 'BTM':
            tool1_x_data = data['BTM_X_mm'] if 'BTM_X_mm' in data.columns else x_data
            tool1_y_data = data['BTM_Y_mm'] if 'BTM_Y_mm' in data.columns else y_data
        else:  # DMT or other
            tool1_x_data, tool1_y_data = x_data, y_data
            
        # Aggregate duplicate coordinates by averaging thickness values
        tool1_df = pd.DataFrame({
            'x': tool1_x_data,
            'y': tool1_y_data, 
            'thickness': data[tool1_col]
        })
        tool1_aggregated = tool1_df.groupby(['x', 'y'])['thickness'].mean().reset_index()
        tool1_x_data = tool1_aggregated['x']
        tool1_y_data = tool1_aggregated['y']
        tool1_thickness_data = tool1_aggregated['thickness']
        print(f"  {tool1_name} coordinates aggregated: {len(data)} -> {len(tool1_aggregated)} unique locations")
            
        xi, yi, zi_tool1 = self.create_contour_grid(tool1_x_data, tool1_y_data, tool1_thickness_data)
        fig.add_trace(
            go.Contour(
                x=xi,
                y=yi,
                z=zi_tool1,
                colorscale='Viridis',
                colorbar=dict(title=f"{tool1_name} Thickness (Å)", x=0.31, len=0.4, y=0.775),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=8, color='white')
                ),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             f'{tool1_name} Thickness: %{{z:.1f}} Å<extra></extra>',
                name=f'{tool1_name} Thickness',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add scatter points overlay for reference
        fig.add_trace(
            go.Scatter(
                x=tool1_x_data,
                y=tool1_y_data,
                mode='markers',
                marker=dict(size=3, color='white', opacity=0.7, line=dict(width=0.5, color='black')),
                text=tool1_thickness_data.round(1),
                hovertemplate=f'<b>Measurement Point</b><br>' +
                             f'X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm<br>' +
                             f'{tool1_name} Thickness: %{{text}} Å<extra></extra>',
                name='Measurement Points',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: Tool 2 mean thickness contour map
        # Aggregate duplicate coordinates for all tools to avoid interpolation artifacts
        if tool2_name == 'TFK':
            tool2_x_data = data['TFK_X_mm'] if 'TFK_X_mm' in data.columns else x_data
            tool2_y_data = data['TFK_Y_mm'] if 'TFK_Y_mm' in data.columns else y_data
        elif tool2_name == 'BTM':
            tool2_x_data = data['BTM_X_mm'] if 'BTM_X_mm' in data.columns else x_data
            tool2_y_data = data['BTM_Y_mm'] if 'BTM_Y_mm' in data.columns else y_data
        else:  # DMT or other
            tool2_x_data, tool2_y_data = x_data, y_data
            
        # Aggregate duplicate coordinates by averaging thickness values
        tool2_df = pd.DataFrame({
            'x': tool2_x_data,
            'y': tool2_y_data, 
            'thickness': data[tool2_col]
        })
        tool2_aggregated = tool2_df.groupby(['x', 'y'])['thickness'].mean().reset_index()
        tool2_x_data = tool2_aggregated['x']
        tool2_y_data = tool2_aggregated['y']
        tool2_thickness_data = tool2_aggregated['thickness']
        print(f"  {tool2_name} coordinates aggregated: {len(data)} -> {len(tool2_aggregated)} unique locations")
            
        xi, yi, zi_tool2 = self.create_contour_grid(tool2_x_data, tool2_y_data, tool2_thickness_data)
        fig.add_trace(
            go.Contour(
                x=xi,
                y=yi,
                z=zi_tool2,
                colorscale='Plasma',
                colorbar=dict(title=f"{tool2_name} Thickness (Å)", x=0.64, len=0.4, y=0.775),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=8, color='white')
                ),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             f'{tool2_name} Thickness: %{{z:.1f}} Å<extra></extra>',
                name=f'{tool2_name} Thickness',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add scatter points overlay for reference
        fig.add_trace(
            go.Scatter(
                x=tool2_x_data,
                y=tool2_y_data,
                mode='markers',
                marker=dict(size=3, color='white', opacity=0.7, line=dict(width=0.5, color='black')),
                text=tool2_thickness_data.round(1),
                hovertemplate=f'<b>Measurement Point</b><br>' +
                             f'X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm<br>' +
                             f'{tool2_name} Thickness: %{{text}} Å<extra></extra>',
                name='Measurement Points',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Delta distribution histogram
        fig.add_trace(
            go.Histogram(
                x=data['Thickness_Delta'],
                nbinsx=50,
                name='Delta Distribution',
                marker_color='lightblue',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Plot 4: High delta locations contour map
        xi, yi, zi_abs_delta = self.create_contour_grid(x_data, y_data, data['Abs_Delta'])
        fig.add_trace(
            go.Contour(
                x=xi,
                y=yi,
                z=zi_abs_delta,
                colorscale='Reds',
                colorbar=dict(title="Absolute Delta (Å)", x=0.31, len=0.4, y=0.225),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=8, color='white')
                ),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             'Absolute Delta: %{z:.1f} Å<extra></extra>',
                name='Absolute Delta Map',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Overlay high delta points
        high_delta_points = data[data['High_Delta']]
        if len(high_delta_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=high_delta_points[x_col],
                    y=high_delta_points[y_col],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='yellow',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    text=high_delta_points['Thickness_Delta'].round(1),
                    hovertemplate=f'<b>HIGH DELTA LOCATION</b><br>' +
                                 f'X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm<br>' +
                                 'Delta: %{text} Å<extra></extra>',
                    name=f'High Delta (|Δ|>{threshold:.1f} Å)'
                ),
                row=2, col=1
            )
        
        # Plot 5: Offset-corrected delta contour map
        # Aggregate duplicate coordinates by averaging corrected delta values
        corrected_delta_df = pd.DataFrame({
            'x': x_data,
            'y': y_data,
            'corrected_delta': data['Corrected_Delta']
        })
        corrected_delta_aggregated = corrected_delta_df.groupby(['x', 'y'])['corrected_delta'].mean().reset_index()
        corrected_x_data = corrected_delta_aggregated['x']
        corrected_y_data = corrected_delta_aggregated['y']
        corrected_delta_data = corrected_delta_aggregated['corrected_delta']
        print(f"  Corrected delta coordinates aggregated: {len(data)} -> {len(corrected_delta_aggregated)} unique locations")
        
        xi, yi, zi_corrected = self.create_contour_grid(corrected_x_data, corrected_y_data, corrected_delta_data)
        
        # Calculate symmetric range around zero for proper positive/negative visualization
        max_abs_corrected = max(abs(corrected_delta_data.min()), abs(corrected_delta_data.max()))
        
        fig.add_trace(
            go.Contour(
                x=xi,
                y=yi,
                z=zi_corrected,
                colorscale='RdBu_r',
                zmid=0,  # Center the colorscale at zero
                zmin=-max_abs_corrected,  # Symmetric range
                zmax=max_abs_corrected,   # Symmetric range
                colorbar=dict(
                    title="Corrected Delta (Å)<br><span style='font-size:8px'>Red=Positive<br>Blue=Negative</span>", 
                    x=0.97, len=0.4, y=0.225
                ),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=8, color='white'),
                    start=-max_abs_corrected,
                    end=max_abs_corrected,
                    size=max_abs_corrected/5  # Create ~10 contour levels
                ),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             'Corrected Delta: %{z:.1f} Å<extra></extra>',
                name='Offset Corrected',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add scatter points overlay for reference
        fig.add_trace(
            go.Scatter(
                x=corrected_x_data,
                y=corrected_y_data,
                mode='markers',
                marker=dict(size=3, color='white', opacity=0.7, line=dict(width=0.5, color='black')),
                text=corrected_delta_data.round(1),
                hovertemplate=f'<b>Measurement Point</b><br>' +
                             f'X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm<br>' +
                             'Corrected Delta: %{text} Å<extra></extra>',
                name='Measurement Points',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Plot 6: Offset correction statistics text
        corrected_std = data['Corrected_Delta'].std()
        original_std = data['Thickness_Delta'].std()
        
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=[f'<b>Offset Correction Analysis</b><br><br>' +
                     f'Average Offset: {avg_offset:.2f} Å<br>' +
                     f'Original Std Dev: {original_std:.2f} Å<br>' +
                     f'Corrected Std Dev: {corrected_std:.2f} Å<br>' +
                     f'Improvement: {((original_std-corrected_std)/original_std*100):.1f}%<br><br>' +
                     f'After offset correction:<br>' +
                     f'• Mean delta ≈ 0.00 Å<br>' +
                     f'• Reduces systematic bias<br>' +
                     f'• Shows remaining variation<br><br>' +
                     f'<b>Contour Plot Features:</b><br>' +
                     f'• Smooth interpolation between points<br>' +
                     f'• Shows spatial trends clearly<br>' +
                     f'• White dots = actual measurements<br>' +
                     f'• Each thickness map uses its own coordinate system<br>' +
                     f'• TFK maps use TFK coordinates for consistency'],
                textfont=dict(size=10),
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_xaxes(title="X Position (mm)", row=1, col=1)
        fig.update_yaxes(title="Y Position (mm)", row=1, col=1)
        fig.update_xaxes(title="X Position (mm)", row=1, col=2)
        fig.update_yaxes(title="Y Position (mm)", row=1, col=2)
        fig.update_xaxes(title="Delta (Å)", row=1, col=3)
        fig.update_yaxes(title="Count", row=1, col=3)
        fig.update_xaxes(title="X Position (mm)", row=2, col=1)
        fig.update_yaxes(title="Y Position (mm)", row=2, col=1)
        fig.update_xaxes(title="X Position (mm)", row=2, col=2)
        fig.update_yaxes(title="Y Position (mm)", row=2, col=2)
        
        # Hide axes for text plot
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=3)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=3)
        
        # Set equal aspect ratio for wafer maps
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
        
        fig.update_layout(
            height=800,
            width=1400,
            title=f'{title_prefix} Thickness and Delta Analysis - Contour Maps (Avg Offset: {avg_offset:.2f} Å)',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        if save_plots:
            Path('output').mkdir(exist_ok=True)
            fig.write_html(f'output/{comparison_type.lower().replace("-", "_")}_contour_map.html')
            print(f"Contour map saved as output/{comparison_type.lower().replace('-', '_')}_contour_map.html")
        
        return fig
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        
        report = []
        report.append("="*80)
        report.append("WAFER MAP ANALYSIS SUMMARY REPORT - CONTOUR VERSION")
        report.append("="*80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append("-" * 40)
        if self.dmt_tfk_data is not None:
            report.append(f"DMT-TFK Comparisons: {len(self.dmt_tfk_data)} matched pairs")
            report.append(f"  Wafers: {', '.join(sorted(self.dmt_tfk_data['WaferID'].unique()))}")
        if self.btm_dmt_data is not None:
            report.append(f"BTM-DMT Comparisons: {len(self.btm_dmt_data)} matched pairs")
            report.append(f"  Wafers: {', '.join(sorted(self.btm_dmt_data['WaferID'].unique()))}")
        if self.btm_tfk_data is not None:
            report.append(f"BTM-TFK Comparisons: {len(self.btm_tfk_data)} matched pairs")
            report.append(f"  Wafers: {', '.join(sorted(self.btm_tfk_data['WaferID'].unique()))}")
        report.append("")
        
        # Location statistics
        if self.location_summary is not None:
            report.append("LOCATION ANALYSIS:")
            report.append("-" * 40)
            report.append(f"Total measurement locations analyzed: {len(self.location_summary)}")
            
            # Highest delta locations by comparison type
            for comp_type in self.location_summary['Comparison_Type'].unique():
                subset = self.location_summary[self.location_summary['Comparison_Type'] == comp_type]
                highest = subset.nlargest(5, 'Abs_Delta')
                report.append(f"\n{comp_type} - Top 5 Highest Delta Locations:")
                for _, row in highest.iterrows():
                    report.append(f"  ({row['X_mm']:6.1f}, {row['Y_mm']:6.1f}) mm: {row['Delta_Mean']:7.1f} ± {row['Delta_Std']:5.1f} Å")
        report.append("")
        
        # Tool statistics
        if self.tool_wafer_summary is not None:
            report.append("TOOL STATISTICS BY WAFER:")
            report.append("-" * 40)
            
            # Group by tool and wafer
            for tool in sorted(self.tool_wafer_summary['Tool'].unique()):
                tool_data = self.tool_wafer_summary[self.tool_wafer_summary['Tool'] == tool]
                report.append(f"\n{tool} Tool:")
                for _, row in tool_data.iterrows():
                    report.append(f"  {row['WaferID']}: {row['Mean_Thickness']:7.1f} ± {row['Std_Thickness']:5.1f} Å ({row['Count']} points)")
        
        report.append("")
        report.append("CONTOUR PLOT FEATURES:")
        report.append("-" * 40)
        report.append("• Smooth interpolated surfaces showing spatial trends")
        report.append("• White dots overlay showing actual measurement locations")
        report.append("• Contour lines with value labels for precise reading")
        report.append("• Color scales optimized for each measurement type")
        report.append("• High delta locations marked with yellow stars")
        report.append("")
        report.append("COORDINATE SYSTEM NOTES:")
        report.append("-" * 40)
        report.append("• Each thickness map uses its tool's native coordinate system")
        report.append("• TFK thickness maps now use TFK coordinates for consistency")
        report.append("• Different tool comparisons may include different measurement subsets")
        report.append("• This is expected due to different matching criteria and algorithms")
        report.append("")
        report.append("="*80)
        
        # Save report
        Path('output').mkdir(exist_ok=True)
        with open('output/wafer_map_contour_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Summary report saved as output/wafer_map_contour_analysis_report.txt")
        return '\n'.join(report)

def main():
    """Run the complete wafer map analysis with contour plots"""
    
    print("Starting Wafer Map Analysis with Contour Plots...")
    analyzer = WaferMapAnalysis()
    
    # Load all data
    analyzer.load_all_data()
    
    # Create location summary table
    print("\nCreating location-based delta summary...")
    analyzer.create_location_summary_table()
    
    # Create tool-wafer summary table
    print("\nCreating tool-by-wafer summary...")
    analyzer.create_tool_wafer_summary()
    
    # Create wafer maps for each comparison type
    print("\nCreating contour wafer maps...")
    
    if analyzer.dmt_tfk_data is not None:
        print("  - Creating DMT-TFK contour map...")
        analyzer.create_wafer_map_plot('DMT-TFK')
    
    if analyzer.btm_dmt_data is not None:
        print("  - Creating BTM-DMT contour map...")
        analyzer.create_wafer_map_plot('BTM-DMT')
        
    if analyzer.btm_tfk_data is not None:
        print("  - Creating BTM-TFK contour map...")
        analyzer.create_wafer_map_plot('BTM-TFK')
    
    # Create summary report
    print("\nGenerating summary report...")
    analyzer.create_summary_report()
    
    print("\nWafer Map Contour Analysis Complete!")
    print("\nGenerated files:")
    print("  - output/location_delta_summary.csv            (X,Y location delta summary)")
    print("  - output/tool_wafer_summary.csv               (Tool statistics by wafer)")
    print("  - output/tool_wafer_summary_pivot.csv         (Tool statistics - pivot format)")
    print("  - output/*_contour_map.html                   (Interactive contour maps)")
    print("  - output/wafer_map_contour_analysis_report.txt (Summary report)")

if __name__ == "__main__":
    main()