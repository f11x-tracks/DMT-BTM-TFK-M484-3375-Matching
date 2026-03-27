#!/usr/bin/env python3
"""
Comprehensive Wafer Map Analysis - Radial Spline Plot Version
Creates radial spline plots showing thickness vs radius, location summaries, and tool comparison tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline, griddata
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')

class WaferMapAnalysisRadial:
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
    
    def calculate_radius(self, x, y):
        """Calculate radius from wafer center (0,0)"""
        return np.sqrt(x**2 + y**2)
    
    def create_radial_bins(self, radius_data, values, bin_size=5.0):
        """Create radial bins and calculate statistics"""
        max_radius = min(150.0, radius_data.max())  # Cap at 150mm
        bin_edges = np.arange(0, max_radius + bin_size, bin_size)
        
        # Calculate binned statistics
        bin_means, _, _ = binned_statistic(radius_data, values, statistic='mean', bins=bin_edges)
        bin_stds, _, _ = binned_statistic(radius_data, values, statistic='std', bins=bin_edges)
        bin_counts, _, _ = binned_statistic(radius_data, values, statistic='count', bins=bin_edges)
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove bins with no data
        valid_bins = ~np.isnan(bin_means)
        bin_centers = bin_centers[valid_bins]
        bin_means = bin_means[valid_bins]
        bin_stds = bin_stds[valid_bins]
        bin_counts = bin_counts[valid_bins]
        
        return bin_centers, bin_means, bin_stds, bin_counts
    
    def create_spline_fit(self, radius_data, values, max_radius=150.0):
        """Create spline fit for radial data"""
        # Filter data within radius range
        valid_mask = radius_data <= max_radius
        radius_filtered = radius_data[valid_mask]
        values_filtered = values[valid_mask]
        
        if len(radius_filtered) < 4:  # Need at least 4 points for spline
            return None, None, None
        
        try:
            # Create spline with smoothing
            spline = UnivariateSpline(radius_filtered, values_filtered, s=len(radius_filtered))
            
            # Generate smooth curve
            radius_smooth = np.linspace(0, max_radius, 300)
            values_smooth = spline(radius_smooth)
            
            return radius_smooth, values_smooth, spline
        except:
            return None, None, None
    
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
        self.location_summary.to_csv('output/location_delta_summary_radial.csv', index=False)
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
        self.tool_wafer_summary.to_csv('output/tool_wafer_summary_radial.csv', index=False)
        pivot_summary.to_csv('output/tool_wafer_summary_pivot_radial.csv', index=False)
        
        print(f"Tool-wafer summary saved with {len(self.tool_wafer_summary)} entries")
        
        return self.tool_wafer_summary, pivot_summary
    
    def create_radial_plot(self, comparison_type='DMT-TFK', save_plots=True):
        """Create radial spline plots showing thickness vs radius"""
        
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
        
        # Calculate radius for each measurement point
        data['Radius_mm'] = self.calculate_radius(data[x_col], data[y_col])
        
        # Calculate absolute delta for highlighting
        data['Abs_Delta'] = abs(data['Thickness_Delta'])
        
        # Calculate average offset between tools
        avg_offset = data['Thickness_Delta'].mean()
        data['Corrected_Delta'] = data['Thickness_Delta'] - avg_offset
        data['Abs_Corrected_Delta'] = abs(data['Corrected_Delta'])
        
        print(f"\n{title_prefix} Average Offset: {avg_offset:.2f} Å ({tool1_name} - {tool2_name})")
        if avg_offset > 0:
            print(f"  → {tool1_name} measures {avg_offset:.2f} Å higher on average")
        else:
            print(f"  → {tool2_name} measures {abs(avg_offset):.2f} Å higher on average")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'{tool1_name} Thickness vs Radius',
                f'{tool2_name} Thickness vs Radius',
                f'{title_prefix} Delta Distribution',
                f'Absolute Delta vs Radius',
                f'Offset-Corrected Delta vs Radius',
                f'Radial Statistics Summary'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Tool 1 thickness vs radius
        # Handle coordinate systems for TFK
        if tool1_name == 'TFK':
            tool1_x_data = data['TFK_X_mm'] if 'TFK_X_mm' in data.columns else data[x_col]
            tool1_y_data = data['TFK_Y_mm'] if 'TFK_Y_mm' in data.columns else data[y_col]
            tool1_radius = self.calculate_radius(tool1_x_data, tool1_y_data)
        else:
            tool1_radius = data['Radius_mm']
        
        # Create binned data for tool 1
        bin_centers_1, bin_means_1, bin_stds_1, bin_counts_1 = self.create_radial_bins(
            tool1_radius, data[tool1_col]
        )
        
        # Create spline fit for tool 1
        radius_smooth_1, values_smooth_1, spline_1 = self.create_spline_fit(
            tool1_radius, data[tool1_col]
        )
        
        # Add raw data points
        fig.add_trace(
            go.Scatter(
                x=tool1_radius,
                y=data[tool1_col],
                mode='markers',
                marker=dict(size=4, color='lightblue', opacity=0.6),
                name=f'{tool1_name} Raw Data',
                hovertemplate=f'<b>{tool1_name} Raw Data</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             f'{tool1_name} Thickness: %{{y:.1f}} Å<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add binned averages with error bars
        fig.add_trace(
            go.Scatter(
                x=bin_centers_1,
                y=bin_means_1,
                error_y=dict(type='data', array=bin_stds_1, visible=True),
                mode='markers',
                marker=dict(size=8, color='darkblue', symbol='diamond'),
                name=f'{tool1_name} Binned Avg',
                hovertemplate=f'<b>{tool1_name} Binned Average</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             f'{tool1_name} Thickness: %{{y:.1f}} ± %{{error_y.array:.1f}} Å<br>' +
                             'Points: %{customdata}<extra></extra>',
                customdata=bin_counts_1,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add spline fit
        if radius_smooth_1 is not None and values_smooth_1 is not None:
            fig.add_trace(
                go.Scatter(
                    x=radius_smooth_1,
                    y=values_smooth_1,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name=f'{tool1_name} Spline Fit',
                    hovertemplate=f'<b>{tool1_name} Spline Fit</b><br>' +
                                 'Radius: %{x:.1f} mm<br>' +
                                 f'{tool1_name} Thickness: %{{y:.1f}} Å<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Tool 2 thickness vs radius
        # Handle coordinate systems for TFK
        if tool2_name == 'TFK':
            tool2_x_data = data['TFK_X_mm'] if 'TFK_X_mm' in data.columns else data[x_col]
            tool2_y_data = data['TFK_Y_mm'] if 'TFK_Y_mm' in data.columns else data[y_col]
            tool2_radius = self.calculate_radius(tool2_x_data, tool2_y_data)
        else:
            tool2_radius = data['Radius_mm']
        
        # Create binned data for tool 2
        bin_centers_2, bin_means_2, bin_stds_2, bin_counts_2 = self.create_radial_bins(
            tool2_radius, data[tool2_col]
        )
        
        # Create spline fit for tool 2
        radius_smooth_2, values_smooth_2, spline_2 = self.create_spline_fit(
            tool2_radius, data[tool2_col]
        )
        
        # Add raw data points
        fig.add_trace(
            go.Scatter(
                x=tool2_radius,
                y=data[tool2_col],
                mode='markers',
                marker=dict(size=4, color='lightcoral', opacity=0.6),
                name=f'{tool2_name} Raw Data',
                hovertemplate=f'<b>{tool2_name} Raw Data</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             f'{tool2_name} Thickness: %{{y:.1f}} Å<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add binned averages with error bars
        fig.add_trace(
            go.Scatter(
                x=bin_centers_2,
                y=bin_means_2,
                error_y=dict(type='data', array=bin_stds_2, visible=True),
                mode='markers',
                marker=dict(size=8, color='darkred', symbol='diamond'),
                name=f'{tool2_name} Binned Avg',
                hovertemplate=f'<b>{tool2_name} Binned Average</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             f'{tool2_name} Thickness: %{{y:.1f}} ± %{{error_y.array:.1f}} Å<br>' +
                             'Points: %{customdata}<extra></extra>',
                customdata=bin_counts_2,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add spline fit
        if radius_smooth_2 is not None and values_smooth_2 is not None:
            fig.add_trace(
                go.Scatter(
                    x=radius_smooth_2,
                    y=values_smooth_2,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=f'{tool2_name} Spline Fit',
                    hovertemplate=f'<b>{tool2_name} Spline Fit</b><br>' +
                                 'Radius: %{x:.1f} mm<br>' +
                                 f'{tool2_name} Thickness: %{{y:.1f}} Å<extra></extra>',
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
        
        # Plot 4: Absolute delta vs radius
        bin_centers_abs, bin_means_abs, bin_stds_abs, bin_counts_abs = self.create_radial_bins(
            data['Radius_mm'], data['Abs_Delta']
        )
        
        # Raw absolute delta vs radius
        fig.add_trace(
            go.Scatter(
                x=data['Radius_mm'],
                y=data['Abs_Delta'],
                mode='markers',
                marker=dict(size=4, color='orange', opacity=0.6),
                name='Abs Delta Raw',
                hovertemplate='<b>Absolute Delta</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             'Abs Delta: %{y:.1f} Å<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Binned absolute delta
        fig.add_trace(
            go.Scatter(
                x=bin_centers_abs,
                y=bin_means_abs,
                error_y=dict(type='data', array=bin_stds_abs, visible=True),
                mode='markers+lines',
                marker=dict(size=8, color='darkorange', symbol='diamond'),
                line=dict(color='darkorange', width=2),
                name='Abs Delta Binned',
                hovertemplate='<b>Binned Absolute Delta</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             'Abs Delta: %{y:.1f} ± %{error_y.array:.1f} Å<br>' +
                             'Points: %{customdata}<extra></extra>',
                customdata=bin_counts_abs,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 5: Offset-corrected delta vs radius
        bin_centers_corr, bin_means_corr, bin_stds_corr, bin_counts_corr = self.create_radial_bins(
            data['Radius_mm'], data['Corrected_Delta']
        )
        
        # Create spline fit for corrected delta
        radius_smooth_corr, values_smooth_corr, spline_corr = self.create_spline_fit(
            data['Radius_mm'], data['Corrected_Delta']
        )
        
        # Raw corrected delta vs radius  
        fig.add_trace(
            go.Scatter(
                x=data['Radius_mm'],
                y=data['Corrected_Delta'],
                mode='markers',
                marker=dict(size=4, color='lightgreen', opacity=0.6),
                name='Corrected Delta Raw',
                hovertemplate='<b>Corrected Delta</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             'Corrected Delta: %{y:.1f} Å<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Binned corrected delta
        fig.add_trace(
            go.Scatter(
                x=bin_centers_corr,
                y=bin_means_corr,
                error_y=dict(type='data', array=bin_stds_corr, visible=True),
                mode='markers',
                marker=dict(size=8, color='darkgreen', symbol='diamond'),
                name='Corrected Delta Binned',
                hovertemplate='<b>Binned Corrected Delta</b><br>' +
                             'Radius: %{x:.1f} mm<br>' +
                             'Corrected Delta: %{y:.1f} ± %{error_y.array:.1f} Å<br>' +
                             'Points: %{customdata}<extra></extra>',
                customdata=bin_counts_corr,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add spline fit for corrected delta
        if radius_smooth_corr is not None and values_smooth_corr is not None:
            fig.add_trace(
                go.Scatter(
                    x=radius_smooth_corr,
                    y=values_smooth_corr,
                    mode='lines',
                    line=dict(color='darkgreen', width=3),
                    name='Corrected Delta Spline',
                    hovertemplate='<b>Corrected Delta Spline</b><br>' +
                                 'Radius: %{x:.1f} mm<br>' +
                                 'Corrected Delta: %{y:.1f} Å<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Add zero line for corrected delta
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        # Plot 6: Summary statistics text
        corrected_std = data['Corrected_Delta'].std()
        original_std = data['Thickness_Delta'].std()
        max_radius = data['Radius_mm'].max()
        
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=[f'<b>Radial Analysis Summary</b><br><br>' +
                     f'Average Offset: {avg_offset:.2f} Å<br>' +
                     f'Original Std Dev: {original_std:.2f} Å<br>' +
                     f'Corrected Std Dev: {corrected_std:.2f} Å<br>' +
                     f'Improvement: {((original_std-corrected_std)/original_std*100):.1f}%<br><br>' +
                     f'Measurement Range: 0 - {max_radius:.1f} mm<br>' +
                     f'Total Points: {len(data)}<br><br>' +
                     f'<b>Radial Plot Features:</b><br>' +
                     f'• Raw data points (light colored)<br>' +
                     f'• Binned averages with error bars (diamonds)<br>' +
                     f'• Spline fits showing radial trends<br>' +
                     f'• 5mm radial bins for statistics<br>' +
                     f'• Separate coordinate systems for each tool<br>' +
                     f'• Zero line for corrected delta reference'],
                textfont=dict(size=10),
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout for all subplots
        fig.update_xaxes(title="Radius (mm)", range=[0, 150], row=1, col=1)
        fig.update_yaxes(title=f"{tool1_name} Thickness (Å)", row=1, col=1)
        
        fig.update_xaxes(title="Radius (mm)", range=[0, 150], row=1, col=2)
        fig.update_yaxes(title=f"{tool2_name} Thickness (Å)", row=1, col=2)
        
        fig.update_xaxes(title="Delta (Å)", row=1, col=3)
        fig.update_yaxes(title="Count", row=1, col=3)
        
        fig.update_xaxes(title="Radius (mm)", range=[0, 150], row=2, col=1)
        fig.update_yaxes(title="Absolute Delta (Å)", row=2, col=1)
        
        fig.update_xaxes(title="Radius (mm)", range=[0, 150], row=2, col=2)
        fig.update_yaxes(title="Corrected Delta (Å)", row=2, col=2)
        
        # Hide axes for text plot
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=3)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=3)
        
        fig.update_layout(
            height=800,
            width=1400,
            title=f'{title_prefix} Radial Thickness and Delta Analysis - Spline Plots (Avg Offset: {avg_offset:.2f} Å)',
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
            fig.write_html(f'output/{comparison_type.lower().replace("-", "_")}_radial_plot.html')
            print(f"Radial plot saved as output/{comparison_type.lower().replace('-', '_')}_radial_plot.html")
        
        return fig
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        
        report = []
        report.append("="*80)
        report.append("WAFER MAP ANALYSIS SUMMARY REPORT - RADIAL SPLINE VERSION")
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
        report.append("RADIAL SPLINE PLOT FEATURES:")
        report.append("-" * 40)
        report.append("• X-axis: Radius from wafer center (0-150 mm)")
        report.append("• Y-axis: Thickness values or delta values")
        report.append("• Raw data points shown as light colored scatter")
        report.append("• Binned averages (5mm bins) with error bars as diamonds")
        report.append("• Spline fits showing smooth radial trends")
        report.append("• Separate plots for each tool and delta analysis")
        report.append("• Zero reference line for corrected delta plots")
        report.append("")
        report.append("COORDINATE SYSTEM NOTES:")
        report.append("-" * 40)
        report.append("• Each tool uses its own native coordinate system for radius calculation")
        report.append("• TFK plots use TFK coordinates when available")
        report.append("• Radius calculated as sqrt(x² + y²) from wafer center")
        report.append("• Different tool comparisons may show different radial coverage")
        report.append("")
        report.append("="*80)
        
        # Save report
        Path('output').mkdir(exist_ok=True)
        with open('output/wafer_map_radial_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Summary report saved as output/wafer_map_radial_analysis_report.txt")
        return '\n'.join(report)

def main():
    """Run the complete wafer map analysis with radial spline plots"""
    
    print("Starting Wafer Map Analysis with Radial Spline Plots...")
    analyzer = WaferMapAnalysisRadial()
    
    # Load all data
    analyzer.load_all_data()
    
    # Create location summary table
    print("\nCreating location-based delta summary...")
    analyzer.create_location_summary_table()
    
    # Create tool-wafer summary table
    print("\nCreating tool-by-wafer summary...")
    analyzer.create_tool_wafer_summary()
    
    # Create radial plots for each comparison type
    print("\nCreating radial spline plots...")
    
    if analyzer.dmt_tfk_data is not None:
        print("  - Creating DMT-TFK radial plot...")
        analyzer.create_radial_plot('DMT-TFK')
    
    if analyzer.btm_dmt_data is not None:
        print("  - Creating BTM-DMT radial plot...")
        analyzer.create_radial_plot('BTM-DMT')
        
    if analyzer.btm_tfk_data is not None:
        print("  - Creating BTM-TFK radial plot...")
        analyzer.create_radial_plot('BTM-TFK')
    
    # Create summary report
    print("\nGenerating summary report...")
    analyzer.create_summary_report()
    
    print("\nWafer Map Radial Analysis Complete!")
    print("\nGenerated files:")
    print("  - output/location_delta_summary_radial.csv        (X,Y location delta summary)")
    print("  - output/tool_wafer_summary_radial.csv           (Tool statistics by wafer)")
    print("  - output/tool_wafer_summary_pivot_radial.csv     (Tool statistics - pivot format)")
    print("  - output/*_radial_plot.html                      (Interactive radial spline plots)")
    print("  - output/wafer_map_radial_analysis_report.txt    (Summary report)")

if __name__ == "__main__":
    main()