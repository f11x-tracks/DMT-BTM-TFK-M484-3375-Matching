#!/usr/bin/env python3
"""
Comprehensive Wafer Map Analysis
Creates wafer maps, location summaries, and tool comparison tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
            self.dmt_tfk_data = pd.read_csv('matched_thickness_data.csv')
            print(f"Loaded {len(self.dmt_tfk_data)} DMT-TFK matched pairs")
        except FileNotFoundError:
            print("Warning: matched_thickness_data.csv not found")
            
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
        self.location_summary.to_csv('location_delta_summary.csv', index=False)
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
        self.tool_wafer_summary.to_csv('tool_wafer_summary.csv', index=False)
        pivot_summary.to_csv('tool_wafer_summary_pivot.csv', index=False)
        
        print(f"Tool-wafer summary saved with {len(self.tool_wafer_summary)} entries")
        
        return self.tool_wafer_summary, pivot_summary
    
    def create_wafer_map_plot(self, comparison_type='DMT-TFK', save_plots=True):
        """Create wafer map showing delta values with highlighting of highest deltas"""
        
        # Filter data for the requested comparison
        if comparison_type == 'DMT-TFK' and self.dmt_tfk_data is not None:
            data = self.dmt_tfk_data.copy()
            x_col, y_col = 'DMT_X_mm', 'DMT_Y_mm'
            title_prefix = "DMT-TFK"
        elif comparison_type == 'BTM-DMT' and self.btm_dmt_data is not None:
            data = self.btm_dmt_data.copy()
            x_col, y_col = 'BTM_X_mm', 'BTM_Y_mm'
            title_prefix = "BTM-DMT"
        elif comparison_type == 'BTM-TFK' and self.btm_tfk_data is not None:
            data = self.btm_tfk_data.copy()
            x_col, y_col = 'BTM_X_mm', 'BTM_Y_mm'
            title_prefix = "BTM-TFK"
        else:
            print(f"No data available for {comparison_type} comparison")
            return None
        
        # Calculate absolute delta for highlighting
        data['Abs_Delta'] = abs(data['Thickness_Delta'])
        
        # Find locations with highest deltas (top 10%)
        threshold = data['Abs_Delta'].quantile(0.9)
        data['High_Delta'] = data['Abs_Delta'] >= threshold
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{title_prefix} Delta Wafer Map',
                f'{title_prefix} Absolute Delta Map',
                f'{title_prefix} Delta Distribution',
                f'{title_prefix} High Delta Locations'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Delta wafer map (signed delta)
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data['Thickness_Delta'],
                    colorscale='RdBu_r',
                    colorbar=dict(title="Delta (Å)", x=0.46),
                    line=dict(width=1, color='black')
                ),
                text=data['Thickness_Delta'].round(1),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             'Delta: %{text} Å<extra></extra>',
                name='Delta Map'
            ),
            row=1, col=1
        )
        
        # Plot 2: Absolute delta map
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data['Abs_Delta'],
                    colorscale='Reds',
                    colorbar=dict(title="Abs Delta (Å)", x=1.02),
                    line=dict(width=1, color='black')
                ),
                text=data['Abs_Delta'].round(1),
                hovertemplate=f'<b>X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm</b><br>' +
                             'Abs Delta: %{text} Å<extra></extra>',
                name='Abs Delta Map'
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
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Plot 4: Highlight highest delta locations
        normal_points = data[~data['High_Delta']]
        high_delta_points = data[data['High_Delta']]
        
        # Normal points
        fig.add_trace(
            go.Scatter(
                x=normal_points[x_col],
                y=normal_points[y_col],
                mode='markers',
                marker=dict(size=6, color='lightgray', opacity=0.6),
                name='Normal Delta',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # High delta points
        fig.add_trace(
            go.Scatter(
                x=high_delta_points[x_col],
                y=high_delta_points[y_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color=high_delta_points['Abs_Delta'],
                    colorscale='Reds',
                    line=dict(width=2, color='black')
                ),
                text=high_delta_points['Abs_Delta'].round(1),
                hovertemplate=f'<b>HIGH DELTA LOCATION</b><br>' +
                             f'X: %{{x:.1f}} mm, Y: %{{y:.1f}} mm<br>' +
                             'Abs Delta: %{text} Å<extra></extra>',
                name=f'High Delta (>{threshold:.1f} Å)'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title="X Position (mm)", row=1, col=1)
        fig.update_yaxes(title="Y Position (mm)", row=1, col=1)
        fig.update_xaxes(title="X Position (mm)", row=1, col=2)
        fig.update_yaxes(title="Y Position (mm)", row=1, col=2)
        fig.update_xaxes(title="Delta (Å)", row=2, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        fig.update_xaxes(title="X Position (mm)", row=2, col=2)
        fig.update_yaxes(title="Y Position (mm)", row=2, col=2)
        
        # Set equal aspect ratio for wafer maps
        for row in [1, 2]:
            for col in [1, 2]:
                if row == 2 and col == 1:  # Skip histogram
                    continue
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
        
        fig.update_layout(
            height=800,
            title=f'{title_prefix} Thickness Comparison - Wafer Map Analysis',
            showlegend=True
        )
        
        if save_plots:
            fig.write_html(f'{comparison_type.lower().replace("-", "_")}_wafer_map.html')
            print(f"Wafer map saved as {comparison_type.lower().replace('-', '_')}_wafer_map.html")
        
        return fig
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        
        report = []
        report.append("="*80)
        report.append("WAFER MAP ANALYSIS SUMMARY REPORT")
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
        report.append("="*80)
        
        # Save report
        with open('wafer_map_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Summary report saved as wafer_map_analysis_report.txt")
        return '\n'.join(report)

def main():
    """Run the complete wafer map analysis"""
    
    print("Starting Wafer Map Analysis...")
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
    print("\nCreating wafer maps...")
    
    if analyzer.dmt_tfk_data is not None:
        print("  - Creating DMT-TFK wafer map...")
        analyzer.create_wafer_map_plot('DMT-TFK')
    
    if analyzer.btm_dmt_data is not None:
        print("  - Creating BTM-DMT wafer map...")
        analyzer.create_wafer_map_plot('BTM-DMT')
        
    if analyzer.btm_tfk_data is not None:
        print("  - Creating BTM-TFK wafer map...")
        analyzer.create_wafer_map_plot('BTM-TFK')
    
    # Create summary report
    print("\nGenerating summary report...")
    analyzer.create_summary_report()
    
    print("\nWafer Map Analysis Complete!")
    print("\nGenerated files:")
    print("  - location_delta_summary.csv          (X,Y location delta summary)")
    print("  - tool_wafer_summary.csv             (Tool statistics by wafer)")
    print("  - tool_wafer_summary_pivot.csv       (Tool statistics - pivot format)")
    print("  - *_wafer_map.html                   (Interactive wafer maps)")
    print("  - wafer_map_analysis_report.txt      (Comprehensive summary report)")

if __name__ == "__main__":
    main()