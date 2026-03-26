"""
Thickness Comparison App: DMT vs TFK Measurement Tools

This application compares thickness measurements between DMT and TFK tools.
- DMT: thickness from 'Datum' with Label='Layer 1 Thickness', coordinates from XWaferLoc/YWaferLoc (in mm)
- TFK: thickness from 'Datum' with Label='T1', coordinates from XNative/YNative (divided by 10000000 for mm)
- Matching criteria: same WaferID and within 2mm distance for X,Y coordinates
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns

class ThicknessComparisonApp:
    def __init__(self, dmt_folder, tfk_folder, distance_threshold=4.0):
        self.dmt_folder = dmt_folder
        self.tfk_folder = tfk_folder
        self.distance_threshold = distance_threshold  # in mm
        self.dmt_data = None
        self.tfk_data = None
        self.matched_data = None
        
    def parse_xml_file(self, file_path, tool_type):
        """Parse XML file and extract relevant data"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find all DataRecord elements
        records = []
        
        if tool_type == 'DMT':
            # DMT structure
            for record in root.findall('.//DataRecord'):
                wafer_id = record.find('WaferID')
                label = record.find('Label')
                datum = record.find('Datum')
                x_loc = record.find('XWaferLoc')
                y_loc = record.find('YWaferLoc')
                
                if (wafer_id is not None and label is not None and 
                    datum is not None and x_loc is not None and y_loc is not None):
                    
                    if label.text == 'Layer 1 Thickness':
                        records.append({
                            'WaferID': wafer_id.text,
                            'X_mm': float(x_loc.text),
                            'Y_mm': float(y_loc.text),
                            'Thickness': float(datum.text),
                            'Tool': 'DMT',
                            'File': os.path.basename(file_path)
                        })
                        
        elif tool_type == 'TFK':
            # TFK structure
            for record in root.findall('.//DataRecord'):
                wafer_id = record.find('WaferID')
                label = record.find('Label')
                datum = record.find('Datum')
                x_native = record.find('XNative')
                y_native = record.find('YNative')
                
                if (wafer_id is not None and label is not None and 
                    datum is not None and x_native is not None and y_native is not None):
                    
                    if label.text == 'T1':
                        records.append({
                            'WaferID': wafer_id.text,
                            'X_mm': float(x_native.text) / 10000000,  # Convert to mm
                            'Y_mm': float(y_native.text) / 10000000,  # Convert to mm
                            'Thickness': float(datum.text),
                            'Tool': 'TFK',
                            'File': os.path.basename(file_path)
                        })
        
        return records
    
    def load_data(self):
        """Load all DMT and TFK data files"""
        print("Loading DMT data...")
        dmt_records = []
        dmt_files = glob.glob(os.path.join(self.dmt_folder, "*.xml"))
        
        for file_path in dmt_files:
            print(f"  Processing: {os.path.basename(file_path)}")
            try:
                records = self.parse_xml_file(file_path, 'DMT')
                dmt_records.extend(records)
                print(f"    Found {len(records)} thickness measurements")
            except Exception as e:
                print(f"    Error processing file: {e}")
        
        print(f"\nLoading TFK data...")
        tfk_records = []
        tfk_files = glob.glob(os.path.join(self.tfk_folder, "*.xml"))
        
        for file_path in tfk_files:
            print(f"  Processing: {os.path.basename(file_path)}")
            try:
                records = self.parse_xml_file(file_path, 'TFK')
                tfk_records.extend(records)
                print(f"    Found {len(records)} thickness measurements")
            except Exception as e:
                print(f"    Error processing file: {e}")
        
        self.dmt_data = pd.DataFrame(dmt_records)
        self.tfk_data = pd.DataFrame(tfk_records)
        
        print(f"\nData Loading Summary:")
        print(f"  DMT measurements: {len(self.dmt_data)}")
        print(f"  TFK measurements: {len(self.tfk_data)}")
        print(f"  DMT wafers: {self.dmt_data['WaferID'].nunique() if len(self.dmt_data) > 0 else 0}")
        print(f"  TFK wafers: {self.tfk_data['WaferID'].nunique() if len(self.tfk_data) > 0 else 0}")
    
    def find_matching_points(self):
        """Match measurements between DMT and TFK based on WaferID and coordinates"""
        if self.dmt_data is None or self.tfk_data is None:
            raise ValueError("Data must be loaded first")
        
        matched_pairs = []
        
        # Get common wafer IDs
        common_wafers = set(self.dmt_data['WaferID'].unique()) & set(self.tfk_data['WaferID'].unique())
        print(f"\nMatching measurements for {len(common_wafers)} common wafers...")
        
        for wafer_id in common_wafers:
            print(f"  Processing wafer: {wafer_id}")
            
            dmt_wafer = self.dmt_data[self.dmt_data['WaferID'] == wafer_id]
            tfk_wafer = self.tfk_data[self.tfk_data['WaferID'] == wafer_id]
            
            if len(dmt_wafer) == 0 or len(tfk_wafer) == 0:
                continue
            
            # Calculate distances between all DMT and TFK points for this wafer
            dmt_coords = dmt_wafer[['X_mm', 'Y_mm']].values
            tfk_coords = tfk_wafer[['X_mm', 'Y_mm']].values
            
            distances = cdist(dmt_coords, tfk_coords, metric='euclidean')
            
            # Find matches within distance threshold
            dmt_indices, tfk_indices = np.where(distances <= self.distance_threshold)
            
            for dmt_idx, tfk_idx in zip(dmt_indices, tfk_indices):
                dmt_point = dmt_wafer.iloc[dmt_idx]
                tfk_point = tfk_wafer.iloc[tfk_idx]
                distance = distances[dmt_idx, tfk_idx]
                
                matched_pairs.append({
                    'WaferID': wafer_id,
                    'DMT_X_mm': dmt_point['X_mm'],
                    'DMT_Y_mm': dmt_point['Y_mm'],
                    'TFK_X_mm': tfk_point['X_mm'],
                    'TFK_Y_mm': tfk_point['Y_mm'],
                    'DMT_Thickness': dmt_point['Thickness'],
                    'TFK_Thickness': tfk_point['Thickness'],
                    'Distance_mm': distance,
                    'Thickness_Delta': dmt_point['Thickness'] - tfk_point['Thickness'],
                    'DMT_File': dmt_point['File'],
                    'TFK_File': tfk_point['File']
                })
            
            print(f"    Found {len([m for m in matched_pairs if m['WaferID'] == wafer_id])} matched pairs")
        
        self.matched_data = pd.DataFrame(matched_pairs)
        print(f"\nTotal matched measurement pairs: {len(self.matched_data)}")
    
    def analyze_thickness_differences(self):
        """Analyze thickness differences between matched points"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for analysis")
            return
        
        print("\n" + "="*60)
        print("THICKNESS COMPARISON ANALYSIS")
        print("="*60)
        
        # Overall statistics
        thickness_deltas = self.matched_data['Thickness_Delta']
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total matched measurement pairs: {len(self.matched_data)}")
        print(f"  Mean thickness difference (DMT - TFK): {thickness_deltas.mean():.3f} Å")
        print(f"  Standard deviation of differences: {thickness_deltas.std():.3f} Å")
        print(f"  Minimum difference: {thickness_deltas.min():.3f} Å")
        print(f"  Maximum difference: {thickness_deltas.max():.3f} Å")
        print(f"  Median difference: {thickness_deltas.median():.3f} Å")
        print(f"  95% of differences within: ±{1.96 * thickness_deltas.std():.3f} Å")
        
        # Summary by wafer
        print(f"\nWAFER-BY-WAFER ANALYSIS:")
        wafer_summary = self.matched_data.groupby('WaferID').agg({
            'Thickness_Delta': ['count', 'mean', 'std', 'min', 'max'],
            'Distance_mm': 'mean',
            'DMT_Thickness': 'mean',
            'TFK_Thickness': 'mean'
        }).round(3)
        
        wafer_summary.columns = ['Count', 'Mean_Delta', 'Std_Delta', 'Min_Delta', 'Max_Delta', 
                               'Avg_Distance', 'DMT_Mean_Thickness', 'TFK_Mean_Thickness']
        
        print(wafer_summary)
        
        # Tool-level statistics
        print(f"\nTOOL-LEVEL STATISTICS:")
        dmt_thickness = self.matched_data['DMT_Thickness']
        tfk_thickness = self.matched_data['TFK_Thickness']
        
        print(f"  DMT Tool:")
        print(f"    Mean thickness: {dmt_thickness.mean():.3f} Å")
        print(f"    Standard deviation: {dmt_thickness.std():.3f} Å")
        print(f"    Range: {dmt_thickness.min():.3f} to {dmt_thickness.max():.3f} Å")
        
        print(f"  TFK Tool:")
        print(f"    Mean thickness: {tfk_thickness.mean():.3f} Å")
        print(f"    Standard deviation: {tfk_thickness.std():.3f} Å")
        print(f"    Range: {tfk_thickness.min():.3f} to {tfk_thickness.max():.3f} Å")
        
        # Delta between tool means and std devs
        mean_delta = dmt_thickness.mean() - tfk_thickness.mean()
        std_delta = dmt_thickness.std() - tfk_thickness.std()
        
        print(f"\nTOOL COMPARISON:")
        print(f"  Mean thickness delta (DMT - TFK): {mean_delta:.3f} Å")
        print(f"  Standard deviation delta (DMT - TFK): {std_delta:.3f} Å")
        
        # Distance analysis
        print(f"\nCOORDINATE MATCHING ANALYSIS:")
        distances = self.matched_data['Distance_mm']
        print(f"  Average distance between matched points: {distances.mean():.3f} mm")
        print(f"  Standard deviation of distances: {distances.std():.3f} mm")
        print(f"  Maximum distance: {distances.max():.3f} mm")
        print(f"  Matches within 0.5mm: {(distances <= 0.5).sum()} ({(distances <= 0.5).mean()*100:.1f}%)")
        print(f"  Matches within 1.0mm: {(distances <= 1.0).sum()} ({(distances <= 1.0).mean()*100:.1f}%)")
    
    def create_visualizations(self, output_dir=None):
        """Create visualizations of the analysis results"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for visualization")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up matplotlib parameters
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Thickness difference histogram
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DMT vs TFK Thickness Measurement Comparison', fontsize=16)
        
        # Subplot 1: Difference histogram
        axes[0,0].hist(self.matched_data['Thickness_Delta'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.matched_data['Thickness_Delta'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.matched_data["Thickness_Delta"].mean():.1f} Å')
        axes[0,0].set_xlabel('Thickness Difference (DMT - TFK) [Å]')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Thickness Differences')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Subplot 2: Scatter plot DMT vs TFK
        axes[0,1].scatter(self.matched_data['TFK_Thickness'], self.matched_data['DMT_Thickness'], 
                         alpha=0.6, s=30)
        min_val = min(self.matched_data['TFK_Thickness'].min(), self.matched_data['DMT_Thickness'].min())
        max_val = max(self.matched_data['TFK_Thickness'].max(), self.matched_data['DMT_Thickness'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
        axes[0,1].set_xlabel('TFK Thickness [Å]')
        axes[0,1].set_ylabel('DMT Thickness [Å]')
        axes[0,1].set_title('DMT vs TFK Thickness Correlation')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Subplot 3: Box plot by wafer
        wafer_data = [self.matched_data[self.matched_data['WaferID'] == wafer]['Thickness_Delta'].values 
                     for wafer in self.matched_data['WaferID'].unique()]
        wafer_labels = self.matched_data['WaferID'].unique()
        
        bp = axes[1,0].boxplot(wafer_data, labels=wafer_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1,0].set_xlabel('Wafer ID')
        axes[1,0].set_ylabel('Thickness Difference (DMT - TFK) [Å]')
        axes[1,0].set_title('Thickness Differences by Wafer')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        
        # Subplot 4: Distance vs thickness difference
        axes[1,1].scatter(self.matched_data['Distance_mm'], self.matched_data['Thickness_Delta'], 
                         alpha=0.6, s=30)
        axes[1,1].set_xlabel('Coordinate Matching Distance [mm]')
        axes[1,1].set_ylabel('Thickness Difference (DMT - TFK) [Å]')
        axes[1,1].set_title('Distance vs Thickness Difference')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'thickness_comparison_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        plt.show()
    
    def create_spatial_delta_plot(self, output_dir=None):
        """Create spatial visualization showing thickness deltas by X,Y coordinates"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for spatial visualization")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up matplotlib parameters
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        
        # Create figure with subplots for each wafer + overall
        wafers = sorted(self.matched_data['WaferID'].unique())
        n_wafers = len(wafers)
        
        # Create subplot layout: if we have multiple wafers, show them separately plus an overall
        if n_wafers == 1:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes = [axes] if n_wafers == 1 else axes
        else:
            # For multiple wafers: create grid layout
            cols = min(3, n_wafers + 1)  # +1 for overall plot
            rows = ((n_wafers + 1) + cols - 1) // cols  # Ceiling division
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            if rows == 1:
                axes = [axes] if isinstance(axes, plt.Axes) else axes
            else:
                axes = [ax for row in axes for ax in row] if hasattr(axes[0], '__iter__') else [axes]
        
        fig.suptitle('Thickness Delta (DMT - TFK) by Spatial Coordinates', fontsize=16, y=0.98)
        
        plot_idx = 0
        
        # Plot overall data first
        ax = axes[plot_idx] if len(axes) > 1 else axes
        
        # Use average coordinates for plotting (DMT and TFK coordinates are close)
        x_coords = (self.matched_data['DMT_X_mm'] + self.matched_data['TFK_X_mm']) / 2
        y_coords = (self.matched_data['DMT_Y_mm'] + self.matched_data['TFK_Y_mm']) / 2
        deltas = self.matched_data['Thickness_Delta']
        
        # Create scatter plot with color-coded deltas
        scatter = ax.scatter(x_coords, y_coords, c=deltas, cmap='RdBu_r', 
                            s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Thickness Delta (DMT - TFK) [Å]', rotation=270, labelpad=20)
        
        ax.set_xlabel('X Coordinate [mm]')
        ax.set_ylabel('Y Coordinate [mm]')
        if n_wafers == 1:
            ax.set_title(f'Overall - Wafer {wafers[0]}')
        else:
            ax.set_title(f'All Wafers Combined (n={len(self.matched_data)})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics annotation
        stats_text = f'Mean Δ: {deltas.mean():.1f} Å\nStd Δ: {deltas.std():.1f} Å\nRange: {deltas.min():.1f} to {deltas.max():.1f} Å'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_idx += 1
        
        # Plot individual wafers if we have multiple
        if n_wafers > 1:
            for i, wafer in enumerate(wafers):
                if plot_idx >= len(axes):
                    break
                    
                ax = axes[plot_idx]
                
                # Filter data for this wafer
                wafer_data = self.matched_data[self.matched_data['WaferID'] == wafer]
                
                x_coords_w = (wafer_data['DMT_X_mm'] + wafer_data['TFK_X_mm']) / 2
                y_coords_w = (wafer_data['DMT_Y_mm'] + wafer_data['TFK_Y_mm']) / 2
                deltas_w = wafer_data['Thickness_Delta']
                
                # Create scatter plot
                scatter = ax.scatter(x_coords_w, y_coords_w, c=deltas_w, cmap='RdBu_r', 
                                   s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Thickness Delta [Å]', rotation=270, labelpad=15)
                
                ax.set_xlabel('X Coordinate [mm]')
                ax.set_ylabel('Y Coordinate [mm]')
                ax.set_title(f'Wafer {wafer} (n={len(wafer_data)})')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add statistics annotation
                stats_text = f'Mean Δ: {deltas_w.mean():.1f} Å\nStd Δ: {deltas_w.std():.1f} Å\nRange: {deltas_w.min():.1f} to {deltas_w.max():.1f} Å'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plot_idx += 1
        
        # Hide any unused subplots
        while plot_idx < len(axes):
            axes[plot_idx].set_visible(False)
            plot_idx += 1
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'spatial_thickness_delta_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Spatial delta visualization saved to: {plot_path}")
        plt.show()
        
        # Also create a detailed analysis of high delta regions
        self._analyze_high_delta_regions()
    
    def create_averaged_wafer_map(self, output_dir=None):
        """Create wafer map showing averaged thickness deltas across all wafers by X,Y coordinates"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for averaged wafer map")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "="*60)
        print("CREATING AVERAGED WAFER MAP")
        print("="*60)
        
        # Calculate average coordinates and group by spatial bins
        x_coords = (self.matched_data['DMT_X_mm'] + self.matched_data['TFK_X_mm']) / 2
        y_coords = (self.matched_data['DMT_Y_mm'] + self.matched_data['TFK_Y_mm']) / 2
        deltas = self.matched_data['Thickness_Delta']
        
        # Create spatial bins (grid) - adjust bin size based on data density
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        # Use reasonable bin sizes (aim for ~20-30 bins across each dimension)
        x_bins = max(20, min(50, int(x_range / 2)))  # ~2mm bins, adjust as needed
        y_bins = max(20, min(50, int(y_range / 2)))
        
        # Create binned data
        # Calculate statistics for each spatial bin
        bin_means, x_edges, y_edges, bin_numbers = binned_statistic_2d(
            x_coords, y_coords, deltas, statistic='mean', bins=[x_bins, y_bins])
        
        bin_counts, _, _, _ = binned_statistic_2d(
            x_coords, y_coords, deltas, statistic='count', bins=[x_bins, y_bins])
        
        bin_stds, _, _, _ = binned_statistic_2d(
            x_coords, y_coords, deltas, statistic='std', bins=[x_bins, y_bins])
        
        print(f"Created {x_bins}x{y_bins} spatial grid")
        print(f"Total bins with data: {np.sum(bin_counts > 0)}")
        
        # Create the visualization
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Averaged Wafer Map - Thickness Delta Analysis Across All Wafers', fontsize=16)
        
        # Subplot 1: Mean delta map
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # Mask bins with no data
        masked_means = np.ma.masked_where(bin_counts.T <= 0, bin_means.T)
        
        im1 = axes[0,0].pcolormesh(X, Y, masked_means, cmap='RdBu_r', shading='auto')
        axes[0,0].set_xlabel('X Coordinate [mm]')
        axes[0,0].set_ylabel('Y Coordinate [mm]')
        axes[0,0].set_title('Mean Thickness Delta (DMT - TFK) [Å]')
        axes[0,0].set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(im1, ax=axes[0,0])
        cbar1.set_label('Delta [Å]', rotation=270, labelpad=15)
        
        # Subplot 2: Data count map
        masked_counts = np.ma.masked_where(bin_counts.T <= 0, bin_counts.T)
        im2 = axes[0,1].pcolormesh(X, Y, masked_counts, cmap='viridis', shading='auto')
        axes[0,1].set_xlabel('X Coordinate [mm]')
        axes[0,1].set_ylabel('Y Coordinate [mm]')
        axes[0,1].set_title('Number of Measurements per Location')
        axes[0,1].set_aspect('equal', adjustable='box')
        cbar2 = plt.colorbar(im2, ax=axes[0,1])
        cbar2.set_label('Count', rotation=270, labelpad=15)
        
        # Subplot 3: Standard deviation map
        masked_stds = np.ma.masked_where(bin_counts.T <= 1, bin_stds.T)  # Need at least 2 points for std
        im3 = axes[1,0].pcolormesh(X, Y, masked_stds, cmap='plasma', shading='auto')
        axes[1,0].set_xlabel('X Coordinate [mm]')
        axes[1,0].set_ylabel('Y Coordinate [mm]')
        axes[1,0].set_title('Standard Deviation of Deltas [Å]')
        axes[1,0].set_aspect('equal', adjustable='box')
        cbar3 = plt.colorbar(im3, ax=axes[1,0])
        cbar3.set_label('Std Dev [Å]', rotation=270, labelpad=15)
        
        # Subplot 4: Reliability map (inverse of std dev, with count weighting)
        # Higher values = more reliable (low std, high count)
        reliability = np.divide(bin_counts.T, bin_stds.T + 0.1, 
                              out=np.zeros_like(bin_counts.T), where=bin_stds.T > 0)
        masked_reliability = np.ma.masked_where(bin_counts.T <= 1, reliability)
        
        im4 = axes[1,1].pcolormesh(X, Y, masked_reliability, cmap='YlGnBu', shading='auto')
        axes[1,1].set_xlabel('X Coordinate [mm]')
        axes[1,1].set_ylabel('Y Coordinate [mm]')
        axes[1,1].set_title('Measurement Reliability (Count/StdDev)')
        axes[1,1].set_aspect('equal', adjustable='box')
        cbar4 = plt.colorbar(im4, ax=axes[1,1])
        cbar4.set_label('Reliability', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'averaged_wafer_map.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Averaged wafer map saved to: {plot_path}")
        plt.show()
        
        # Store results for ranking
        self.wafer_map_results = {
            'x_centers': x_centers,
            'y_centers': y_centers, 
            'bin_means': bin_means,
            'bin_counts': bin_counts,
            'bin_stds': bin_stds,
            'x_edges': x_edges,
            'y_edges': y_edges
        }
        
        return self.wafer_map_results
    
    def create_location_ranking(self, output_dir=None):
        """Create ranking of X,Y locations by thickness delta performance"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for location ranking")
            return
        
        if not hasattr(self, 'wafer_map_results') or self.wafer_map_results is None:
            print("Creating wafer map first...")
            self.create_averaged_wafer_map(output_dir)
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "="*60)
        print("LOCATION PERFORMANCE RANKING")
        print("="*60)
        
        results = self.wafer_map_results
        x_centers = results['x_centers']
        y_centers = results['y_centers']
        bin_means = results['bin_means']
        bin_counts = results['bin_counts']
        bin_stds = results['bin_stds']
        
        # Create ranking data
        ranking_data = []
        
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                if bin_counts[i,j] > 0:  # Only include locations with data
                    mean_delta = bin_means[i,j]
                    count = bin_counts[i,j]
                    std_delta = bin_stds[i,j] if not np.isnan(bin_stds[i,j]) else 0
                    
                    # Calculate performance metrics
                    abs_delta = abs(mean_delta)
                    reliability = count / (std_delta + 0.1)  # High count, low std = reliable
                    
                    ranking_data.append({
                        'X_mm': x_centers[i],
                        'Y_mm': y_centers[j],
                        'Mean_Delta': mean_delta,
                        'Abs_Delta': abs_delta,
                        'Std_Delta': std_delta,
                        'Count': count,
                        'Reliability': reliability
                    })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        if len(ranking_df) == 0:
            print("No valid locations found for ranking")
            return
        
        print(f"Analyzing {len(ranking_df)} spatial locations...")
        
        # Create different rankings
        rankings = {}
        
        # 1. Best Agreement (lowest absolute delta)
        rankings['Best_Agreement'] = ranking_df.nsmallest(10, 'Abs_Delta')[
            ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Count', 'Reliability']].round(3)
        
        # 2. Worst Agreement (highest absolute delta)
        rankings['Worst_Agreement'] = ranking_df.nlargest(10, 'Abs_Delta')[
            ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Count', 'Reliability']].round(3)
        
        # 3. Most Reliable (highest reliability score, with minimum count threshold)
        reliable_locations = ranking_df[ranking_df['Count'] >= 3]  # At least 3 measurements
        if len(reliable_locations) > 0:
            rankings['Most_Reliable'] = reliable_locations.nlargest(10, 'Reliability')[
                ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Count', 'Reliability']].round(3)
        
        # 4. Most Consistent (lowest standard deviation, with minimum count threshold)
        consistent_locations = ranking_df[(ranking_df['Count'] >= 3) & (ranking_df['Std_Delta'] > 0)]
        if len(consistent_locations) > 0:
            rankings['Most_Consistent'] = consistent_locations.nsmallest(10, 'Std_Delta')[
                ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Std_Delta', 'Count']].round(3)
        
        # 5. DMT Biased High (highest positive delta - DMT reads higher than TFK)
        rankings['DMT_Biased_High'] = ranking_df.nlargest(10, 'Mean_Delta')[
            ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Count', 'Reliability']].round(3)
        
        # 6. TFK Biased High (lowest/most negative delta - TFK reads higher than DMT)
        rankings['TFK_Biased_High'] = ranking_df.nsmallest(10, 'Mean_Delta')[
            ['X_mm', 'Y_mm', 'Mean_Delta', 'Abs_Delta', 'Count', 'Reliability']].round(3)
        
        # Print rankings
        for rank_name, rank_data in rankings.items():
            if len(rank_data) > 0:
                print(f"\n{rank_name.replace('_', ' ').upper()}:")
                print(rank_data.to_string(index=False))
        
        # Create rankings directory
        rankings_dir = os.path.join(output_dir, 'rankings')
        os.makedirs(rankings_dir, exist_ok=True)
        
        # Save rankings to CSV files
        for rank_name, rank_data in rankings.items():
            if len(rank_data) > 0:
                rank_file = os.path.join(rankings_dir, f'ranking_{rank_name.lower()}.csv')
                rank_data.to_csv(rank_file, index=False, encoding='utf-8')
                print(f"\n{rank_name} ranking saved to: {rank_file}")
        
        # Save complete location data
        complete_file = os.path.join(output_dir, 'complete_location_analysis.csv')
        ranking_df.round(3).to_csv(complete_file, index=False, encoding='utf-8')
        print(f"\nComplete location analysis saved to: {complete_file}")
        
        return rankings
    
    def create_enhanced_summary(self, output_dir=None):
        """Create comprehensive summary report with wafer map and ranking results"""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        summary_file = os.path.join(output_dir, 'enhanced_thickness_summary.txt')
        
        print("\n" + "="*60)
        print("CREATING ENHANCED SUMMARY REPORT")
        print("="*60)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED THICKNESS COMPARISON ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Date: March 26, 2026\n\n")
            
            if self.matched_data is not None and len(self.matched_data) > 0:
                thickness_deltas = self.matched_data['Thickness_Delta']
                dmt_thickness = self.matched_data['DMT_Thickness']
                tfk_thickness = self.matched_data['TFK_Thickness']
                
                # Basic statistics
                f.write("BASIC STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total matched measurement pairs: {len(self.matched_data)}\n")
                f.write(f"Unique wafers analyzed: {self.matched_data['WaferID'].nunique()}\n")
                f.write(f"Wafer IDs: {', '.join(sorted(self.matched_data['WaferID'].unique()))}\n\n")
                
                f.write(f"Mean thickness difference (DMT - TFK): {thickness_deltas.mean():.3f} ± {thickness_deltas.std():.3f} Å\n")
                f.write(f"Median thickness difference: {thickness_deltas.median():.3f} Å\n")
                f.write(f"Range: {thickness_deltas.min():.3f} to {thickness_deltas.max():.3f} Å\n")
                f.write(f"95% of differences within: ±{1.96 * thickness_deltas.std():.3f} Å\n\n")
                
                # Tool comparison
                f.write("TOOL COMPARISON\n")
                f.write("-" * 30 + "\n")
                f.write(f"DMT Tool - Mean: {dmt_thickness.mean():.3f} Å, Std: {dmt_thickness.std():.3f} Å\n")
                f.write(f"TFK Tool - Mean: {tfk_thickness.mean():.3f} Å, Std: {tfk_thickness.std():.3f} Å\n")
                f.write(f"Tool bias (DMT - TFK): {dmt_thickness.mean() - tfk_thickness.mean():.3f} Å\n")
                f.write(f"Precision difference (DMT std - TFK std): {dmt_thickness.std() - tfk_thickness.std():.3f} Å\n\n")
                
                # Coordinate matching
                distances = self.matched_data['Distance_mm']
                f.write("COORDINATE MATCHING QUALITY\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average matching distance: {distances.mean():.3f} ± {distances.std():.3f} mm\n")
                f.write(f"Matches within 0.5mm: {(distances <= 0.5).sum()}/{len(distances)} ({(distances <= 0.5).mean()*100:.1f}%)\n")
                f.write(f"Matches within 1.0mm: {(distances <= 1.0).sum()}/{len(distances)} ({(distances <= 1.0).mean()*100:.1f}%)\n")
                f.write(f"Maximum matching distance: {distances.max():.3f} mm\n\n")
                
                # Spatial analysis summary
                if hasattr(self, 'wafer_map_results') and self.wafer_map_results is not None:
                    results = self.wafer_map_results
                    valid_bins = np.sum(results['bin_counts'] > 0)
                    total_bins = results['bin_counts'].size
                    
                    f.write("SPATIAL ANALYSIS SUMMARY\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Spatial grid resolution: {results['bin_means'].shape[0]}x{results['bin_means'].shape[1]}\n")
                    f.write(f"Bins with data: {valid_bins}/{total_bins} ({valid_bins/total_bins*100:.1f}%)\n")
                    f.write(f"Spatial coverage: X=[{results['x_edges'][0]:.1f}, {results['x_edges'][-1]:.1f}]mm, ")
                    f.write(f"Y=[{results['y_edges'][0]:.1f}, {results['y_edges'][-1]:.1f}]mm\n\n")
                
                # Key findings
                f.write("KEY FINDINGS\n")
                f.write("-" * 30 + "\n")
                
                if abs(thickness_deltas.mean()) < thickness_deltas.std():
                    f.write("✓ Tools show good overall agreement (low systematic bias)\n")
                else:
                    bias_direction = "DMT reads higher" if thickness_deltas.mean() > 0 else "TFK reads higher"
                    f.write(f"⚠ Systematic bias detected: {bias_direction} by {abs(thickness_deltas.mean()):.1f} Å\n")
                
                if thickness_deltas.std() < 10:  # Arbitrary threshold - adjust as needed
                    f.write("✓ Good precision matching between tools\n")
                else:
                    f.write("⚠ High measurement scatter between tools\n")
                
                cv = (thickness_deltas.std() / abs(thickness_deltas.mean())) * 100 if thickness_deltas.mean() != 0 else float('inf')
                if cv < 50:  # Arbitrary threshold
                    f.write("✓ Good relative precision\n")
                else:
                    f.write("⚠ High coefficient of variation in differences\n")
                    
                f.write(f"\nCoefficient of variation: {cv:.1f}%\n")
                
            else:
                f.write("No matched data found for analysis\n")
        
        print(f"Enhanced summary report saved to: {summary_file}")
        return summary_file

    def _analyze_high_delta_regions(self):
        """Analyze regions with highest thickness deltas"""
        if self.matched_data is None or len(self.matched_data) == 0:
            return
        
        print("\n" + "="*60)
        print("SPATIAL DELTA ANALYSIS")
        print("="*60)
        
        deltas = self.matched_data['Thickness_Delta']
        
        # Find extreme delta points
        high_threshold = deltas.quantile(0.9)  # Top 10%
        low_threshold = deltas.quantile(0.1)   # Bottom 10%
        
        high_delta_points = self.matched_data[deltas >= high_threshold]
        low_delta_points = self.matched_data[deltas <= low_threshold]
        
        print(f"\nHIGH DELTA REGIONS (Δ ≥ {high_threshold:.1f} Å):")
        print(f"  Number of points: {len(high_delta_points)}")
        if len(high_delta_points) > 0:
            avg_x = (high_delta_points['DMT_X_mm'] + high_delta_points['TFK_X_mm']) / 2
            avg_y = (high_delta_points['DMT_Y_mm'] + high_delta_points['TFK_Y_mm']) / 2
            print(f"  Average coordinates: X={avg_x.mean():.1f}mm, Y={avg_y.mean():.1f}mm")
            print(f"  Coordinate ranges: X=[{avg_x.min():.1f}, {avg_x.max():.1f}]mm, Y=[{avg_y.min():.1f}, {avg_y.max():.1f}]mm")
            print(f"  Delta range: [{high_delta_points['Thickness_Delta'].min():.1f}, {high_delta_points['Thickness_Delta'].max():.1f}] Å")
        
        print(f"\nLOW DELTA REGIONS (Δ ≤ {low_threshold:.1f} Å):")
        print(f"  Number of points: {len(low_delta_points)}")
        if len(low_delta_points) > 0:
            avg_x = (low_delta_points['DMT_X_mm'] + low_delta_points['TFK_X_mm']) / 2
            avg_y = (low_delta_points['DMT_Y_mm'] + low_delta_points['TFK_Y_mm']) / 2
            print(f"  Average coordinates: X={avg_x.mean():.1f}mm, Y={avg_y.mean():.1f}mm")
            print(f"  Coordinate ranges: X=[{avg_x.min():.1f}, {avg_x.max():.1f}]mm, Y=[{avg_y.min():.1f}, {avg_y.max():.1f}]mm")
            print(f"  Delta range: [{low_delta_points['Thickness_Delta'].min():.1f}, {low_delta_points['Thickness_Delta'].max():.1f}] Å")
        
        # Quadrant analysis
        print(f"\nQUADRANT ANALYSIS:")
        x_coords = (self.matched_data['DMT_X_mm'] + self.matched_data['TFK_X_mm']) / 2
        y_coords = (self.matched_data['DMT_Y_mm'] + self.matched_data['TFK_Y_mm']) / 2
        
        quadrants = {
            'Q1 (+X, +Y)': self.matched_data[(x_coords >= 0) & (y_coords >= 0)],
            'Q2 (-X, +Y)': self.matched_data[(x_coords < 0) & (y_coords >= 0)],
            'Q3 (-X, -Y)': self.matched_data[(x_coords < 0) & (y_coords < 0)],
            'Q4 (+X, -Y)': self.matched_data[(x_coords >= 0) & (y_coords < 0)]
        }
        
        for quad_name, quad_data in quadrants.items():
            if len(quad_data) > 0:
                mean_delta = quad_data['Thickness_Delta'].mean()
                std_delta = quad_data['Thickness_Delta'].std()
                print(f"  {quad_name}: n={len(quad_data)}, Mean Δ={mean_delta:.1f}±{std_delta:.1f} Å")
    
    def create_coordinate_delta_summary(self, output_dir=None):
        """Create summary of thickness deltas by X,Y coordinate combinations."""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for coordinate summary!")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use average coordinates for site identification
        site_data = self.matched_data.copy()
        site_data['Avg_X_mm'] = (site_data['DMT_X_mm'] + site_data['TFK_X_mm']) / 2
        site_data['Avg_Y_mm'] = (site_data['DMT_Y_mm'] + site_data['TFK_Y_mm']) / 2
        
        # Round coordinates to nearest mm for grouping
        site_data['Site_X'] = site_data['Avg_X_mm'].round(1)
        site_data['Site_Y'] = site_data['Avg_Y_mm'].round(1)
        
        # Group by coordinates and calculate statistics
        coord_summary = site_data.groupby(['Site_X', 'Site_Y']).agg({
            'Thickness_Delta': ['count', 'mean', 'std', 'min', 'max'],
            'Distance_mm': ['mean'],
            'DMT_Thickness': ['mean'],
            'TFK_Thickness': ['mean'],
            'WaferID': ['first']
        }).round(3)
        
        # Flatten column names
        coord_summary.columns = ['Count', 'Delta_Mean', 'Delta_Std', 'Delta_Min', 'Delta_Max', 
                                'Avg_Distance', 'DMT_Mean_Thickness', 'TFK_Mean_Thickness', 'Wafer_Example']
        
        # Reset index to make coordinates regular columns
        coord_summary = coord_summary.reset_index()
        
        # Sort by X then Y coordinates
        coord_summary = coord_summary.sort_values(['Site_X', 'Site_Y'])
        
        # Save coordinate summary
        output_path = os.path.join(output_dir, 'dmt_tfk_coordinate_delta_summary.csv')
        coord_summary.to_csv(output_path, index=False)
        print(f"Coordinate delta summary saved to {output_path}")
        
        # Print key statistics
        print(f"\nCoordinate Delta Summary (DMT vs TFK):")
        print(f"  Unique coordinate locations: {len(coord_summary)}")
        print(f"  Most measurements per site: {coord_summary['Count'].max()}")
        print(f"  Average delta range: {coord_summary['Delta_Mean'].min():.2f} to {coord_summary['Delta_Mean'].max():.2f} Å")
        print(f"  Most variable location std: {coord_summary['Delta_Std'].max():.2f} Å")
        
        return coord_summary
    
    def save_results(self, output_dir=None):
        """Save analysis results to files"""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Save detailed matched data
        if self.matched_data is not None and len(self.matched_data) > 0:
            matched_file = os.path.join(output_dir, 'matched_thickness_data.csv')
            self.matched_data.to_csv(matched_file, index=False, encoding='utf-8')
            print(f"Detailed matched data saved to: {matched_file}")
            
            # Create coordinate delta summary
            self.create_coordinate_delta_summary(output_dir)
        
        # Save summary statistics
        summary_file = os.path.join(output_dir, 'thickness_comparison_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("THICKNESS COMPARISON ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            if self.matched_data is not None and len(self.matched_data) > 0:
                thickness_deltas = self.matched_data['Thickness_Delta']
                dmt_thickness = self.matched_data['DMT_Thickness']
                tfk_thickness = self.matched_data['TFK_Thickness']
                
                f.write(f"Total matched measurement pairs: {len(self.matched_data)}\n")
                f.write(f"Mean thickness difference (DMT - TFK): {thickness_deltas.mean():.3f} Å\n")
                f.write(f"Standard deviation of differences: {thickness_deltas.std():.3f} Å\n")
                f.write(f"Minimum difference: {thickness_deltas.min():.3f} Å\n")
                f.write(f"Maximum difference: {thickness_deltas.max():.3f} Å\n")
                f.write(f"Median difference: {thickness_deltas.median():.3f} Å\n\n")
                
                f.write(f"DMT Tool - Mean thickness: {dmt_thickness.mean():.3f} Å, Std: {dmt_thickness.std():.3f} Å\n")
                f.write(f"TFK Tool - Mean thickness: {tfk_thickness.mean():.3f} Å, Std: {tfk_thickness.std():.3f} Å\n\n")
                
                mean_delta = dmt_thickness.mean() - tfk_thickness.mean()
                std_delta = dmt_thickness.std() - tfk_thickness.std()
                f.write(f"Tool mean thickness delta (DMT - TFK): {mean_delta:.3f} Å\n")
                f.write(f"Tool std deviation delta (DMT - TFK): {std_delta:.3f} Å\n")
            else:
                f.write("No matched data found for analysis\n")
        
        print(f"Summary report saved to: {summary_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        print("Starting Thickness Comparison Analysis...")
        print("="*60)
        
        try:
            # Load data
            self.load_data()
            
            if len(self.dmt_data) == 0:
                print("ERROR: No DMT data loaded!")
                return
            
            if len(self.tfk_data) == 0:
                print("ERROR: No TFK data loaded!")
                return
            
            # Find matching points
            self.find_matching_points()
            
            # Analyze differences
            self.analyze_thickness_differences()
            
            # Create visualizations
            self.create_visualizations()
            
            # Create spatial delta visualization
            self.create_spatial_delta_plot()
            
            # Create averaged wafer map
            self.create_averaged_wafer_map()
            
            # Create location ranking
            self.create_location_ranking()
            
            # Create enhanced summary
            self.create_enhanced_summary()
            
            # Create new trend plots
            self.create_thickness_trend_plots()
            
            # Create standard deviation trend plots
            self.create_std_dev_trend_plots()
            
            # Create spline plots
            self.create_spline_plots()
            
            # Save results
            self.save_results()
            
            print("\n" + "="*60)
            print("Analysis completed successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"ERROR during analysis: {e}")
            import traceback
            traceback.print_exc()

    def create_thickness_trend_plots(self, output_dir=None):
        """Create trend plots showing raw and mean thickness data by wafer"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for trend plots")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "="*60)
        print("CREATING THICKNESS TREND PLOTS")
        print("="*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Thickness Trend Analysis: DMT vs TFK', fontsize=16, fontweight='bold')
        
        wafers = sorted(self.matched_data['WaferID'].unique())
        wafer_positions = range(len(wafers))
        
        # Plot 1: Raw thickness data (box plots)
        ax1 = axes[0, 0]
        dmt_data = [self.matched_data[self.matched_data['WaferID'] == w]['DMT_Thickness'].values for w in wafers]
        tfk_data = [self.matched_data[self.matched_data['WaferID'] == w]['TFK_Thickness'].values for w in wafers]
        
        bp1 = ax1.boxplot(dmt_data, positions=[p - 0.2 for p in wafer_positions], 
                         widths=0.3, patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7),
                         medianprops=dict(color='darkred', linewidth=2))
        bp2 = ax1.boxplot(tfk_data, positions=[p + 0.2 for p in wafer_positions], 
                         widths=0.3, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='darkblue', linewidth=2))
        
        ax1.set_xlabel('Wafer')
        ax1.set_ylabel('Thickness [Å]')
        ax1.set_title('Raw Thickness Distribution by Wafer')
        ax1.set_xticks(wafer_positions)
        ax1.set_xticklabels(wafers, rotation=45)
        ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DMT', 'TFK'])
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean thickness by wafer
        ax2 = axes[0, 1]
        wafer_stats = self.matched_data.groupby('WaferID').agg({
            'DMT_Thickness': ['mean', 'std'],
            'TFK_Thickness': ['mean', 'std']
        }).reset_index()
        
        dmt_means = wafer_stats['DMT_Thickness']['mean'].values
        dmt_stds = wafer_stats['DMT_Thickness']['std'].values
        tfk_means = wafer_stats['TFK_Thickness']['mean'].values
        tfk_stds = wafer_stats['TFK_Thickness']['std'].values
        
        ax2.errorbar(wafer_positions, dmt_means, yerr=dmt_stds, 
                    fmt='o-', color='red', label='DMT', linewidth=2, markersize=8, capsize=5)
        ax2.errorbar(wafer_positions, tfk_means, yerr=tfk_stds, 
                    fmt='s-', color='blue', label='TFK', linewidth=2, markersize=8, capsize=5)
        
        ax2.set_xlabel('Wafer')
        ax2.set_ylabel('Mean Thickness ± Std [Å]')
        ax2.set_title('Mean Thickness Trend by Wafer')
        ax2.set_xticks(wafer_positions)
        ax2.set_xticklabels(wafers, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Thickness delta trend
        ax3 = axes[1, 0]
        delta_stats = self.matched_data.groupby('WaferID')['Thickness_Delta'].agg(['mean', 'std']).reset_index()
        
        ax3.errorbar(wafer_positions, delta_stats['mean'], yerr=delta_stats['std'], 
                    fmt='o-', color='purple', linewidth=2, markersize=8, capsize=5)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Wafer')
        ax3.set_ylabel('Thickness Delta (DMT - TFK) ± Std [Å]')
        ax3.set_title('Thickness Delta Trend by Wafer')
        ax3.set_xticks(wafer_positions)
        ax3.set_xticklabels(wafers, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample count by wafer
        ax4 = axes[1, 1]
        sample_counts = self.matched_data.groupby('WaferID').size()
        
        bars = ax4.bar(wafer_positions, sample_counts.values, 
                      color='green', alpha=0.7, edgecolor='darkgreen')
        ax4.set_xlabel('Wafer')
        ax4.set_ylabel('Number of Matched Points')
        ax4.set_title('Sample Count by Wafer')
        ax4.set_xticks(wafer_positions)
        ax4.set_xticklabels(wafers, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, sample_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'thickness_trend_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Thickness trend plots saved to: {plot_path}")
        plt.show()

    def create_std_dev_trend_plots(self, output_dir=None):
        """Create standard deviation trend comparison plots"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for std dev trend plots")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "="*60)
        print("CREATING STANDARD DEVIATION TREND PLOTS")
        print("="*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Standard Deviation Trend Analysis: DMT vs TFK', fontsize=16, fontweight='bold')
        
        wafers = sorted(self.matched_data['WaferID'].unique())
        wafer_positions = range(len(wafers))
        
        wafer_stats = self.matched_data.groupby('WaferID').agg({
            'DMT_Thickness': ['mean', 'std'],
            'TFK_Thickness': ['mean', 'std'],
            'Thickness_Delta': ['std']
        }).reset_index()
        
        dmt_stds = wafer_stats['DMT_Thickness']['std'].values
        tfk_stds = wafer_stats['TFK_Thickness']['std'].values
        delta_stds = wafer_stats['Thickness_Delta']['std'].values
        
        # Plot 1: Individual tool standard deviations
        ax1 = axes[0, 0]
        ax1.plot(wafer_positions, dmt_stds, 'o-', color='red', label='DMT Std Dev', linewidth=2, markersize=8)
        ax1.plot(wafer_positions, tfk_stds, 's-', color='blue', label='TFK Std Dev', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Wafer')
        ax1.set_ylabel('Standard Deviation [Å]')
        ax1.set_title('Tool Standard Deviation by Wafer')
        ax1.set_xticks(wafer_positions)
        ax1.set_xticklabels(wafers, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation ratio (DMT/TFK)
        ax2 = axes[0, 1]
        std_ratio = dmt_stds / tfk_stds
        ax2.plot(wafer_positions, std_ratio, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Variation')
        
        ax2.set_xlabel('Wafer')
        ax2.set_ylabel('Std Dev Ratio (DMT/TFK)')
        ax2.set_title('Standard Deviation Ratio by Wafer')
        ax2.set_xticks(wafer_positions)
        ax2.set_xticklabels(wafers, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Delta standard deviation
        ax3 = axes[1, 0]
        ax3.plot(wafer_positions, delta_stds, 'o-', color='green', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Wafer')
        ax3.set_ylabel('Delta Standard Deviation [Å]')
        ax3.set_title('Thickness Delta Variation by Wafer')
        ax3.set_xticks(wafer_positions)
        ax3.set_xticklabels(wafers, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Coefficient of Variation comparison
        ax4 = axes[1, 1]
        wafer_means = self.matched_data.groupby('WaferID').agg({
            'DMT_Thickness': 'mean',
            'TFK_Thickness': 'mean'
        })
        
        dmt_cv = (dmt_stds / wafer_means['DMT_Thickness'].values) * 100
        tfk_cv = (tfk_stds / wafer_means['TFK_Thickness'].values) * 100
        
        ax4.plot(wafer_positions, dmt_cv, 'o-', color='red', label='DMT CV%', linewidth=2, markersize=8)
        ax4.plot(wafer_positions, tfk_cv, 's-', color='blue', label='TFK CV%', linewidth=2, markersize=8)
        
        ax4.set_xlabel('Wafer')
        ax4.set_ylabel('Coefficient of Variation [%]')
        ax4.set_title('Coefficient of Variation by Wafer')
        ax4.set_xticks(wafer_positions)
        ax4.set_xticklabels(wafers, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'std_dev_trend_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Standard deviation trend plots saved to: {plot_path}")
        plt.show()

    def create_spline_plots(self, output_dir=None):
        """Create spline plots showing DMT and TFK thickness from 0 to 150mm"""
        if self.matched_data is None or len(self.matched_data) == 0:
            print("No matched data available for spline plots")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "="*60)
        print("CREATING SPLINE PLOTS")
        print("="*60)
        
        # Calculate radial distance from center for each point
        x_coords = (self.matched_data['DMT_X_mm'] + self.matched_data['TFK_X_mm']) / 2
        y_coords = (self.matched_data['DMT_Y_mm'] + self.matched_data['TFK_Y_mm']) / 2
        radial_distance = np.sqrt(x_coords**2 + y_coords**2)
        
        # Filter data within 0-150mm range
        mask = (radial_distance >= 0) & (radial_distance <= 150)
        filtered_data = self.matched_data[mask].copy()
        filtered_distance = radial_distance[mask]
        
        if len(filtered_data) == 0:
            print("No data points found in 0-150mm range")
            return
        
        print(f"Using {len(filtered_data)} data points for spline analysis (0-150mm range)")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Radial Spline Analysis: DMT vs TFK (0-150mm)', fontsize=16, fontweight='bold')
        
        # Create evenly spaced radial points for spline evaluation
        r_smooth = np.linspace(0, 150, 300)
        
        # Plot 1: DMT and TFK thickness vs radial distance with splines
        ax1 = axes[0, 0]
        
        # Sort data by radial distance for spline fitting
        sort_idx = np.argsort(filtered_distance)
        sorted_distance = filtered_distance.iloc[sort_idx]
        sorted_dmt = filtered_data['DMT_Thickness'].iloc[sort_idx]
        sorted_tfk = filtered_data['TFK_Thickness'].iloc[sort_idx]
        
        # Fit splines (smooth with reasonable smoothing factor)
        try:
            dmt_spline = UnivariateSpline(sorted_distance, sorted_dmt, s=len(sorted_distance)*100)
            tfk_spline = UnivariateSpline(sorted_distance, sorted_tfk, s=len(sorted_distance)*100)
            
            dmt_smooth = dmt_spline(r_smooth)
            tfk_smooth = tfk_spline(r_smooth)
            
            # Plot raw data points
            ax1.scatter(filtered_distance, filtered_data['DMT_Thickness'], 
                       alpha=0.3, color='red', s=20, label='DMT Raw Data')
            ax1.scatter(filtered_distance, filtered_data['TFK_Thickness'], 
                       alpha=0.3, color='blue', s=20, label='TFK Raw Data')
            
            # Plot splines
            ax1.plot(r_smooth, dmt_smooth, color='red', linewidth=3, label='DMT Spline')
            ax1.plot(r_smooth, tfk_smooth, color='blue', linewidth=3, label='TFK Spline')
            
        except Exception as e:
            print(f"Warning: Could not fit splines: {e}")
            # Fallback to scatter plot only
            ax1.scatter(filtered_distance, filtered_data['DMT_Thickness'], 
                       alpha=0.6, color='red', s=30, label='DMT')
            ax1.scatter(filtered_distance, filtered_data['TFK_Thickness'], 
                       alpha=0.6, color='blue', s=30, label='TFK')
        
        ax1.set_xlabel('Radial Distance from Center [mm]')
        ax1.set_ylabel('Thickness [Å]')
        ax1.set_title('Thickness vs Radial Distance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 150)
        
        # Plot 2: Thickness delta vs radial distance
        ax2 = axes[0, 1]
        try:
            sorted_delta = filtered_data['Thickness_Delta'].iloc[sort_idx]
            delta_spline = UnivariateSpline(sorted_distance, sorted_delta, s=len(sorted_distance)*50)
            delta_smooth = delta_spline(r_smooth)
            
            ax2.scatter(filtered_distance, filtered_data['Thickness_Delta'], 
                       alpha=0.4, color='purple', s=20, label='Raw Delta')
            ax2.plot(r_smooth, delta_smooth, color='purple', linewidth=3, label='Delta Spline')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
        except Exception as e:
            ax2.scatter(filtered_distance, filtered_data['Thickness_Delta'], 
                       alpha=0.6, color='purple', s=30, label='Thickness Delta')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Radial Distance from Center [mm]')
        ax2.set_ylabel('Thickness Delta (DMT - TFK) [Å]')
        ax2.set_title('Thickness Delta vs Radial Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 150)
        
        # Plot 3: Binned analysis
        ax3 = axes[1, 0]
        # Create radial bins
        n_bins = 15
        bin_edges = np.linspace(0, 150, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        dmt_binned = []
        tfk_binned = []
        delta_binned = []
        count_binned = []
        
        for i in range(n_bins):
            mask_bin = (filtered_distance >= bin_edges[i]) & (filtered_distance < bin_edges[i+1])
            if np.any(mask_bin):
                dmt_binned.append(filtered_data[mask_bin]['DMT_Thickness'].mean())
                tfk_binned.append(filtered_data[mask_bin]['TFK_Thickness'].mean())
                delta_binned.append(filtered_data[mask_bin]['Thickness_Delta'].mean())
                count_binned.append(np.sum(mask_bin))
            else:
                dmt_binned.append(np.nan)
                tfk_binned.append(np.nan)
                delta_binned.append(np.nan)
                count_binned.append(0)
        
        # Remove NaN values for plotting
        valid_bins = ~np.isnan(dmt_binned)
        valid_centers = bin_centers[valid_bins]
        valid_dmt = np.array(dmt_binned)[valid_bins]
        valid_tfk = np.array(tfk_binned)[valid_bins]
        
        ax3.plot(valid_centers, valid_dmt, 'o-', color='red', linewidth=2, markersize=8, label='DMT Binned')
        ax3.plot(valid_centers, valid_tfk, 's-', color='blue', linewidth=2, markersize=8, label='TFK Binned')
        
        ax3.set_xlabel('Radial Distance from Center [mm]')
        ax3.set_ylabel('Mean Thickness [Å]')
        ax3.set_title(f'Binned Thickness Analysis ({n_bins} bins)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 150)
        
        # Plot 4: Sample density vs radial distance
        ax4 = axes[1, 1]
        ax4.bar(bin_centers[valid_bins], np.array(count_binned)[valid_bins], 
                width=bin_edges[1] - bin_edges[0], alpha=0.7, color='green', edgecolor='darkgreen')
        
        ax4.set_xlabel('Radial Distance from Center [mm]')
        ax4.set_ylabel('Number of Sample Points')
        ax4.set_title('Sample Density vs Radial Distance')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 150)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'radial_spline_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Radial spline plots saved to: {plot_path}")
        plt.show()

def create_spatial_plot_from_data(csv_file_path=None):
    """Create spatial delta plot from existing matched data CSV"""
    if csv_file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_dir, 'matched_thickness_data.csv')
    
    if not os.path.exists(csv_file_path):
        print(f"ERROR: Matched data file not found at {csv_file_path}")
        print("Please run the complete analysis first to generate the matched data.")
        return
    
    # Load matched data
    matched_data = pd.read_csv(csv_file_path)
    print(f"Loaded {len(matched_data)} matched measurement pairs from {csv_file_path}")
    
    # Create a temporary app instance just for visualization
    app = ThicknessComparisonApp("", "")  # Empty folders since we're loading from CSV
    app.matched_data = matched_data
    
    # Create the spatial plot
    app.create_spatial_delta_plot()

def create_summary_and_ranking_from_data(csv_file_path=None, output_dir=None):
    """Create wafer map, ranking, and enhanced summary from existing matched data CSV"""
    if csv_file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_dir, 'matched_thickness_data.csv')
    
    if not os.path.exists(csv_file_path):
        print(f"ERROR: Matched data file not found at {csv_file_path}")
        print("Please run the complete analysis first to generate the matched data.")
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load matched data
    matched_data = pd.read_csv(csv_file_path)
    print(f"Loaded {len(matched_data)} matched measurement pairs from {csv_file_path}")
    
    # Create a temporary app instance 
    app = ThicknessComparisonApp("", "")  # Empty folders since we're loading from CSV
    app.matched_data = matched_data
    
    # Create all the new visualizations and analyses
    print("\nGenerating averaged wafer map...")
    app.create_averaged_wafer_map(output_dir)
    
    print("\nGenerating location ranking...")
    rankings = app.create_location_ranking(output_dir)
    
    print("\nGenerating enhanced summary...")
    summary_file = app.create_enhanced_summary(output_dir)
    
    print("\nGenerating thickness trend plots...")
    app.create_thickness_trend_plots(output_dir)
    
    print("\nGenerating standard deviation trend plots...")
    app.create_std_dev_trend_plots(output_dir)
    
    print("\nGenerating radial spline plots...")
    app.create_spline_plots(output_dir)
    
    print(f"\n{'='*60}")
    print("SUMMARY AND RANKING ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"Wafer map and ranking files saved to: {output_dir}")
    
    return rankings

def main():
    # Define folder paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dmt_folder = os.path.join(script_dir, "DMT")
    tfk_folder = os.path.join(script_dir, "TFK")
    
    # Check if folders exist
    if not os.path.exists(dmt_folder):
        print(f"ERROR: DMT folder not found at {dmt_folder}")
        return
    
    if not os.path.exists(tfk_folder):
        print(f"ERROR: TFK folder not found at {tfk_folder}")
        return
    
    # Create and run the comparison app
    app = ThicknessComparisonApp(
        dmt_folder=dmt_folder,
        tfk_folder=tfk_folder,
        distance_threshold=4.0  # 4mm threshold for coordinate matching
    )
    
    app.run_complete_analysis()

if __name__ == "__main__":
    main()