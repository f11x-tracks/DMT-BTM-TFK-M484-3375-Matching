import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d
from scipy.interpolate import UnivariateSpline
import xml.etree.ElementTree as ET
import os
import seaborn as sns
from pathlib import Path

class BTMDMTComparisonApp:
    def __init__(self, btm_folder, dmt_folder, distance_threshold=4.0):
        self.btm_folder = btm_folder
        self.dmt_folder = dmt_folder
        self.distance_threshold = distance_threshold
        self.btm_data = pd.DataFrame()
        self.dmt_data = pd.DataFrame()
        self.matched_data = pd.DataFrame()
        self.analysis_results = {}
        
    def parse_btm_csv(self, file_path):
        """Parse BTM CSV file to extract thickness measurements."""
        records = []
        try:
            df = pd.read_csv(file_path)
            # Clean up column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            for _, row in df.iterrows():
                record = {
                    'WaferID': str(row['WaferID']).strip(),
                    'X_mm': float(row['X[mm]']),
                    'Y_mm': float(row['Y[mm]']),
                    'Thickness': float(row['Film Thickness']),
                    'Tool': 'BTM',
                    'File': os.path.basename(file_path)
                }
                records.append(record)
        except Exception as e:
            print(f"Warning: Could not parse BTM file {file_path}: {e}")
        
        return records
    
    def parse_dmt_xml(self, file_path):
        """Parse DMT XML file to extract thickness measurements."""
        records = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all DataRecord elements
            for record in root.findall('.//DataRecord'):
                label_elem = record.find('Label')
                if label_elem is not None and label_elem.text == "Layer 1 Thickness":
                    wafer_elem = record.find('WaferID')
                    x_elem = record.find('XWaferLoc')
                    y_elem = record.find('YWaferLoc')
                    datum_elem = record.find('Datum')
                    
                    if all(elem is not None for elem in [wafer_elem, x_elem, y_elem, datum_elem]):
                        record_dict = {
                            'WaferID': wafer_elem.text,
                            'X_mm': float(x_elem.text),
                            'Y_mm': float(y_elem.text),
                            'Thickness': float(datum_elem.text),
                            'Tool': 'DMT',
                            'File': os.path.basename(file_path)
                        }
                        records.append(record_dict)
        except Exception as e:
            print(f"Warning: Could not parse DMT file {file_path}: {e}")
        
        return records
    
    def load_data(self):
        """Load BTM and DMT data from their respective folders."""
        print("Loading BTM data...")
        btm_records = []
        btm_files = list(Path(self.btm_folder).glob("*.csv"))
        
        if not btm_files:
            print("Warning: No CSV files found in BTM folder!")
        
        for file_path in btm_files:
            records = self.parse_btm_csv(file_path)
            btm_records.extend(records)
            print(f"BTM: Loaded {len(records)} records from {file_path.name}")
        
        self.btm_data = pd.DataFrame(btm_records)
        print(f"Total BTM records: {len(self.btm_data)}")
        
        print("Loading DMT data...")
        dmt_records = []
        dmt_files = list(Path(self.dmt_folder).glob("*.xml"))
        
        if not dmt_files:
            print("Warning: No XML files found in DMT folder!")
        
        for file_path in dmt_files:
            records = self.parse_dmt_xml(file_path)
            dmt_records.extend(records)
            print(f"DMT: Loaded {len(records)} records from {file_path.name}")
        
        self.dmt_data = pd.DataFrame(dmt_records)
        print(f"Total DMT records: {len(self.dmt_data)}")
        
        if len(self.btm_data) == 0 or len(self.dmt_data) == 0:
            print("Warning: One or both datasets are empty!")
            return
        
        print(f"BTM Wafer IDs: {sorted(self.btm_data['WaferID'].unique())}")
        print(f"DMT Wafer IDs: {sorted(self.dmt_data['WaferID'].unique())}")
    
    def find_matching_points(self):
        """Find matching points between BTM and DMT measurements."""
        print("Finding matching points...")
        
        # Find common wafer IDs
        btm_wafers = set(self.btm_data['WaferID'].unique())
        dmt_wafers = set(self.dmt_data['WaferID'].unique())
        common_wafers = btm_wafers.intersection(dmt_wafers)
        
        print(f"Common wafers: {sorted(common_wafers)}")
        
        if not common_wafers:
            print("No common wafers found between BTM and DMT data!")
            return
        
        matched_records = []
        
        for wafer_id in common_wafers:
            btm_wafer = self.btm_data[self.btm_data['WaferID'] == wafer_id]
            dmt_wafer = self.dmt_data[self.dmt_data['WaferID'] == wafer_id]
            
            # Extract coordinates
            btm_coords = btm_wafer[['X_mm', 'Y_mm']].values
            dmt_coords = dmt_wafer[['X_mm', 'Y_mm']].values
            
            # Calculate distance matrix
            distances = cdist(btm_coords, dmt_coords)
            
            # Find matches within threshold
            btm_indices, dmt_indices = np.where(distances <= self.distance_threshold)
            
            for btm_idx, dmt_idx in zip(btm_indices, dmt_indices):
                btm_row = btm_wafer.iloc[btm_idx]
                dmt_row = dmt_wafer.iloc[dmt_idx]
                
                matched_record = {
                    'WaferID': wafer_id,
                    'BTM_X_mm': btm_row['X_mm'],
                    'BTM_Y_mm': btm_row['Y_mm'],
                    'BTM_Thickness': btm_row['Thickness'],
                    'DMT_X_mm': dmt_row['X_mm'],
                    'DMT_Y_mm': dmt_row['Y_mm'],
                    'DMT_Thickness': dmt_row['Thickness'],
                    'Distance_mm': distances[btm_idx, dmt_idx],
                    'Thickness_Delta': btm_row['Thickness'] - dmt_row['Thickness'],  # BTM - DMT
                    'BTM_File': btm_row['File'],
                    'DMT_File': dmt_row['File'],
                }
                matched_records.append(matched_record)
        
        self.matched_data = pd.DataFrame(matched_records)
        print(f"Found {len(self.matched_data)} matching point pairs")
        
        if len(self.matched_data) > 0:
            print(f"Distance range: {self.matched_data['Distance_mm'].min():.2f} - {self.matched_data['Distance_mm'].max():.2f} mm")
            print(f"Thickness delta range: {self.matched_data['Thickness_Delta'].min():.2f} - {self.matched_data['Thickness_Delta'].max():.2f} Å")
    
    def analyze_thickness_differences(self):
        """Analyze thickness differences between BTM and DMT."""
        if len(self.matched_data) == 0:
            print("No matched data to analyze!")
            return
        
        print("Analyzing thickness differences...")
        
        delta = self.matched_data['Thickness_Delta']
        
        self.analysis_results = {
            'total_matches': len(self.matched_data),
            'unique_wafers': len(self.matched_data['WaferID'].unique()),
            'mean_delta': delta.mean(),
            'std_delta': delta.std(),
            'min_delta': delta.min(),
            'max_delta': delta.max(),
            'median_delta': delta.median(),
            'q25_delta': delta.quantile(0.25),
            'q75_delta': delta.quantile(0.75),
            'mean_distance': self.matched_data['Distance_mm'].mean(),
        }
        
        print(f"Analysis Results:")
        print(f"  Total matched pairs: {self.analysis_results['total_matches']}")
        print(f"  Unique wafers: {self.analysis_results['unique_wafers']}")
        print(f"  Mean thickness delta (BTM - DMT): {self.analysis_results['mean_delta']:.2f} ± {self.analysis_results['std_delta']:.2f} Å")
        print(f"  Median delta: {self.analysis_results['median_delta']:.2f} Å")
        print(f"  Delta range: {self.analysis_results['min_delta']:.2f} to {self.analysis_results['max_delta']:.2f} Å")
        print(f"  Mean matching distance: {self.analysis_results['mean_distance']:.2f} mm")
    
    def create_visualizations(self, output_dir="btm_dmt_comparison_results"):
        """Create comprehensive visualizations."""
        if len(self.matched_data) == 0:
            print("No data to visualize!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        
        # 1. Thickness Comparison Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot BTM vs DMT
        ax1.scatter(self.matched_data['DMT_Thickness'], self.matched_data['BTM_Thickness'], 
                   alpha=0.6, s=20)
        min_thick = min(self.matched_data['DMT_Thickness'].min(), self.matched_data['BTM_Thickness'].min())
        max_thick = max(self.matched_data['DMT_Thickness'].max(), self.matched_data['BTM_Thickness'].max())
        ax1.plot([min_thick, max_thick], [min_thick, max_thick], 'r--', alpha=0.8)
        ax1.set_xlabel('DMT Thickness (Å)')
        ax1.set_ylabel('BTM Thickness (Å)')
        ax1.set_title('BTM vs DMT Thickness Correlation')
        ax1.grid(True, alpha=0.3)
        
        # Thickness Delta histogram
        ax2.hist(self.matched_data['Thickness_Delta'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(self.analysis_results['mean_delta'], color='red', linestyle='--', 
                   label=f'Mean: {self.analysis_results["mean_delta"]:.1f}Å')
        ax2.set_xlabel('Thickness Delta (BTM - DMT) [Å]')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Thickness Difference Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Box plots by wafer
        wafer_list = sorted(self.matched_data['WaferID'].unique())
        if len(wafer_list) > 1:
            wafer_data = [self.matched_data[self.matched_data['WaferID'] == w]['Thickness_Delta'].values 
                         for w in wafer_list]
            ax3.boxplot(wafer_data, labels=wafer_list)
            ax3.set_xlabel('Wafer ID')
            ax3.set_ylabel('Thickness Delta (BTM - DMT) [Å]')
            ax3.set_title('Delta Distribution by Wafer')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Only one wafer\navailable', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Delta Distribution by Wafer')
        
        # Spatial delta map
        self.create_spatial_delta_plot(ax4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'btm_dmt_thickness_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def create_spatial_delta_plot(self, ax):
        """Create spatial map of thickness deltas."""
        # Use average coordinates for spatial analysis
        avg_x = (self.matched_data['BTM_X_mm'] + self.matched_data['DMT_X_mm']) / 2
        avg_y = (self.matched_data['BTM_Y_mm'] + self.matched_data['DMT_Y_mm']) / 2
        
        scatter = ax.scatter(avg_x, avg_y, c=self.matched_data['Thickness_Delta'], 
                           cmap='RdBu_r', s=30, alpha=0.7)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('Spatial Distribution of Thickness Deltas')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Thickness Delta (BTM - DMT) [Å]')
    
    def create_coordinate_delta_summary(self, output_dir="btm_dmt_comparison_results"):
        """Create summary of thickness deltas by X,Y coordinate combinations."""
        if len(self.matched_data) == 0:
            print("No data available for coordinate summary!")
            return
        
        # Use average coordinates for site identification
        site_data = self.matched_data.copy()
        site_data['Avg_X_mm'] = (site_data['BTM_X_mm'] + site_data['DMT_X_mm']) / 2
        site_data['Avg_Y_mm'] = (site_data['BTM_Y_mm'] + site_data['DMT_Y_mm']) / 2
        
        # Round coordinates to nearest mm for grouping
        site_data['Site_X'] = site_data['Avg_X_mm'].round(1)
        site_data['Site_Y'] = site_data['Avg_Y_mm'].round(1)
        
        # Group by coordinates and calculate statistics
        coord_summary = site_data.groupby(['Site_X', 'Site_Y']).agg({
            'Thickness_Delta': ['count', 'mean', 'std', 'min', 'max'],
            'Distance_mm': ['mean'],
            'BTM_Thickness': ['mean'],
            'DMT_Thickness': ['mean'],
            'WaferID': ['first']
        }).round(3)
        
        # Flatten column names
        coord_summary.columns = ['Count', 'Delta_Mean', 'Delta_Std', 'Delta_Min', 'Delta_Max', 
                                'Avg_Distance', 'BTM_Mean_Thickness', 'DMT_Mean_Thickness', 'Wafer_Example']
        
        # Reset index to make coordinates regular columns
        coord_summary = coord_summary.reset_index()
        
        # Sort by X then Y coordinates
        coord_summary = coord_summary.sort_values(['Site_X', 'Site_Y'])
        
        # Save coordinate summary
        output_path = os.path.join(output_dir, 'btm_dmt_coordinate_delta_summary.csv')
        coord_summary.to_csv(output_path, index=False)
        print(f"Coordinate delta summary saved to {output_path}")
        
        # Print key statistics
        print(f"\nCoordinate Delta Summary (BTM vs DMT):")
        print(f"  Unique coordinate locations: {len(coord_summary)}")
        print(f"  Most measurements per site: {coord_summary['Count'].max()}")
        print(f"  Average delta range: {coord_summary['Delta_Mean'].min():.2f} to {coord_summary['Delta_Mean'].max():.2f} Å")
        print(f"  Most variable location std: {coord_summary['Delta_Std'].max():.2f} Å")
        
        return coord_summary
    
    def save_results(self, output_dir="btm_dmt_comparison_results"):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matched data
        if len(self.matched_data) > 0:
            self.matched_data.to_csv(os.path.join(output_dir, 'btm_dmt_matched_data.csv'), index=False)
            print(f"Matched data saved to {output_dir}/btm_dmt_matched_data.csv")
            
            # Create coordinate delta summary
            self.create_coordinate_delta_summary(output_dir)
        
        # Save analysis summary
        with open(os.path.join(output_dir, 'btm_dmt_analysis_summary.txt'), 'w') as f:
            f.write("BTM vs DMT Thickness Comparison Analysis\n")
            f.write("="*50 + "\n\n")
            
            if self.analysis_results:
                f.write(f"Total matched pairs: {self.analysis_results['total_matches']}\n")
                f.write(f"Unique wafers: {self.analysis_results['unique_wafers']}\n")
                f.write(f"Mean thickness delta (BTM - DMT): {self.analysis_results['mean_delta']:.3f} ± {self.analysis_results['std_delta']:.3f} Å\n")
                f.write(f"Median delta: {self.analysis_results['median_delta']:.3f} Å\n")
                f.write(f"Delta range: {self.analysis_results['min_delta']:.3f} to {self.analysis_results['max_delta']:.3f} Å\n")
                f.write(f"25th percentile: {self.analysis_results['q25_delta']:.3f} Å\n")
                f.write(f"75th percentile: {self.analysis_results['q75_delta']:.3f} Å\n")
                f.write(f"Mean matching distance: {self.analysis_results['mean_distance']:.3f} mm\n")
                f.write(f"Distance threshold used: {self.distance_threshold} mm\n")
        
        print(f"Analysis summary saved to {output_dir}/btm_dmt_analysis_summary.txt")
    
    def run_complete_analysis(self):
        """Run the complete BTM vs DMT analysis workflow."""
        print("Starting BTM vs DMT Thickness Comparison Analysis")
        print("="*60)
        
        self.load_data()
        
        if len(self.btm_data) == 0 or len(self.dmt_data) == 0:
            print("Cannot proceed with empty datasets!")
            return
        
        self.find_matching_points()
        self.analyze_thickness_differences()
        self.create_visualizations()
        self.save_results()
        
        print("\nBTM vs DMT analysis completed!")

def main():
    # Define folders
    btm_folder = "BTM"
    dmt_folder = "DMT"
    
    # Create and run analysis
    app = BTMDMTComparisonApp(btm_folder, dmt_folder, distance_threshold=4.0)
    app.run_complete_analysis()

if __name__ == "__main__":
    main()