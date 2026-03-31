import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d, ttest_ind, ttest_rel, mannwhitneyu
from scipy.interpolate import UnivariateSpline
import xml.etree.ElementTree as ET
import os
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class DMTTFKComparisonApp:
    def __init__(self, dmt_folder, tfk_folder, distance_threshold=4.0):
        self.dmt_folder = dmt_folder
        self.tfk_folder = tfk_folder
        self.distance_threshold = distance_threshold
        self.dmt_data = pd.DataFrame()
        self.tfk_data = pd.DataFrame()
        self.matched_data = pd.DataFrame()
        self.analysis_results = {}
        
    def parse_dmt_xml(self, file_path):
        """Parse DMT XML file to extract thickness measurements."""
        records = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all DataRecord elements
            for record in root.findall('.//DataRecord'):
                wafer_id = record.find('WaferID')
                label = record.find('Label')
                datum = record.find('Datum')
                x_loc = record.find('XWaferLoc')
                y_loc = record.find('YWaferLoc')
                
                if (wafer_id is not None and label is not None and 
                    datum is not None and x_loc is not None and y_loc is not None):
                    
                    if label.text == 'Layer 1 Thickness':
                        record_dict = {
                            'WaferID': wafer_id.text,
                            'X_mm': float(x_loc.text),
                            'Y_mm': float(y_loc.text),
                            'Thickness': float(datum.text),
                            'Tool': 'DMT',
                            'File': os.path.basename(file_path)
                        }
                        records.append(record_dict)
        except Exception as e:
            print(f"Warning: Could not parse DMT file {file_path}: {e}")
        
        return records
    
    def parse_tfk_xml(self, file_path):
        """Parse TFK XML file to extract thickness measurements."""
        records = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all DataRecord elements
            for record in root.findall('.//DataRecord'):
                label_elem = record.find('Label')
                if label_elem is not None and label_elem.text == "T1":
                    wafer_elem = record.find('WaferID')
                    x_elem = record.find('XNative')
                    y_elem = record.find('YNative')
                    datum_elem = record.find('Datum')
                    
                    if all(elem is not None for elem in [wafer_elem, x_elem, y_elem, datum_elem]):
                        # Convert TFK coordinates from native units to mm
                        x_mm = float(x_elem.text) / 10_000_000
                        y_mm = float(y_elem.text) / 10_000_000
                        
                        record_dict = {
                            'WaferID': wafer_elem.text,
                            'X_mm': x_mm,
                            'Y_mm': y_mm,
                            'Thickness': float(datum_elem.text),
                            'Tool': 'TFK',
                            'File': os.path.basename(file_path)
                        }
                        records.append(record_dict)
        except Exception as e:
            print(f"Warning: Could not parse TFK file {file_path}: {e}")
        
        return records
    
    def load_data(self):
        """Load DMT and TFK data from their respective folders."""
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
        
        print("Loading TFK data...")
        tfk_records = []
        tfk_files = list(Path(self.tfk_folder).glob("*.xml"))
        
        if not tfk_files:
            print("Warning: No XML files found in TFK folder!")
        
        for file_path in tfk_files:
            records = self.parse_tfk_xml(file_path)
            tfk_records.extend(records)
            print(f"TFK: Loaded {len(records)} records from {file_path.name}")
        
        self.tfk_data = pd.DataFrame(tfk_records)
        print(f"Total TFK records: {len(self.tfk_data)}")
        
        if len(self.dmt_data) == 0 or len(self.tfk_data) == 0:
            print("Warning: One or both datasets are empty!")
            return
        
        print(f"DMT Wafer IDs: {sorted(self.dmt_data['WaferID'].unique())}")
        print(f"TFK Wafer IDs: {sorted(self.tfk_data['WaferID'].unique())}")
    
    def find_matching_points(self):
        """Find matching points between DMT and TFK measurements."""
        print("Finding matching points...")
        
        # Find common wafer IDs
        dmt_wafers = set(self.dmt_data['WaferID'].unique())
        tfk_wafers = set(self.tfk_data['WaferID'].unique())
        common_wafers = dmt_wafers.intersection(tfk_wafers)
        
        print(f"Common wafers: {sorted(common_wafers)}")
        
        if not common_wafers:
            print("No common wafers found between DMT and TFK data!")
            return
        
        matched_records = []
        
        for wafer_id in common_wafers:
            dmt_wafer = self.dmt_data[self.dmt_data['WaferID'] == wafer_id]
            tfk_wafer = self.tfk_data[self.tfk_data['WaferID'] == wafer_id]
            
            # Extract coordinates
            dmt_coords = dmt_wafer[['X_mm', 'Y_mm']].values
            tfk_coords = tfk_wafer[['X_mm', 'Y_mm']].values
            
            # Calculate distance matrix
            distances = cdist(dmt_coords, tfk_coords)
            
            # Find matches within threshold
            dmt_indices, tfk_indices = np.where(distances <= self.distance_threshold)
            
            for dmt_idx, tfk_idx in zip(dmt_indices, tfk_indices):
                dmt_row = dmt_wafer.iloc[dmt_idx]
                tfk_row = tfk_wafer.iloc[tfk_idx]
                
                matched_record = {
                    'WaferID': wafer_id,
                    'DMT_X_mm': dmt_row['X_mm'],
                    'DMT_Y_mm': dmt_row['Y_mm'],
                    'DMT_Thickness': dmt_row['Thickness'],
                    'TFK_X_mm': tfk_row['X_mm'],
                    'TFK_Y_mm': tfk_row['Y_mm'],
                    'TFK_Thickness': tfk_row['Thickness'],
                    'Distance_mm': distances[dmt_idx, tfk_idx],
                    'Thickness_Delta': dmt_row['Thickness'] - tfk_row['Thickness'],  # DMT - TFK
                    'DMT_File': dmt_row['File'],
                    'TFK_File': tfk_row['File'],
                }
                matched_records.append(matched_record)
        
        self.matched_data = pd.DataFrame(matched_records)
        print(f"Found {len(self.matched_data)} matching point pairs")
        
        # Apply radial adjustment to DMT thickness data (add 20 to points > 146mm radius)
        if len(self.matched_data) > 0:
            print("\nApplying radial adjustment to DMT thickness data...")
            
            # Calculate radius from center (0,0) for each DMT measurement point 
            dmt_radius = np.sqrt(self.matched_data['DMT_X_mm']**2 + self.matched_data['DMT_Y_mm']**2)
            
            # Find points with radius > 146mm
            radius_mask = dmt_radius > 146.0
            num_adjusted = radius_mask.sum()
            
            if num_adjusted > 0:
                # Add 20 Å to DMT thickness for points > 146mm radius
                self.matched_data.loc[radius_mask, 'DMT_Thickness'] += 20.0
                
                # Recalculate thickness delta with adjusted DMT values
                self.matched_data['Thickness_Delta'] = (self.matched_data['DMT_Thickness'] - 
                                                       self.matched_data['TFK_Thickness'])
                
                print(f"  Adjusted {num_adjusted} DMT thickness points (radius > 146mm) by +20 Å")
                print(f"  Radius range of adjusted points: {dmt_radius[radius_mask].min():.1f} - {dmt_radius[radius_mask].max():.1f} mm")
            else:
                print("  No DMT thickness points found with radius > 146mm")
                
            print(f"Radial adjustment complete.")
        
        if len(self.matched_data) > 0:
            print(f"Distance range: {self.matched_data['Distance_mm'].min():.2f} - {self.matched_data['Distance_mm'].max():.2f} mm")
            print(f"Thickness delta range: {self.matched_data['Thickness_Delta'].min():.2f} - {self.matched_data['Thickness_Delta'].max():.2f} Å")
    
    def analyze_thickness_differences(self):
        """Analyze thickness differences between DMT and TFK."""
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
        print(f"  Mean thickness delta (DMT - TFK): {self.analysis_results['mean_delta']:.2f} ± {self.analysis_results['std_delta']:.2f} Å")
        print(f"  Median delta: {self.analysis_results['median_delta']:.2f} Å")
        print(f"  Delta range: {self.analysis_results['min_delta']:.2f} to {self.analysis_results['max_delta']:.2f} Å")
        print(f"  Mean matching distance: {self.analysis_results['mean_distance']:.2f} mm")
    
    def create_visualizations(self, output_dir="dmt_tfk_comparison_results"):
        """Create comprehensive visualizations."""
        if len(self.matched_data) == 0:
            print("No data to visualize!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        
        # 1. Thickness Comparison Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot DMT vs TFK
        ax1.scatter(self.matched_data['TFK_Thickness'], self.matched_data['DMT_Thickness'], 
                   alpha=0.6, s=20)
        min_thick = min(self.matched_data['TFK_Thickness'].min(), self.matched_data['DMT_Thickness'].min())
        max_thick = max(self.matched_data['TFK_Thickness'].max(), self.matched_data['DMT_Thickness'].max())
        
        # Linear correlation line instead of 1:1
        x = self.matched_data['TFK_Thickness'].values
        y = self.matched_data['DMT_Thickness'].values
        coeffs = np.polyfit(x, y, 1)
        correlation_coeff = np.corrcoef(x, y)[0, 1]
        r_squared = correlation_coeff**2
        
        x_line = np.linspace(min_thick, max_thick, 100)
        y_line = np.polyval(coeffs, x_line)
        ax1.plot(x_line, y_line, 'r--', alpha=0.8, label='Linear Fit')
        
        ax1.set_xlabel('TFK Thickness (Å)')
        ax1.set_ylabel('DMT Thickness (Å)')
        ax1.set_title('DMT vs TFK Thickness Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Thickness Delta histogram
        ax2.hist(self.matched_data['Thickness_Delta'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(self.analysis_results['mean_delta'], color='red', linestyle='--', 
                   label=f'Mean: {self.analysis_results["mean_delta"]:.1f}Å')
        ax2.set_xlabel('Thickness Delta (DMT - TFK) [Å]')
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
            ax3.set_ylabel('Thickness Delta (DMT - TFK) [Å]')
            ax3.set_title('Delta Distribution by Wafer')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Only one wafer\navailable', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Delta Distribution by Wafer')
        
        # Spatial delta map
        self.create_spatial_delta_plot(ax4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dmt_tfk_thickness_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def create_spatial_delta_plot(self, ax):
        """Create spatial map of thickness deltas."""
        # Use average coordinates for spatial analysis
        avg_x = (self.matched_data['DMT_X_mm'] + self.matched_data['TFK_X_mm']) / 2
        avg_y = (self.matched_data['DMT_Y_mm'] + self.matched_data['TFK_Y_mm']) / 2
        
        scatter = ax.scatter(avg_x, avg_y, c=self.matched_data['Thickness_Delta'], 
                           cmap='RdBu_r', s=30, alpha=0.7)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('Spatial Distribution of Thickness Deltas')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Thickness Delta (DMT - TFK) [Å]')
    
    def create_statistical_summary(self, data_dmt, data_tfk, data_name):
        """Create statistical summary table for equivalence testing."""
        from scipy import stats
        
        # Basic statistics
        stats_dmt = {
            'mean': np.mean(data_dmt),
            'std': np.std(data_dmt, ddof=1),
            'min': np.min(data_dmt),
            'max': np.max(data_dmt),
            'n': len(data_dmt)
        }
        
        stats_tfk = {
            'mean': np.mean(data_tfk),
            'std': np.std(data_tfk, ddof=1),
            'min': np.min(data_tfk),
            'max': np.max(data_tfk),
            'n': len(data_tfk)
        }
        
        # Statistical tests
        # Paired t-test (since these are matched measurements)
        t_stat, p_value_paired = ttest_rel(data_dmt, data_tfk)
        
        # Independent t-test for comparison
        t_stat_ind, p_value_ind = ttest_ind(data_dmt, data_tfk)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = mannwhitneyu(data_dmt, data_tfk, alternative='two-sided')
        
        # Equivalence testing (TOST)
        equivalence_results = self.perform_equivalence_test(data_dmt, data_tfk)
        
        # Significance level
        alpha = 0.05
        
        return {
            'dmt_stats': stats_dmt,
            'tfk_stats': stats_tfk,
            'paired_t_test': {'t_stat': t_stat, 'p_value': p_value_paired, 'alpha': alpha},
            'independent_t_test': {'t_stat': t_stat_ind, 'p_value': p_value_ind, 'alpha': alpha},
            'mann_whitney': {'u_stat': u_stat, 'p_value': p_value_mw, 'alpha': alpha},
            'equivalence_test': equivalence_results,
            'data_name': data_name
        }
    
    def perform_equivalence_test(self, data_dmt, data_tfk, equivalence_margin=5.0):
        """Perform Two One-Sided Tests (TOST) for equivalence.
        
        Parameters:
        - equivalence_margin: δ value in same units as data (e.g., 5 Å for thickness)
        """
        from scipy import stats
        
        # Calculate differences (paired data)
        differences = np.array(data_dmt) - np.array(data_tfk)
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        
        # Two one-sided tests
        # Test 1: H0: mean_diff <= -delta (test if mean_diff > -delta)
        t1 = (mean_diff + equivalence_margin) / se_diff
        p1 = stats.t.sf(t1, n-1)  # Upper tail
        
        # Test 2: H0: mean_diff >= +delta (test if mean_diff < +delta)  
        t2 = (mean_diff - equivalence_margin) / se_diff
        p2 = stats.t.cdf(t2, n-1)  # Lower tail
        
        # TOST p-value is the maximum of the two p-values
        tost_p_value = max(p1, p2)
        
        # 90% Confidence interval for equivalence (α=0.05)
        t_critical = stats.t.ppf(0.95, n-1)  # 90% CI
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Equivalence conclusion
        is_equivalent = tost_p_value < 0.05 and abs(ci_lower) < equivalence_margin and abs(ci_upper) < equivalence_margin
        
        return {
            'mean_difference': mean_diff,
            'se_difference': se_diff,
            'equivalence_margin': equivalence_margin,
            't1_stat': t1,
            't2_stat': t2,
            'p1_value': p1,
            'p2_value': p2,
            'tost_p_value': tost_p_value,
            'ci_90_lower': ci_lower,
            'ci_90_upper': ci_upper,
            'is_equivalent': is_equivalent,
            'conclusion': 'EQUIVALENT' if is_equivalent else 'NOT EQUIVALENT'
        }
    
    def create_raw_data_comparison(self, output_dir="dmt_tfk_comparison_results"):
        """Create raw data comparison plots and statistical analysis using Plotly."""
        if len(self.matched_data) == 0:
            print("No data available for raw data comparison")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract raw thickness data
        dmt_raw = self.matched_data['DMT_Thickness'].values
        tfk_raw = self.matched_data['TFK_Thickness'].values
        
        # Statistical analysis
        stats_summary = self.create_statistical_summary(dmt_raw, tfk_raw, "Raw Thickness Data")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            subplot_titles=['Raw Thickness Data Trend', 'Raw Thickness Distribution', 'Statistical Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2, "type": "table"}, None]]
        )
        
        # Trend plot
        indices = np.arange(len(dmt_raw))
        fig.add_trace(
            go.Scatter(x=indices, y=dmt_raw, mode='lines', name='DMT Raw',
                      line=dict(color='blue', width=2), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=indices, y=tfk_raw, mode='lines', name='TFK Raw',
                      line=dict(color='red', width=2), opacity=0.7),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=dmt_raw, name='DMT', boxpoints='outliers',
                  marker_color='lightblue', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=tfk_raw, name='TFK', boxpoints='outliers',
                  marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        # Statistical summary table
        table_headers = ['Statistic', 'DMT', 'TFK']
        table_values = [
            ['Mean (Å)', 'Std Dev (Å)', 'Min (Å)', 'Max (Å)', 'N', '', 
             'Statistical Tests', 'Paired t-test', 'Independent t-test', 'Mann-Whitney U', 'Alpha (α)', '',
             'Equivalence Test (TOST)', 'Mean Difference', 'Equivalence Margin (±δ)', 'TOST p-value',
             '90% CI Lower', '90% CI Upper', 'Conclusion'],
            [f"{stats_summary['dmt_stats']['mean']:.2f}",
             f"{stats_summary['dmt_stats']['std']:.2f}",
             f"{stats_summary['dmt_stats']['min']:.2f}",
             f"{stats_summary['dmt_stats']['max']:.2f}",
             f"{stats_summary['dmt_stats']['n']}", '',
             'Test Statistic',
             f"{stats_summary['paired_t_test']['t_stat']:.3f}",
             f"{stats_summary['independent_t_test']['t_stat']:.3f}",
             f"{stats_summary['mann_whitney']['u_stat']:.1f}",
             f"{stats_summary['paired_t_test']['alpha']}", '',
             'Value',
             f"{stats_summary['equivalence_test']['mean_difference']:.2f} Å",
             f"{stats_summary['equivalence_test']['equivalence_margin']:.1f} Å",
             f"{stats_summary['equivalence_test']['tost_p_value']:.3f}",
             f"{stats_summary['equivalence_test']['ci_90_lower']:.2f} Å",
             f"{stats_summary['equivalence_test']['ci_90_upper']:.2f} Å",
             stats_summary['equivalence_test']['conclusion']],
            [f"{stats_summary['tfk_stats']['mean']:.2f}",
             f"{stats_summary['tfk_stats']['std']:.2f}",
             f"{stats_summary['tfk_stats']['min']:.2f}",
             f"{stats_summary['tfk_stats']['max']:.2f}",
             f"{stats_summary['tfk_stats']['n']}", '',
             'p-value',
             f"{stats_summary['paired_t_test']['p_value']:.3f}",
             f"{stats_summary['independent_t_test']['p_value']:.3f}",
             f"{stats_summary['mann_whitney']['p_value']:.3f}",
             '', '',
             'Result',
             '',
             '',
             '',
             '',
             '',
             '']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, fill_color='#E6E6FA', align='center'),
                cells=dict(values=table_values, fill_color='white', align='center'),
                columnwidth=[0.4, 0.3, 0.3]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Measurement Index", row=1, col=1)
        fig.update_yaxes(title_text="Thickness (Å)", row=1, col=1)
        fig.update_yaxes(title_text="Thickness (Å)", row=1, col=2)
        
        fig.update_layout(
            height=800,
            title_text="DMT vs TFK Raw Data Comparison",
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save as HTML
        output_path = os.path.join(output_dir, 'dmt_tfk_raw_data_comparison.html')
        fig.write_html(output_path)
        print(f"Raw data comparison saved to {output_dir}/dmt_tfk_raw_data_comparison.html")
    
    def create_mean_data_comparison(self, output_dir="dmt_tfk_comparison_results"):
        """Create mean data comparison plots by wafer using Plotly."""
        if len(self.matched_data) == 0:
            print("No data available for mean data comparison")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate means by wafer
        wafer_means = self.matched_data.groupby('WaferID').agg({
            'DMT_Thickness': 'mean',
            'TFK_Thickness': 'mean'
        }).reset_index()
        
        dmt_means = wafer_means['DMT_Thickness'].values
        tfk_means = wafer_means['TFK_Thickness'].values
        wafer_ids = wafer_means['WaferID'].values
        
        # Statistical analysis
        stats_summary = self.create_statistical_summary(dmt_means, tfk_means, "Mean Thickness Data by Wafer")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            subplot_titles=['Mean Thickness by Wafer', 'Mean Thickness Distribution', 'Statistical Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2, "type": "table"}, None]]
        )
        
        # Trend plot by wafer
        x_pos = np.arange(len(wafer_ids))
        wafer_labels = [f'W{i+1}' for i in range(len(wafer_ids))]
        
        fig.add_trace(
            go.Scatter(x=wafer_labels, y=dmt_means, mode='lines+markers', name='DMT Means',
                      line=dict(color='blue', width=2), marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=wafer_labels, y=tfk_means, mode='lines+markers', name='TFK Means',
                      line=dict(color='red', width=2), marker=dict(size=8)),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=dmt_means, name='DMT', boxpoints='outliers',
                  marker_color='lightblue', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=tfk_means, name='TFK', boxpoints='outliers',
                  marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        # Statistical summary table
        table_headers = ['Statistic', 'DMT', 'TFK']
        table_values = [
            ['Mean (Å)', 'Std Dev (Å)', 'Min (Å)', 'Max (Å)', 'N Wafers', '',
             'Statistical Tests', 'Paired t-test', 'Independent t-test', 'Mann-Whitney U', 'Alpha (α)'],
            [f"{stats_summary['dmt_stats']['mean']:.2f}",
             f"{stats_summary['dmt_stats']['std']:.2f}",
             f"{stats_summary['dmt_stats']['min']:.2f}",
             f"{stats_summary['dmt_stats']['max']:.2f}",
             f"{stats_summary['dmt_stats']['n']}", '',
             'Test Statistic',
             f"{stats_summary['paired_t_test']['t_stat']:.3f}",
             f"{stats_summary['independent_t_test']['t_stat']:.3f}",
             f"{stats_summary['mann_whitney']['u_stat']:.1f}",
             f"{stats_summary['paired_t_test']['alpha']}"],
            [f"{stats_summary['tfk_stats']['mean']:.2f}",
             f"{stats_summary['tfk_stats']['std']:.2f}",
             f"{stats_summary['tfk_stats']['min']:.2f}",
             f"{stats_summary['tfk_stats']['max']:.2f}",
             f"{stats_summary['tfk_stats']['n']}", '',
             'p-value',
             f"{stats_summary['paired_t_test']['p_value']:.3f}",
             f"{stats_summary['independent_t_test']['p_value']:.3f}",
             f"{stats_summary['mann_whitney']['p_value']:.3f}",
             '']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, fill_color='#E6E6FA', align='center'),
                cells=dict(values=table_values, fill_color='white', align='center'),
                columnwidth=[0.4, 0.3, 0.3]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Wafer Index", row=1, col=1)
        fig.update_yaxes(title_text="Mean Thickness (Å)", row=1, col=1)
        fig.update_yaxes(title_text="Mean Thickness (Å)", row=1, col=2)
        
        fig.update_layout(
            height=800,
            title_text="DMT vs TFK Mean Data Comparison by Wafer",
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save as HTML
        output_path = os.path.join(output_dir, 'dmt_tfk_mean_data_comparison.html')
        fig.write_html(output_path)
        print(f"Mean data comparison saved to {output_dir}/dmt_tfk_mean_data_comparison.html")
    
    def create_std_data_comparison(self, output_dir="dmt_tfk_comparison_results"):
        """Create standard deviation data comparison plots by wafer using Plotly."""
        if len(self.matched_data) == 0:
            print("No data available for standard deviation comparison")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate standard deviations by wafer
        wafer_stds = self.matched_data.groupby('WaferID').agg({
            'DMT_Thickness': 'std',
            'TFK_Thickness': 'std'
        }).reset_index()
        
        dmt_stds = wafer_stds['DMT_Thickness'].values
        tfk_stds = wafer_stds['TFK_Thickness'].values
        wafer_ids = wafer_stds['WaferID'].values
        
        # Statistical analysis
        stats_summary = self.create_statistical_summary(dmt_stds, tfk_stds, "Standard Deviation Data by Wafer")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            subplot_titles=['Thickness Standard Deviation by Wafer', 'Std Dev Distribution', 'Statistical Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2, "type": "table"}, None]]
        )
        
        # Trend plot by wafer
        x_pos = np.arange(len(wafer_ids))
        wafer_labels = [f'W{i+1}' for i in range(len(wafer_ids))]
        
        fig.add_trace(
            go.Scatter(x=wafer_labels, y=dmt_stds, mode='lines+markers', name='DMT Std Dev',
                      line=dict(color='blue', width=2), marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=wafer_labels, y=tfk_stds, mode='lines+markers', name='TFK Std Dev',
                      line=dict(color='red', width=2), marker=dict(size=8)),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=dmt_stds, name='DMT', boxpoints='outliers',
                  marker_color='lightblue', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=tfk_stds, name='TFK', boxpoints='outliers',
                  marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        # Statistical summary table
        table_headers = ['Statistic', 'DMT', 'TFK']
        table_values = [
            ['Mean Std (Å)', 'Std of Std (Å)', 'Min Std (Å)', 'Max Std (Å)', 'N Wafers', '',
             'Statistical Tests', 'Paired t-test', 'Independent t-test', 'Mann-Whitney U', 'Alpha (α)'],
            [f"{stats_summary['dmt_stats']['mean']:.2f}",
             f"{stats_summary['dmt_stats']['std']:.2f}",
             f"{stats_summary['dmt_stats']['min']:.2f}",
             f"{stats_summary['dmt_stats']['max']:.2f}",
             f"{stats_summary['dmt_stats']['n']}", '',
             'Test Statistic',
             f"{stats_summary['paired_t_test']['t_stat']:.3f}",
             f"{stats_summary['independent_t_test']['t_stat']:.3f}",
             f"{stats_summary['mann_whitney']['u_stat']:.1f}",
             f"{stats_summary['paired_t_test']['alpha']}"],
            [f"{stats_summary['tfk_stats']['mean']:.2f}",
             f"{stats_summary['tfk_stats']['std']:.2f}",
             f"{stats_summary['tfk_stats']['min']:.2f}",
             f"{stats_summary['tfk_stats']['max']:.2f}",
             f"{stats_summary['tfk_stats']['n']}", '',
             'p-value',
             f"{stats_summary['paired_t_test']['p_value']:.3f}",
             f"{stats_summary['independent_t_test']['p_value']:.3f}",
             f"{stats_summary['mann_whitney']['p_value']:.3f}",
             '']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, fill_color='#E6E6FA', align='center'),
                cells=dict(values=table_values, fill_color='white', align='center'),
                columnwidth=[0.4, 0.3, 0.3]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Wafer Index", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation (Å)", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation (Å)", row=1, col=2)
        
        fig.update_layout(
            height=800,
            title_text="DMT vs TFK Standard Deviation Comparison by Wafer",
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save as HTML
        output_path = os.path.join(output_dir, 'dmt_tfk_std_data_comparison.html')
        fig.write_html(output_path)
        print(f"Standard deviation data comparison saved to {output_dir}/dmt_tfk_std_data_comparison.html")
    
    def create_offset_corrected_comparison(self, output_dir="dmt_tfk_comparison_results"):
        """Create comparison after applying systematic offset correction."""
        if len(self.matched_data) == 0:
            print("No data available for offset-corrected comparison")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate systematic offset (mean difference)
        systematic_offset = self.matched_data['Thickness_Delta'].mean()
        
        # Apply offset correction to DMT data
        # If DMT measures lower than TFK (negative offset), we need to ADD the absolute value to DMT
        dmt_corrected = self.matched_data['DMT_Thickness'].values - systematic_offset
        tfk_raw = self.matched_data['TFK_Thickness'].values
        
        # Statistical analysis on corrected data
        stats_summary = self.create_statistical_summary(dmt_corrected, tfk_raw, "Offset-Corrected Data")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.25, 0.75],
            subplot_titles=['Corrected Data Comparison', 'Corrected Distribution', 'Offset-Corrected Equivalence Analysis'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2, "type": "table"}, None]]
        )
        
        # Corrected data trend
        indices = np.arange(len(dmt_corrected))
        fig.add_trace(
            go.Scatter(x=indices, y=dmt_corrected, mode='lines', name='DMT Corrected',
                      line=dict(color='green', width=2), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=indices, y=tfk_raw, mode='lines', name='TFK Raw',
                      line=dict(color='red', width=2), opacity=0.7),
            row=1, col=1
        )
        
        # Box plot comparison
        fig.add_trace(
            go.Box(y=dmt_corrected, name='DMT Corrected', boxpoints='outliers',
                  marker_color='lightgreen', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=tfk_raw, name='TFK', boxpoints='outliers',
                  marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        # Create comprehensive table data
        table_headers = ['Parameter', 'Value', 'Interpretation']
        table_rows = [
            ['Systematic Offset', f"{systematic_offset:.2f} Å", 'Applied to DMT'],
            ['', '', ''],
            ['Corrected Statistics', 'DMT Corrected', 'TFK'],
            ['Mean (Å)', f"{stats_summary['dmt_stats']['mean']:.2f}", f"{stats_summary['tfk_stats']['mean']:.2f}"],
            ['Std Dev (Å)', f"{stats_summary['dmt_stats']['std']:.2f}", f"{stats_summary['tfk_stats']['std']:.2f}"],
            ['', '', ''],
            ['Equivalence Test (TOST)', 'Value', 'Result'],
            ['Mean Difference', f"{stats_summary['equivalence_test']['mean_difference']:.2f} Å", ''],
            ['Equivalence Margin', f"±{stats_summary['equivalence_test']['equivalence_margin']:.1f} Å", ''],
            ['TOST p-value', f"{stats_summary['equivalence_test']['tost_p_value']:.3f}", '< 0.05 = Equivalent'],
            ['90% CI Range', f"[{stats_summary['equivalence_test']['ci_90_lower']:.2f}, {stats_summary['equivalence_test']['ci_90_upper']:.2f}]", ''],
            ['Conclusion', stats_summary['equivalence_test']['conclusion'], ''],
            ['', '', ''],
            ['Interpretation:', 'After applying systematic offset correction,', ''],
            ['', 'the datasets are statistically', ''],
            ['', stats_summary['equivalence_test']['conclusion'].lower(), '']
        ]
        
        # Transpose for table display
        table_transposed = list(map(list, zip(*table_rows)))
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, 
                           fill_color='#E6E6FA', align='center'),
                cells=dict(values=table_transposed, 
                          fill_color='white', 
                          align='center'),
                columnwidth=[0.35, 0.35, 0.30]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Measurement Index", row=1, col=1)
        fig.update_yaxes(title_text="Thickness (Å)", row=1, col=1)
        fig.update_yaxes(title_text="Thickness (Å)", row=1, col=2)
        
        fig.update_layout(
            height=800,
            title_text="DMT vs TFK Offset-Corrected Comparison",
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save as HTML
        output_path = os.path.join(output_dir, 'dmt_tfk_offset_corrected_comparison.html')
        fig.write_html(output_path)
        
        print(f"Offset-corrected comparison saved to {output_dir}/dmt_tfk_offset_corrected_comparison.html")
        print(f"Applied systematic offset: {systematic_offset:.2f} Å to DMT measurements")
        print(f"Equivalence test result: {stats_summary['equivalence_test']['conclusion']}")
    
    def create_coordinate_delta_summary(self, output_dir="dmt_tfk_comparison_results"):
        """Create summary of thickness deltas by X,Y coordinate combinations."""
        if len(self.matched_data) == 0:
            print("No data available for coordinate summary!")
            return
        
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
    
    def save_results(self, output_dir="dmt_tfk_comparison_results"):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matched data
        if len(self.matched_data) > 0:
            self.matched_data.to_csv(os.path.join(output_dir, 'dmt_tfk_matched_data.csv'), index=False)
            print(f"Matched data saved to {output_dir}/dmt_tfk_matched_data.csv")
            
            # Create coordinate delta summary
            self.create_coordinate_delta_summary(output_dir)
        
        # Save analysis summary
        with open(os.path.join(output_dir, 'dmt_tfk_analysis_summary.txt'), 'w') as f:
            f.write("DMT vs TFK Thickness Comparison Analysis\n")
            f.write("="*50 + "\n\n")
            
            if self.analysis_results:
                f.write(f"Total matched pairs: {self.analysis_results['total_matches']}\n")
                f.write(f"Unique wafers: {self.analysis_results['unique_wafers']}\n")
                f.write(f"Mean thickness delta (DMT - TFK): {self.analysis_results['mean_delta']:.3f} ± {self.analysis_results['std_delta']:.3f} Å\n")
                f.write(f"Median delta: {self.analysis_results['median_delta']:.3f} Å\n")
                f.write(f"Delta range: {self.analysis_results['min_delta']:.3f} to {self.analysis_results['max_delta']:.3f} Å\n")
                f.write(f"25th percentile: {self.analysis_results['q25_delta']:.3f} Å\n")
                f.write(f"75th percentile: {self.analysis_results['q75_delta']:.3f} Å\n")
                f.write(f"Mean matching distance: {self.analysis_results['mean_distance']:.3f} mm\n")
                f.write(f"Distance threshold used: {self.distance_threshold} mm\n")
        
        print(f"Analysis summary saved to {output_dir}/dmt_tfk_analysis_summary.txt")
    
    def run_complete_analysis(self):
        """Run the complete DMT vs TFK analysis workflow."""
        print("Starting DMT vs TFK Thickness Comparison Analysis")
        print("="*60)
        
        self.load_data()
        
        if len(self.dmt_data) == 0 or len(self.tfk_data) == 0:
            print("Cannot proceed with empty datasets!")
            return
        
        self.find_matching_points()
        self.analyze_thickness_differences()
        self.create_visualizations()
        
        # Create additional statistical comparison plots
        print("\nGenerating additional statistical comparison plots...")
        self.create_raw_data_comparison()
        self.create_mean_data_comparison()
        self.create_std_data_comparison()
        self.create_offset_corrected_comparison()
        
        self.save_results()
        
        print("\nDMT vs TFK analysis completed!")

def main():
    # Define folders
    dmt_folder = "DMT"
    tfk_folder = "TFK"
    
    # Create and run analysis
    app = DMTTFKComparisonApp(dmt_folder, tfk_folder, distance_threshold=4.0)
    app.run_complete_analysis()

if __name__ == "__main__":
    main()