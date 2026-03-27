run
thickness_comparison_app - to create data used for other apps
run 
btm_dmt and btm_tfk comparisons
run wafer_map_analys and wafer_map_analysis_contour

# Thickness Measurement Comparison System

A comprehensive tool for comparing thickness measurements between different measurement tools: DMT, BTM, and TFK.

## Overview

This system analyzes and compares thickness measurements from three different tools:
- **DMT**: XML format with coordinates in mm
- **BTM**: CSV format with coordinates in mm  
- **TFK**: XML format with native coordinate units (converted to mm)

## Features

### Three Comparison Types
1. **DMT vs TFK** (Original) - Compare Direct Measurement Tool vs Thin Film Kit
2. **BTM vs DMT** (New) - Compare Bottom Tool Measurement vs Direct Measurement Tool
3. **BTM vs TFK** (New) - Compare Bottom Tool Measurement vs Thin Film Kit

### Analysis Features
- Spatial coordinate matching within 4mm threshold
- Thickness delta analysis with statistical summaries
- Spatial visualization of measurement differences
- Box plots and distribution analysis by wafer
- Interactive web dashboard for DMT vs TFK comparison

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas>=1.3.0
- numpy>=1.21.0  
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scipy>=1.8.0
- dash>=2.14.0
- plotly>=5.15.0

## Data Formats

### DMT Data (XML)
- Files: `DMT/*.xml`
- Thickness: `<Label>Layer 1 Thickness</Label>` → `<Datum>` (Angstrom)
- Coordinates: `<XWaferLoc>`, `<YWaferLoc>` (mm)

### TFK Data (XML)  
- Files: `TFK/*.xml`
- Thickness: `<Label>T1</Label>` → `<Datum>` (Angstrom)
- Coordinates: `<XNative>`, `<YNative>` (converted: divide by 10,000,000 to get mm)

### BTM Data (CSV)
- Files: `BTM/*.csv`
- Thickness: `Film Thickness` column (Angstrom)
- Coordinates: `X[mm]`, `Y[mm]` columns (mm)

## Usage

### Quick Start
Run the interactive launcher:
```bash
python launch_dashboard.py
```

This provides a menu with options:
1. DMT vs TFK Comparison (Original)
2. BTM vs DMT Comparison (New)  
3. BTM vs TFK Comparison (New)
4. Run All Comparisons
5. Launch Dashboard (DMT vs TFK)
6. Exit

### Individual Scripts

#### DMT vs TFK Comparison
```bash
python thickness_comparison_app.py
```
- Creates: `matched_thickness_data.csv`, various plots
- Output folder: `comparison_results/`

#### BTM vs DMT Comparison  
```bash
python btm_dmt_comparison.py
```
- Creates: `btm_dmt_matched_data.csv`, analysis plots
- Output folder: `btm_dmt_comparison_results/`

#### BTM vs TFK Comparison
```bash
python btm_tfk_comparison.py  
```
- Creates: `btm_tfk_matched_data.csv`, analysis plots
- Output folder: `btm_tfk_comparison_results/`

#### Interactive Dashboard
```bash
python thickness_dashboard.py
```
- Requires: `matched_thickness_data.csv` from DMT vs TFK comparison
- Opens web interface at: http://127.0.0.1:8050/

## Output Files

### Analysis Results
Each comparison generates:
- **Matched Data CSV**: All matched point pairs with deltas
- **Summary Report**: Statistical analysis in text format
- **Visualization Plots**: 2x2 grid with correlation, distribution, and spatial analysis

### DMT vs TFK Extended Output
- Spatial delta plots with color mapping
- Radial analysis with spline fits
- Location ranking tables (best/worst agreement)
- Wafer-averaged spatial maps

## Key Analysis Parameters

- **Distance Threshold**: 4.0mm (points matched within this distance)
- **Coordinate System**: All measurements converted to mm units
- **Thickness Units**: Angstrom (Å) for all tools
- **Matching Algorithm**: Euclidean distance using scipy.spatial.distance.cdist

## Results Interpretation

### Thickness Delta
- **Positive Delta**: First tool reads higher than second tool
- **Negative Delta**: First tool reads lower than second tool
- Example: BTM - DMT = +2.5Å means BTM reads 2.5Å thicker than DMT

### Spatial Analysis
- Color maps show spatial patterns in measurement differences
- Wafer maps identify regions with systematic bias
- Radial analysis reveals center-to-edge trends

## File Structure
```
project/
├── BTM/                    # BTM CSV data files
├── DMT/                    # DMT XML data files  
├── TFK/                    # TFK XML data files
├── btm_dmt_comparison.py   # BTM vs DMT analysis
├── btm_tfk_comparison.py   # BTM vs TFK analysis
├── thickness_comparison_app.py    # DMT vs TFK analysis
├── thickness_dashboard.py         # Interactive dashboard
├── launch_dashboard.py     # Main launcher script
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## Advanced Features

### Coordinate Matching
- Uses scipy.spatial.distance for efficient large-scale matching
- Handles multiple measurements per location
- Reports actual matching distances for quality assessment

### Statistical Analysis
- Mean, median, standard deviation of deltas
- Quartile analysis (25th, 75th percentiles)
- Per-wafer and overall statistics

### Visualization Options
- Scatter plots with 1:1 reference lines
- Histogram distributions with mean indicators
- Box plots for wafer-by-wafer comparison
- Spatial heatmaps with customizable color scales

## Troubleshooting

### Common Issues
1. **No matching points found**: Check wafer IDs match between datasets
2. **Coordinate mismatch**: Verify coordinate units and TFK conversion factor
3. **Empty datasets**: Check file paths and data format compatibility

### Debug Tips
- Check console output for data loading statistics
- Verify wafer ID lists match between tools
- Confirm coordinate ranges are reasonable (typically -150 to +150 mm)

## Technical Details

### Coordinate Conversion
- **DMT**: Direct use of XWaferLoc/YWaferLoc (already in mm)
- **TFK**: XNative/10,000,000 and YNative/10,000,000 to convert to mm
- **BTM**: Direct use of X[mm]/Y[mm] columns (already in mm)

### Performance
- Efficient matrix operations using NumPy
- Vectorized distance calculations with scipy
- Memory-efficient processing for large datasets