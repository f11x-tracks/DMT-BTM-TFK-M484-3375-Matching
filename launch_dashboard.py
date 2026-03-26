#!/usr/bin/env python3
"""
Launch script for the Thickness Analysis System
Supports DMT vs TFK, BTM vs DMT, and BTM vs TFK comparisons
"""

import subprocess
import sys
import os
from pathlib import Path

def show_menu():
    """Display the main menu options"""
    print("=" * 70)
    print("THICKNESS MEASUREMENT COMPARISON SYSTEM")
    print("=" * 70)
    print()
    print("Available comparison types:")
    print("1. DMT vs TFK Comparison (Original)")
    print("2. BTM vs DMT Comparison (New)")
    print("3. BTM vs TFK Comparison (New)")
    print("4. Run All Comparisons")
    print("5. Launch Dashboard (DMT vs TFK)")
    print("6. Exit")
    print()
    return input("Select an option (1-6): ").strip()

def run_comparison(comparison_type, script_dir):
    """Run a specific thickness comparison analysis"""
    
    comparison_scripts = {
        'dmt_tfk': 'thickness_comparison_app.py',
        'btm_dmt': 'btm_dmt_comparison.py', 
        'btm_tfk': 'btm_tfk_comparison.py'
    }
    
    script_name = comparison_scripts.get(comparison_type)
    if not script_name:
        print(f"Error: Unknown comparison type '{comparison_type}'")
        return 1
        
    script_path = script_dir / script_name
    
    # Check if the script exists
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return 1
    
    # Use the virtual environment Python executable
    python_exe = script_dir / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        # Fallback to sys.executable if venv not found
        python_exe = sys.executable
    
    # Get comparison name for display
    comparison_names = {
        'dmt_tfk': 'DMT vs TFK',
        'btm_dmt': 'BTM vs DMT',
        'btm_tfk': 'BTM vs TFK'
    }
    
    print(f"\nRunning {comparison_names[comparison_type]} comparison...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            str(python_exe),
            str(script_path)
        ], cwd=script_dir)
        
        if result.returncode == 0:
            print(f"\n{comparison_names[comparison_type]} analysis completed successfully!")
        else:
            print(f"\nError: {comparison_names[comparison_type]} analysis failed with return code {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running {comparison_names[comparison_type]} analysis: {e}")
        return 1

def launch_dashboard(script_dir):
    """Launch the thickness analysis dashboard"""
    
    dashboard_script = script_dir / "thickness_dashboard.py"
    
    # Check if the dashboard file exists
    if not dashboard_script.exists():
        print(f"Error: Dashboard script not found at {dashboard_script}")
        return 1
        
    # Check if required data file exists
    data_file = script_dir / "matched_thickness_data.csv"
    if not data_file.exists():
        print("Error: matched_thickness_data.csv not found.")
        print("Please run the DMT vs TFK comparison (option 1) first to generate the data.")
        return 1
    
    # Use the virtual environment Python executable
    python_exe = script_dir / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        # Fallback to sys.executable if venv not found
        python_exe = sys.executable
    
    print("=" * 60)
    print("DMT vs TFK Thickness Analysis Dashboard")
    print("=" * 60)
    print()
    print("Starting dashboard server...")
    print("Once started, open your web browser to: http://127.0.0.1:8050/")
    print()
    print("The dashboard includes:")
    print("• Site-by-site thickness trends with delta box plots")
    print("• Wafer average thickness trends with distributions")
    print("• Wafer standard deviation analysis")
    print("• Detailed summary table")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run the dashboard
        result = subprocess.run([
            str(python_exe), 
            str(dashboard_script)
        ], cwd=script_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
        return 0
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return 1

def main():
    """Main program loop"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # DMT vs TFK comparison
            result = run_comparison('dmt_tfk', script_dir)
            if result == 0:
                print("\nData file 'matched_thickness_data.csv' is ready for the dashboard.")
            
        elif choice == '2':
            # BTM vs DMT comparison
            run_comparison('btm_dmt', script_dir)
            
        elif choice == '3':
            # BTM vs TFK comparison 
            run_comparison('btm_tfk', script_dir)
            
        elif choice == '4':
            # Run all comparisons
            print("\nRunning all thickness comparisons...")
            print("=" * 50)
            
            comparisons = [('dmt_tfk', 'DMT vs TFK'), 
                          ('btm_dmt', 'BTM vs DMT'), 
                          ('btm_tfk', 'BTM vs TFK')]
            
            all_successful = True
            for comp_type, comp_name in comparisons:
                print(f"\nStarting {comp_name}...")
                result = run_comparison(comp_type, script_dir)
                if result != 0:
                    all_successful = False
                    print(f"Warning: {comp_name} comparison failed!")
                    
            if all_successful:
                print("\nAll comparisons completed successfully!")
            else:
                print("\nSome comparisons failed. Check the output above.")
                
        elif choice == '5':
            # Launch dashboard
            return launch_dashboard(script_dir)
            
        elif choice == '6':
            # Exit
            print("Goodbye!")
            return 0
            
        else:
            print("Invalid choice. Please select 1-6.")
        
        # Ask if user wants to continue
        print("\n" + "-" * 50)
        continue_choice = input("Would you like to run another analysis? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main())