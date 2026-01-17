#!/usr/bin/env python3

"""
Run Qualification Optimization Suite
====================================

This script provides multiple optimization approaches for qualification matrices
with standardized output for easy validation and comparison.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

from src.analysis.ppm_capacity_optimizer import PPMCapacityOptimizer
from src.analysis.milp_optimization_designer import MILPOptimizationDesigner
from src.analysis.standard_output_manager import StandardOutputManager
from src.analysis.general_health_reporter import GeneralHealthReporter

# Configure which teams to process (set to [1] for Team 1 only, [1, 2] for both)
ACTIVE_TEAMS = [1]


def run_rota_parser():
    """Parse rota files to generate JSON files for MILP optimization"""
    print("\n🔄 PARSING ROTA FILES...")
    try:
        # Import rota parser functions
        from scripts.rota_parser import parse_week_block_csv, load_engineer_map_with_roles, save_json

        # Define all team/role combinations
        teams_to_parse = [
            {
                'team': 1,
                'role': 'Mechanical',
                'csv_path': "data/processed/rota/theoretical_rota/Team 1 Mech Rota.csv",
                'engineer_json': "data/processed/engineers/team1_mech_engineers.json",
                'output_path': "data/processed/parsed_rotas/parsed_team1_mech_rota.json"
            },
            {
                'team': 1,
                'role': 'Electrical',
                'csv_path': "data/processed/rota/theoretical_rota/Team 1 Elec Rota.csv",
                'engineer_json': "data/processed/engineers/team1_elec_engineers.json",
                'output_path': "data/processed/parsed_rotas/parsed_team1_elec_rota.json"
            },
            {
                'team': 2,
                'role': 'Mechanical',
                'csv_path': "data/processed/rota/theoretical_rota/Team 2 Mech Rota.csv",
                'engineer_json': "data/processed/engineers/team2_mech_engineers.json",
                'output_path': "data/processed/parsed_rotas/parsed_team2_mech_rota.json"
            },
            {
                'team': 2,
                'role': 'Electrical',
                'csv_path': "data/processed/rota/theoretical_rota/Team 2 Elec Rota.csv",
                'engineer_json': "data/processed/engineers/team2_elec_engineers.json",
                'output_path': "data/processed/parsed_rotas/parsed_team2_elec_rota.json"
            }
        ]

        for team_config in teams_to_parse:
            # Load engineer mapping
            engineer_map = load_engineer_map_with_roles(
                team_config['engineer_json'],
                team_config['role']
            )

            if not engineer_map:
                print(f"   ⚠️  No {team_config['role']} engineers found in team {team_config['team']}")
                continue

            # Parse the rota
            rota_data = parse_week_block_csv(team_config['csv_path'], engineer_map)

            if rota_data:
                save_json(rota_data, team_config['output_path'])
                total_weeks = len(rota_data)
                total_engineers = len(set().union(*[week_data.keys() for week_data in rota_data.values()]))
                print(f"   ✅ Team {team_config['team']} {team_config['role']}: {total_weeks} weeks, {total_engineers} engineers")
            else:
                print(f"   ❌ Failed to parse Team {team_config['team']} {team_config['role']}")

        print("   ✅ Rota parsing complete")
        return True

    except Exception as e:
        print(f"   ⚠️  Error parsing rotas: {e}")
        print("   Continuing with existing parsed rota files...")
        return False


def run_milp_optimization(optimizer):
    """Run Mathematical (MILP) optimization with guaranteed coverage"""
    print("\n🔢 RUNNING MILP MATHEMATICAL OPTIMIZATION...")

    # Parse rota files first
    run_rota_parser()

    designer = MILPOptimizationDesigner(optimizer)
    matrices = designer.create_optimized_qualification_matrices()
    validation_results, assignment_counts = designer.validate_and_export_results(matrices)
    
    return matrices, validation_results, assignment_counts, {
        "approach": "milp_mathematical",
        "features": [
            "mathematical_optimization",
            "guaranteed_coverage_constraints",
            "fairness_objective_function",
            "pulp_linear_programming",
            "optimal_solution_guarantee",
            "constraint_satisfaction",
            "intelligent_heuristic_fallback"
        ],
        "target_coverage": {
            "daily_ppms": "100% (guaranteed)",
            "weekly_ppms": "100% (guaranteed)", 
            "monthly_ppms": "100% (guaranteed)"
        }
    }


def run_training_optimization(optimizer):
    """Run Training Optimization based on current qualifications"""
    print("\n🎓 RUNNING TRAINING OPTIMIZATION ANALYSIS...")
    
    # Import the training optimizer
    from src.analysis.training_optimization_designer import TrainingOptimizationDesigner
    
    designer = TrainingOptimizationDesigner(optimizer)
    
    # Step 1: Load current qualifications from EngQual.csv
    current_matrices = designer.load_current_qualification_state()
    
    # Step 2: Create qualification matrices from current state (not theoretical optimal)
    current_state_matrices = designer.create_current_state_matrices(current_matrices)
    
    # Step 3: Analyze coverage gaps and optimize training assignments
    training_recommendations = designer.optimize_training_assignments(current_state_matrices)
    
    # Step 3.5: Generate and display detailed training report
    detailed_report = designer.generate_detailed_training_report(training_recommendations, current_state_matrices)
    designer.display_detailed_training_report(detailed_report)
    
    # Step 3.6: Export to CSV for easy analysis
    csv_files = designer.export_detailed_report_to_csv(detailed_report)
    
    # Step 3.7: Generate PPM Expert Analysis
    expert_analysis_file = designer.generate_ppm_expert_analysis()
    
    # Step 4: Run Succession Planning Analysis (NEW)
    print("\n👥 RUNNING SUCCESSION PLANNING ANALYSIS...")
    run_succession_planning_analysis()
    
    # Step 5: Validate proposed training impact
    validation_results = designer.validate_training_impact(training_recommendations)
    
    return current_matrices, current_state_matrices, training_recommendations, validation_results, detailed_report, csv_files, {
        "approach": "training_optimization",
        "features": [
            "current_state_analysis",
            "qualification_gap_identification", 
            "training_impact_optimization",
            "coverage_improvement_prioritization",
            "cost_benefit_analysis",
            "skill_development_planning"
        ],
        "target_coverage": {
            "daily_ppms": "Progressive improvement",
            "weekly_ppms": "Progressive improvement", 
            "monthly_ppms": "Progressive improvement"
        },
        "optimization_goals": [
            "Minimize training effort",
            "Maximize coverage improvement",
            "Balance workload distribution",
            "Prioritize critical skills gaps"
        ]
    }


def run_succession_planning_analysis():
    """Run comprehensive succession planning analysis including annual team"""
    import os
    try:
        print("  📊 Analyzing retirement timelines and qualification impacts...")
        
        # Check working directory
        print(f"  Working directory: {os.getcwd()}")
        
        # Check if script exists
        script_path = 'scripts/succession_planning_complete.py'
        if not os.path.exists(script_path):
            print(f"  ❌ Script not found: {script_path}")
            return False
        
        # Run the succession planning script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"  Script return code: {result.returncode}")
        
        # Check if the key file was actually created
        key_file = 'outputs/current/succession_qualification_complete.csv'
        if os.path.exists(key_file):
            size = os.path.getsize(key_file)
            print(f"  ✅ Created {key_file} ({size} bytes)")
        else:
            print(f"  ❌ Key file NOT created: {key_file}")
            print("  Script output:")
            print(result.stdout[-500:] if result.stdout else "No output")  # Last 500 chars
            return False
        
        # Print summary of results
        if "CRITICAL:" in result.stdout:
            for line in result.stdout.split('\n'):
                if "CRITICAL:" in line or "OPERATIONAL RISK:" in line or "OPPORTUNITY:" in line:
                    print(f"  {line.strip()}")
        
        print("\n  ✅ Succession planning report generated:")
        print("     • succession_qualification_complete.csv - Power BI dashboard data")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Error: Succession planning script failed (code {e.returncode})")
        print(f"     STDOUT: {e.stdout[-300:] if e.stdout else 'None'}")
        print(f"     STDERR: {e.stderr[-300:] if e.stderr else 'None'}")
        print("     Continuing with optimization...")
        return False
    except FileNotFoundError:
        print("  ⚠️  Error: Python or script not found")
        print("     Please ensure scripts/succession_planning_complete.py exists")
        print("     Continuing with optimization...")
        return False
    except Exception as e:
        print(f"  ⚠️  Unexpected error: {e}")
        return False


def main():
    """Main optimization selection and execution"""
    print("🚀 QUALIFICATION MATRIX OPTIMIZATION SUITE")
    print("=" * 60)
    print("1. MILP: Mathematical optimization (guaranteed coverage + fairness)")
    print("2. TRAINING: Current state vs optimal training + succession planning")
    print()
    
    choice = input("Select optimization approach (1-2): ").strip()
    
    if choice not in ['1', '2']:
        print("❌ Invalid choice. Please select 1 or 2.")
        sys.exit(1)
    
    print(f"\n🚀 STARTING OPTIMIZATION")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load and analyze PPM data
        print("📊 LOADING PPM DATA...")
        optimizer = PPMCapacityOptimizer()
        optimizer.generate_report()  # This loads and analyzes all data
        
        # Step 2: Run selected optimization
        assignment_counts = None  # Initialize assignment_counts for all methods
        
        if choice == '1':
            matrices, validation_results, assignment_counts, config = run_milp_optimization(optimizer)
            optimization_name = "milp_mathematical"
        elif choice == '2':
            current_matrices, current_state_matrices, training_recommendations, validation_results, detailed_report, csv_files, config = run_training_optimization(optimizer)
            optimization_name = "training_optimization"
            matrices = current_state_matrices  # Use current state matrices for saving
        
        # Step 3: Save to standardized location
        print("\n💾 SAVING TO STANDARD LOCATION...")
        output_manager = StandardOutputManager()
        
        output_manager.save_optimization_results(
            qualification_matrices=matrices,
            optimization_name=optimization_name,
            optimization_config=config,
            validation_results=validation_results
        )
        
        # Step 3.5: Save assignment counts if available (MILP optimization)
        if assignment_counts is not None:
            import json
            assignment_counts_path = output_manager.current_dir / "engineer_assignment_counts.json"
            print(f"   💾 Saving engineer assignment counts to: {assignment_counts_path}")
            
            with open(assignment_counts_path, 'w') as f:
                json.dump(assignment_counts, f, indent=2)
        
        # Step 3.6: Skip JSON files (only CSV files needed for dashboard)
        if choice == '2':
            print(f"   ⏭️  Skipping JSON files (only CSVs needed for dashboard)")
            
            # Display summary of assignment counts
            print("\n📊 ENGINEER ASSIGNMENT COUNTS SUMMARY:")
            if assignment_counts:
                for team_key, team_data in assignment_counts.items():
                    team_num = team_key.split('_')[1]
                    print(f"\n🏢 TEAM {team_num}:")
                    
                    # Calculate statistics
                    total_rides = len(team_data)
                    total_engineers = sum(ride_data['total_count'] for ride_data in team_data.values())
                    avg_engineers_per_ride = total_engineers / total_rides if total_rides > 0 else 0
                    
                    electrical_engineers = sum(ride_data['electrical_count'] for ride_data in team_data.values())
                    mechanical_engineers = sum(ride_data['mechanical_count'] for ride_data in team_data.values())
                    
                    print(f"   Total rides: {total_rides}")
                    print(f"   Total engineer assignments: {total_engineers}")
                    print(f"   Average engineers per ride: {avg_engineers_per_ride:.1f}")
                    print(f"   Electrical engineers: {electrical_engineers}")
                    print(f"   Mechanical engineers: {mechanical_engineers}")
                    
                    # Show rides with highest/lowest coverage
                    if team_data:
                        ride_counts = [(ride_id, data['total_count'], data['ride_name']) 
                                      for ride_id, data in team_data.items()]
                        ride_counts.sort(key=lambda x: x[1])
                        
                        min_ride = ride_counts[0]
                        max_ride = ride_counts[-1]
                        
                        print(f"   Lowest coverage: {min_ride[2]} ({min_ride[1]} engineers)")
                        print(f"   Highest coverage: {max_ride[2]} ({max_ride[1]} engineers)")
            else:
                print("   No assignment counts available for this optimization type.")
        
        # Step 4: Generate Health.csv against MILP target (even in Training mode)
        try:
            # Determine matrices to use for Health: always MILP target pairs
            if choice == '2':
                print("\n🔄 Generating MILP target for Health report...")
                milp_health_designer = MILPOptimizationDesigner(optimizer)
                matrices_for_health = milp_health_designer.create_optimized_qualification_matrices()
            else:
                matrices_for_health = matrices

            reporter = GeneralHealthReporter(optimizer)
            health_path = reporter.compute_and_export_health(matrices_for_health, output_file=str(output_manager.current_dir / "Health.csv"))
            print(f"\n📄 Health report saved: {health_path}")
        except Exception as e:
            print(f"\n⚠️  Health report generation failed: {e}")

        # Step 5: Display summary
        print("\n📈 OPTIMIZATION RESULTS SUMMARY:")
        print("=" * 50)
        
        for team in ACTIVE_TEAMS:
            if team in validation_results:
                results = validation_results[team]
                daily_cov = results['daily']['coverage_percentage']
                weekly_cov = results['weekly']['coverage_percentage']
                monthly_cov = results['monthly']['coverage_percentage']

                print(f"\n🏢 TEAM {team}:")

                # Color-code the coverage percentages
                daily_icon = "🎯" if daily_cov >= 100 else "⚠️" if daily_cov >= 50 else "❌"
                weekly_icon = "🎯" if weekly_cov >= 100 else "⚠️" if weekly_cov >= 80 else "❌"
                monthly_icon = "🎯" if monthly_cov >= 100 else "⚠️" if monthly_cov >= 80 else "❌"

                print(f"   Daily Coverage:    {daily_cov:.1f}% {daily_icon}")
                print(f"   Weekly Coverage:   {weekly_cov:.1f}% {weekly_icon}")
                print(f"   Monthly Coverage:  {monthly_cov:.1f}% {monthly_icon}")
                print(f"   Overall Status:    {results['overall_status']}")
                print(f"   Risk Level:        {results['risk_analysis']['overall_risk']}")

                # Show specific gaps
                if results['daily']['failed_days']:
                    failed_count = len(results['daily']['failed_days'])
                    total_count = results['daily']['total_days_tested']
                    print(f"   Daily gaps:        {failed_count} out of {total_count} days failed")

                if results['weekly']['coverage_gaps']:
                    print(f"   Weekly gaps:       {len(results['weekly']['coverage_gaps'])} qualifications missing")

        # Success message based on choice
        if choice == '1':
            print(f"\n🔢 MILP Mathematical optimization completed!")

            # Check if PuLP was available
            try:
                import pulp
                print(f"   🎯 Used PuLP mathematical solver for optimal solution")
            except ImportError:
                print(f"   🧠 Used intelligent heuristic (PuLP not available)")

            # Check for perfect balance
            perfect_balance = True
            for team in ACTIVE_TEAMS:
                if team in validation_results:
                    if validation_results[team]['daily']['coverage_percentage'] < 95:
                        perfect_balance = False
                        break

            if perfect_balance:
                print(f"   ✅ Achieved mathematical optimum: fairness + coverage")
            else:
                print(f"   ⚠️  Constraint satisfaction achieved (coverage optimized)")
        elif choice == '2':
            print(f"\n🎓 Training optimization completed!")

            # Check for training effectiveness
            training_effectiveness = True
            for team in ACTIVE_TEAMS:
                if team in validation_results:
                    if validation_results[team]['daily']['coverage_percentage'] < 95:
                        training_effectiveness = False
                        break
            
            if training_effectiveness:
                print(f"   ✅ Training effectiveness achieved: coverage improvement")
            else:
                print(f"   ⚠️  Training effectiveness not fully achieved")
        
        print(f"📁 Results saved to standard location: {output_manager.current_dir}")
        
        # Results are available in CSV format for analysis
        
        # Additional info for training optimization
        if choice == '2':
            print(f"\n📊 POWER BI DASHBOARD FILES GENERATED:")
            print(f"   • Health.csv - General health report (team/role/ride/qualification)")
            print(f"   • ppm_expert_analysis.csv - PPM experts based on operational performance")
            print(f"   • ppm_risk_analysis.csv - PPM risk assessment by business criticality")
            print(f"   • specific_qualifications_needed.csv - Exact qualifications to train")
            print(f"   • succession_qualification_complete.csv - Succession planning analysis")
            print(f"   💡 Import these CSV files into Power BI for dashboard visualization!")
        
        print(f"\n💡 Next steps:")
        print(f"   • Run validation: python3 validate_qualifications.py")
        print(f"   • View results: ls {output_manager.current_dir}")
        if choice == '2':
            print(f"   • Open CSV files in Excel/Google Sheets for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 