#!/usr/bin/env python3

"""
Validate Current Qualification Matrices
=======================================

This script validates the current qualification matrices using the
standardized output approach. Much simpler and more maintainable!
"""

from src.analysis.coverage_validator import CoverageValidator
from src.analysis.standard_output_manager import StandardOutputManager
import json
from pathlib import Path

def main():
    print("🧪 VALIDATE CURRENT QUALIFICATION MATRICES")
    print("=" * 60)
    
    # Initialize managers
    output_manager = StandardOutputManager()
    validator = CoverageValidator()
    
    # Check if current matrices exist
    current_matrices = output_manager.load_current_matrices()
    current_metadata = output_manager.load_current_metadata()
    
    if not current_matrices:
        print("\n❌ No current qualification matrices found!")
        print("   Run an optimization first:")
        print("   python3 run_optimization.py")
        return
    
    # Show what we're validating
    print(f"\n📋 CURRENT MATRICES TO VALIDATE:")
    if current_metadata:
        print(f"   Optimization: {current_metadata['optimization_name']}")
        print(f"   Created: {current_metadata['created_timestamp']}")
        print(f"   Teams: {current_metadata['teams_included']}")
        print(f"   Engineers: {current_metadata['total_engineers']}")
    else:
        print("   (No metadata found - matrices may be from older format)")
    
    print(f"   Location: {output_manager.current_dir}")
    
    # Run validation
    print(f"\n🔍 RUNNING VALIDATION...")
    validation_results = validator.validate_assignment_coverage(current_matrices)
    
    # Save validation results
    output_manager.save_optimization_results(
        qualification_matrices=current_matrices,
        optimization_name=current_metadata['optimization_name'] if current_metadata else 'unknown',
        optimization_config=current_metadata.get('optimization_config', {}) if current_metadata else {},
        validation_results=validation_results
    )
    
    # Display results summary
    print(f"\n📊 VALIDATION RESULTS SUMMARY:")
    print("=" * 50)
    
    for team in [1, 2]:
        if team in validation_results:
            result = validation_results[team]
            print(f"\n🏢 TEAM {team}:")
            print(f"   Daily PPMs:    {result['daily']['coverage_percentage']:.1f}% coverage")
            print(f"   Weekly PPMs:   {result['weekly']['coverage_percentage']:.1f}% coverage")
            print(f"   Monthly PPMs:  {result['monthly']['coverage_percentage']:.1f}% coverage")
            print(f"   Status:        {result['overall_status']}")
            print(f"   Risk Level:    {result['risk_analysis']['overall_risk']}")
            
            if result['daily']['failed_days']:
                print(f"   ⚠️  Daily Gaps: {len(result['daily']['failed_days'])} out of {result['daily']['total_days_tested']} days")
            
            if result['weekly']['coverage_gaps']:
                print(f"   ⚠️  Weekly Gaps: {len(result['weekly']['coverage_gaps'])} qualifications")
    
    print(f"\n✅ Validation complete!")
    print(f"📄 Results saved to: {output_manager.current_dir}/validation_results.json")


def show_archive_status():
    """Show archived optimization results"""
    output_manager = StandardOutputManager()
    archives = output_manager.list_archive()
    
    if archives:
        print(f"\n📦 ARCHIVED RESULTS ({len(archives)} found):")
        for i, archive in enumerate(archives[:10], 1):  # Show last 10
            print(f"   {i:2d}. {archive['optimization_name']} - {archive['timestamp'][:16]}")
    else:
        print(f"\n📦 No archived results found")


if __name__ == "__main__":
    main()
    show_archive_status() 