"""
Enhanced Coverage Validation Framework
=====================================

This module provides comprehensive testing framework to validate qualification
assignments against FULL operational constraints including:
- Complete shift rotation cycles (36 weeks = 2 mechanical cycles + 4 electrical cycles)
- Proper PPM scheduling windows (AM preference for weekly PPMs)
- Real-world scheduling constraints and handover times

The framework validates any qualification optimization results to ensure they
provide adequate coverage for daily, weekly, and monthly PPMs across complete
rotation cycles.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configure which teams to process (set to [1] for Team 1 only, [1, 2] for both)
ACTIVE_TEAMS = [1]


class CoverageValidator:
    """Validate qualification assignments against full operational rotation cycles"""
    
    def _extend_rota_to_weeks(self, rota_data, target_weeks):
        """Extend rota data by cycling to reach target weeks"""
        original_weeks = len(rota_data)
        cycles_needed = target_weeks // original_weeks
        
        extended_rota = {}
        for cycle in range(cycles_needed):
            for week_num in range(1, original_weeks + 1):
                original_key = f'Week {week_num}'
                new_week_num = week_num + (cycle * original_weeks)
                new_key = f'Week {new_week_num}'
                
                if original_key in rota_data:
                    extended_rota[new_key] = rota_data[original_key].copy()
        
        return extended_rota
    
    def __init__(self, optimizer_results=None):
        """
        Initialize coverage validator
        
        Args:
            optimizer_results: Optional object containing PPM data and ride information
                             If None, will load data directly from files
        """
        self.optimizer = optimizer_results
        if self.optimizer is None:
            self._load_data_directly()
    
    def _load_data_directly(self):
        """Load PPM and ride data directly from files"""
        import glob
        
        # Create mock optimizer object with required data structure
        class MockOptimizer:
            def __init__(self):
                self.ppms_by_type = {'daily': {}, 'weekly': {}, 'monthly': {}}
                self.rides_info = {}
        
        self.optimizer = MockOptimizer()
        
        # Load ride info
        try:
            with open('data/processed/ride_info.json', 'r') as f:
                ride_data = json.load(f)
                # Extract rides from nested structure
                self.optimizer.rides_info = ride_data.get('rides', {})
        except FileNotFoundError:
            print("⚠️  Warning: ride_info.json not found, will skip team filtering")
            self.optimizer.rides_info = {}
        
        # Load PPM data by type
        for ppm_type in ['daily', 'weekly', 'monthly']:
            ppm_files = glob.glob(f'data/raw/ppms/{ppm_type}/*.json')
            for file_path in ppm_files:
                with open(file_path, 'r') as f:
                    ppm_data = json.load(f)
                    ride_id = ppm_data['ride_id']
                    self.optimizer.ppms_by_type[ppm_type][ride_id] = ppm_data
        
    def validate_assignment_coverage(self, qualification_matrices):
        """
        Test if proposed qualification assignments provide adequate coverage
        across FULL rotation cycles
        
        Args:
            qualification_matrices: Dict of {team: {engineer_id: assignment_data}}
                                  Where assignment_data contains:
                                  - 'name': engineer name
                                  - 'role': 'electrical' or 'mechanical' 
                                  - 'qualifications': list of qualification codes
                                  - 'rota_number': rota position (optional)
        
        Returns:
            Dict containing comprehensive coverage test results over full cycles
        """
        print("\n🧪 ENHANCED COVERAGE VALIDATION FRAMEWORK")
        print("=" * 60)
        print("Testing qualification assignments against FULL rotation cycles...")
        print("• Mechanical teams: 36 weeks (2 full rotations)")
        print("• Electrical teams: 36 weeks (4 full rotations)")
        print("• Proper PPM scheduling windows and preferences")
        
        test_results = {}
        
        for team in ACTIVE_TEAMS:
            if team not in qualification_matrices:
                print(f"\n⚠️  Team {team} not found in qualification matrices")
                continue
                
            print(f"\n🏢 TEAM {team} FULL CYCLE COVERAGE TESTING:")
            
            # Load rota data
            elec_rota_file = f'data/processed/parsed_rotas/parsed_team{team}_elec_rota.json'
            mech_rota_file = f'data/processed/parsed_rotas/parsed_team{team}_mech_rota.json'
            
            try:
                with open(elec_rota_file, 'r') as f:
                    elec_rota = json.load(f)
                with open(mech_rota_file, 'r') as f:
                    mech_rota = json.load(f)
                
                # Extend rotas to 36-week cycle to match MILP optimizer
                elec_rota = self._extend_rota_to_weeks(elec_rota, 36)
                mech_rota = self._extend_rota_to_weeks(mech_rota, 36)
                
            except FileNotFoundError as e:
                print(f"   ❌ Rota files not found for Team {team}: {e}")
                continue
            
            # Determine full rotation cycles
            elec_weeks_available = len(elec_rota)
            mech_weeks_available = len(mech_rota)
            
            print(f"   📊 ROTATION CYCLE ANALYSIS:")
            print(f"      Electrical: {elec_weeks_available} weeks available")
            print(f"      Mechanical: {mech_weeks_available} weeks available")
            
            # Test coverage across FULL rotation cycles
            daily_results = self._test_daily_ppm_coverage_full_cycle(
                team, qualification_matrices[team], elec_rota, mech_rota
            )
            weekly_results = self._test_weekly_ppm_coverage_full_cycle(
                team, qualification_matrices[team], elec_rota, mech_rota
            )
            monthly_results = self._test_monthly_ppm_coverage_full_cycle(
                team, qualification_matrices[team], elec_rota, mech_rota
            )

            # Overall assessment
            test_results[team] = {
                'daily': daily_results,
                'weekly': weekly_results,
                'monthly': monthly_results,
                'overall_status': self._assess_overall_coverage(daily_results, weekly_results, monthly_results),
                'risk_analysis': self._analyze_coverage_risks(team, qualification_matrices[team]),
                'rotation_info': {
                    'electrical_weeks': elec_weeks_available,
                    'mechanical_weeks': mech_weeks_available,
                    'total_days_tested': daily_results['total_days_tested'],
                    'total_weeks_tested': weekly_results['total_weeks_tested']
                }
            }

            # Print summary
            print(f"\n   📊 FULL CYCLE COVERAGE SUMMARY:")
            print(f"      Daily PPMs:   {daily_results['coverage_percentage']:.1f}% coverage ({daily_results['successful_days']}/{daily_results['total_days_tested']} days)")
            print(f"      Weekly PPMs:  {weekly_results['coverage_percentage']:.1f}% coverage ({weekly_results['successful_weeks']}/{weekly_results['total_weeks_tested']} weeks)")
            print(f"      Monthly PPMs: {monthly_results['coverage_percentage']:.1f}% coverage ({monthly_results['successful_months']}/{monthly_results['total_months_tested']} months)")
            print(f"      Status:       {test_results[team]['overall_status']}")
            print(f"      Risk Level:   {test_results[team]['risk_analysis']['overall_risk']}")
            
            if daily_results['failed_days']:
                print(f"      ⚠️  CRITICAL: {len(daily_results['failed_days'])} days with coverage gaps across full rotation!")
        
        return test_results
    
    def _test_daily_ppm_coverage_full_cycle(self, team, engineer_assignments, elec_rota, mech_rota):
        """Test daily PPM coverage across FULL rotation cycles"""
        print(f"\n   🌅 TESTING DAILY PPM COVERAGE (FULL ROTATION):")
        
        # Get all daily PPMs for this team
        team_daily_ppms = []
        for ride_id, ppm_data in self.optimizer.ppms_by_type['daily'].items():
            if ride_id in self.optimizer.rides_info and self.optimizer.rides_info[ride_id]['team_responsible'] == team:
                team_daily_ppms.extend(ppm_data['ppms'])
        
        # Build qualification lookup
        qual_to_engineers = {}
        for engineer_id, assignment in engineer_assignments.items():
            for qual in assignment['qualifications']:
                if qual not in qual_to_engineers:
                    qual_to_engineers[qual] = []
                qual_to_engineers[qual].append({
                    'id': engineer_id,
                    'role': assignment['role']
                })
        
        failed_days = []
        coverage_gaps = []
        total_days_tested = 0
        successful_days = 0
        
        # Test across FULL rotation cycles (36 weeks) - 2 mech + 4 elec cycles
        max_weeks = min(len(mech_rota), 36)  # Use actual available weeks, up to 36
        
        print(f"      Testing across {max_weeks} weeks of full rotation...")
        
        for week_num in range(1, max_weeks + 1):
            week_key = f'Week {week_num}'
            
            # Both electrical and mechanical are now 36-week rotas
            elec_week_key = f'Week {week_num}'
            
            if week_key not in mech_rota or elec_week_key not in elec_rota:
                continue
                
            mech_week = mech_rota[week_key]
            elec_week = elec_rota[elec_week_key]
            
            # Test each day of the week (Mon-Fri for daily PPMs)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            for day_idx, day_name in enumerate(days):
                total_days_tested += 1
                
                # Get engineers available on Early shift this day (6:00-9:00 AM window)
                available_engineers = []
                
                # Check electrical engineers (using their cycled week)
                for engineer_id, shifts in elec_week.items():
                    if day_idx < len(shifts) and shifts[day_idx] == 'E':
                        if engineer_id in engineer_assignments:
                            available_engineers.append({
                                'id': engineer_id,
                                'role': 'electrical'
                            })
                
                # Check mechanical engineers
                for engineer_id, shifts in mech_week.items():
                    if day_idx < len(shifts) and shifts[day_idx] == 'E':
                        if engineer_id in engineer_assignments:
                            available_engineers.append({
                                'id': engineer_id,
                                'role': 'mechanical'
                            })
                
                # Group PPMs by ride and maintenance type
                ride_ppm_groups = {}
                for ppm in team_daily_ppms:
                    # Get ride_id from the PPM data structure
                    ride_id = None
                    for rid, ppm_data in self.optimizer.ppms_by_type['daily'].items():
                        if ppm in ppm_data['ppms']:
                            ride_id = rid
                            break
                    
                    if ride_id is None:
                        continue
                        
                    if ride_id not in ride_ppm_groups:
                        ride_ppm_groups[ride_id] = {'ELECTRICAL': [], 'MECHANICAL': []}
                    
                    ride_ppm_groups[ride_id][ppm['maintenance_type']].append(ppm)
                
                # Test each ride's PPM groups
                day_successful = True
                daily_gaps = []
                
                for ride_id, maintenance_groups in ride_ppm_groups.items():
                    for maintenance_type, ppms in maintenance_groups.items():
                        if not ppms:  # Skip empty groups
                            continue
                            
                        # Calculate total duration for this maintenance type
                        total_duration = sum(ppm['duration_hours'] for ppm in ppms)
                        
                        # Calculate engineers needed based on 3-hour AM window
                        import math
                        engineers_needed = math.ceil(total_duration / 3.0)
                        
                        # Find ALL qualified engineers available for ANY of these PPMs
                        all_qualified_available = set()
                        for ppm in ppms:
                            qual_needed = ppm['qualification_code']
                            if qual_needed in qual_to_engineers:
                                for qualified_eng in qual_to_engineers[qual_needed]:
                                    # Check if this engineer is available today and right role
                                    for available_eng in available_engineers:
                                        # Check if this engineer is available today and right role
                                        id_match = qualified_eng['id'] == available_eng['id']
                                        role_match = qualified_eng['role'] == available_eng['role']
                                        type_match = qualified_eng['role'] == maintenance_type.lower()
                                        
                                        if id_match and role_match and type_match:
                                            all_qualified_available.add(qualified_eng['id'])
                                            # Debug: Log successful matches
                                            # print(f"    ✅ Match: {qualified_eng['id']} for {qual_needed}")
                                            break
                        
                        if len(all_qualified_available) < engineers_needed:
                            day_successful = False
                            daily_gaps.append({
                                'ride_id': ride_id,
                                'maintenance_type': maintenance_type,
                                'total_duration_hours': total_duration,
                                'engineers_needed': engineers_needed,
                                'engineers_available': len(all_qualified_available),
                                'available_engineers_total': len(available_engineers),
                                'ppm_count': len(ppms),
                                'ppm_codes': [ppm['ppm_code'] for ppm in ppms],
                                'week': week_num,
                                'elec_week': week_num if maintenance_type == 'ELECTRICAL' else None
                            })
                
                if day_successful:
                    successful_days += 1
                else:
                    failed_days.append({
                        'week': week_num,
                        'elec_week': week_num,  # Now same as week_num since both are 36-week rotas
                        'day': day_name,
                        'gaps': daily_gaps,
                        'available_engineers': len(available_engineers)
                    })
                    coverage_gaps.extend(daily_gaps)
        
        coverage_percentage = (successful_days / total_days_tested * 100) if total_days_tested > 0 else 0
        
        print(f"      Days tested: {total_days_tested} (across {max_weeks} weeks)")
        print(f"      Successful days: {successful_days}")
        print(f"      Failed days: {len(failed_days)}")
        print(f"      Coverage: {coverage_percentage:.1f}%")
        
        if failed_days:
            print(f"      📅 SAMPLE FAILED DAYS:")
            for i, failure in enumerate(failed_days[:5]):  # Show first 5 failures
                elec_info = f" (Elec Week {failure['elec_week']})" if failure['elec_week'] else ""
                print(f"         Week {failure['week']}{elec_info} {failure['day']}: {len(failure['gaps'])} PPM gaps")
        
        return {
            'coverage_percentage': coverage_percentage,
            'total_days_tested': total_days_tested,
            'successful_days': successful_days,
            'failed_days': failed_days,
            'coverage_gaps': coverage_gaps,
            'unique_gap_rides': list(set(gap['ride_id'] for gap in coverage_gaps)),
            'weeks_tested': max_weeks
        }
    
    def _test_weekly_ppm_coverage_full_cycle(self, team, engineer_assignments, elec_rota, mech_rota):
        """Test weekly PPM coverage with AM preference across FULL rotation cycles"""
        print(f"\n   📅 TESTING WEEKLY PPM COVERAGE (FULL ROTATION + AM PREFERENCE):")
        
        # Get all weekly PPMs for this team
        team_weekly_ppms = []
        for ride_id, ppm_data in self.optimizer.ppms_by_type['weekly'].items():
            if ride_id in self.optimizer.rides_info and self.optimizer.rides_info[ride_id]['team_responsible'] == team:
                team_weekly_ppms.extend(ppm_data['ppms'])
        
        # Build qualification lookup
        qual_to_engineers = {}
        for engineer_id, assignment in engineer_assignments.items():
            for qual in assignment['qualifications']:
                if qual not in qual_to_engineers:
                    qual_to_engineers[qual] = []
                qual_to_engineers[qual].append({
                    'id': engineer_id,
                    'role': assignment['role']
                })
        
        failed_weeks = []
        coverage_gaps = []
        total_weeks_tested = 0
        successful_weeks = 0
        am_scheduled = 0
        pm_scheduled = 0
        
        # Test across FULL rotation cycles (36 weeks)
        max_weeks = min(len(mech_rota), 36)
        
        print(f"      Testing across {max_weeks} weeks with AM preference logic...")
        
        for week_num in range(1, max_weeks + 1):
            week_key = f'Week {week_num}'
            
            # Both electrical and mechanical are now 36-week rotas
            elec_week_key = f'Week {week_num}'
            
            if week_key not in mech_rota or elec_week_key not in elec_rota:
                continue
                
            total_weeks_tested += 1
            mech_week = mech_rota[week_key]
            elec_week = elec_rota[elec_week_key]
            
            # Get engineers available for AM window (Early shift Mon-Fri)
            am_available_engineers = set()
            # Get engineers available for PM window (Late shift Mon-Fri)
            pm_available_engineers = set()
            
            for day_idx in range(5):  # Mon-Fri
                # Check electrical engineers
                for engineer_id, shifts in elec_week.items():
                    if day_idx < len(shifts):
                        if shifts[day_idx] == 'E' and engineer_id in engineer_assignments:
                            am_available_engineers.add(engineer_id)
                        elif shifts[day_idx] == 'L' and engineer_id in engineer_assignments:
                            pm_available_engineers.add(engineer_id)
                
                # Check mechanical engineers
                for engineer_id, shifts in mech_week.items():
                    if day_idx < len(shifts):
                        if shifts[day_idx] == 'E' and engineer_id in engineer_assignments:
                            am_available_engineers.add(engineer_id)
                        elif shifts[day_idx] == 'L' and engineer_id in engineer_assignments:
                            pm_available_engineers.add(engineer_id)
            
            # Test each weekly PPM with AM preference
            week_successful = True
            weekly_gaps = []
            week_am_count = 0
            week_pm_count = 0
            
            for ppm in team_weekly_ppms:
                qual_needed = ppm['qualification_code']
                maintenance_type = ppm['maintenance_type']
                
                # First try AM window (PREFERRED)
                am_qualified_available = []
                if qual_needed in qual_to_engineers:
                    for qualified_eng in qual_to_engineers[qual_needed]:
                        if (qualified_eng['id'] in am_available_engineers and
                            qualified_eng['role'] == maintenance_type.lower()):
                            am_qualified_available.append(qualified_eng['id'])
                
                am_qualified_available = list(set(am_qualified_available))
                
                if len(am_qualified_available) >= 1:
                    # Can schedule in AM window (PREFERRED)
                    week_am_count += 1
                    continue
                
                # If AM not available, try PM window (FALLBACK)
                pm_qualified_available = []
                if qual_needed in qual_to_engineers:
                    for qualified_eng in qual_to_engineers[qual_needed]:
                        if (qualified_eng['id'] in pm_available_engineers and
                            qualified_eng['role'] == maintenance_type.lower()):
                            pm_qualified_available.append(qualified_eng['id'])
                
                pm_qualified_available = list(set(pm_qualified_available))
                
                if len(pm_qualified_available) >= 1:
                    # Can schedule in PM window (FALLBACK)
                    week_pm_count += 1
                    continue
                
                # Cannot schedule in either window - FAILURE
                week_successful = False
                
                # Get ride_id for this PPM
                ride_id = None
                for rid, ppm_data in self.optimizer.ppms_by_type['weekly'].items():
                    if ppm in ppm_data['ppms']:
                        ride_id = rid
                        break
                
                weekly_gaps.append({
                    'ppm_code': ppm['ppm_code'],
                    'ride_id': ride_id or 'Unknown',
                    'qualification_code': qual_needed,
                    'maintenance_type': maintenance_type,
                    'duration_hours': ppm['duration_hours'],
                    'engineers_needed': 1,
                    'am_qualified_available': len(am_qualified_available),
                    'pm_qualified_available': len(pm_qualified_available),
                    'week': week_num,
                    'elec_week': week_num
                })
            
            if week_successful:
                successful_weeks += 1
                am_scheduled += week_am_count
                pm_scheduled += week_pm_count
            else:
                failed_weeks.append({
                    'week': week_num,
                    'elec_week': week_num,
                    'gaps': weekly_gaps,
                    'am_available_engineers': len(am_available_engineers),
                    'pm_available_engineers': len(pm_available_engineers)
                })
                coverage_gaps.extend(weekly_gaps)
        
        coverage_percentage = (successful_weeks / total_weeks_tested * 100) if total_weeks_tested > 0 else 0
        
        # Calculate AM preference success rate
        total_scheduled = am_scheduled + pm_scheduled
        am_preference_rate = (am_scheduled / total_scheduled * 100) if total_scheduled > 0 else 0
        
        print(f"      Weeks tested: {total_weeks_tested} (across {max_weeks} weeks)")
        print(f"      Successful weeks: {successful_weeks}")
        print(f"      Coverage: {coverage_percentage:.1f}%")
        print(f"      AM scheduling: {am_scheduled} PPMs ({am_preference_rate:.1f}% of scheduled)")
        print(f"      PM fallback: {pm_scheduled} PPMs ({100-am_preference_rate:.1f}% of scheduled)")
        
        return {
            'coverage_percentage': coverage_percentage,
            'total_weeks_tested': total_weeks_tested,
            'successful_weeks': successful_weeks,
            'failed_weeks': failed_weeks,
            'coverage_gaps': coverage_gaps,
            'unique_gap_qualifications': list(set(gap['qualification_code'] for gap in coverage_gaps)),
            'am_scheduled': am_scheduled,
            'pm_scheduled': pm_scheduled,
            'am_preference_rate': am_preference_rate,
            'weeks_tested': max_weeks
        }
    
    def _test_monthly_ppm_coverage_full_cycle(self, team, engineer_assignments, elec_rota, mech_rota):
        """Test monthly PPM coverage across full rotation cycle (proper monthly scheduling)"""
        print(f"\n   📆 TESTING MONTHLY PPM COVERAGE (FULL ROTATION):")
        
        # Get all monthly PPMs for this team
        team_monthly_ppms = []
        for ride_id, ppm_data in self.optimizer.ppms_by_type['monthly'].items():
            if ride_id in self.optimizer.rides_info and self.optimizer.rides_info[ride_id]['team_responsible'] == team:
                team_monthly_ppms.extend(ppm_data['ppms'])
        
        # Build qualification lookup
        qual_to_engineers = {}
        for engineer_id, assignment in engineer_assignments.items():
            for qual in assignment['qualifications']:
                if qual not in qual_to_engineers:
                    qual_to_engineers[qual] = []
                qual_to_engineers[qual].append({
                    'id': engineer_id,
                    'role': assignment['role']
                })
        
        # Test across ALL 36 weeks = 9 full months
        max_weeks = min(len(mech_rota), 36)
        months_to_test = 9  # 9 full months to cover all 36 weeks
        
        print(f"      Testing {months_to_test} months across {max_weeks} weeks...")
        
        coverage_gaps = []
        successful_months = 0
        
        for month_num in range(1, months_to_test + 1):
            # Each "month" is approximately 4 weeks
            month_start_week = ((month_num - 1) * 4) + 1
            month_end_week = min(month_start_week + 3, max_weeks)
            
            # Get all engineers available during this month
            month_available_engineers = set()
            
            for week_offset in range(month_end_week - month_start_week + 1):
                week_num = month_start_week + week_offset
                week_key = f'Week {week_num}'
                
                # Both electrical and mechanical are now 36-week rotas
                elec_week_key = f'Week {week_num}'
                
                if week_key not in mech_rota or elec_week_key not in elec_rota:
                    continue
                    
                mech_week = mech_rota[week_key]
                elec_week = elec_rota[elec_week_key]
                
                # Get engineers available any day Mon-Fri (Early or Late shift)
                for day_idx in range(5):  # Mon-Fri
                    # Check electrical engineers
                    for engineer_id, shifts in elec_week.items():
                        if day_idx < len(shifts) and shifts[day_idx] in ['E', 'L']:
                            if engineer_id in engineer_assignments:
                                month_available_engineers.add(engineer_id)
                    
                    # Check mechanical engineers
                    for engineer_id, shifts in mech_week.items():
                        if day_idx < len(shifts) and shifts[day_idx] in ['E', 'L']:
                            if engineer_id in engineer_assignments:
                                month_available_engineers.add(engineer_id)
            
            # Test each monthly PPM for this month
            month_successful = True
            month_gaps = []
            
            for ppm in team_monthly_ppms:
                qual_needed = ppm['qualification_code']
                maintenance_type = ppm['maintenance_type']
                
                # Find qualified engineers available during this month
                qualified_available = []
                if qual_needed in qual_to_engineers:
                    for qualified_eng in qual_to_engineers[qual_needed]:
                        if (qualified_eng['id'] in month_available_engineers and
                            qualified_eng['role'] == maintenance_type.lower()):
                            qualified_available.append(qualified_eng['id'])
                
                qualified_available = list(set(qualified_available))
                
                # Each monthly PPM needs at least 1 qualified engineer available during the month
                if len(qualified_available) < 1:
                    month_successful = False
                    
                    # Get ride_id for this PPM
                    ride_id = None
                    for rid, ppm_data in self.optimizer.ppms_by_type['monthly'].items():
                        if ppm in ppm_data['ppms']:
                            ride_id = rid
                            break
                    
                    month_gaps.append({
                        'ppm_code': ppm['ppm_code'],
                        'ride_id': ride_id or 'Unknown',
                        'qualification_code': qual_needed,
                        'maintenance_type': maintenance_type,
                        'duration_hours': ppm['duration_hours'],
                        'engineers_needed': 1,
                        'engineers_available': len(qualified_available),
                        'month': month_num,
                        'weeks': f"{month_start_week}-{month_end_week}"
                    })
            
            if month_successful:
                successful_months += 1
            else:
                coverage_gaps.extend(month_gaps)
        
        coverage_percentage = (successful_months / months_to_test * 100) if months_to_test > 0 else 100
        
        print(f"      Months tested: {months_to_test} (across {max_weeks} weeks)")
        print(f"      Successfully covered: {successful_months}")
        print(f"      Coverage: {coverage_percentage:.1f}%")
        
        return {
            'coverage_percentage': coverage_percentage,
            'total_months_tested': months_to_test,
            'successful_months': successful_months,
            'coverage_gaps': coverage_gaps,
            'unique_gap_qualifications': list(set(gap['qualification_code'] for gap in coverage_gaps)),
            'weeks_tested': max_weeks
        }

    def _assess_overall_coverage(self, daily_results, weekly_results, monthly_results):
        """Assess overall coverage status"""
        daily_coverage = daily_results['coverage_percentage']
        weekly_coverage = weekly_results['coverage_percentage']
        monthly_coverage = monthly_results['coverage_percentage']

        if daily_coverage >= 95 and weekly_coverage >= 95 and monthly_coverage >= 95:
            return "✅ EXCELLENT"
        elif daily_coverage >= 90 and weekly_coverage >= 90 and monthly_coverage >= 90:
            return "🟡 GOOD"
        elif daily_coverage >= 80 and weekly_coverage >= 85 and monthly_coverage >= 85:
            return "🟠 ACCEPTABLE"
        else:
            return "❌ INSUFFICIENT"
    
    def _analyze_coverage_risks(self, team, engineer_assignments):
        """Analyze coverage risks and redundancy"""
        
        # Count how many engineers have each qualification
        qual_coverage = defaultdict(int)
        for assignment in engineer_assignments.values():
            for qual in assignment['qualifications']:
                qual_coverage[qual] += 1
        
        # Identify single points of failure
        single_point_failures = [qual for qual, count in qual_coverage.items() if count == 1]
        good_coverage = [qual for qual, count in qual_coverage.items() if count >= 2]
        excellent_coverage = [qual for qual, count in qual_coverage.items() if count >= 3]
        
        # Calculate risk level
        total_quals = len(qual_coverage)
        spf_ratio = len(single_point_failures) / total_quals if total_quals > 0 else 0
        
        if spf_ratio > 0.5:
            risk_level = "CRITICAL"
        elif spf_ratio > 0.3:
            risk_level = "HIGH"
        elif spf_ratio > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'single_point_failures': len(single_point_failures),
            'good_coverage': len(good_coverage),
            'excellent_coverage': len(excellent_coverage),
            'overall_risk': risk_level,
            'spf_ratio': spf_ratio,
            'total_qualifications': total_quals,
            'spf_qualifications': single_point_failures[:10]  # Show first 10 for debugging
        }
    
    def export_validation_results(self, test_results, output_file=None):
        """Export validation results to file"""
        if output_file is None:
            output_file = f"coverage_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path("outputs/validation") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n📄 Validation results exported to: {output_path}")
        return output_path
    
    def generate_validation_report(self, test_results, output_file=None):
        """Generate a human-readable validation report"""
        if output_file is None:
            output_file = f"coverage_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        output_path = Path("outputs/validation") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Qualification Assignment Coverage Validation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            for team in ACTIVE_TEAMS:
                if team not in test_results:
                    continue
                    
                results = test_results[team]
                f.write(f"### Team {team}\n\n")
                
                f.write(f"**Coverage Status:** {results['overall_status']}\n\n")
                f.write(f"**Risk Level:** {results['risk_analysis']['overall_risk']}\n\n")
                
                f.write(f"**Coverage Percentages:**\n")
                f.write(f"- Daily PPMs: {results['daily']['coverage_percentage']:.1f}%\n")
                f.write(f"- Weekly PPMs: {results['weekly']['coverage_percentage']:.1f}%\n")
                f.write(f"- Monthly PPMs: {results['monthly']['coverage_percentage']:.1f}%\n\n")
                
                if results['daily']['failed_days']:
                    f.write(f"**⚠️ Critical Issues:**\n")
                    f.write(f"- {len(results['daily']['failed_days'])} days with coverage gaps\n")
                    f.write(f"- {len(results['daily']['unique_gap_qualifications'])} unique qualifications causing gaps\n\n")
                
                f.write(f"**Risk Analysis:**\n")
                f.write(f"- Single Points of Failure: {results['risk_analysis']['single_point_failures']}\n")
                f.write(f"- Good Coverage (2+ engineers): {results['risk_analysis']['good_coverage']}\n")
                f.write(f"- Excellent Coverage (3+ engineers): {results['risk_analysis']['excellent_coverage']}\n\n")
        
        print(f"\n📊 Validation report exported to: {output_path}")
        return output_path


def main():
    """Example usage of the Coverage Validation Framework"""
    # This would typically be called with results from any optimization system
    print("Coverage Validation Framework - Standalone Testing System")
    print("Usage: validator = CoverageValidator(optimizer_results)")
    print("       results = validator.validate_assignment_coverage(qualification_matrices)")


if __name__ == "__main__":
    main() 