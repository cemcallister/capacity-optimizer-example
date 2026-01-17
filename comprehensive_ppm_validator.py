#!/usr/bin/env python3
"""
Comprehensive PPM Coverage Validation Script

This script validates that the MILP qualification assignments can actually handle:
1. Daily PPMs - covered every single day
2. Weekly PPMs - each one scheduled once per week
3. Monthly PPMs - each one scheduled once every 4 weeks

Validates across the full 36-week rotation cycle for both teams.
"""

import json
import glob
import math
from collections import defaultdict, deque
from pathlib import Path


class ComprehensivePPMValidator:
    """Validates PPM coverage across full rotation cycles"""
    
    def __init__(self):
        print("🔍 COMPREHENSIVE PPM COVERAGE VALIDATOR")
        print("=" * 60)
        self.load_data()
        
    def load_data(self):
        """Load all necessary data"""
        print("📊 Loading data...")
        
        # Load qualification matrices
        with open('outputs/current/team_1_qualification_matrix.json', 'r') as f:
            self.team1_quals = json.load(f)
        with open('outputs/current/team_2_qualification_matrix.json', 'r') as f:
            self.team2_quals = json.load(f)
            
        # Load rotas
        with open('data/processed/parsed_rotas/parsed_team1_elec_rota.json', 'r') as f:
            self.team1_elec_rota = json.load(f)
        with open('data/processed/parsed_rotas/parsed_team1_mech_rota.json', 'r') as f:
            self.team1_mech_rota = json.load(f)
        with open('data/processed/parsed_rotas/parsed_team2_elec_rota.json', 'r') as f:
            self.team2_elec_rota = json.load(f)
        with open('data/processed/parsed_rotas/parsed_team2_mech_rota.json', 'r') as f:
            self.team2_mech_rota = json.load(f)
            
        # Load ride assignments
        with open('data/processed/ride_info.json', 'r') as f:
            self.ride_info = json.load(f)['rides']
            
        # Load all PPM data
        self.ppms = {'daily': {}, 'weekly': {}, 'monthly': {}}
        for ppm_type in ['daily', 'weekly', 'monthly']:
            files = glob.glob(f'data/raw/ppms/{ppm_type}/*.json')
            for file_path in files:
                with open(file_path, 'r') as f:
                    ppm_data = json.load(f)
                    self.ppms[ppm_type][ppm_data['ride_id']] = ppm_data['ppms']
        
        print("✅ Data loaded successfully")
        
    def name_to_code(self, name):
        """Convert 'Adrian Williams' to 'AWILLIAMS'"""
        parts = name.split()
        if len(parts) >= 2:
            return parts[0][0].upper() + parts[1].upper()
        return name.upper()
    
    def get_engineers_on_shift(self, team, week_key, day_idx, shift_type='E'):
        """Get engineers on specified shift for a team on a specific day"""
        team_quals = self.team1_quals if team == 1 else self.team2_quals
        elec_rota = self.team1_elec_rota if team == 1 else self.team2_elec_rota
        mech_rota = self.team1_mech_rota if team == 1 else self.team2_mech_rota
        
        engineers_on_shift = {'electrical': [], 'mechanical': []}
        
        # Check electrical engineers
        if week_key in elec_rota:
            week_data = elec_rota[week_key]
            for eng_code, shifts in week_data.items():
                if day_idx < len(shifts) and shifts[day_idx] == shift_type:
                    # Find matching engineer in qualification matrix
                    for eng_id, eng_data in team_quals.items():
                        if self.name_to_code(eng_data['name']) == eng_code and eng_data['role'] == 'electrical':
                            engineers_on_shift['electrical'].append({
                                'id': eng_id,
                                'name': eng_data['name'],
                                'qualifications': set(eng_data.get('daily_qualifications', [])),
                                'all_qualifications': set(eng_data.get('qualifications', []))
                            })
                            break
        
        # Check mechanical engineers
        if week_key in mech_rota:
            week_data = mech_rota[week_key]
            for eng_code, shifts in week_data.items():
                if day_idx < len(shifts) and shifts[day_idx] == shift_type:
                    # Find matching engineer in qualification matrix
                    for eng_id, eng_data in team_quals.items():
                        if self.name_to_code(eng_data['name']) == eng_code and eng_data['role'] == 'mechanical':
                            engineers_on_shift['mechanical'].append({
                                'id': eng_id,
                                'name': eng_data['name'],
                                'qualifications': set(eng_data.get('daily_qualifications', [])),
                                'all_qualifications': set(eng_data.get('qualifications', []))
                            })
                            break
        
        return engineers_on_shift
    
    def check_daily_ppm_coverage(self, team, week_key, day_idx, day_name):
        """Check if daily PPMs can be covered on a specific day"""
        team_rides = [ride for ride, info in self.ride_info.items() if info.get('team_responsible') == team]
        
        # Get engineers on early shift (daily PPMs must be done in AM)
        early_engineers = self.get_engineers_on_shift(team, week_key, day_idx, 'E')
        
        coverage_results = []
        total_workload = {'electrical': 0, 'mechanical': 0}
        engineer_workloads = defaultdict(float)
        
        for ride_id in team_rides:
            if ride_id not in self.ppms['daily']:
                continue
                
            ppms = self.ppms['daily'][ride_id]
            
            # Group by maintenance type
            maintenance_groups = {'ELECTRICAL': [], 'MECHANICAL': []}
            for ppm in ppms:
                maintenance_groups[ppm['maintenance_type']].append(ppm)
            
            for maint_type, type_ppms in maintenance_groups.items():
                if not type_ppms:
                    continue
                    
                total_hours = sum(ppm['duration_hours'] for ppm in type_ppms)
                needed_quals = set(ppm['qualification_code'] for ppm in type_ppms)
                total_workload[maint_type.lower()] += total_hours
                
                # Find available qualified engineers
                role = maint_type.lower()
                available_qualified = []
                
                for eng in early_engineers[role]:
                    if needed_quals.intersection(eng['qualifications']):
                        available_qualified.append(eng)
                
                engineers_needed = math.ceil(total_hours / 3.0)  # 3-hour AM window
                
                if len(available_qualified) >= engineers_needed:
                    coverage_results.append({
                        'ride': ride_id,
                        'type': maint_type,
                        'status': 'COVERED',
                        'hours': total_hours,
                        'engineers_needed': engineers_needed,
                        'engineers_available': len(available_qualified),
                        'engineers': [eng['name'] for eng in available_qualified]
                    })
                    
                    # Distribute workload
                    hours_per_eng = total_hours / len(available_qualified)
                    for eng in available_qualified:
                        engineer_workloads[eng['name']] += hours_per_eng
                else:
                    coverage_results.append({
                        'ride': ride_id,
                        'type': maint_type,
                        'status': 'UNCOVERED',
                        'hours': total_hours,
                        'engineers_needed': engineers_needed,
                        'engineers_available': len(available_qualified),
                        'gap': engineers_needed - len(available_qualified)
                    })
        
        # Check for overloaded engineers (>3.5h)
        overloaded = [(name, hours) for name, hours in engineer_workloads.items() if hours > 3.5]
        
        success = all(result['status'] == 'COVERED' for result in coverage_results) and len(overloaded) == 0
        
        return {
            'success': success,
            'day': day_name,
            'week': week_key,
            'total_engineers': len(early_engineers['electrical']) + len(early_engineers['mechanical']),
            'total_workload': sum(total_workload.values()),
            'coverage_results': coverage_results,
            'overloaded_engineers': overloaded,
            'engineer_workloads': dict(engineer_workloads)
        }
    
    def check_weekly_ppm_scheduling(self, team, week_key):
        """Check if all weekly PPMs can be scheduled once during the week"""
        team_rides = [ride for ride, info in self.ride_info.items() if info.get('team_responsible') == team]
        
        # Get all weekly PPMs for this team
        team_weekly_ppms = []
        for ride_id in team_rides:
            if ride_id in self.ppms['weekly']:
                for ppm in self.ppms['weekly'][ride_id]:
                    team_weekly_ppms.append({
                        'ride': ride_id,
                        'ppm_code': ppm['ppm_code'],
                        'qualification': ppm['qualification_code'],
                        'maintenance_type': ppm['maintenance_type'],
                        'hours': ppm['duration_hours']
                    })
        
        scheduled_ppms = []
        unscheduled_ppms = []
        
        for ppm in team_weekly_ppms:
            scheduled = False
            
            # Try each day of the week (Mon-Fri)
            for day_idx in range(5):
                if scheduled:
                    break
                    
                # Try AM shift first (preferred)
                am_engineers = self.get_engineers_on_shift(team, week_key, day_idx, 'E')
                role = ppm['maintenance_type'].lower()
                
                qualified_am = [eng for eng in am_engineers[role] 
                              if ppm['qualification'] in eng['all_qualifications']]
                
                if len(qualified_am) >= 1:
                    scheduled_ppms.append({
                        **ppm,
                        'scheduled_day': day_idx,
                        'shift': 'AM',
                        'engineers': [eng['name'] for eng in qualified_am[:1]]
                    })
                    scheduled = True
                    continue
                
                # Try PM shift as fallback
                pm_engineers = self.get_engineers_on_shift(team, week_key, day_idx, 'L')
                qualified_pm = [eng for eng in pm_engineers[role] 
                              if ppm['qualification'] in eng['all_qualifications']]
                
                if len(qualified_pm) >= 1:
                    scheduled_ppms.append({
                        **ppm,
                        'scheduled_day': day_idx,
                        'shift': 'PM',
                        'engineers': [eng['name'] for eng in qualified_pm[:1]]
                    })
                    scheduled = True
            
            if not scheduled:
                unscheduled_ppms.append(ppm)
        
        am_count = len([p for p in scheduled_ppms if p['shift'] == 'AM'])
        pm_count = len([p for p in scheduled_ppms if p['shift'] == 'PM'])
        
        return {
            'success': len(unscheduled_ppms) == 0,
            'week': week_key,
            'total_weekly_ppms': len(team_weekly_ppms),
            'scheduled_ppms': len(scheduled_ppms),
            'unscheduled_ppms': len(unscheduled_ppms),
            'am_scheduled': am_count,
            'pm_scheduled': pm_count,
            'am_preference_rate': (am_count / len(scheduled_ppms) * 100) if scheduled_ppms else 0,
            'unscheduled_details': unscheduled_ppms
        }
    
    def check_monthly_ppm_scheduling(self, team, month_start_week, weeks_in_month):
        """Check if all monthly PPMs can be scheduled once during a 4-week period"""
        team_rides = [ride for ride, info in self.ride_info.items() if info.get('team_responsible') == team]
        
        # Get all monthly PPMs for this team
        team_monthly_ppms = []
        for ride_id in team_rides:
            if ride_id in self.ppms['monthly']:
                for ppm in self.ppms['monthly'][ride_id]:
                    team_monthly_ppms.append({
                        'ride': ride_id,
                        'ppm_code': ppm['ppm_code'],
                        'qualification': ppm['qualification_code'],
                        'maintenance_type': ppm['maintenance_type'],
                        'hours': ppm['duration_hours']
                    })
        
        scheduled_ppms = []
        unscheduled_ppms = []
        
        for ppm in team_monthly_ppms:
            scheduled = False
            
            # Try each week in the month
            for week_offset in range(weeks_in_month):
                if scheduled:
                    break
                    
                week_num = month_start_week + week_offset
                week_key = f'Week {week_num}'
                
                # Try each day of the week (Mon-Fri)
                for day_idx in range(5):
                    if scheduled:
                        break
                        
                    # Try AM shift first, then PM
                    for shift_type, shift_name in [('E', 'AM'), ('L', 'PM')]:
                        engineers = self.get_engineers_on_shift(team, week_key, day_idx, shift_type)
                        role = ppm['maintenance_type'].lower()
                        
                        if role in engineers:
                            qualified = [eng for eng in engineers[role] 
                                       if ppm['qualification'] in eng['all_qualifications']]
                            
                            if len(qualified) >= 1:
                                scheduled_ppms.append({
                                    **ppm,
                                    'scheduled_week': week_num,
                                    'scheduled_day': day_idx,
                                    'shift': shift_name,
                                    'engineers': [eng['name'] for eng in qualified[:1]]
                                })
                                scheduled = True
                                break
            
            if not scheduled:
                unscheduled_ppms.append(ppm)
        
        return {
            'success': len(unscheduled_ppms) == 0,
            'month_weeks': f'{month_start_week}-{month_start_week + weeks_in_month - 1}',
            'total_monthly_ppms': len(team_monthly_ppms),
            'scheduled_ppms': len(scheduled_ppms),
            'unscheduled_ppms': len(unscheduled_ppms),
            'unscheduled_details': unscheduled_ppms
        }
    
    def validate_team(self, team, weeks_to_test=36):
        """Validate a team across multiple weeks"""
        print(f"\n🏢 TEAM {team} COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        daily_results = []
        weekly_results = []
        monthly_results = []
        
        # Test daily PPMs for each day across multiple weeks
        print(f"📅 Testing daily PPM coverage across {weeks_to_test} weeks...")
        
        for week_num in range(1, weeks_to_test + 1):
            week_key = f'Week {week_num}'
            
            for day_idx, day_name in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
                result = self.check_daily_ppm_coverage(team, week_key, day_idx, day_name)
                daily_results.append(result)
        
        # Test weekly PPMs for each week
        print(f"📅 Testing weekly PPM coverage across {weeks_to_test} weeks...")
        
        for week_num in range(1, weeks_to_test + 1):
            week_key = f'Week {week_num}'
            result = self.check_weekly_ppm_scheduling(team, week_key)
            weekly_results.append(result)
        
        # Test monthly PPMs every 4 weeks
        print(f"📅 Testing monthly PPM coverage across {weeks_to_test//4} months...")
        
        for month_num in range(weeks_to_test // 4):
            month_start_week = month_num * 4 + 1
            weeks_in_month = min(4, weeks_to_test - month_start_week + 1)
            
            if weeks_in_month > 0:
                result = self.check_monthly_ppm_scheduling(team, month_start_week, weeks_in_month)
                monthly_results.append(result)
        
        return self.analyze_results(team, daily_results, weekly_results, monthly_results)
    
    def analyze_results(self, team, daily_results, weekly_results, monthly_results):
        """Analyze and summarize validation results"""
        print(f"\n📊 ANALYSIS RESULTS - TEAM {team}")
        print("=" * 40)
        
        # Daily analysis
        daily_failures = [r for r in daily_results if not r['success']]
        daily_success_rate = (len(daily_results) - len(daily_failures)) / len(daily_results) * 100
        
        print(f"📅 DAILY PPM RESULTS:")
        print(f"   Days tested: {len(daily_results)}")
        print(f"   Successful days: {len(daily_results) - len(daily_failures)}")
        print(f"   Failed days: {len(daily_failures)}")
        print(f"   Success rate: {daily_success_rate:.1f}%")
        
        if daily_failures:
            print(f"   📋 Sample failures:")
            for failure in daily_failures[:3]:
                uncovered = [r for r in failure['coverage_results'] if r['status'] == 'UNCOVERED']
                overloaded = failure['overloaded_engineers']
                issues = []
                if uncovered:
                    issues.append(f"{len(uncovered)} uncovered PPMs")
                if overloaded:
                    issues.append(f"{len(overloaded)} overloaded engineers")
                print(f"     {failure['week']} {failure['day']}: {', '.join(issues)}")
        
        # Weekly analysis
        weekly_failures = [r for r in weekly_results if not r['success']]
        weekly_success_rate = (len(weekly_results) - len(weekly_failures)) / len(weekly_results) * 100
        
        avg_am_preference = sum(r['am_preference_rate'] for r in weekly_results) / len(weekly_results) if weekly_results else 0
        
        print(f"\n📅 WEEKLY PPM RESULTS:")
        print(f"   Weeks tested: {len(weekly_results)}")
        print(f"   Successful weeks: {len(weekly_results) - len(weekly_failures)}")
        print(f"   Failed weeks: {len(weekly_failures)}")
        print(f"   Success rate: {weekly_success_rate:.1f}%")
        print(f"   AM preference rate: {avg_am_preference:.1f}%")
        
        # Monthly analysis
        monthly_failures = [r for r in monthly_results if not r['success']]
        monthly_success_rate = (len(monthly_results) - len(monthly_failures)) / len(monthly_results) * 100
        
        print(f"\n📅 MONTHLY PPM RESULTS:")
        print(f"   Months tested: {len(monthly_results)}")
        print(f"   Successful months: {len(monthly_results) - len(monthly_failures)}")
        print(f"   Failed months: {len(monthly_failures)}")
        print(f"   Success rate: {monthly_success_rate:.1f}%")
        
        # Overall assessment
        if daily_success_rate >= 95 and weekly_success_rate >= 95 and monthly_success_rate >= 95:
            overall_status = "✅ EXCELLENT"
        elif daily_success_rate >= 90 and weekly_success_rate >= 90 and monthly_success_rate >= 90:
            overall_status = "🟡 GOOD"
        else:
            overall_status = "❌ NEEDS IMPROVEMENT"
        
        print(f"\n🎯 OVERALL STATUS: {overall_status}")
        
        return {
            'team': team,
            'daily': {
                'success_rate': daily_success_rate,
                'failures': len(daily_failures),
                'total_tests': len(daily_results)
            },
            'weekly': {
                'success_rate': weekly_success_rate,
                'failures': len(weekly_failures),
                'am_preference_rate': avg_am_preference,
                'total_tests': len(weekly_results)
            },
            'monthly': {
                'success_rate': monthly_success_rate,
                'failures': len(monthly_failures),
                'total_tests': len(monthly_results)
            },
            'overall_status': overall_status,
            'detailed_failures': {
                'daily': daily_failures[:5],  # First 5 failures for debugging
                'weekly': weekly_failures[:5],
                'monthly': monthly_failures[:5]
            }
        }
    
    def run_validation(self, weeks_to_test=36):
        """Run comprehensive validation for both teams"""
        print(f"🚀 Starting comprehensive PPM validation")
        print(f"📊 Testing {weeks_to_test} weeks (full rotation cycle)")
        print(f"📅 Total daily tests: {weeks_to_test * 5 * 2} (both teams)")
        print(f"📅 Total weekly tests: {weeks_to_test * 2} (both teams)")
        print(f"📅 Total monthly tests: {weeks_to_test // 4 * 2} (both teams)")
        
        results = {}
        
        # Validate both teams
        for team in [1, 2]:
            results[team] = self.validate_team(team, weeks_to_test)
        
        # Generate summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 50)
        
        for team in [1, 2]:
            result = results[team]
            print(f"\n🏢 TEAM {team}: {result['overall_status']}")
            print(f"   Daily: {result['daily']['success_rate']:.1f}% ({result['daily']['failures']} failures)")
            print(f"   Weekly: {result['weekly']['success_rate']:.1f}% ({result['weekly']['failures']} failures)")
            print(f"   Monthly: {result['monthly']['success_rate']:.1f}% ({result['monthly']['failures']} failures)")
        
        # Save results
        output_file = 'outputs/comprehensive_ppm_validation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return results


def main():
    """Run the comprehensive PPM validation"""
    validator = ComprehensivePPMValidator()
    results = validator.run_validation(weeks_to_test=36)
    
    print(f"\n✅ Comprehensive PPM validation complete!")
    print(f"📊 Check the results above and in outputs/comprehensive_ppm_validation.json")


if __name__ == "__main__":
    main()
