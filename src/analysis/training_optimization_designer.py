#!/usr/bin/env python3

"""
Training Optimization Designer
==============================

This module analyzes current engineer qualifications against optimal requirements
and uses optimization techniques to recommend training that maximizes coverage
improvement while minimizing training effort.

Key Features:
- Loads current qualifications from EngQual.csv
- Compares against optimal MILP-generated matrices
- Identifies critical skill gaps
- Optimizes training assignments for maximum coverage impact
- Provides cost-benefit analysis of training recommendations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Configure which teams to process (set to [1] for Team 1 only, [1, 2] for both)
ACTIVE_TEAMS = [1]

from .milp_optimization_designer import MILPOptimizationDesigner
from .coverage_validator import CoverageValidator
from .training_progress_analyzer import TrainingProgressAnalyzer
from .qualification_duration_predictor import QualificationDurationPredictor

# Add logging and error handling
from ..utils.logger import get_logger, log_function_entry, log_function_exit, log_optimization_progress
from ..utils.exceptions import OptimizationError, DataValidationError


class TrainingOptimizationDesigner:
    """Training optimization using current vs optimal state analysis"""
    
    def __init__(self, optimizer_results):
        """Initialize with PPM optimization results"""
        self.logger = get_logger(__name__)
        log_function_entry(self.logger, "__init__")
        
        if not optimizer_results:
            raise DataValidationError(
                "Optimizer results are required but not provided",
                data_source="optimizer_results"
            )
        
        self.optimizer = optimizer_results
        self.logger.info("Initializing training optimization components")
        
        try:
            self.milp_designer = MILPOptimizationDesigner(optimizer_results)
            self.coverage_validator = CoverageValidator()
            self.training_analyzer = TrainingProgressAnalyzer("data")
            self.duration_predictor = QualificationDurationPredictor()
            self.current_date = datetime.now()
            
            self.logger.info("Training optimization designer initialized successfully")
            
            # Console output for backwards compatibility
            print("🎓 TRAINING OPTIMIZATION DESIGNER INITIALIZED")
            print("   Approach: Current state vs optimal training analysis")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training optimization designer: {e}")
            raise OptimizationError(
                f"Training designer initialization failed: {e}",
                algorithm="Training"
            )
        print("   Goal: Maximize coverage improvement with minimal training effort")
        print("   Data: EngQual.csv current qualifications vs MILP optimal state")
        print("   Training Timeline: Ongoing training analysis with bias weights")
    
    def load_current_qualification_state(self):
        """Load current engineer qualifications from EngQual.csv"""
        print("\n📊 LOADING CURRENT QUALIFICATION STATE FROM ENGQUAL.CSV")
        
        try:
            # Load EngQual.csv
            df = pd.read_csv('data/raw/EngQual.csv')
            print(f"   📁 Loaded {len(df)} qualification records")
            
            # Filter for active qualifications (not expired, not temp disqualified)
            current_date = self.current_date
            
            # Convert date columns
            df['Qualification Start'] = pd.to_datetime(df['Qualification Start'], errors='coerce')
            df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
            
            # Handle expired qualifications using available data format
            if 'expired' in df.columns:
                # Use pre-calculated expired field if available
                # Handle string values: 'False' = active, 'True' = expired
                expired_mask = (df['expired'].astype(str).str.lower() == 'false')
                expired_count = len(df[~expired_mask])
                print(f"   🔍 Using 'expired' field for filtering")
                print(f"   ❌ Found {expired_count} expired qualifications to exclude")
            else:
                # Use date comparison for expiration filtering
                expired_mask = (df['Expiration'] > current_date)
                print(f"   🔍 Using date comparison for expiration filtering")
            
            # Filter active qualifications
            active_mask = (
                expired_mask &  # Not expired (using appropriate method)
                (df['Temp Disqualified'] != '+') &   # Not temp disqualified
                (df['Employee Name'] != 'OUT OF SERVICE')  # Not out of service
            )
            
            active_df = df[active_mask].copy()
            print(f"   ✅ {len(active_df)} active qualifications")
            
            # Build current qualification matrices by team
            current_matrices = {}
            
            # Load engineer data to get team assignments
            engineers_by_team = self._load_engineer_team_assignments()
            
            for team in ACTIVE_TEAMS:
                current_matrices[team] = {}
                team_engineers = engineers_by_team[team]
                
                for eng_code in team_engineers:
                    eng_name = team_engineers[eng_code]['name']
                    eng_role = team_engineers[eng_code]['role']
                    
                    # Get current qualifications for this engineer
                    eng_quals = active_df[active_df['Employee Code'] == eng_code]['Qualification'].tolist()
                    
                    # Filter to PPM-relevant qualifications only
                    ppm_quals = [q for q in eng_quals if self._is_ppm_qualification(q)]
                    daily_quals = [q for q in ppm_quals if self._is_daily_qualification(q, team)]
                    
                    # Extract ride assignments from qualifications
                    assigned_rides = list(set([q.split('.')[0] for q in ppm_quals]))
                    
                    # Categorize by ride type
                    type_a_rides = [r for r in assigned_rides if self._get_ride_type(r) == 'A']
                    type_b_rides = [r for r in assigned_rides if self._get_ride_type(r) == 'B']
                    type_c_rides = [r for r in assigned_rides if self._get_ride_type(r) == 'C']
                    
                    current_matrices[team][eng_code] = {
                        'name': eng_name,
                        'role': eng_role,
                        'rota_number': team_engineers[eng_code].get('rota_number', 1),
                        'assigned_rides': assigned_rides,
                        'type_a_rides': type_a_rides,
                        'type_b_rides': type_b_rides,
                        'type_c_rides': type_c_rides,
                        'qualifications': ppm_quals,
                        'daily_qualifications': daily_quals,
                        'coverage_score': len(ppm_quals),
                        'current_state': True  # Flag to indicate this is current state
                    }
                
                print(f"   🏢 Team {team}: {len(current_matrices[team])} engineers with qualifications")
            
            return current_matrices
            
        except FileNotFoundError:
            print("   ⚠️  EngQual.csv not found in data/raw/")
            return {}
        except Exception as e:
            print(f"   ❌ Error loading qualifications: {e}")
            return {}
    
    def create_current_state_matrices(self, current_matrices):
        """Convert current qualifications into proper qualification matrices format"""
        print("\n🎯 CREATING QUALIFICATION MATRICES FROM CURRENT STATE")
        
        # The current_matrices from load_current_qualification_state() are already in the right format
        # We just need to ensure they're properly structured for the validation system
        
        formatted_matrices = {}
        
        for team in ACTIVE_TEAMS:
            if team not in current_matrices:
                continue
                
            formatted_matrices[team] = {}
            
            for eng_code, eng_data in current_matrices[team].items():
                # Ensure the engineer data has all required fields for validation
                formatted_matrices[team][eng_code] = {
                    'name': eng_data['name'],
                    'role': eng_data['role'],
                    'rota_number': eng_data.get('rota_number', 1),
                    'assigned_rides': eng_data['assigned_rides'],
                    'type_a_rides': eng_data['type_a_rides'],
                    'type_b_rides': eng_data['type_b_rides'],
                    'type_c_rides': eng_data['type_c_rides'],
                    'qualifications': eng_data['qualifications'],
                    'daily_qualifications': eng_data.get('daily_qualifications', []),
                    'coverage_score': eng_data.get('coverage_score', len(eng_data['qualifications'])),
                    'current_state': True
                }
        
        for team in formatted_matrices:
            print(f"   🏢 Team {team}: {len(formatted_matrices[team])} engineers with current qualifications")
            
        return formatted_matrices
    
    def optimize_training_assignments(self, current_matrices):
        """Generate ride assignments and complete qualification requirements from current state"""
        print("\n🧠 OPTIMIZING TRAINING ASSIGNMENTS FROM CURRENT STATE")
        print("Approach: Assign rides to engineers, then determine ALL required qualifications")
        print("=" * 70)
        
        # Step 0: Analyze ongoing training to get bias weights and timeline data
        print("\n📈 ANALYZING ONGOING TRAINING FOR BIAS WEIGHTS AND TIMELINE DATA...")
        training_progress_report = self.training_analyzer.analyze_ongoing_training()
        
        # Create a lookup dictionary for bias recommendations
        bias_recommendations = {}
        for rec in self.training_analyzer.bias_recommendations:
            qual_key = f"{rec['engineer']}_{rec['required_qual']}"
            bias_recommendations[qual_key] = rec
        
        training_progress_data = {
            'bias_recommendations': bias_recommendations,
            'report': training_progress_report
        }
        print(f"   ✅ Training analysis complete: {len(bias_recommendations)} bias recommendations")
        
        training_recommendations = {}
        
        for team in ACTIVE_TEAMS:
            if team not in current_matrices:
                continue
                
            print(f"\n🏢 TEAM {team} RIDE ASSIGNMENT & QUALIFICATION ANALYSIS:")
            
            # Step 1: Use MILP designer to get optimal ride assignments
            optimal_ride_assignments = self._get_milp_ride_assignments(team)
            
            # Step 2: For each engineer, determine what qualifications they need for their assigned rides
            qualification_requirements = self._determine_qualification_requirements(
                optimal_ride_assignments, team
            )
            
            # Step 3: Compare current qualifications vs required qualifications
            training_gaps = self._compare_current_vs_required(
                current_matrices[team], qualification_requirements, team
            )
            
            # Step 4: Generate training recommendations for the gaps with timeline data
            optimized_recommendations = self._generate_training_recommendations_with_timeline(
                training_gaps, team, training_progress_data
            )
            
            training_recommendations[team] = optimized_recommendations
        
        return training_recommendations
    
    def _get_milp_ride_assignments(self, team):
        """Get optimal ride assignments from MILP designer"""
        print(f"   🎯 Getting optimal ride assignments for Team {team} from MILP...")
        
        # Use the MILP designer to get optimal assignments
        optimal_matrices = self.milp_designer.create_optimized_qualification_matrices()
        
        if team not in optimal_matrices:
            return {}
        
        ride_assignments = {}
        for eng_code, eng_data in optimal_matrices[team].items():
            ride_assignments[eng_code] = {
                'name': eng_data['name'],
                'role': eng_data['role'],
                'assigned_rides': eng_data['assigned_rides'],
                'type_a_rides': eng_data['type_a_rides'],
                'type_b_rides': eng_data['type_b_rides'],
                'type_c_rides': eng_data['type_c_rides']
            }
        
        print(f"      ✅ Got ride assignments for {len(ride_assignments)} engineers")
        return ride_assignments
    
    def _determine_qualification_requirements(self, ride_assignments, team):
        """Determine ALL qualifications needed for each engineer's assigned rides"""
        print(f"   📋 Determining qualification requirements for assigned rides...")
        
        qualification_requirements = {}
        
        for eng_code, assignment in ride_assignments.items():
            required_quals = set()
            
            # For each assigned ride, get ALL required qualifications
            for ride_code in assignment['assigned_rides']:
                # Get daily qualifications for this ride
                if ride_code in self.optimizer.ppms_by_type['daily']:
                    for ppm in self.optimizer.ppms_by_type['daily'][ride_code]['ppms']:
                        # Check if this qualification matches the engineer's role
                        qual_role = self._get_qualification_role(ppm['qualification_code'])
                        if qual_role == 'any' or assignment['role'] == qual_role:
                            required_quals.add(ppm['qualification_code'])
                
                # Get weekly qualifications for this ride
                if ride_code in self.optimizer.ppms_by_type['weekly']:
                    for ppm in self.optimizer.ppms_by_type['weekly'][ride_code]['ppms']:
                        qual_role = self._get_qualification_role(ppm['qualification_code'])
                        if qual_role == 'any' or assignment['role'] == qual_role:
                            required_quals.add(ppm['qualification_code'])
                
                # Get monthly qualifications for this ride
                if ride_code in self.optimizer.ppms_by_type['monthly']:
                    for ppm in self.optimizer.ppms_by_type['monthly'][ride_code]['ppms']:
                        qual_role = self._get_qualification_role(ppm['qualification_code'])
                        if qual_role == 'any' or assignment['role'] == qual_role:
                            required_quals.add(ppm['qualification_code'])

                # Get reactive qualifications for this ride (discipline-agnostic)
                if ride_code in self.optimizer.ppms_by_type['reactive']:
                    for ppm in self.optimizer.ppms_by_type['reactive'][ride_code]['ppms']:
                        # Reactive is discipline-agnostic - all engineers with ride get reactive
                        required_quals.add(ppm['qualification_code'])

            # ALSO check if this engineer has reactive-only assignments from gap filling
            # These are stored in the assignment['qualifications'] list
            if 'qualifications' in assignment:
                for qual in assignment['qualifications']:
                    if '.5.' in qual:  # Reactive qualifications
                        required_quals.add(qual)

            qualification_requirements[eng_code] = {
                'name': assignment['name'],
                'role': assignment['role'],
                'assigned_rides': assignment['assigned_rides'],
                'required_qualifications': list(required_quals),
                'total_required': len(required_quals)
            }
        
        print(f"      ✅ Determined qualification requirements for {len(qualification_requirements)} engineers")
        return qualification_requirements
    
    def _compare_current_vs_required(self, current_team, qualification_requirements, team):
        """Compare current qualifications vs required qualifications"""
        print(f"   🔍 Comparing current vs required qualifications...")
        
        training_gaps = {}
        
        for eng_code, requirements in qualification_requirements.items():
            # Get current qualifications (if engineer exists)
            current_quals = set()
            if eng_code in current_team:
                current_quals = set(current_team[eng_code]['qualifications'])
            
            # Calculate gaps
            required_quals = set(requirements['required_qualifications'])
            missing_quals = required_quals - current_quals
            
            if missing_quals:
                # Categorize missing qualifications
                daily_missing = [q for q in missing_quals if self._is_daily_qualification(q, team)]
                weekly_missing = [q for q in missing_quals if self._is_weekly_qualification(q, team)]
                monthly_missing = [q for q in missing_quals if self._is_monthly_qualification(q, team)]
                reactive_missing = [q for q in missing_quals if self._is_reactive_qualification(q, team)]

                training_gaps[eng_code] = {
                    'name': requirements['name'],
                    'role': requirements['role'],
                    'assigned_rides': requirements['assigned_rides'],
                    'current_qualifications': len(current_quals),
                    'required_qualifications': len(required_quals),
                    'missing_qualifications': list(missing_quals),
                    'daily_missing': daily_missing,
                    'weekly_missing': weekly_missing,
                    'monthly_missing': monthly_missing,
                    'reactive_missing': reactive_missing,
                    'total_missing': len(missing_quals),
                    'is_vacancy': eng_code.startswith('VACANCY') or eng_code not in current_team
                }
        
        # Display summary
        total_engineers = len(training_gaps)
        vacancy_engineers = len([gap for gap in training_gaps.values() if gap['is_vacancy']])
        real_engineers = total_engineers - vacancy_engineers
        
        print(f"      📊 Training gap summary:")
        print(f"         Real engineers needing training: {real_engineers}")
        print(f"         Vacant positions needing qualifications: {vacancy_engineers}")
        print(f"         Total positions: {total_engineers}")
        
        # Show top gaps
        sorted_gaps = sorted(training_gaps.items(), key=lambda x: x[1]['total_missing'], reverse=True)
        print(f"      🎯 Top 3 training priorities:")
        for i, (eng_code, gap) in enumerate(sorted_gaps[:3]):
            status = "VACANT" if gap['is_vacancy'] else "CURRENT"
            print(f"         {i+1}. {gap['name']} ({status}): {gap['total_missing']} quals needed")
        
        return training_gaps
    
    def _generate_training_recommendations_with_timeline(self, training_gaps, team, training_progress_data):
        """Generate training recommendations from training gaps with timeline data"""
        print(f"   💡 Generating training recommendations with timeline analysis...")
        
        optimized_assignments = []
        bias_recommendations = training_progress_data.get('bias_recommendations', {})
        
        for eng_code, gap in training_gaps.items():
            # For all engineers (including vacancies), recommend all missing qualifications
            if gap['missing_qualifications']:
                # Enhance each qualification with training timeline data
                enhanced_qualifications = []
                for qual in gap['missing_qualifications']:
                    qual_key = f"{eng_code}_{qual}"
                    timeline_data = bias_recommendations.get(qual_key, {})
                    
                    # Convert Timestamp objects to strings for JSON serialization
                    start_date = timeline_data.get('training_start_date')
                    last_date = timeline_data.get('last_training_date')
                    
                    enhanced_qual = {
                        'qualification_code': qual,
                        'training_start_date': start_date.isoformat() if start_date is not None else None,
                        'last_training_date': last_date.isoformat() if last_date is not None else None,
                        'training_duration_days': timeline_data.get('training_duration_days'),
                        'sessions': timeline_data.get('sessions', 0),
                        'total_hours': timeline_data.get('total_hours', 0),
                        'bias_weight': timeline_data.get('bias_weight', 1.0),
                        'training_frequency': timeline_data.get('training_frequency', 0),
                        'urgency': timeline_data.get('urgency', 'Normal'),
                        'priority': timeline_data.get('priority', 'Medium'),
                        'ongoing_training': len(timeline_data) > 0  # True if timeline data exists
                    }
                    enhanced_qualifications.append(enhanced_qual)
                
                # Calculate weighted training effort using bias weights
                weighted_effort = sum(qual.get('bias_weight', 1.0) for qual in enhanced_qualifications)
                ongoing_training_count = sum(1 for qual in enhanced_qualifications if qual.get('ongoing_training', False))
                
                # Calculate improved priority score based on business logic
                priority_score = self._calculate_improved_priority_score(
                    gap, enhanced_qualifications, ongoing_training_count
                )
                
                optimized_assignments.append({
                    'engineer_code': eng_code,
                    'engineer_name': gap['name'],
                    'role': gap['role'],
                    'assigned_rides': gap['assigned_rides'],
                    'recommended_qualifications': gap['missing_qualifications'],  # Keep simple list for compatibility
                    'enhanced_qualifications': enhanced_qualifications,  # New: detailed timeline data
                    'training_effort': gap['total_missing'],
                    'weighted_training_effort': weighted_effort,  # New: bias-weighted effort
                    'ongoing_training_count': ongoing_training_count,  # New: count of ongoing training
                    'priority_score': priority_score,  # New: improved business logic scoring
                    'daily_impact': len(gap['daily_missing']),
                    'weekly_impact': len(gap['weekly_missing']),
                    'monthly_impact': len(gap['monthly_missing']),
                    'reactive_impact': len(gap.get('reactive_missing', [])),
                    'is_vacancy': gap['is_vacancy']
                })
        
        # Sort by improved priority score (business logic prioritization)
        optimized_assignments.sort(key=lambda x: x['priority_score'], reverse=True)
        
        total_effort = sum(a['training_effort'] for a in optimized_assignments)
        vacancy_effort = sum(a['training_effort'] for a in optimized_assignments if a['is_vacancy'])
        ongoing_training_total = sum(a['ongoing_training_count'] for a in optimized_assignments)
        
        print(f"      ✅ Generated recommendations for {len(optimized_assignments)} engineers")
        print(f"      📚 Total training effort: {total_effort} qualifications")
        print(f"      🔄 Ongoing training qualifications: {ongoing_training_total}")
        print(f"      🏢 Vacancy training effort: {vacancy_effort} qualifications")
        
        return {
            'optimized_assignments': optimized_assignments,
            'total_training_effort': total_effort,
            'vacancy_training_effort': vacancy_effort,
            'ongoing_training_count': ongoing_training_total,
            'method': 'MILP_Ride_Assignment_With_Timeline'
        }
    
    def _calculate_improved_priority_score(self, gap, enhanced_qualifications, ongoing_training_count):
        """Calculate priority score based on business logic:
        1. Highest Priority: Ongoing Training (Quick ROI)
        2. High Priority: Portfolio Completion + Vacancy Comprehensive Training 
        3. Low Priority: New Rides for Well-Qualified Engineers
        """
        base_score = 0.0
        
        # 1. HIGHEST PRIORITY: Ongoing Training (Quick ROI)
        ongoing_bonus = ongoing_training_count * 50.0  # 50 points per ongoing training
        
        # 2. HIGH PRIORITY: Vacancy Comprehensive Training
        vacancy_bonus = 30.0 if gap['is_vacancy'] else 0.0
        
        # 3. HIGH PRIORITY: Portfolio Completion Bonus
        # Note: current_qualifications in gap is a count (int), not a list
        # We'll use a simplified approach based on training effort and ongoing training
        portfolio_completion_bonus = 0.0
        assigned_rides = gap.get('assigned_rides', [])
        
        # Portfolio completion bonus based on ongoing training on assigned rides
        for ride in assigned_rides:
            ride_ongoing_count = sum(1 for qual in enhanced_qualifications 
                                   if qual.get('ongoing_training', False) and 
                                   qual['qualification_code'].startswith(ride + '.'))
            if ride_ongoing_count > 0:  # Has ongoing training on this ride
                portfolio_completion_bonus += 20.0  # 20 points per ride with ongoing training
        
        # 4. LOW PRIORITY: Penalty for Over-Qualified Engineers Starting New Rides
        total_current_quals = gap.get('current_qualifications', 0)
        if isinstance(total_current_quals, list):
            total_current_quals = len(total_current_quals)
        
        over_qualification_penalty = 0.0
        
        if total_current_quals > 25:  # Well-qualified engineer
            # Penalize based on lack of ongoing training (indicates new rides)
            total_recommendations = len(enhanced_qualifications)
            ongoing_recommendations = sum(1 for qual in enhanced_qualifications 
                                        if qual.get('ongoing_training', False))
            
            if ongoing_recommendations == 0 and total_recommendations > 10:  # Many new qualifications, no ongoing
                over_qualification_penalty = 15.0  # Penalty for over-training experts on new rides
        
        # 5. Medium Priority: Daily Impact Bonus (operational importance)
        daily_missing = gap.get('daily_missing', [])
        if isinstance(daily_missing, list):
            daily_impact_bonus = len(daily_missing) * 2.0  # 2 points per daily qual needed
        else:
            daily_impact_bonus = daily_missing * 2.0  # Handle integer case
        
        # Final score calculation
        priority_score = (
            base_score + 
            ongoing_bonus +                    # Highest: Complete ongoing training
            vacancy_bonus +                    # High: Build up new hires
            portfolio_completion_bonus +       # High: Complete ride portfolios
            daily_impact_bonus -               # Medium: Operational impact
            over_qualification_penalty        # Low: Don't over-train experts
        )
        
        return priority_score
    
    def _generate_training_recommendations(self, training_gaps, team):
        """Generate training recommendations from training gaps (legacy method)"""
        print(f"   💡 Generating training recommendations...")
        
        optimized_assignments = []
        
        for eng_code, gap in training_gaps.items():
            # For all engineers (including vacancies), recommend all missing qualifications
            if gap['missing_qualifications']:
                optimized_assignments.append({
                    'engineer_code': eng_code,
                    'engineer_name': gap['name'],
                    'role': gap['role'],
                    'assigned_rides': gap['assigned_rides'],
                    'recommended_qualifications': gap['missing_qualifications'],
                    'training_effort': gap['total_missing'],
                    'daily_impact': len(gap['daily_missing']),
                    'weekly_impact': len(gap['weekly_missing']),
                    'monthly_impact': len(gap['monthly_missing']),
                    'reactive_impact': len(gap.get('reactive_missing', [])),
                    'is_vacancy': gap['is_vacancy']
                })
        
        # Sort by training effort (vacancies typically need more)
        optimized_assignments.sort(key=lambda x: x['training_effort'], reverse=True)
        
        total_effort = sum(a['training_effort'] for a in optimized_assignments)
        vacancy_effort = sum(a['training_effort'] for a in optimized_assignments if a['is_vacancy'])
        
        print(f"      ✅ Generated recommendations for {len(optimized_assignments)} engineers")
        print(f"      📚 Total training effort: {total_effort} qualifications")
        print(f"      🏢 Vacancy training effort: {vacancy_effort} qualifications")
        
        return {
            'optimized_assignments': optimized_assignments,
            'total_training_effort': total_effort,
            'vacancy_training_effort': vacancy_effort,
            'method': 'MILP_Ride_Assignment'
        }
    
    def generate_detailed_training_report(self, training_recommendations, current_matrices):
        """Generate detailed training report with current vs recommended breakdown"""
        print("\n📊 GENERATING DETAILED TRAINING REPORT")
        print("=" * 70)
        
        detailed_report = {}
        
        for team in ACTIVE_TEAMS:
            if team not in training_recommendations or team not in current_matrices:
                continue
                
            print(f"\n🏢 TEAM {team} DETAILED TRAINING ANALYSIS:")
            
            team_report = {
                'engineers': {},
                'summary': {
                    'total_engineers': 0,
                    'engineers_needing_training': 0,
                    'vacant_positions': 0,
                    'total_training_effort': 0,
                    'high_impact_training': 0,
                    'medium_impact_training': 0,
                    'low_impact_training': 0
                },
                'priority_training': []
            }
            
            team_current = current_matrices[team]
            team_recommendations = training_recommendations[team]['optimized_assignments']
            
            # Create lookup for easy access
            rec_lookup = {rec['engineer_code']: rec for rec in team_recommendations}
            
            for eng_code, current_profile in team_current.items():
                engineer_report = self._generate_engineer_detailed_report(
                    eng_code, current_profile, rec_lookup.get(eng_code), team
                )
                
                if engineer_report:
                    team_report['engineers'][eng_code] = engineer_report
                    
                    # Update summary
                    team_report['summary']['total_engineers'] += 1
                    if engineer_report['needs_training']:
                        team_report['summary']['engineers_needing_training'] += 1
                        team_report['summary']['total_training_effort'] += engineer_report['training_effort']
                        
                        # Categorize impact
                        if engineer_report['daily_impact'] >= 3:
                            team_report['summary']['high_impact_training'] += 1
                        elif engineer_report['daily_impact'] >= 1:
                            team_report['summary']['medium_impact_training'] += 1
                        else:
                            team_report['summary']['low_impact_training'] += 1
                    
                    if engineer_report['is_vacancy']:
                        team_report['summary']['vacant_positions'] += 1
            
            # Generate priority training list
            team_report['priority_training'] = self._generate_priority_training_list(team_report['engineers'])
            
            # Display summary
            summary = team_report['summary']
            print(f"   📈 TEAM {team} SUMMARY:")
            print(f"      Total Engineers: {summary['total_engineers']}")
            print(f"      Need Training: {summary['engineers_needing_training']}")
            print(f"      Vacant Positions: {summary['vacant_positions']}")
            print(f"      Total Training Effort: {summary['total_training_effort']} qualifications")
            print(f"      High Impact Training: {summary['high_impact_training']} engineers (≥3 daily)")
            print(f"      Medium Impact Training: {summary['medium_impact_training']} engineers (1-2 daily)")
            print(f"      Low Impact Training: {summary['low_impact_training']} engineers (0 daily)")
            
            detailed_report[team] = team_report
        
        return detailed_report
    
    def _generate_engineer_detailed_report(self, eng_code, current_profile, recommendation, team):
        """Generate detailed report for individual engineer"""
        if not recommendation:
            return {
                'engineer_code': eng_code,
                'engineer_name': current_profile['name'],
                'role': current_profile['role'],
                'needs_training': False,
                'is_vacancy': eng_code.startswith('VACANCY'),
                'current_total_qualifications': len(current_profile['qualifications']),
                'current_daily_qualifications': len(current_profile['daily_qualifications']),
                'assigned_rides': current_profile.get('assigned_rides', []),
                'ride_breakdown': {},
                'training_effort': 0,
                'daily_impact': 0,
                'weekly_impact': 0,
                'monthly_impact': 0,
                'reactive_impact': 0
            }
        
        # Generate ride-by-ride breakdown
        ride_breakdown = {}
        for ride_code in recommendation['assigned_rides']:
            ride_analysis = self._analyze_ride_qualifications(
                ride_code, current_profile['qualifications'], 
                recommendation['recommended_qualifications'], team
            )
            ride_breakdown[ride_code] = ride_analysis
        
        return {
            'engineer_code': eng_code,
            'engineer_name': current_profile['name'],
            'role': current_profile['role'],
            'needs_training': True,
            'is_vacancy': recommendation['is_vacancy'],
            'current_total_qualifications': len(current_profile['qualifications']),
            'current_daily_qualifications': len(current_profile['daily_qualifications']),
            'assigned_rides': recommendation['assigned_rides'],
            'ride_breakdown': ride_breakdown,
            'recommended_qualifications': recommendation['recommended_qualifications'],
            'enhanced_qualifications': recommendation.get('enhanced_qualifications', []),  # New: timeline data
            'training_effort': recommendation['training_effort'],
            'weighted_training_effort': recommendation.get('weighted_training_effort', recommendation['training_effort']),  # New
            'ongoing_training_count': recommendation.get('ongoing_training_count', 0),  # New
            'daily_impact': recommendation['daily_impact'],
            'weekly_impact': recommendation['weekly_impact'],
            'monthly_impact': recommendation['monthly_impact'],
            'reactive_impact': recommendation.get('reactive_impact', 0),
            'training_priority_score': self._calculate_training_priority_score(recommendation)
        }
    
    def _analyze_ride_qualifications(self, ride_code, current_quals, recommended_quals, team):
        """Analyze qualifications for a specific ride"""
        # Get all possible qualifications for this ride
        ride_quals = {
            'daily': [],
            'weekly': [],
            'monthly': [],
            'reactive': []
        }

        # Check daily PPMs
        if ride_code in self.optimizer.ppms_by_type['daily']:
            for ppm in self.optimizer.ppms_by_type['daily'][ride_code]['ppms']:
                ride_quals['daily'].append(ppm['qualification_code'])

        # Check weekly PPMs
        if ride_code in self.optimizer.ppms_by_type['weekly']:
            for ppm in self.optimizer.ppms_by_type['weekly'][ride_code]['ppms']:
                ride_quals['weekly'].append(ppm['qualification_code'])

        # Check monthly PPMs
        if ride_code in self.optimizer.ppms_by_type['monthly']:
            for ppm in self.optimizer.ppms_by_type['monthly'][ride_code]['ppms']:
                ride_quals['monthly'].append(ppm['qualification_code'])

        # Check reactive PPMs (discipline-agnostic)
        if ride_code in self.optimizer.ppms_by_type['reactive']:
            for ppm in self.optimizer.ppms_by_type['reactive'][ride_code]['ppms']:
                ride_quals['reactive'].append(ppm['qualification_code'])

        # Analyze current vs required
        all_ride_quals = ride_quals['daily'] + ride_quals['weekly'] + ride_quals['monthly'] + ride_quals['reactive']
        current_ride_quals = [q for q in current_quals if q.startswith(ride_code + '.')]
        recommended_ride_quals = [q for q in recommended_quals if q.startswith(ride_code + '.')]
        
        return {
            'ride_code': ride_code,
            'ride_type': self._get_ride_type(ride_code),
            'total_possible_qualifications': len(all_ride_quals),
            'current_qualifications': current_ride_quals,
            'current_count': len(current_ride_quals),
            'recommended_additional': recommended_ride_quals,
            'recommended_count': len(recommended_ride_quals),
            'final_count': len(current_ride_quals) + len(recommended_ride_quals),
            'daily_qualifications': {
                'current': [q for q in current_ride_quals if q in ride_quals['daily']],
                'recommended': [q for q in recommended_ride_quals if q in ride_quals['daily']]
            },
            'weekly_qualifications': {
                'current': [q for q in current_ride_quals if q in ride_quals['weekly']],
                'recommended': [q for q in recommended_ride_quals if q in ride_quals['weekly']]
            },
            'monthly_qualifications': {
                'current': [q for q in current_ride_quals if q in ride_quals['monthly']],
                'recommended': [q for q in recommended_ride_quals if q in ride_quals['monthly']]
            },
            'reactive_qualifications': {
                'current': [q for q in current_ride_quals if q in ride_quals['reactive']],
                'recommended': [q for q in recommended_ride_quals if q in ride_quals['reactive']]
            }
        }
    
    def _calculate_training_priority_score(self, recommendation):
        """Calculate priority score using improved business logic"""
        # Use the new improved priority score if available
        if 'priority_score' in recommendation:
            return recommendation['priority_score']
        
        # Fallback to old calculation for compatibility
        score = (recommendation['daily_impact'] * 10 + 
                recommendation['weekly_impact'] * 5 + 
                recommendation['monthly_impact'] * 2)
        
        # Old logic penalized vacancies - new logic prioritizes them
        if recommendation['is_vacancy']:
            score += 30  # High priority for vacant positions (changed from -20)
            
        return score
    
    def _generate_priority_training_list(self, engineers_report):
        """Generate prioritized training list"""
        # Get engineers needing training and sort by priority
        training_engineers = [
            eng for eng in engineers_report.values() 
            if eng['needs_training']
        ]
        
        # Sort by priority score (high to low)
        training_engineers.sort(key=lambda x: x['training_priority_score'], reverse=True)
        
        priority_list = []
        for i, eng in enumerate(training_engineers[:10]):  # Top 10
            priority_item = {
                'rank': i + 1,
                'engineer_code': eng['engineer_code'],
                'engineer_name': eng['engineer_name'],
                'is_vacancy': eng['is_vacancy'],
                'training_effort': eng['training_effort'],
                'daily_impact': eng['daily_impact'],
                'priority_score': eng['training_priority_score'],
                'top_rides_needing_training': []
            }
            
            # Get top 3 rides with most training needed
            ride_training_needs = []
            for ride_code, ride_data in eng['ride_breakdown'].items():
                if ride_data['recommended_count'] > 0:
                    ride_training_needs.append({
                        'ride_code': ride_code,
                        'ride_type': ride_data['ride_type'],
                        'training_needed': ride_data['recommended_count'],
                        'daily_training': len(ride_data['daily_qualifications']['recommended'])
                    })
            
            ride_training_needs.sort(key=lambda x: (x['daily_training'], x['training_needed']), reverse=True)
            priority_item['top_rides_needing_training'] = ride_training_needs[:3]
            
            priority_list.append(priority_item)
        
        return priority_list
    
    def display_detailed_training_report(self, detailed_report):
        """Display the detailed training report in a readable format"""
        print("\n📋 DETAILED TRAINING ANALYSIS REPORT")
        print("=" * 80)
        
        for team, team_data in detailed_report.items():
            print(f"\n🏢 TEAM {team} DETAILED BREAKDOWN:")
            print("-" * 50)
            
            # Show priority training first
            print(f"\n🎯 TOP TRAINING PRIORITIES:")
            for priority in team_data['priority_training'][:5]:  # Top 5
                status = "VACANT" if priority['is_vacancy'] else "CURRENT"
                print(f"   {priority['rank']}. {priority['engineer_name']} ({status})")
                print(f"      📚 Training Effort: {priority['training_effort']} qualifications")
                print(f"      🌅 Daily Impact: {priority['daily_impact']} qualifications")
                print(f"      🎯 Priority Score: {priority['priority_score']}")
                print(f"      🎢 Top Rides Needing Training:")
                for ride in priority['top_rides_needing_training']:
                    print(f"         - {ride['ride_code']} (Type {ride['ride_type']}): {ride['training_needed']} quals ({ride['daily_training']} daily)")
                print()
            
            # Show detailed breakdown for top engineers
            print(f"\n📊 DETAILED ENGINEER BREAKDOWN (Top 3):")
            top_engineers = sorted(
                [eng for eng in team_data['engineers'].values() if eng['needs_training']], 
                key=lambda x: x['training_priority_score'], reverse=True
            )[:3]
            
            for eng in top_engineers:
                self._display_engineer_breakdown(eng)
    
    def _display_engineer_breakdown(self, engineer_report):
        """Display detailed breakdown for individual engineer"""
        status = "VACANT" if engineer_report['is_vacancy'] else "CURRENT"
        print(f"\n   👤 {engineer_report['engineer_name']} ({status}) - {engineer_report['role'].upper()}")
        print(f"      Current Qualifications: {engineer_report['current_total_qualifications']} total, {engineer_report['current_daily_qualifications']} daily")
        print(f"      Training Needed: {engineer_report['training_effort']} qualifications")
        print(f"      Impact: {engineer_report['daily_impact']} daily, {engineer_report['weekly_impact']} weekly, {engineer_report['monthly_impact']} monthly")
        print(f"      Assigned Rides: {', '.join(engineer_report['assigned_rides'])}")
        
        print(f"      🎢 RIDE-BY-RIDE BREAKDOWN:")
        for ride_code, ride_data in engineer_report['ride_breakdown'].items():
            if ride_data['recommended_count'] > 0:
                current_count = ride_data['current_count']
                recommended_count = ride_data['recommended_count']
                final_count = ride_data['final_count']
                
                print(f"         {ride_code} (Type {ride_data['ride_type']}): {current_count} → {final_count} (+{recommended_count})")
                
                # Show daily qualifications breakdown
                daily_current = len(ride_data['daily_qualifications']['current'])
                daily_recommended = len(ride_data['daily_qualifications']['recommended'])
                if daily_recommended > 0:
                    print(f"           🌅 Daily: {daily_current} → {daily_current + daily_recommended} (+{daily_recommended})")
                
                # Show weekly qualifications breakdown
                weekly_current = len(ride_data['weekly_qualifications']['current'])
                weekly_recommended = len(ride_data['weekly_qualifications']['recommended'])
                if weekly_recommended > 0:
                    print(f"           📅 Weekly: {weekly_current} → {weekly_current + weekly_recommended} (+{weekly_recommended})")
                
                # Show monthly qualifications breakdown
                monthly_current = len(ride_data['monthly_qualifications']['current'])
                monthly_recommended = len(ride_data['monthly_qualifications']['recommended'])
                if monthly_recommended > 0:
                    print(f"           📆 Monthly: {monthly_current} → {monthly_current + monthly_recommended} (+{monthly_recommended})")
        
        print()
        
    def export_detailed_report_to_csv(self, detailed_report, output_dir="outputs/current"):
        """Export detailed training report to CSV files for easy analysis - minimal version for dashboard"""
        import csv
        from pathlib import Path
        
        print(f"\n📊 EXPORTING TRAINING REPORT (dashboard files only)")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create specific qualifications needed CSV with duration estimates (1 of 5 essential files)
        quals_file = output_path / "specific_qualifications_needed.csv"
        print(f"   📄 Creating specific qualifications: {quals_file}")
        
        with open(quals_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'Team', 'Engineer_Code', 'Engineer_Name', 'Role', 'Status',
                'Ride_Code', 'Ride_Type', 'Qualification_Code', 'PPM_Type',
                'Priority_Level', 'Training_Start_Date', 'Last_Training_Date',
                'Training_Duration_Days', 'Sessions', 'Total_Hours', 'Bias_Weight',
                'Training_Frequency', 'Urgency', 'Ongoing_Training'
            ])
            
            # Data rows
            for team, team_data in detailed_report.items():
                for eng_code, eng_data in team_data['engineers'].items():
                    if eng_data['needs_training']:
                        # Check if enhanced qualifications are available
                        enhanced_quals = eng_data.get('enhanced_qualifications', [])
                        enhanced_lookup = {q['qualification_code']: q for q in enhanced_quals} if enhanced_quals else {}
                        
                        for ride_code, ride_data in eng_data['ride_breakdown'].items():
                            # Daily qualifications (highest priority)
                            for qual in ride_data['daily_qualifications']['recommended']:
                                timeline_data = enhanced_lookup.get(qual, {})
                                writer.writerow([
                                    team, eng_data['engineer_code'], eng_data['engineer_name'],
                                    eng_data['role'].title(), 'VACANT' if eng_data['is_vacancy'] else 'CURRENT',
                                    ride_code, ride_data['ride_type'], qual, 'Daily', 'HIGH',
                                    timeline_data.get('training_start_date', ''),
                                    timeline_data.get('last_training_date', ''),
                                    timeline_data.get('training_duration_days', ''),
                                    timeline_data.get('sessions', ''),
                                    timeline_data.get('total_hours', ''),
                                    timeline_data.get('bias_weight', 1.0),
                                    timeline_data.get('training_frequency', 'New'),
                                    timeline_data.get('urgency', 'Normal'),
                                    timeline_data.get('ongoing_training', False)
                                ])
                            
                            # Weekly qualifications (medium priority)
                            for qual in ride_data['weekly_qualifications']['recommended']:
                                timeline_data = enhanced_lookup.get(qual, {})
                                writer.writerow([
                                    team, eng_data['engineer_code'], eng_data['engineer_name'],
                                    eng_data['role'].title(), 'VACANT' if eng_data['is_vacancy'] else 'CURRENT',
                                    ride_code, ride_data['ride_type'], qual, 'Weekly', 'MEDIUM',
                                    timeline_data.get('training_start_date', ''),
                                    timeline_data.get('last_training_date', ''),
                                    timeline_data.get('training_duration_days', ''),
                                    timeline_data.get('sessions', ''),
                                    timeline_data.get('total_hours', ''),
                                    timeline_data.get('bias_weight', 1.0),
                                    timeline_data.get('training_frequency', 'New'),
                                    timeline_data.get('urgency', 'Normal'),
                                    timeline_data.get('ongoing_training', False)
                                ])
                            
                            # Monthly qualifications (lower priority)
                            for qual in ride_data['monthly_qualifications']['recommended']:
                                timeline_data = enhanced_lookup.get(qual, {})
                                writer.writerow([
                                    team, eng_data['engineer_code'], eng_data['engineer_name'],
                                    eng_data['role'].title(), 'VACANT' if eng_data['is_vacancy'] else 'CURRENT',
                                    ride_code, ride_data['ride_type'], qual, 'Monthly', 'LOW',
                                    timeline_data.get('training_start_date', ''),
                                    timeline_data.get('last_training_date', ''),
                                    timeline_data.get('training_duration_days', ''),
                                    timeline_data.get('sessions', ''),
                                    timeline_data.get('total_hours', ''),
                                    timeline_data.get('bias_weight', 1.0),
                                    timeline_data.get('training_frequency', 'New'),
                                    timeline_data.get('urgency', 'Normal'),
                                    timeline_data.get('ongoing_training', False)
                                ])

                            # Reactive qualifications (lowest priority - discipline agnostic)
                            if 'reactive_qualifications' in ride_data and ride_data['reactive_qualifications']['recommended']:
                                for qual in ride_data['reactive_qualifications']['recommended']:
                                    timeline_data = enhanced_lookup.get(qual, {})
                                    writer.writerow([
                                        team, eng_data['engineer_code'], eng_data['engineer_name'],
                                        eng_data['role'].title(), 'VACANT' if eng_data['is_vacancy'] else 'CURRENT',
                                        ride_code, ride_data['ride_type'], qual, 'Reactive', 'LOWEST',
                                        timeline_data.get('training_start_date', ''),
                                        timeline_data.get('last_training_date', ''),
                                        timeline_data.get('training_duration_days', ''),
                                        timeline_data.get('sessions', ''),
                                        timeline_data.get('total_hours', ''),
                                        timeline_data.get('bias_weight', 1.0),
                                        timeline_data.get('training_frequency', 'New'),
                                        timeline_data.get('urgency', 'Normal'),
                                        timeline_data.get('ongoing_training', False)
                                    ])

        print("   ✅ Essential CSV file generated for Power BI dashboard")
        return []
    
    def validate_training_impact(self, training_recommendations):
        """Validate the impact of proposed training on coverage"""
        print("\n🧪 VALIDATING TRAINING IMPACT ON COVERAGE")
        print("=" * 70)
        
        # Create projected qualification matrices after training
        projected_matrices = self._apply_training_to_current_state(training_recommendations)
        
        # Validate coverage of projected state
        validation_results = self.coverage_validator.validate_assignment_coverage(projected_matrices)
        
        for team in ACTIVE_TEAMS:
            if team in validation_results:
                results = validation_results[team]
                daily_cov = results['daily']['coverage_percentage']
                weekly_cov = results['weekly']['coverage_percentage']
                monthly_cov = results['monthly']['coverage_percentage']
                
                print(f"\n🏢 TEAM {team} POST-TRAINING COVERAGE PROJECTION:")
                print(f"   Daily Coverage:   {daily_cov:.1f}% {'🎯' if daily_cov >= 90 else '⚠️' if daily_cov >= 60 else '❌'}")
                print(f"   Weekly Coverage:  {weekly_cov:.1f}% {'🎯' if weekly_cov >= 90 else '⚠️' if weekly_cov >= 60 else '❌'}")
                print(f"   Monthly Coverage: {monthly_cov:.1f}% {'🎯' if monthly_cov >= 90 else '⚠️' if monthly_cov >= 60 else '❌'}")
        
        return validation_results
    
    def _load_engineer_team_assignments(self):
        """Load engineer team assignments from processed data"""
        engineers_by_team = {1: {}, 2: {}}
        
        for team in ACTIVE_TEAMS:
            for role in ['elec', 'mech']:
                file_path = f'data/processed/engineers/team{team}_{role}_engineers.json'
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for eng in data.get('engineers', []):
                            eng_code = eng['employee_code']
                            engineers_by_team[team][eng_code] = {
                                'name': eng['timeplan_name'],
                                'role': 'electrical' if role == 'elec' else 'mechanical',
                                'rota_number': eng.get('rota_number', 1),
                                'active': eng.get('active', True),
                                'vacancy': eng.get('vacancy', False)
                            }
                except FileNotFoundError:
                    continue
        
        return engineers_by_team
    
    def _is_ppm_qualification(self, qualification):
        """Check if qualification is PPM-related (format: RIDE.X.TYPE.S)"""
        parts = qualification.split('.')
        if len(parts) < 3:
            return False
        
        # Check if it's a ride code we know about
        ride_code = parts[0]
        return ride_code in self.optimizer.rides_info
    
    def _is_daily_qualification(self, qualification, team):
        """Check if qualification is for daily PPMs"""
        ride_code = qualification.split('.')[0]
        
        if ride_code in self.optimizer.ppms_by_type['daily']:
            ppm_data = self.optimizer.ppms_by_type['daily'][ride_code]
            for ppm in ppm_data['ppms']:
                if ppm['qualification_code'] == qualification:
                    return True
        return False
    
    def _get_ride_type(self, ride_code):
        """Get ride complexity type (A, B, or C)"""
        if ride_code in self.optimizer.rides_info:
            return self.optimizer.rides_info[ride_code].get('type', 'C')
        return 'C'
    
    def _analyze_current_coverage_gaps(self, team_current, team):
        """Analyze coverage gaps in current state"""
        print(f"   📊 Analyzing coverage gaps in current Team {team} state...")
        
        # Get all required qualifications for this team
        all_required_quals = set()
        
        # Get daily qualifications (highest priority)
        for ride_code, ppm_data in self.optimizer.ppms_by_type['daily'].items():
            if ride_code in self.optimizer.rides_info:
                team_responsible = self.optimizer.rides_info[ride_code].get('team_responsible')
                if team_responsible == team:
                    for ppm in ppm_data['ppms']:
                        all_required_quals.add(ppm['qualification_code'])
        
        # Get weekly qualifications
        for ride_code, ppm_data in self.optimizer.ppms_by_type['weekly'].items():
            if ride_code in self.optimizer.rides_info:
                team_responsible = self.optimizer.rides_info[ride_code].get('team_responsible')
                if team_responsible == team:
                    for ppm in ppm_data['ppms']:
                        all_required_quals.add(ppm['qualification_code'])
        
        # Get monthly qualifications
        for ride_code, ppm_data in self.optimizer.ppms_by_type['monthly'].items():
            if ride_code in self.optimizer.rides_info:
                team_responsible = self.optimizer.rides_info[ride_code].get('team_responsible')
                if team_responsible == team:
                    for ppm in ppm_data['ppms']:
                        all_required_quals.add(ppm['qualification_code'])
        
        # Get qualifications currently held by team
        current_quals = set()
        for eng_data in team_current.values():
            current_quals.update(eng_data['qualifications'])
        
        # Identify gaps
        missing_quals = all_required_quals - current_quals
        
        # Categorize gaps by priority
        daily_gaps = [q for q in missing_quals if self._is_daily_qualification(q, team)]
        weekly_gaps = [q for q in missing_quals if self._is_weekly_qualification(q, team)]
        monthly_gaps = [q for q in missing_quals if self._is_monthly_qualification(q, team)]
        
        gaps = {
            'total_required': len(all_required_quals),
            'currently_covered': len(all_required_quals - missing_quals),
            'missing_qualifications': list(missing_quals),
            'daily_gaps': daily_gaps,
            'weekly_gaps': weekly_gaps,
            'monthly_gaps': monthly_gaps,
            'coverage_percentage': ((len(all_required_quals) - len(missing_quals)) / len(all_required_quals)) * 100 if all_required_quals else 100
        }
        
        print(f"      Required qualifications: {gaps['total_required']}")
        print(f"      Currently covered: {gaps['currently_covered']} ({gaps['coverage_percentage']:.1f}%)")
        print(f"      Missing daily PPM quals: {len(gaps['daily_gaps'])}")
        print(f"      Missing weekly PPM quals: {len(gaps['weekly_gaps'])}")
        print(f"      Missing monthly PPM quals: {len(gaps['monthly_gaps'])}")
        
        return gaps
    
    def _is_weekly_qualification(self, qualification, team):
        """Check if qualification is for weekly PPMs"""
        ride_code = qualification.split('.')[0]
        
        if ride_code in self.optimizer.ppms_by_type['weekly']:
            ppm_data = self.optimizer.ppms_by_type['weekly'][ride_code]
            for ppm in ppm_data['ppms']:
                if ppm['qualification_code'] == qualification:
                    return True
        return False
    
    def _is_monthly_qualification(self, qualification, team):
        """Check if qualification is for monthly PPMs"""
        ride_code = qualification.split('.')[0]

        if ride_code in self.optimizer.ppms_by_type['monthly']:
            ppm_data = self.optimizer.ppms_by_type['monthly'][ride_code]
            for ppm in ppm_data['ppms']:
                if ppm['qualification_code'] == qualification:
                    return True
        return False

    def _is_reactive_qualification(self, qualification, team):
        """Check if qualification is for reactive PPMs"""
        ride_code = qualification.split('.')[0]

        if ride_code in self.optimizer.ppms_by_type['reactive']:
            ppm_data = self.optimizer.ppms_by_type['reactive'][ride_code]
            for ppm in ppm_data['ppms']:
                if ppm['qualification_code'] == qualification:
                    return True
        return False

    def _optimize_training_for_coverage_gaps(self, team_current, coverage_gaps, team):
        """Use MILP to optimize training to fill coverage gaps"""
        print(f"   🔢 Using MILP to optimize Team {team} training for coverage gaps...")
        
        if not coverage_gaps['missing_qualifications']:
            return {'optimized_assignments': [], 'total_training_effort': 0, 'method': 'MILP'}
        
        # Create MILP problem
        prob = pulp.LpProblem(f"Team_{team}_Coverage_Gap_Training", pulp.LpMinimize)
        
        # Decision variables: train[engineer][qualification] = 1 if we train this engineer on this qual
        train_vars = {}
        total_training_effort = 0
        
        # Only consider engineers who could potentially learn the missing qualifications
        for eng_code, eng_data in team_current.items():
            train_vars[eng_code] = {}
            for qual in coverage_gaps['missing_qualifications']:
                # Check if this engineer's role matches the qualification
                qual_role = self._get_qualification_role(qual)
                if qual_role == 'any' or eng_data['role'] == qual_role:
                    var_name = f"train_{eng_code}_{qual.replace('.', '_')}"
                    train_vars[eng_code][qual] = pulp.LpVariable(var_name, cat='Binary')
                    total_training_effort += train_vars[eng_code][qual]
        
        # Objective: Minimize total training effort
        prob += total_training_effort, "Minimize_Training_Effort"
        
        # Constraints: Ensure each missing qualification is covered by at least one engineer
        for qual in coverage_gaps['missing_qualifications']:
            engineers_who_can_learn = []
            for eng_code in train_vars:
                if qual in train_vars[eng_code]:
                    engineers_who_can_learn.append(eng_code)
            
            if engineers_who_can_learn:
                coverage_sum = pulp.lpSum([
                    train_vars[eng_code][qual] 
                    for eng_code in engineers_who_can_learn
                ])
                
                # Prioritize daily qualifications (require at least 1 person)
                min_coverage = 2 if qual in coverage_gaps['daily_gaps'] else 1
                prob += coverage_sum >= min_coverage, f"Coverage_{qual.replace('.', '_')}"
        
        # Constraint: Limit training load per engineer
        for eng_code in train_vars:
            if train_vars[eng_code]:
                total_quals_for_eng = pulp.lpSum([
                    train_vars[eng_code][qual] 
                    for qual in train_vars[eng_code]
                ])
                prob += total_quals_for_eng <= 8, f"Max_Training_{eng_code}"  # Max 8 new quals per engineer
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        optimized_assignments = []
        if pulp.LpStatus[prob.status] == 'Optimal':
            for eng_code in train_vars:
                recommended_quals = []
                for qual in train_vars[eng_code]:
                    if train_vars[eng_code][qual].varValue == 1:
                        recommended_quals.append(qual)
                
                if recommended_quals:
                    eng_data = team_current[eng_code]
                    daily_impact = len([q for q in recommended_quals if q in coverage_gaps['daily_gaps']])
                    
                    optimized_assignments.append({
                        'engineer_code': eng_code,
                        'engineer_name': eng_data['name'],
                        'role': eng_data['role'],
                        'recommended_qualifications': recommended_quals,
                        'training_effort': len(recommended_quals),
                        'daily_impact': daily_impact,
                        'coverage_improvement': len(recommended_quals)
                    })
            
            print(f"      ✅ Optimal solution: {len(optimized_assignments)} engineers need training")
            total_effort = sum(a['training_effort'] for a in optimized_assignments)
            daily_coverage = sum(a['daily_impact'] for a in optimized_assignments)
            print(f"      📚 Total training effort: {total_effort} qualifications")
            print(f"      🌅 Daily PPM improvement: {daily_coverage} qualifications")
        else:
            print(f"      ⚠️  No optimal solution found, using heuristic")
            return self._optimize_training_heuristically_for_gaps(team_current, coverage_gaps, team)
        
        return {
            'optimized_assignments': optimized_assignments,
            'total_training_effort': sum(a['training_effort'] for a in optimized_assignments),
            'coverage_improvement': len(coverage_gaps['missing_qualifications']),
            'method': 'MILP'
        }
    
    def _optimize_training_heuristically_for_gaps(self, team_current, coverage_gaps, team):
        """Use heuristic optimization for coverage gaps"""
        print(f"   🧠 Using heuristic optimization for Team {team} coverage gaps...")
        
        optimized_assignments = []
        
        # Prioritize daily gaps first, then weekly, then monthly
        prioritized_gaps = (
            coverage_gaps['daily_gaps'] + 
            coverage_gaps['weekly_gaps'] + 
            coverage_gaps['monthly_gaps']
        )
        
        # Simple heuristic: assign qualifications to engineers with compatible roles
        engineer_loads = {eng_code: 0 for eng_code in team_current.keys()}
        
        for qual in prioritized_gaps:
            qual_role = self._get_qualification_role(qual)
            
            # Find engineers who can learn this qualification
            candidates = []
            for eng_code, eng_data in team_current.items():
                if qual_role == 'any' or eng_data['role'] == qual_role:
                    if engineer_loads[eng_code] < 8:  # Max 8 new quals per engineer
                        candidates.append((eng_code, engineer_loads[eng_code], eng_data))
            
            if candidates:
                # Assign to engineer with lowest current training load
                candidates.sort(key=lambda x: x[1])
                selected_eng = candidates[0]
                eng_code, current_load, eng_data = selected_eng
                
                # Add to existing assignment or create new one
                existing = next((a for a in optimized_assignments if a['engineer_code'] == eng_code), None)
                if existing:
                    existing['recommended_qualifications'].append(qual)
                    existing['training_effort'] += 1
                    if qual in coverage_gaps['daily_gaps']:
                        existing['daily_impact'] += 1
                else:
                    daily_impact = 1 if qual in coverage_gaps['daily_gaps'] else 0
                    optimized_assignments.append({
                        'engineer_code': eng_code,
                        'engineer_name': eng_data['name'],
                        'role': eng_data['role'],
                        'recommended_qualifications': [qual],
                        'training_effort': 1,
                        'daily_impact': daily_impact,
                        'coverage_improvement': 1
                    })
                
                engineer_loads[eng_code] += 1
        
        print(f"      ✅ Heuristic solution: {len(optimized_assignments)} engineers need training")
        
        return {
            'optimized_assignments': optimized_assignments,
            'total_training_effort': sum(a['training_effort'] for a in optimized_assignments),
            'coverage_improvement': len(coverage_gaps['missing_qualifications']),
            'method': 'Heuristic'
        }
    
    def _get_qualification_role(self, qualification):
        """Determine if qualification is electrical, mechanical, or either"""
        # Look for electrical patterns
        if any(pattern in qualification for pattern in ['DE', 'ME', 'WE']):
            return 'electrical'
        # Look for mechanical patterns  
        elif any(pattern in qualification for pattern in ['DM', 'MM', 'WM', '3MM', 'MMS', 'QM']):
            return 'mechanical'
        else:
            return 'any'  # Could be learned by either role

    def _apply_training_to_current_state(self, training_recommendations):
        """Apply training recommendations to current state to create projected matrices"""
        print("   🔮 Simulating post-training qualification state...")
        
        # Load current state
        current_matrices = self.load_current_qualification_state()
        
        # Apply training recommendations to create projected state
        projected_matrices = {}
        
        for team in ACTIVE_TEAMS:
            if team not in current_matrices or team not in training_recommendations:
                continue
                
            projected_matrices[team] = {}
            team_current = current_matrices[team]
            team_recommendations = training_recommendations[team]
            
            # Create lookup for training assignments
            training_lookup = {}
            for assignment in team_recommendations.get('optimized_assignments', []):
                eng_code = assignment['engineer_code']
                training_lookup[eng_code] = assignment['recommended_qualifications']
            
            # For each engineer in current state, apply training if recommended
            for eng_code, current_profile in team_current.items():
                # Start with current profile
                projected_profile = current_profile.copy()
                projected_profile['qualifications'] = current_profile['qualifications'].copy()
                projected_profile['daily_qualifications'] = current_profile['daily_qualifications'].copy()
                
                # Add training qualifications if engineer has training recommendations
                if eng_code in training_lookup:
                    new_quals = training_lookup[eng_code]
                    
                    # Add new qualifications (avoid duplicates)
                    for qual in new_quals:
                        if qual not in projected_profile['qualifications']:
                            projected_profile['qualifications'].append(qual)
                            
                            # Check if it's a daily qualification
                            if self._is_daily_qualification(qual, team):
                                projected_profile['daily_qualifications'].append(qual)
                    
                    # Recalculate ride assignments from updated qualifications
                    updated_rides = list(set([q.split('.')[0] for q in projected_profile['qualifications']]))
                    projected_profile['assigned_rides'] = updated_rides
                    
                    # Recategorize by ride type
                    projected_profile['type_a_rides'] = [r for r in updated_rides if self._get_ride_type(r) == 'A']
                    projected_profile['type_b_rides'] = [r for r in updated_rides if self._get_ride_type(r) == 'B']
                    projected_profile['type_c_rides'] = [r for r in updated_rides if self._get_ride_type(r) == 'C']
                    
                    # Update coverage score
                    projected_profile['coverage_score'] = len(projected_profile['qualifications'])
                    
                    # Mark as trained
                    projected_profile['training_applied'] = True
                    projected_profile['new_qualifications'] = new_quals
                
                projected_matrices[team][eng_code] = projected_profile
            
            trained_count = len([eng for eng in projected_matrices[team].values() if eng.get('training_applied', False)])
            print(f"      Team {team}: Applied training to {trained_count} engineers")
        
        return projected_matrices
    
    def generate_ppm_expert_analysis(self, output_dir="outputs/current"):
        """Generate PPM Expert Analysis based on operational performance (Hours Type = N)"""
        print(f"\n🎯 GENERATING PPM EXPERT ANALYSIS")
        print("=" * 60)
        print("   📊 Identifying experts based on operational performance (Hours Type = N)")
        
        import pandas as pd
        import csv
        from pathlib import Path
        from collections import defaultdict
        from datetime import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load work order data
        print("   📂 Loading work order data...")
        mech_file = Path("data/raw/MechWOFeb2023.csv")
        elec_file = Path("data/raw/ElecWOFeb2023.csv")
        
        mech_wo = pd.read_csv(mech_file) if mech_file.exists() else pd.DataFrame()
        elec_wo = pd.read_csv(elec_file) if elec_file.exists() else pd.DataFrame()
        wo_data = pd.concat([mech_wo, elec_wo], ignore_index=True)
        
        # Convert date column
        if 'Date' in wo_data.columns:
            wo_data['Date'] = pd.to_datetime(wo_data['Date'], errors='coerce')
        
        print(f"      📊 Loaded {len(wo_data):,} work orders")
        
        # Load current qualifications
        print("   📂 Loading current qualifications...")
        qual_file = Path("data/raw/EngQual.csv")
        if qual_file.exists():
            eng_qual = pd.read_csv(qual_file)
            current_qualifications = defaultdict(list)
            for _, row in eng_qual.iterrows():
                if pd.notna(row['Employee Code']) and pd.notna(row['Qualification']):
                    current_qualifications[row['Employee Code']].append(row['Qualification'])
        else:
            current_qualifications = {}
        
        print(f"      📊 Loaded qualifications for {len(current_qualifications)} engineers")
        
        # Load engineer team information
        print("   📂 Loading engineer team information...")
        engineer_details = {}
        team_files = [
            'data/processed/engineers/team1_elec_engineers.json',
            'data/processed/engineers/team1_mech_engineers.json', 
            'data/processed/engineers/team2_elec_engineers.json',
            'data/processed/engineers/team2_mech_engineers.json'
        ]
        
        for team_file in team_files:
            file_path = Path(team_file)
            if file_path.exists():
                import json
                with open(file_path) as f:
                    data = json.load(f)
                    for engineer in data.get('engineers', []):
                        emp_code = engineer.get('employee_code')
                        if emp_code and engineer.get('active', False):
                            engineer_details[emp_code] = {
                                'name': engineer.get('timeplan_name'),
                                'team': engineer.get('team'),
                                'role': engineer.get('role')
                            }
        
        print(f"      📊 Loaded details for {len(engineer_details)} engineers")
        
        # Load PPM mapping
        print("   📂 Loading PPM mappings...")
        ppm_to_qual = {}
        for ppm_type in ['daily', 'weekly', 'monthly']:
            ppm_dir = Path(f"data/raw/ppms/{ppm_type}")
            if ppm_dir.exists():
                for ppm_file in ppm_dir.glob("*.json"):
                    import json
                    with open(ppm_file) as f:
                        data = json.load(f)
                        for ppm in data.get('ppms', []):
                            ppm_code = ppm.get('ppm_code')
                            qual_code = ppm.get('qualification_code')
                            if ppm_code and qual_code:
                                ppm_to_qual[ppm_code] = {
                                    'qualification_code': qual_code,
                                    'ppm_type': ppm_type.title(),
                                    'ride_code': ppm_code.split('.')[0] if '.' in ppm_code else ppm_code
                                }
        
        print(f"      📊 Loaded {len(ppm_to_qual)} PPM mappings")

        # Filter to operational work orders (Hours Type = N)
        print("   🔍 Filtering to operational performance data...")

        # Handle empty work order data
        if wo_data.empty or 'Type' not in wo_data.columns:
            print("      ⚠️  No work order data available - skipping PPM expert analysis")
            print("      📄 PPM expert analysis requires work order history (ElecWO/MechWO CSV files)")
            return None

        operational_wo = wo_data[
            (wo_data['Type'] == 'PM') &
            (wo_data['Hours Type'] == 'N') &  # Only operational work, not training
            (wo_data['Person'].isin(engineer_details.keys())) &  # Current engineers only
            (wo_data['PM code'].isin(ppm_to_qual.keys()))  # Valid PPM codes only
        ].copy()
        
        print(f"      📊 Filtered to {len(operational_wo):,} operational work records")
        
        # Analyze expert performance by PPM
        print("   📈 Analyzing expert performance...")
        expert_analysis = defaultdict(lambda: defaultdict(lambda: {
            'sessions': 0,
            'total_hours': 0.0,
            'first_date': None,
            'last_date': None
        }))
        
        for _, row in operational_wo.iterrows():
            engineer = row['Person']
            ppm_code = row['PM code']
            hours = row.get('Hours', 0)
            date = row['Date']
            
            # Skip if invalid data
            if pd.isna(hours) or pd.isna(date):
                continue
            
            expert_data = expert_analysis[ppm_code][engineer]
            expert_data['sessions'] += 1
            expert_data['total_hours'] += hours
            
            if expert_data['first_date'] is None or date < expert_data['first_date']:
                expert_data['first_date'] = date
            if expert_data['last_date'] is None or date > expert_data['last_date']:
                expert_data['last_date'] = date
        
        # Generate expert rankings
        print("   🏆 Generating expert rankings...")
        expert_rankings = []
        
        for ppm_code, engineers in expert_analysis.items():
            ppm_info = ppm_to_qual.get(ppm_code, {})
            qual_code = ppm_info.get('qualification_code')
            
            # Get ride responsibility information
            ride_code = ppm_info.get('ride_code', '')
            team_responsible = None
            if ride_code and hasattr(self.optimizer, 'rides_info') and ride_code in self.optimizer.rides_info:
                team_responsible = self.optimizer.rides_info[ride_code].get('team_responsible')
            
            # Get engineers who are currently qualified and have operational experience
            qualified_experts = []
            for engineer, performance in engineers.items():
                # Must be currently qualified - use EXACT qualification mapping from PPM file
                engineer_quals = current_qualifications.get(engineer, [])
                # CRITICAL BUG FIX: Ensure EXACT qualification match
                if qual_code and qual_code in engineer_quals:
                    eng_details = engineer_details.get(engineer, {})
                    eng_team = eng_details.get('team')
                    
                    # CRITICAL FIX: Only include engineers from the team responsible for this ride
                    if team_responsible is not None and eng_team != team_responsible:
                        continue  # Skip engineers from wrong team
                    
                    qualified_experts.append({
                        'ppm_code': ppm_code,
                        'qualification_code': qual_code,
                        'ride_code': ppm_info.get('ride_code', ''),
                        'ppm_type': ppm_info.get('ppm_type', ''),
                        'engineer_code': engineer,
                        'engineer_name': eng_details.get('name', engineer),
                        'team': eng_details.get('team', ''),
                        'role': eng_details.get('role', '').title(),
                        'current_qualified': True,
                        'operational_sessions': performance['sessions'],
                        'operational_hours': performance['total_hours'],
                        'avg_hours_per_session': performance['total_hours'] / performance['sessions'] if performance['sessions'] > 0 else 0,
                        'first_performed_date': performance['first_date'].strftime('%Y-%m-%d') if performance['first_date'] else '',
                        'last_performed_date': performance['last_date'].strftime('%Y-%m-%d') if performance['last_date'] else '',
                    })
            
            # Sort by operational experience (sessions first, then hours)
            qualified_experts.sort(key=lambda x: (x['operational_sessions'], x['operational_hours']), reverse=True)
            
            # Add ranking and experience level
            for rank, expert in enumerate(qualified_experts, 1):
                expert['rank_on_ppm'] = rank
                
                # Determine experience level
                sessions = expert['operational_sessions']
                hours = expert['operational_hours']
                
                if sessions >= 20 and hours >= 30:
                    expert['experience_level'] = 'Expert'
                elif sessions >= 10 and hours >= 15:
                    expert['experience_level'] = 'Experienced'
                elif sessions >= 5 and hours >= 5:
                    expert['experience_level'] = 'Competent'
                else:
                    expert['experience_level'] = 'Novice'
                
                expert_rankings.append(expert)
        
        # Export to CSV
        expert_file = output_path / "ppm_expert_analysis.csv"
        print(f"   📄 Creating expert analysis: {expert_file}")
        
        with open(expert_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'PPM_Code', 'Qualification_Code', 'Ride_Code', 'PPM_Type',
                'Expert_Engineer', 'Engineer_Name', 'Team', 'Role',
                'Current_Qualified', 'Operational_Sessions', 'Operational_Hours',
                'Avg_Hours_Per_Session', 'First_Performed_Date', 'Last_Performed_Date',
                'Experience_Level', 'Rank_On_PPM'
            ])
            
            # Data rows
            for expert in expert_rankings:
                writer.writerow([
                    expert['ppm_code'],
                    expert['qualification_code'],
                    expert['ride_code'],
                    expert['ppm_type'],
                    expert['engineer_code'],
                    expert['engineer_name'],
                    expert['team'],
                    expert['role'],
                    expert['current_qualified'],
                    expert['operational_sessions'],
                    f"{expert['operational_hours']:.2f}",
                    f"{expert['avg_hours_per_session']:.2f}",
                    expert['first_performed_date'],
                    expert['last_performed_date'],
                    expert['experience_level'],
                    expert['rank_on_ppm']
                ])
        
        # Generate summary statistics
        print(f"\n   📊 EXPERT ANALYSIS SUMMARY:")
        total_ppms = len(set(expert['ppm_code'] for expert in expert_rankings))
        total_experts = len(set(expert['engineer_code'] for expert in expert_rankings))
        experts_by_level = defaultdict(int)
        for expert in expert_rankings:
            experts_by_level[expert['experience_level']] += 1
        
        print(f"      PPMs with qualified experts: {total_ppms}")
        print(f"      Total qualified experts: {total_experts}")
        print(f"      Experience distribution:")
        for level, count in experts_by_level.items():
            print(f"         {level}: {count}")
        
        # Find PPMs with limited expertise
        ppm_expert_counts = defaultdict(int)
        for expert in expert_rankings:
            ppm_expert_counts[expert['ppm_code']] += 1
        
        single_expert_ppms = [ppm for ppm, count in ppm_expert_counts.items() if count == 1]
        if single_expert_ppms:
            print(f"      ⚠️  PPMs with only 1 expert: {len(single_expert_ppms)} (knowledge risk)")
        
        print(f"   ✅ Expert analysis exported to: {expert_file}")
        
        # Generate risk analysis CSV
        risk_file = self._generate_risk_analysis_csv(expert_rankings, output_path)
        
        return expert_file
    
    def _generate_risk_analysis_csv(self, expert_rankings, output_path):
        """Generate ride risk analysis based on expert coverage"""
        import csv
        from collections import defaultdict
        
        print(f"   📊 Generating ride risk analysis...")
        
        # Analyze expert coverage by PPM (not ride) - business criticality matters
        ppm_coverage = defaultdict(lambda: {
            'total_experts': 0,
            'expert_level': 0,
            'experienced_level': 0, 
            'competent_level': 0,
            'novice_level': 0,
            'total_sessions': 0,
            'total_hours': 0,
            'top_expert': None,
            'ppm_type': '',
            'ride_code': '',
            'qualification_code': ''
        })
        
        # Aggregate data by PPM
        for expert in expert_rankings:
            ppm_code = expert['ppm_code']
            level = expert['experience_level']
            
            coverage = ppm_coverage[ppm_code]
            coverage['ppm_type'] = expert['ppm_type']
            coverage['ride_code'] = expert['ride_code']
            coverage['qualification_code'] = expert['qualification_code']
            coverage['total_experts'] += 1
            coverage['total_sessions'] += expert['operational_sessions']
            coverage['total_hours'] += expert['operational_hours']
            
            # Count by experience level
            if level == 'Expert':
                coverage['expert_level'] += 1
            elif level == 'Experienced':
                coverage['experienced_level'] += 1
            elif level == 'Competent':
                coverage['competent_level'] += 1
            else:
                coverage['novice_level'] += 1
            
            # Track top expert (rank 1)
            if expert['rank_on_ppm'] == 1:
                if not coverage['top_expert'] or expert['operational_sessions'] > coverage['top_expert']['sessions']:
                    coverage['top_expert'] = {
                        'name': expert['engineer_name'],
                        'code': expert['engineer_code'],
                        'sessions': expert['operational_sessions'],
                        'hours': expert['operational_hours']
                    }
        
        # Calculate risk scores and classifications based on PPM criticality
        risk_analysis = []
        for ppm_code, coverage in ppm_coverage.items():
            # Risk scoring based on PPM type criticality
            expert_count = coverage['expert_level']
            experienced_count = coverage['experienced_level']
            total_qualified = coverage['total_experts']
            ppm_type = coverage['ppm_type']
            
            # Base risk classification by total qualified engineers (not just experts)
            # Critical = very few people can do this PPM on the responsible team
            if total_qualified <= 1:
                base_risk_level = 'CRITICAL'
                base_risk_score = 100
            elif total_qualified <= 2:
                base_risk_level = 'HIGH'
                base_risk_score = 85
            elif total_qualified <= 4:
                base_risk_level = 'MEDIUM'
                base_risk_score = 65
            elif total_qualified <= 7:
                base_risk_level = 'LOW'
                base_risk_score = 45
            else:
                base_risk_level = 'MINIMAL'
                base_risk_score = 25
            
            # PPM Type Risk Multiplier (business criticality)
            if ppm_type == 'Daily':
                risk_multiplier = 1.0  # Highest priority
                criticality = 'DAILY'
            elif ppm_type == 'Weekly':
                risk_multiplier = 0.8  # Medium priority
                criticality = 'WEEKLY'
            elif ppm_type == 'Monthly':
                risk_multiplier = 0.6  # Lower priority
                criticality = 'MONTHLY'
            else:
                risk_multiplier = 1.0
                criticality = 'UNKNOWN'
            
            # Final risk score with PPM type weighting
            final_risk_score = int(base_risk_score * risk_multiplier)
            
            # Adjust risk level based on final score and ensure logic consistency
            if final_risk_score >= 85:
                final_risk_level = 'CRITICAL'
            elif final_risk_score >= 65:
                final_risk_level = 'HIGH'  
            elif final_risk_score >= 45:
                final_risk_level = 'MEDIUM'
            elif final_risk_score >= 25:
                final_risk_level = 'LOW'
            else:
                final_risk_level = 'MINIMAL'
            
            # Single point of failure check (1 or fewer qualified engineers)
            single_point_failure = total_qualified <= 1
            if single_point_failure:
                final_risk_level = 'CRITICAL'
                final_risk_score = 100
            
            # Boost risk for daily PPMs with very few qualified people
            if ppm_type == 'Daily' and total_qualified <= 2:
                if final_risk_score < 85:
                    final_risk_score = 85
                    final_risk_level = 'HIGH'
            
            # Adjust for experienced engineers backup
            if experienced_count >= 2 and not single_point_failure:
                final_risk_score = max(25, final_risk_score - 10)  # Don't go below minimal
            
            risk_analysis.append({
                'ppm_code': ppm_code,
                'qualification_code': coverage['qualification_code'],
                'ride_code': coverage['ride_code'],
                'ppm_type': ppm_type,
                'criticality': criticality,
                'risk_level': final_risk_level,
                'risk_score': final_risk_score,
                'total_experts': total_qualified,
                'expert_level_count': expert_count,
                'experienced_level_count': experienced_count,
                'competent_level_count': coverage['competent_level'],
                'novice_level_count': coverage['novice_level'],
                'single_point_failure': single_point_failure,
                'total_operational_sessions': coverage['total_sessions'],
                'total_operational_hours': f"{coverage['total_hours']:.2f}",
                'top_expert_name': coverage['top_expert']['name'] if coverage['top_expert'] else '',
                'top_expert_code': coverage['top_expert']['code'] if coverage['top_expert'] else '',
                'top_expert_sessions': coverage['top_expert']['sessions'] if coverage['top_expert'] else 0,
                'top_expert_hours': f"{coverage['top_expert']['hours']:.2f}" if coverage['top_expert'] else '0.00'
            })
        
        # Sort by risk (highest first)
        risk_analysis.sort(key=lambda x: (x['risk_score'], -x['total_experts']), reverse=True)
        
        # Export risk analysis CSV
        risk_file = output_path / "ppm_risk_analysis.csv"
        print(f"   📄 Creating PPM risk analysis: {risk_file}")
        
        with open(risk_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'PPM_Code', 'Qualification_Code', 'Ride_Code', 'PPM_Type', 'Criticality',
                'Risk_Level', 'Risk_Score', 'Total_Qualified',
                'Expert_Level_Count', 'Experienced_Level_Count', 'Competent_Level_Count', 'Novice_Level_Count',
                'Single_Point_Failure', 'Total_Operational_Sessions', 'Total_Operational_Hours',
                'Top_Expert_Name', 'Top_Expert_Code', 'Top_Expert_Sessions', 'Top_Expert_Hours'
            ])
            
            # Data rows
            for risk in risk_analysis:
                writer.writerow([
                    risk['ppm_code'],
                    risk['qualification_code'],
                    risk['ride_code'],
                    risk['ppm_type'],
                    risk['criticality'],
                    risk['risk_level'],
                    risk['risk_score'],
                    risk['total_experts'],
                    risk['expert_level_count'],
                    risk['experienced_level_count'],
                    risk['competent_level_count'],
                    risk['novice_level_count'],
                    risk['single_point_failure'],
                    risk['total_operational_sessions'],
                    risk['total_operational_hours'],
                    risk['top_expert_name'],
                    risk['top_expert_code'],
                    risk['top_expert_sessions'],
                    risk['top_expert_hours']
                ])
        
        # Display risk summary
        print(f"\n   ⚠️  PPM RISK ANALYSIS SUMMARY:")
        critical_ppms = [r for r in risk_analysis if r['risk_level'] == 'CRITICAL']
        high_risk_ppms = [r for r in risk_analysis if r['risk_level'] == 'HIGH']
        spf_ppms = [r for r in risk_analysis if r['single_point_failure']]
        daily_critical = [r for r in critical_ppms if r['ppm_type'] == 'Daily']
        
        print(f"      Critical risk PPMs: {len(critical_ppms)} (Daily: {len(daily_critical)})")
        print(f"      High risk PPMs: {len(high_risk_ppms)}")
        print(f"      Single point failures: {len(spf_ppms)}")
        
        if daily_critical:
            print(f"      🚨 Most critical DAILY PPMs:")
            for ppm in daily_critical[:3]:
                expert_info = f" (Expert: {ppm['top_expert_name']})" if ppm['top_expert_name'] else " (No experts!)"
                print(f"         {ppm['ppm_code']} ({ppm['ride_code']}): {ppm['total_experts']} experts{expert_info}")
        
        print(f"   ✅ PPM risk analysis exported to: {risk_file}")
        return risk_file


def main():
    """Run training optimization"""
    print("Training Optimization Designer")
    print("Requires PPM data and current qualifications to be loaded first")


if __name__ == "__main__":
    main() 