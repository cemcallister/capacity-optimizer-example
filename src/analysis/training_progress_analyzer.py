#!/usr/bin/env python3

"""
Training Progress Analyzer
==========================

Production-ready analyzer for ongoing training patterns and MILP bias calculation.
Integrates training history analysis with qualification optimization to preserve
sunk cost investments and align with real-world training decisions.

Features:
- Current engineer filtering (only active employees)
- PPM-to-qualification mapping from optimization files
- Training timeline analysis (start/end dates, progression)
- Sophisticated bias weight calculation
- MILP integration-ready outputs

Usage:
    analyzer = TrainingProgressAnalyzer()
    bias_data = analyzer.analyze_ongoing_training()
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingProgressAnalyzer:
    """Analyzes training progress and calculates MILP bias weights"""
    
    def __init__(self, data_dir="data"):
        """Initialize analyzer with data directory"""
        self.data_dir = Path(data_dir)
        
        # Core data
        self.current_engineers = set()
        self.engineer_details = {}
        self.ppm_to_qual = {}
        self.qual_to_ppm = defaultdict(list)
        self.current_qualifications = {}
        self.wo_data = None
        self.ride_info = {}  # Ride responsibility data
        
        # Analysis results
        self.ongoing_training = defaultdict(lambda: defaultdict(list))
        self.completed_training = []
        self.bias_recommendations = []
        
        logger.info("Training Progress Analyzer initialized")
    
    def load_all_data(self):
        """Load all required data sources"""
        logger.info("Loading all data sources...")
        
        self._load_current_engineers()
        self._load_ride_info()
        self._load_ppm_mapping()
        self._load_work_order_data()
        self._load_current_qualifications()
        
        logger.info("All data loaded successfully")
    
    def _load_current_engineers(self):
        """Load current active engineers from team files"""
        logger.info("Loading current engineers...")
        
        team_files = [
            'data/processed/engineers/team1_elec_engineers.json',
            'data/processed/engineers/team1_mech_engineers.json', 
            'data/processed/engineers/team2_elec_engineers.json',
            'data/processed/engineers/team2_mech_engineers.json'
        ]
        
        for team_file in team_files:
            file_path = self.data_dir / team_file.replace('data/', '')
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    for engineer in data.get('engineers', []):
                        emp_code = engineer.get('employee_code')
                        # Only include active engineers, exclude vacancies
                        if (emp_code and 
                            engineer.get('active', False) and 
                            not emp_code.startswith('VACANCY') and
                            not emp_code.startswith('VACENC')):
                            self.current_engineers.add(emp_code)
                            self.engineer_details[emp_code] = {
                                'name': engineer.get('timeplan_name'),
                                'team': engineer.get('team'),
                                'role': engineer.get('role'),
                                'rota_number': engineer.get('rota_number')
                            }
        
        logger.info(f"Loaded {len(self.current_engineers)} current engineers")
    
    def _load_ride_info(self):
        """Load ride information including team responsibilities"""
        logger.info("Loading ride information...")
        
        ride_info_file = self.data_dir / "processed/ride_info.json"
        if ride_info_file.exists():
            with open(ride_info_file) as f:
                data = json.load(f)
                self.ride_info = data.get('rides', {})
        
        logger.info(f"Loaded info for {len(self.ride_info)} rides")
    
    def _load_ppm_mapping(self):
        """Load PPM code to qualification code mapping"""
        logger.info("Loading PPM mappings...")
        
        # Load all PPM files from daily, weekly, monthly
        for ppm_type in ['daily', 'weekly', 'monthly']:
            ppm_dir = self.data_dir / f"raw/ppms/{ppm_type}"
            
            if ppm_dir.exists():
                for ppm_file in ppm_dir.glob("*.json"):
                    with open(ppm_file) as f:
                        data = json.load(f)
                        
                        for ppm in data.get('ppms', []):
                            ppm_code = ppm.get('ppm_code')
                            qual_code = ppm.get('qualification_code')
                            
                            if ppm_code and qual_code:
                                self.ppm_to_qual[ppm_code] = qual_code
                                self.qual_to_ppm[qual_code].append(ppm_code)
        
        logger.info(f"Loaded {len(self.ppm_to_qual)} PPM mappings")
    
    def _load_work_order_data(self):
        """Load work order data from CSV files"""
        logger.info("Loading work order data...")
        
        # Load WO files
        mech_file = self.data_dir / "raw/MechWOFeb2023.csv"
        elec_file = self.data_dir / "raw/ElecWOFeb2023.csv"
        training_file = self.data_dir / "raw/Training.csv"

        mech_wo = pd.read_csv(mech_file) if mech_file.exists() else pd.DataFrame()
        elec_wo = pd.read_csv(elec_file) if elec_file.exists() else pd.DataFrame()
        training_wo = pd.read_csv(training_file) if training_file.exists() else pd.DataFrame()

        # Combine datasets: include Training.csv so Hours Type == 'T' rows are present
        self.wo_data = pd.concat([mech_wo, elec_wo, training_wo], ignore_index=True)
        
        # Convert date column to datetime for timeline analysis
        if 'Date' in self.wo_data.columns:
            self.wo_data['Date'] = pd.to_datetime(self.wo_data['Date'], errors='coerce')
        
        logger.info(f"Loaded {len(self.wo_data):,} work orders")
    
    def _load_current_qualifications(self):
        """Load current engineer qualifications"""
        logger.info("Loading current qualifications...")
        
        qual_file = self.data_dir / "raw/EngQual.csv"
        if qual_file.exists():
            eng_qual = pd.read_csv(qual_file)
            
            # Process into dictionary format
            quals = defaultdict(list)
            for _, row in eng_qual.iterrows():
                if pd.notna(row['Employee Code']) and pd.notna(row['Qualification']):
                    quals[row['Employee Code']].append(row['Qualification'])
            
            # Filter to current engineers only
            self.current_qualifications = {
                emp: qual_list for emp, qual_list in quals.items() 
                if emp in self.current_engineers
            }
        
        logger.info(f"Loaded qualifications for {len(self.current_qualifications)} current engineers")
    
    def analyze_ongoing_training(self):
        """Main analysis function - returns bias data for MILP"""
        logger.info("Starting ongoing training analysis...")
        
        # Load all data
        self.load_all_data()
        
        # Filter to relevant training data
        training_data = self._filter_relevant_training()
        
        # Separate ongoing vs completed training
        self._categorize_training(training_data)
        
        # Calculate bias weights with timeline analysis
        self._calculate_bias_weights_with_timeline()
        
        logger.info(f"Analysis complete: {len(self.bias_recommendations)} bias recommendations")
        
        return {
            'bias_recommendations': self.bias_recommendations,
            'ongoing_training_summary': dict(self.ongoing_training),
            'completed_training_count': len(self.completed_training),
            'engineer_details': self.engineer_details
        }
    
    def _filter_relevant_training(self):
        """Filter work orders to relevant training data"""
        logger.info("Filtering to relevant training data...")
        
        if self.wo_data is None or self.wo_data.empty:
            logger.warning("No work order data available")
            return pd.DataFrame()
        
        # Filter to training work by current engineers on relevant PPMs
        training_data = self.wo_data[
            (self.wo_data['Type'] == 'PM') & 
            (self.wo_data['Hours Type'] == 'T') &
            (self.wo_data['Person'].isin(self.current_engineers)) &
            (self.wo_data['PM code'].isin(self.ppm_to_qual.keys()))
        ].copy()
        
        logger.info(f"Filtered to {len(training_data):,} relevant training records")
        return training_data
    
    def _categorize_training(self, training_data):
        """Separate ongoing training from completed training"""
        logger.info("Categorizing ongoing vs completed training (with team responsibility check)...")
        
        ongoing_count = 0
        completed_count = 0
        team_mismatch_count = 0
        
        for _, row in training_data.iterrows():
            engineer = row['Person']
            ppm_code = row['PM code']
            
            # Get required qualification
            required_qual = self.ppm_to_qual.get(ppm_code)
            if not required_qual:
                continue
            
            # TEAM RESPONSIBILITY CHECK
            # Extract ride code from PPM code (e.g., APEX.E.1D.S.02 -> APEX)
            ride_code = ppm_code.split('.')[0] if '.' in str(ppm_code) else str(ppm_code)
            
            # Get engineer's current team
            engineer_team = self.engineer_details.get(engineer, {}).get('team')
            
            # Get ride's responsible team
            ride_responsible_team = self.ride_info.get(ride_code, {}).get('team_responsible')
            
            # Skip if engineer's team doesn't match ride's responsible team
            if engineer_team and ride_responsible_team and engineer_team != ride_responsible_team:
                team_mismatch_count += 1
                continue  # Skip this training - not relevant to current team
            
            # Check if engineer has qualification
            engineer_quals = self.current_qualifications.get(engineer, [])
            has_qualification = required_qual in engineer_quals
            
            training_record = {
                'engineer': engineer,
                'ppm_code': ppm_code,
                'required_qual': required_qual,
                'date': row['Date'],
                'hours': row['Hours'],
                'wo_number': row.get('WO', ''),
                'description': row.get('WO Description', ''),
                'has_qualification': has_qualification
            }
            
            if has_qualification:
                self.completed_training.append(training_record)
                completed_count += 1
            else:
                # Group by qualification (not PPM) for bias calculation
                self.ongoing_training[engineer][required_qual].append(training_record)
                ongoing_count += 1
        
        logger.info(f"Categorized: {ongoing_count:,} ongoing, {completed_count:,} completed")
        logger.info(f"Filtered out: {team_mismatch_count:,} team mismatches (engineers moved teams)")
    
    def _calculate_bias_weights_with_timeline(self):
        """Calculate bias weights including timeline analysis"""
        logger.info("Calculating bias weights with timeline analysis...")
        
        for engineer, qualifications in self.ongoing_training.items():
            engineer_info = self.engineer_details.get(engineer, {})
            
            for required_qual, sessions in qualifications.items():
                # Basic statistics
                session_count = len(sessions)
                total_hours = sum(s['hours'] for s in sessions if pd.notna(s['hours']))
                
                # Timeline analysis
                timeline_data = self._analyze_training_timeline(sessions)
                
                # Calculate bias weight
                bias_weight = self._calculate_individual_bias_weight(
                    session_count, total_hours, timeline_data
                )
                
                # Collect PPM codes for this qualification
                pmp_codes = list(set(s['ppm_code'] for s in sessions))
                
                recommendation = {
                    'engineer': engineer,
                    'engineer_team': engineer_info.get('team'),
                    'engineer_role': engineer_info.get('role'),
                    'engineer_name': engineer_info.get('name'),
                    'required_qual': required_qual,
                    'ppm_codes': pmp_codes,
                    'sessions': session_count,
                    'total_hours': total_hours,
                    'bias_weight': bias_weight,
                    'training_start_date': timeline_data['start_date'],
                    'last_training_date': timeline_data['end_date'],
                    'training_duration_days': timeline_data['duration_days'],
                    'training_frequency': timeline_data['frequency_per_month'],
                    'priority': self._determine_priority(bias_weight, timeline_data),
                    'urgency': self._determine_urgency(timeline_data),
                    'reasoning': self._generate_reasoning(session_count, total_hours, timeline_data)
                }
                
                self.bias_recommendations.append(recommendation)
        
        # Sort by bias weight and urgency
        self.bias_recommendations.sort(
            key=lambda x: (x['bias_weight'], x['urgency'] == 'HIGH', x['sessions']), 
            reverse=True
        )
    
    def _analyze_training_timeline(self, sessions):
        """Analyze training timeline for a specific qualification"""
        # Filter out sessions with invalid dates
        valid_sessions = [s for s in sessions if pd.notna(s['date'])]
        
        if not valid_sessions:
            return {
                'start_date': None,
                'end_date': None,
                'duration_days': 0,
                'frequency_per_month': 0,
                'recent_activity': False
            }
        
        # Sort by date
        valid_sessions.sort(key=lambda x: x['date'])
        
        start_date = valid_sessions[0]['date']
        end_date = valid_sessions[-1]['date']
        duration_days = (end_date - start_date).days if start_date != end_date else 0
        
        # Calculate frequency (sessions per month)
        if duration_days > 0:
            frequency_per_month = (len(valid_sessions) * 30) / duration_days
        else:
            frequency_per_month = len(valid_sessions)  # All on same day
        
        # Check if training is recent (within last 90 days)
        recent_activity = (datetime.now() - end_date.replace(tzinfo=None)).days <= 90
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'frequency_per_month': frequency_per_month,
            'recent_activity': recent_activity
        }
    
    def _calculate_individual_bias_weight(self, session_count, total_hours, timeline_data):
        """Calculate bias weight for individual qualification"""
        # Base bias for ongoing training (sunk cost principle)
        base_bias = 2.0
        
        # Investment level bias
        investment_bias = 0.0
        if session_count >= 10 or total_hours >= 15:
            investment_bias = 1.0  # High investment
        elif session_count >= 5 or total_hours >= 8:
            investment_bias = 0.5  # Medium investment
        elif session_count >= 2 or total_hours >= 3:
            investment_bias = 0.3  # Some investment
        
        # Timeline bias (recent and frequent training gets higher priority)
        timeline_bias = 0.0
        if timeline_data['recent_activity']:
            timeline_bias += 0.2
        if timeline_data['frequency_per_month'] >= 4:  # Weekly or more
            timeline_bias += 0.3
        elif timeline_data['frequency_per_month'] >= 2:  # Bi-weekly
            timeline_bias += 0.2
        
        # Cap total bias at 3.0
        return min(base_bias + investment_bias + timeline_bias, 3.0)
    
    def _determine_priority(self, bias_weight, timeline_data):
        """Determine priority level based on bias weight and timeline"""
        if bias_weight >= 2.8:
            return "HIGH"
        elif bias_weight >= 2.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_urgency(self, timeline_data):
        """Determine urgency based on training timeline"""
        if not timeline_data['recent_activity']:
            return "HIGH"  # Training has stalled
        elif timeline_data['frequency_per_month'] >= 4:
            return "HIGH"  # Very active training
        elif timeline_data['frequency_per_month'] >= 2:
            return "MEDIUM"  # Regular training
        else:
            return "LOW"  # Infrequent training
    
    def _generate_reasoning(self, session_count, total_hours, timeline_data):
        """Generate human-readable reasoning for bias recommendation"""
        reasons = []
        
        reasons.append(f"{session_count} training sessions")
        reasons.append(f"{total_hours:.1f} hours invested")
        
        if timeline_data['start_date'] and timeline_data['end_date']:
            if timeline_data['duration_days'] > 0:
                reasons.append(f"training over {timeline_data['duration_days']} days")
            
            if timeline_data['recent_activity']:
                reasons.append("recent activity")
            else:
                days_since = (datetime.now() - timeline_data['end_date'].replace(tzinfo=None)).days
                reasons.append(f"last training {days_since} days ago")
        
        return "Ongoing training: " + ", ".join(reasons)
    
    def generate_training_report(self):
        """Generate comprehensive training progress report"""
        if not self.bias_recommendations:
            logger.warning("No bias recommendations available. Run analyze_ongoing_training() first.")
            return {}
        
        # Summary statistics
        total_engineers = len(self.ongoing_training)
        total_recommendations = len(self.bias_recommendations)
        high_priority = len([r for r in self.bias_recommendations if r['priority'] == 'HIGH'])
        
        # Team breakdown
        team_breakdown = defaultdict(int)
        for rec in self.bias_recommendations:
            team = rec['engineer_team']
            if team:
                team_breakdown[team] += 1
        
        # Priority breakdown
        priority_breakdown = Counter(rec['priority'] for rec in self.bias_recommendations)
        urgency_breakdown = Counter(rec['urgency'] for rec in self.bias_recommendations)
        
        report = {
            'summary': {
                'total_engineers_with_ongoing_training': total_engineers,
                'total_bias_recommendations': total_recommendations,
                'high_priority_cases': high_priority,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'team_breakdown': dict(team_breakdown),
            'priority_breakdown': dict(priority_breakdown),
            'urgency_breakdown': dict(urgency_breakdown),
            'top_recommendations': self.bias_recommendations[:20],  # Top 20 for detailed view
            'milp_integration': {
                'normal_training_weight': 1.0,
                'ongoing_training_weight_range': '2.0x - 3.0x',
                'total_bias_cases': total_recommendations,
                'implementation_ready': True
            }
        }
        
        return report
    
    def export_bias_weights_for_milp(self):
        """Export bias weights in format ready for MILP integration"""
        milp_bias_data = {}
        
        for rec in self.bias_recommendations:
            engineer = rec['engineer']
            required_qual = rec['required_qual']
            bias_weight = rec['bias_weight']
            
            if engineer not in milp_bias_data:
                milp_bias_data[engineer] = {}
            
            milp_bias_data[engineer][required_qual] = {
                'weight': bias_weight,
                'priority': rec['priority'],
                'sessions': rec['sessions'],
                'hours': rec['total_hours'],
                'reasoning': rec['reasoning']
            }
        
        return milp_bias_data


def main():
    """Test the analyzer with current data"""
    print("🎯 TRAINING PROGRESS ANALYZER - PRODUCTION TEST")
    print("=" * 60)
    
    try:
        # Initialize and run analysis
        analyzer = TrainingProgressAnalyzer()
        results = analyzer.analyze_ongoing_training()
        
        # Generate comprehensive report
        report = analyzer.generate_training_report()
        
        # Display key results
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Engineers with ongoing training: {report['summary']['total_engineers_with_ongoing_training']}")
        print(f"   Total bias recommendations: {report['summary']['total_bias_recommendations']}")
        print(f"   High priority cases: {report['summary']['high_priority_cases']}")
        
        print(f"\n🏢 TEAM BREAKDOWN:")
        for team, count in report['team_breakdown'].items():
            print(f"   Team {team}: {count} recommendations")
        
        print(f"\n🎯 TOP 5 RECOMMENDATIONS:")
        for i, rec in enumerate(report['top_recommendations'][:5]):
            print(f"\n   {i+1}. {rec['engineer']} → {rec['required_qual']}")
            print(f"      Training: {rec['sessions']} sessions, {rec['total_hours']:.1f} hours")
            print(f"      Timeline: {rec['training_start_date'].strftime('%Y-%m-%d') if rec['training_start_date'] else 'N/A'} to {rec['last_training_date'].strftime('%Y-%m-%d') if rec['last_training_date'] else 'N/A'}")
            print(f"      Bias weight: {rec['bias_weight']:.1f}x ({rec['priority']} priority, {rec['urgency']} urgency)")
        
        # Export MILP bias data
        milp_data = analyzer.export_bias_weights_for_milp()
        print(f"\n✅ MILP bias data ready for {len(milp_data)} engineers")
        
        print(f"\n🎉 TRAINING PROGRESS ANALYZER READY FOR INTEGRATION!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()