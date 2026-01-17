#!/usr/bin/env python3
"""
Analyze time to attain qualifications from training start to operational status
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict

class QualificationTimeAnalyzer:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load work order and qualification data"""
        print("Loading work order data...")
        
        # Load work orders
        mech_wo = pd.read_csv("data/raw/MechWOFeb2023.csv")
        elec_wo = pd.read_csv("data/raw/ElecWOFeb2023.csv")
        self.wo_data = pd.concat([mech_wo, elec_wo], ignore_index=True)
        
        # Convert date
        self.wo_data['Date'] = pd.to_datetime(self.wo_data['Date'], errors='coerce')
        
        # Load current qualifications
        self.eng_qual = pd.read_csv("data/raw/EngQual.csv")
        
        # Load PPM mappings
        self.ppm_to_qual = self._load_ppm_mappings()
        
        print(f"Loaded {len(self.wo_data):,} work orders")
        print(f"Loaded {len(self.ppm_to_qual)} PPM mappings")
        
    def _load_ppm_mappings(self):
        """Load PPM to qualification mappings"""
        ppm_to_qual = {}
        
        for ppm_type in ['daily', 'weekly', 'monthly']:
            ppm_dir = Path(f"data/raw/ppms/{ppm_type}")
            if ppm_dir.exists():
                for ppm_file in ppm_dir.glob("*.json"):
                    with open(ppm_file) as f:
                        data = json.load(f)
                        for ppm in data.get('ppms', []):
                            if ppm.get('ppm_code') and ppm.get('qualification_code'):
                                ppm_to_qual[ppm['ppm_code']] = ppm['qualification_code']
        
        return ppm_to_qual
    
    def analyze_qualification_times(self):
        """Find time from first training to qualification achievement"""
        
        # Get current qualifications per engineer
        current_quals = defaultdict(set)
        for _, row in self.eng_qual.iterrows():
            if pd.notna(row['Employee Code']) and pd.notna(row['Qualification']):
                current_quals[row['Employee Code']].add(row['Qualification'])
        
        # Filter to PM work orders with valid PPM codes
        pm_orders = self.wo_data[
            (self.wo_data['Type'] == 'PM') & 
            (self.wo_data['PM code'].isin(self.ppm_to_qual.keys())) &
            (pd.notna(self.wo_data['Date']))
        ].copy()
        
        # Add qualification code
        pm_orders['qualification_code'] = pm_orders['PM code'].map(self.ppm_to_qual)
        
        # Sort by engineer, qualification, date
        pm_orders = pm_orders.sort_values(['Person', 'qualification_code', 'Date'])
        
        training_sequences = []
        
        for (engineer, qual_code), group in pm_orders.groupby(['Person', 'qualification_code']):
            # Skip if engineer doesn't currently have this qualification
            if qual_code not in current_quals.get(engineer, set()):
                continue
                
            # Find training sequence
            sequence = self._find_training_sequence(group, engineer, qual_code)
            if sequence:
                training_sequences.append(sequence)
        
        return pd.DataFrame(training_sequences)
    
    def _find_training_sequence(self, group, engineer, qual_code):
        """Find the training sequence that led to qualification"""
        
        # Separate training (T) and operational (N) records
        training_records = group[group['Hours Type'] == 'T'].copy()
        operational_records = group[group['Hours Type'] == 'N'].copy()
        
        if len(training_records) == 0 or len(operational_records) == 0:
            return None
            
        # Find first operational date
        first_operational_date = operational_records['Date'].min()
        
        # Find all training before first operational
        relevant_training = training_records[training_records['Date'] < first_operational_date]
        
        if len(relevant_training) == 0:
            return None
            
        # Calculate training sequence
        first_training_date = relevant_training['Date'].min()
        last_training_date = relevant_training['Date'].max()
        
        # Calculate metrics
        days_to_qualify = (first_operational_date - first_training_date).days
        training_sessions = len(relevant_training)
        total_training_hours = relevant_training['Hours'].sum()
        
        # Find gaps in training
        training_dates = relevant_training['Date'].sort_values()
        gaps = []
        for i in range(1, len(training_dates)):
            gap_days = (training_dates.iloc[i] - training_dates.iloc[i-1]).days
            if gap_days > 30:  # Significant gap
                gaps.append(gap_days)
        
        return {
            'engineer_code': engineer,
            'qualification_code': qual_code,
            'first_training_date': first_training_date,
            'last_training_date': last_training_date,
            'first_operational_date': first_operational_date,
            'days_to_qualify': days_to_qualify,
            'training_sessions': training_sessions,
            'total_training_hours': total_training_hours,
            'avg_hours_per_session': total_training_hours / training_sessions if training_sessions > 0 else 0,
            'training_gaps': len(gaps),
            'max_gap_days': max(gaps) if gaps else 0,
            'continuous_training': len(gaps) == 0
        }
    
    def generate_summary_statistics(self, training_data):
        """Generate summary statistics by qualification"""
        
        summary = []
        
        for qual_code, group in training_data.groupby('qualification_code'):
            stats = {
                'qualification_code': qual_code,
                'engineers_qualified': len(group),
                'avg_days_to_qualify': group['days_to_qualify'].mean(),
                'median_days_to_qualify': group['days_to_qualify'].median(),
                'std_days_to_qualify': group['days_to_qualify'].std(),
                'min_days_to_qualify': group['days_to_qualify'].min(),
                'max_days_to_qualify': group['days_to_qualify'].max(),
                'avg_training_sessions': group['training_sessions'].mean(),
                'avg_training_hours': group['total_training_hours'].mean(),
                'continuous_training_pct': (group['continuous_training'].sum() / len(group)) * 100,
                'avg_gaps_in_training': group['training_gaps'].mean()
            }
            
            # Categorize by time
            if stats['median_days_to_qualify'] <= 30:
                stats['qualification_difficulty'] = 'EASY'
            elif stats['median_days_to_qualify'] <= 90:
                stats['qualification_difficulty'] = 'MODERATE'
            elif stats['median_days_to_qualify'] <= 180:
                stats['qualification_difficulty'] = 'DIFFICULT'
            else:
                stats['qualification_difficulty'] = 'VERY_DIFFICULT'
                
            summary.append(stats)
            
        return pd.DataFrame(summary)
    
    def export_results(self, training_data, summary_data):
        """Export analysis results"""
        
        output_dir = Path("outputs/current")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export detailed training sequences
        training_file = output_dir / "qualification_training_sequences.csv"
        training_data.to_csv(training_file, index=False)
        print(f"\nExported training sequences to: {training_file}")
        
        # Export summary statistics
        summary_file = output_dir / "qualification_time_statistics.csv"
        summary_data.to_csv(summary_file, index=False)
        print(f"Exported summary statistics to: {summary_file}")
        
        # Print key insights
        print("\n📊 KEY INSIGHTS:")
        print(f"Total qualification sequences analyzed: {len(training_data)}")
        print(f"Unique qualifications: {training_data['qualification_code'].nunique()}")
        print(f"Average time to qualify: {training_data['days_to_qualify'].mean():.1f} days")
        print(f"Continuous training (no gaps): {(training_data['continuous_training'].sum() / len(training_data)) * 100:.1f}%")
        
        # Most difficult qualifications
        print("\n🔴 Most Difficult Qualifications (by median days):")
        difficult = summary_data.nlargest(5, 'median_days_to_qualify')
        for _, row in difficult.iterrows():
            print(f"  {row['qualification_code']}: {row['median_days_to_qualify']:.0f} days "
                  f"({row['engineers_qualified']} engineers)")
        
        # Easiest qualifications
        print("\n🟢 Easiest Qualifications (by median days):")
        easy = summary_data.nsmallest(5, 'median_days_to_qualify')
        for _, row in easy.iterrows():
            print(f"  {row['qualification_code']}: {row['median_days_to_qualify']:.0f} days "
                  f"({row['engineers_qualified']} engineers)")
        
        return training_file, summary_file


def main():
    print("🎓 QUALIFICATION TIME ANALYSIS")
    print("=" * 60)
    
    analyzer = QualificationTimeAnalyzer()
    
    print("\n📊 Analyzing qualification training sequences...")
    training_data = analyzer.analyze_qualification_times()
    
    if len(training_data) == 0:
        print("❌ No training sequences found")
        return
        
    print(f"✅ Found {len(training_data)} qualification sequences")
    
    print("\n📈 Generating summary statistics...")
    summary_data = analyzer.generate_summary_statistics(training_data)
    
    # Export results
    analyzer.export_results(training_data, summary_data)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()