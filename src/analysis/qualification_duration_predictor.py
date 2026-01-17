"""
Qualification Duration Predictor
Uses historical data to estimate realistic training timeframes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

class QualificationDurationPredictor:
    """Predicts training duration based on historical data"""
    
    def __init__(self):
        self.duration_stats = None
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load pre-calculated qualification statistics"""
        stats_file = Path("outputs/current/qualification_time_statistics.csv")
        
        if stats_file.exists():
            self.duration_stats = pd.read_csv(stats_file)
            print(f"✅ Loaded duration statistics for {len(self.duration_stats)} qualifications")
        else:
            print("⚠️  No historical duration data found. Run qualification_time_analysis.py first.")
            self.duration_stats = pd.DataFrame()
    
    def get_estimated_duration(self, qualification_code: str, confidence_level: str = 'median') -> Dict:
        """
        Get estimated training duration for a qualification
        
        Args:
            qualification_code: The qualification to estimate
            confidence_level: 'optimistic', 'median', 'pessimistic'
        
        Returns:
            Dict with duration estimates and metadata
        """
        if self.duration_stats is None or len(self.duration_stats) == 0:
            return self._fallback_estimate(qualification_code)
        
        # Find qualification in historical data
        qual_stats = self.duration_stats[
            self.duration_stats['qualification_code'] == qualification_code
        ]
        
        if len(qual_stats) == 0:
            return self._fallback_estimate(qualification_code)
        
        stats = qual_stats.iloc[0]
        
        # Apply frequency-based adjustment to historical data
        frequency_multiplier = self._get_frequency_multiplier(qualification_code)
        
        # Calculate different confidence levels with frequency adjustment
        raw_optimistic = max(1, stats['min_days_to_qualify'])
        raw_median = stats['median_days_to_qualify']
        raw_pessimistic = min(365, stats['max_days_to_qualify'])
        
        estimates = {
            'qualification_code': qualification_code,
            'optimistic_days': max(1, raw_optimistic * frequency_multiplier),
            'median_days': raw_median * frequency_multiplier,
            'pessimistic_days': min(365, raw_pessimistic * frequency_multiplier),
            'average_days': stats['avg_days_to_qualify'] * frequency_multiplier,
            'difficulty_level': stats['qualification_difficulty'],
            'historical_sample_size': int(stats['engineers_qualified']),
            'avg_training_sessions': stats['avg_training_sessions'],
            'avg_training_hours': stats['avg_training_hours'],
            'continuous_training_pct': stats['continuous_training_pct'],
            'confidence': 'HIGH' if stats['engineers_qualified'] >= 3 else 'LOW',
            'frequency_multiplier': frequency_multiplier,
            'ppm_frequency': self._get_ppm_frequency(qualification_code)
        }
        
        # Return requested estimate
        if confidence_level == 'optimistic':
            estimates['selected_estimate'] = estimates['optimistic_days']
        elif confidence_level == 'pessimistic':
            estimates['selected_estimate'] = estimates['pessimistic_days']
        else:
            estimates['selected_estimate'] = estimates['median_days']
        
        return estimates
    
    def _get_frequency_multiplier(self, qualification_code: str) -> float:
        """
        Calculate frequency-based adjustment multiplier
        
        Higher frequency PPMs (daily) offer more training opportunities,
        so should train faster. Lower frequency (monthly) offers fewer
        opportunities, so takes longer.
        """
        if '.2.D' in qualification_code:  # Daily - 365 opportunities/year
            return 0.4  # Much faster due to daily practice
        elif '.3.W' in qualification_code:  # Weekly - 52 opportunities/year
            return 0.7  # Moderately faster
        elif '.4.M' in qualification_code or '.4.Q' in qualification_code:  # Monthly/Quarterly - 12-4 opportunities/year
            return 1.8  # Much slower due to limited practice
        else:
            return 1.0  # Default - no adjustment
    
    def _get_ppm_frequency(self, qualification_code: str) -> str:
        """Extract PPM frequency from qualification code"""
        if '.2.D' in qualification_code:
            return 'DAILY'
        elif '.3.W' in qualification_code:
            return 'WEEKLY'
        elif '.4.M' in qualification_code:
            return 'MONTHLY'
        elif '.4.Q' in qualification_code:
            return 'QUARTERLY'
        else:
            return 'UNKNOWN'

    def _fallback_estimate(self, qualification_code: str) -> Dict:
        """Fallback estimates when no historical data available"""
        
        # Basic rules based on qualification pattern with frequency adjustment
        frequency_multiplier = self._get_frequency_multiplier(qualification_code)
        
        if '.2.D' in qualification_code:  # Daily qualifications
            base_days = 30
        elif '.3.W' in qualification_code:  # Weekly qualifications
            base_days = 60
        elif '.4.M' in qualification_code:  # Monthly qualifications
            base_days = 90
        else:
            base_days = 45  # Default
        
        # Apply frequency adjustment to base estimates
        adjusted_days = base_days * frequency_multiplier
        
        return {
            'qualification_code': qualification_code,
            'optimistic_days': max(7, adjusted_days * 0.5),
            'median_days': adjusted_days,
            'pessimistic_days': adjusted_days * 2,
            'average_days': adjusted_days,
            'difficulty_level': 'UNKNOWN',
            'historical_sample_size': 0,
            'avg_training_sessions': 5,  # Estimate
            'avg_training_hours': 10,   # Estimate
            'continuous_training_pct': 70,  # Average
            'confidence': 'ESTIMATE',
            'frequency_multiplier': frequency_multiplier,
            'ppm_frequency': self._get_ppm_frequency(qualification_code),
            'selected_estimate': adjusted_days
        }
    
    def bulk_estimate(self, qualification_codes: list, confidence_level: str = 'median') -> pd.DataFrame:
        """Get estimates for multiple qualifications"""
        
        estimates = []
        for qual_code in qualification_codes:
            estimate = self.get_estimated_duration(qual_code, confidence_level)
            estimates.append(estimate)
        
        return pd.DataFrame(estimates)
    
    def get_training_plan_timeline(self, engineer_training_plan: Dict) -> Dict:
        """
        Calculate realistic timeline for an engineer's training plan
        
        Args:
            engineer_training_plan: Dict with engineer code and list of qualifications needed
            
        Returns:
            Timeline with parallel and sequential training estimates
        """
        
        engineer = engineer_training_plan['engineer_code']
        qualifications = engineer_training_plan['qualifications_needed']
        
        # Get estimates for all qualifications
        estimates = self.bulk_estimate(qualifications)
        
        # Calculate different scenarios
        timeline = {
            'engineer_code': engineer,
            'total_qualifications': len(qualifications),
            'sequential_timeline_days': estimates['selected_estimate'].sum(),
            'parallel_timeline_days': estimates['selected_estimate'].max(),
            'total_training_hours': estimates['avg_training_hours'].sum(),
            'high_risk_qualifications': len(estimates[estimates['difficulty_level'].isin(['DIFFICULT', 'VERY_DIFFICULT'])]),
            'low_confidence_estimates': len(estimates[estimates['confidence'] != 'HIGH']),
            'recommended_approach': self._recommend_approach(estimates)
        }
        
        return timeline
    
    def _recommend_approach(self, estimates: pd.DataFrame) -> str:
        """Recommend training approach based on estimates"""
        
        difficult_quals = estimates[estimates['difficulty_level'].isin(['DIFFICULT', 'VERY_DIFFICULT'])]
        easy_quals = estimates[estimates['difficulty_level'] == 'EASY']
        
        if len(difficult_quals) > len(easy_quals):
            return "FOCUS_ON_DIFFICULT_FIRST"
        elif len(easy_quals) > 3:
            return "QUICK_WINS_FIRST"
        else:
            return "BALANCED_APPROACH"
    
    def export_duration_lookup(self, output_path: str = "outputs/current/qualification_duration_lookup.csv"):
        """Export duration estimates for all known qualifications"""
        
        if self.duration_stats is None or len(self.duration_stats) == 0:
            print("❌ No duration statistics available for export")
            return
        
        # Create enhanced duration data with frequency adjustments
        enhanced_data = []
        
        for _, row in self.duration_stats.iterrows():
            qual_code = row['qualification_code']
            
            # Get frequency-adjusted estimates
            estimate = self.get_estimated_duration(qual_code, 'median')
            
            enhanced_record = {
                'qualification_code': qual_code,
                'ppm_frequency': estimate['ppm_frequency'],
                'frequency_multiplier': estimate['frequency_multiplier'],
                'raw_median_days': row['median_days_to_qualify'],
                'adjusted_median_days': estimate['median_days'],
                'adjusted_median_weeks': round(estimate['median_days'] / 7, 1),
                'optimistic_days': estimate['optimistic_days'],
                'pessimistic_days': estimate['pessimistic_days'],
                'difficulty_level': estimate['difficulty_level'],
                'historical_sample_size': estimate['historical_sample_size'],
                'confidence_level': estimate['confidence'],
                'avg_training_sessions': estimate['avg_training_sessions'],
                'avg_training_hours': estimate['avg_training_hours'],
                'resource_intensity': self._calculate_intensity(row),
                'training_opportunities_per_year': self._get_annual_opportunities(qual_code)
            }
            enhanced_data.append(enhanced_record)
        
        # Convert to DataFrame and sort by frequency then difficulty
        enhanced_df = pd.DataFrame(enhanced_data)
        enhanced_df = enhanced_df.sort_values(['ppm_frequency', 'adjusted_median_days'])
        
        enhanced_df.to_csv(output_path, index=False)
        print(f"✅ Exported frequency-adjusted duration lookup to: {output_path}")
        
        return output_path
    
    def _get_annual_opportunities(self, qualification_code: str) -> int:
        """Calculate annual training opportunities based on PPM frequency"""
        if '.2.D' in qualification_code:  # Daily
            return 365
        elif '.3.W' in qualification_code:  # Weekly
            return 52
        elif '.4.M' in qualification_code:  # Monthly
            return 12
        elif '.4.Q' in qualification_code:  # Quarterly
            return 4
        else:
            return 26  # Default bi-weekly
    
    def _calculate_intensity(self, row):
        """Calculate resource intensity category"""
        hours = row['avg_training_hours']
        sessions = row['avg_training_sessions']
        
        if hours > 15 and sessions > 10:
            return 'HIGH'
        elif hours > 8 and sessions > 5:
            return 'MEDIUM'
        else:
            return 'LOW'


def main():
    """Example usage"""
    predictor = QualificationDurationPredictor()
    
    # Test single qualification
    estimate = predictor.get_estimated_duration('WICK.2.DM.T.S')
    print("\n📊 Sample Qualification Estimate:")
    print(f"Qualification: {estimate['qualification_code']}")
    print(f"Median Duration: {estimate['median_days']} days ({estimate['median_days']/7:.1f} weeks)")
    print(f"Difficulty: {estimate['difficulty_level']}")
    print(f"Sample Size: {estimate['historical_sample_size']} engineers")
    print(f"Confidence: {estimate['confidence']}")
    
    # Export lookup table
    predictor.export_duration_lookup()


if __name__ == "__main__":
    main()