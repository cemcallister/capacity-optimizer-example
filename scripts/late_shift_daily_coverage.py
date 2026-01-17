#!/usr/bin/env python3
"""
Late Shift Daily Coverage Analysis - 18 Week Rotation
======================================================

For each day of the 18-week rotation, show which engineers are on late shift
and which rides are covered. Identify any gaps where a ride has NO qualified
engineer on site during a late shift.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_engineer_qualifications():
    """Load engineer qualifications from EngQual.csv"""
    df = pd.read_csv('data/raw/EngQual.csv')

    # Filter active qualifications
    if 'expired' in df.columns:
        active_df = df[df['expired'].astype(str).str.lower() == 'false'].copy()
    else:
        active_df = df.copy()

    # Build engineer -> rides mapping
    engineer_rides = defaultdict(set)

    for _, row in active_df.iterrows():
        eng_code = row['Employee Code']
        qual = row['Qualification']

        # Extract ride code from qualification
        if pd.notna(qual) and '.' in str(qual):
            ride_code = qual.split('.')[0]
            engineer_rides[eng_code].add(ride_code)

    return dict(engineer_rides)


def load_ride_info():
    """Load ride information"""
    with open('data/processed/ride_info.json', 'r') as f:
        ride_data = json.load(f)
    return ride_data.get('rides', {})


def load_engineer_names(team):
    """Load engineer names for display"""
    eng_names = {}

    for role in ['elec', 'mech']:
        eng_file = f'data/processed/engineers/team{team}_{role}_engineers.json'
        try:
            with open(eng_file, 'r') as f:
                data = json.load(f)
                for eng in data.get('engineers', []):
                    eng_names[eng['employee_code']] = eng.get('timeplan_name', eng['employee_code'])
        except FileNotFoundError:
            pass

    return eng_names


def load_engineer_rota_mapping(team, role):
    """Load engineer data to detect job sharing"""
    eng_file = f'data/processed/engineers/team{team}_{role}_engineers.json'

    try:
        with open(eng_file, 'r') as f:
            data = json.load(f)
            engineers = data.get('engineers', [])

        # Group by rota position
        rota_positions = defaultdict(list)
        for eng in engineers:
            rota_num = eng.get('rota_number')
            if rota_num:
                rota_positions[rota_num].append(eng['employee_code'])

        # Find job sharing pairs/groups
        job_sharing_groups = {}
        for rota_num, eng_codes in rota_positions.items():
            if len(eng_codes) > 1:
                for code in eng_codes:
                    job_sharing_groups[code] = eng_codes

        return job_sharing_groups
    except FileNotFoundError:
        return {}


def analyze_daily_coverage_18_weeks(team):
    """Analyze day-by-day coverage across 18-week rotation"""
    print(f"\n{'='*80}")
    print(f"LATE SHIFT DAILY COVERAGE - TEAM {team} - 18 WEEK ROTATION")
    print(f"{'='*80}")

    # Load data for both electrical and mechanical
    rotas = {}
    job_sharing = {}

    for role in ['elec', 'mech']:
        rota_file = f'data/processed/parsed_rotas/parsed_team{team}_{role}_rota.json'
        try:
            with open(rota_file, 'r') as f:
                rotas[role] = json.load(f)
            job_sharing[role] = load_engineer_rota_mapping(team, role)
        except FileNotFoundError:
            print(f"   ⚠️  Rota file not found for {role}")
            rotas[role] = {}
            job_sharing[role] = {}

    engineer_rides = load_engineer_qualifications()
    rides_info = load_ride_info()
    engineer_names = load_engineer_names(team)

    # Get all team rides
    team_rides = sorted([rid for rid, info in rides_info.items()
                        if info.get('team_responsible') == team])

    print(f"\n📊 Team {team}: {len(team_rides)} rides to cover")
    print(f"   Rides: {', '.join(team_rides)}")

    # Determine max weeks (use mechanical as it's typically 18 weeks)
    max_weeks_mech = len(rotas.get('mech', {}))
    max_weeks_elec = len(rotas.get('elec', {}))

    # Use 18 weeks or max available
    max_weeks = min(18, max(max_weeks_mech, max_weeks_elec))

    print(f"\n📅 Analyzing {max_weeks} weeks of rotation")

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Track coverage gaps
    coverage_gaps = []
    daily_coverage_data = []

    # Analyze each day
    for week_num in range(1, max_weeks + 1):
        week_key = f'Week {week_num}'

        for day_idx, day_name in enumerate(days):
            # Get late shift engineers from BOTH electrical and mechanical (handling job sharing)
            late_shift_engineers = []
            seen_positions = set()

            for role in ['elec', 'mech']:
                if week_key not in rotas.get(role, {}):
                    continue

                week_data = rotas[role][week_key]

                for eng_code, shifts in week_data.items():
                    if day_idx < len(shifts) and shifts[day_idx] == 'L':
                        # Handle job sharing
                        if eng_code in job_sharing.get(role, {}):
                            shared_group = job_sharing[role][eng_code]
                            position_id = tuple(sorted(shared_group))

                            if position_id in seen_positions:
                                continue
                            else:
                                seen_positions.add(position_id)
                                late_shift_engineers.append(eng_code)
                        else:
                            late_shift_engineers.append(eng_code)

            # Get all rides covered by late shift engineers today
            covered_rides = set()
            for eng_code in late_shift_engineers:
                if eng_code in engineer_rides:
                    covered_rides.update(engineer_rides[eng_code])

            # Check each team ride
            uncovered_rides = []
            for ride_code in team_rides:
                if ride_code not in covered_rides:
                    uncovered_rides.append(ride_code)

            # Record this day
            daily_coverage_data.append({
                'Week': week_num,
                'Day': day_name,
                'Day_Index': day_idx,
                'Engineers_On_Late': len(late_shift_engineers),
                'Engineer_Codes': ', '.join(sorted(late_shift_engineers)),
                'Engineer_Names': ', '.join([engineer_names.get(e, e) for e in sorted(late_shift_engineers)]),
                'Rides_Covered': len(covered_rides),
                'Uncovered_Rides': len(uncovered_rides),
                'Uncovered_Ride_Codes': ', '.join(uncovered_rides) if uncovered_rides else ''
            })

            # If there are gaps, record them
            if uncovered_rides:
                for ride_code in uncovered_rides:
                    ride_info = rides_info.get(ride_code, {})
                    coverage_gaps.append({
                        'Week': week_num,
                        'Day': day_name,
                        'Ride_Code': ride_code,
                        'Ride_Name': ride_info.get('name', ride_code),
                        'Ride_Type': ride_info.get('type', '?'),
                        'Engineers_On_Late': len(late_shift_engineers)
                    })

    # Print summary
    print(f"\n📊 COVERAGE SUMMARY:")
    total_days = len(daily_coverage_data)
    days_with_gaps = sum(1 for d in daily_coverage_data if d['Uncovered_Rides'] > 0)
    days_full_coverage = total_days - days_with_gaps

    print(f"   Total days analyzed: {total_days}")
    print(f"   Days with FULL coverage: {days_full_coverage} ({days_full_coverage/total_days*100:.1f}%)")
    print(f"   Days with gaps: {days_with_gaps} ({days_with_gaps/total_days*100:.1f}%)")
    print(f"   Total coverage gaps: {len(coverage_gaps)}")

    if coverage_gaps:
        print(f"\n❌ COVERAGE GAPS FOUND:")

        # Group by ride
        gaps_by_ride = defaultdict(list)
        for gap in coverage_gaps:
            gaps_by_ride[gap['Ride_Code']].append(gap)

        for ride_code in sorted(gaps_by_ride.keys()):
            gaps = gaps_by_ride[ride_code]
            ride_name = gaps[0]['Ride_Name']
            ride_type = gaps[0]['Ride_Type']
            gap_count = len(gaps)

            print(f"\n   {ride_code} - {ride_name} (Type {ride_type})")
            print(f"   ⚠️  {gap_count} days with NO coverage ({gap_count/total_days*100:.1f}% of rotation)")

            # Show first 5 gaps
            print(f"   Sample gaps:")
            for gap in gaps[:5]:
                print(f"      • Week {gap['Week']:2} {gap['Day']:9} - {gap['Engineers_On_Late']} engineers on site (none qualified)")

            if len(gaps) > 5:
                print(f"      ... and {len(gaps)-5} more days")

    else:
        print(f"\n✅ FULL COVERAGE - Every ride has at least one qualified engineer on every late shift!")

    return {
        'team': team,
        'daily_coverage': daily_coverage_data,
        'coverage_gaps': coverage_gaps,
        'total_days': total_days,
        'days_with_gaps': days_with_gaps
    }


def export_results(all_results):
    """Export detailed results to CSV"""
    output_dir = Path('outputs/current')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export daily coverage
    all_daily = []
    for result in all_results:
        for day in result['daily_coverage']:
            day['Team'] = result['team']
            all_daily.append(day)

    if all_daily:
        df_daily = pd.DataFrame(all_daily)
        df_daily = df_daily[['Team', 'Week', 'Day', 'Day_Index', 'Engineers_On_Late',
                             'Engineer_Names', 'Rides_Covered', 'Uncovered_Rides', 'Uncovered_Ride_Codes']]

        output_file = output_dir / 'late_shift_daily_coverage_18weeks.csv'
        df_daily.to_csv(output_file, index=False)
        print(f"\n💾 Daily coverage exported to: {output_file}")

    # Export coverage gaps
    all_gaps = []
    for result in all_results:
        for gap in result['coverage_gaps']:
            gap['Team'] = result['team']
            all_gaps.append(gap)

    if all_gaps:
        df_gaps = pd.DataFrame(all_gaps)
        df_gaps = df_gaps[['Team', 'Week', 'Day', 'Ride_Code', 'Ride_Name', 'Ride_Type', 'Engineers_On_Late']]
        df_gaps = df_gaps.sort_values(['Team', 'Ride_Code', 'Week', 'Day'])

        output_file = output_dir / 'late_shift_coverage_gaps_18weeks.csv'
        df_gaps.to_csv(output_file, index=False)
        print(f"💾 Coverage gaps exported to: {output_file}")

    # Print overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY:")
    for result in all_results:
        team = result['team']
        total_days = result['total_days']
        days_with_gaps = result['days_with_gaps']
        coverage_pct = ((total_days - days_with_gaps) / total_days * 100) if total_days > 0 else 0

        print(f"\n   Team {team}:")
        print(f"      Coverage: {coverage_pct:.1f}% ({total_days - days_with_gaps}/{total_days} days)")

        if result['coverage_gaps']:
            # Count unique rides with gaps
            unique_rides = len(set(g['Ride_Code'] for g in result['coverage_gaps']))
            print(f"      Rides with gaps: {unique_rides}")
            print(f"      Total gap instances: {len(result['coverage_gaps'])}")
        else:
            print(f"      ✅ FULL COVERAGE - no gaps!")


def main():
    """Run daily coverage analysis for all teams"""
    print("📅 LATE SHIFT DAILY COVERAGE ANALYSIS - 18 WEEK ROTATION")
    print("="*80)
    print("Checking day-by-day if each ride has at least one qualified")
    print("engineer on site during late shifts\n")

    all_results = []

    for team in [1, 2]:
        result = analyze_daily_coverage_18_weeks(team)
        all_results.append(result)

    export_results(all_results)

    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
