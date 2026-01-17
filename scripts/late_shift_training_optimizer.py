#!/usr/bin/env python3
"""
Late Shift Training Optimizer - Set Cover Approach
===================================================

Find the MINIMUM set of training assignments needed to achieve 100%
late shift coverage across the 18-week rotation.

Uses a greedy set cover algorithm:
1. Identify all coverage gaps (ride + day instances)
2. For each gap, determine which engineers could close it
3. Greedily select training assignments that close the most uncovered gaps
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter


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
    """Load engineer names and details"""
    eng_details = {}

    for role in ['elec', 'mech']:
        eng_file = f'data/processed/engineers/team{team}_{role}_engineers.json'
        try:
            with open(eng_file, 'r') as f:
                data = json.load(f)
                for eng in data.get('engineers', []):
                    eng_details[eng['employee_code']] = {
                        'name': eng.get('timeplan_name', eng['employee_code']),
                        'role': eng.get('role', role),
                        'team': team,
                        'rota_number': eng.get('rota_number', 0)
                    }
        except FileNotFoundError:
            pass

    return eng_details


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


def identify_all_gaps(team, max_weeks=18):
    """Identify ALL gap instances across the rotation"""
    print(f"\n{'='*80}")
    print(f"IDENTIFYING COVERAGE GAPS - TEAM {team}")
    print(f"{'='*80}")

    # Load data
    rotas = {}
    job_sharing = {}

    for role in ['elec', 'mech']:
        rota_file = f'data/processed/parsed_rotas/parsed_team{team}_{role}_rota.json'
        try:
            with open(rota_file, 'r') as f:
                rotas[role] = json.load(f)
            job_sharing[role] = load_engineer_rota_mapping(team, role)
        except FileNotFoundError:
            rotas[role] = {}
            job_sharing[role] = {}

    engineer_rides = load_engineer_qualifications()
    rides_info = load_ride_info()

    # Get all team rides
    team_rides = sorted([rid for rid, info in rides_info.items()
                        if info.get('team_responsible') == team])

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Track all gaps and which engineers could close them
    all_gaps = []
    gap_to_engineers = {}  # gap_id -> set of engineers who could close it

    # Analyze each day
    for week_num in range(1, max_weeks + 1):
        week_key = f'Week {week_num}'

        for day_idx, day_name in enumerate(days):
            # Get late shift engineers (handling job sharing)
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

            # Get covered rides
            covered_rides = set()
            for eng_code in late_shift_engineers:
                if eng_code in engineer_rides:
                    covered_rides.update(engineer_rides[eng_code])

            # For each uncovered ride, create a gap instance
            for ride_code in team_rides:
                if ride_code not in covered_rides:
                    # Create unique gap ID
                    gap_id = f"W{week_num:02d}_D{day_idx}_{ride_code}"

                    gap_info = {
                        'gap_id': gap_id,
                        'week': week_num,
                        'day': day_name,
                        'day_idx': day_idx,
                        'ride_code': ride_code,
                        'engineers_on_shift': late_shift_engineers.copy()
                    }

                    all_gaps.append(gap_info)

                    # Record which engineers could close this gap
                    gap_to_engineers[gap_id] = set(late_shift_engineers)

    print(f"\n📊 GAP ANALYSIS:")
    print(f"   Total gap instances: {len(all_gaps)}")
    print(f"   Unique rides with gaps: {len(set(g['ride_code'] for g in all_gaps))}")

    # Analyze rotation symmetry
    print(f"\n🔄 ROTATION ANALYSIS:")

    # Count how many times each engineer appears on late shift
    late_shift_counts = Counter()
    for week_num in range(1, max_weeks + 1):
        week_key = f'Week {week_num}'
        for day_idx in range(7):
            seen_positions = set()
            for role in ['elec', 'mech']:
                if week_key not in rotas.get(role, {}):
                    continue

                week_data = rotas[role][week_key]

                for eng_code, shifts in week_data.items():
                    if day_idx < len(shifts) and shifts[day_idx] == 'L':
                        if eng_code in job_sharing.get(role, {}):
                            shared_group = job_sharing[role][eng_code]
                            position_id = tuple(sorted(shared_group))

                            if position_id in seen_positions:
                                continue
                            else:
                                seen_positions.add(position_id)
                                late_shift_counts[eng_code] += 1
                        else:
                            late_shift_counts[eng_code] += 1

    # Show distribution
    if late_shift_counts:
        print(f"   Engineers on late shifts: {len(late_shift_counts)}")
        print(f"   Late shift appearances per engineer:")
        print(f"      Min: {min(late_shift_counts.values())}")
        print(f"      Max: {max(late_shift_counts.values())}")
        print(f"      Mean: {sum(late_shift_counts.values()) / len(late_shift_counts):.1f}")

        # Show a few examples
        print(f"\n   Sample engineers:")
        for eng_code, count in sorted(late_shift_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {eng_code}: {count} late shifts")

    return all_gaps, gap_to_engineers


def greedy_set_cover_optimization(team, all_gaps, gap_to_engineers):
    """Use greedy set cover to find minimal training assignments"""
    print(f"\n{'='*80}")
    print(f"OPTIMIZING TRAINING PLAN - TEAM {team}")
    print(f"{'='*80}")
    print(f"Finding MINIMUM set of training assignments to close all gaps\n")

    engineer_details = load_engineer_names(team)
    rides_info = load_ride_info()

    # Build reverse mapping: (engineer, ride) -> set of gaps it would cover
    training_coverage = defaultdict(set)

    for gap in all_gaps:
        gap_id = gap['gap_id']
        ride_code = gap['ride_code']

        # Which engineers could close this gap?
        for eng_code in gap_to_engineers[gap_id]:
            training_key = (eng_code, ride_code)
            training_coverage[training_key].add(gap_id)

    # Greedy set cover algorithm
    uncovered_gaps = set(g['gap_id'] for g in all_gaps)
    selected_trainings = []

    iteration = 0
    while uncovered_gaps:
        iteration += 1

        # Find the training assignment that covers the most uncovered gaps
        best_training = None
        best_coverage = set()

        for training_key, gaps_covered in training_coverage.items():
            # How many uncovered gaps would this training close?
            newly_covered = gaps_covered & uncovered_gaps

            if len(newly_covered) > len(best_coverage):
                best_training = training_key
                best_coverage = newly_covered

        if best_training is None:
            # No more trainings can help (shouldn't happen)
            break

        # Select this training
        eng_code, ride_code = best_training
        selected_trainings.append({
            'engineer_code': eng_code,
            'ride_code': ride_code,
            'gaps_closed': len(best_coverage),
            'iteration': iteration
        })

        # Mark these gaps as covered
        uncovered_gaps -= best_coverage

        # Progress update
        if iteration % 10 == 0 or len(uncovered_gaps) == 0:
            print(f"   Iteration {iteration}: Selected {eng_code} -> {ride_code} "
                  f"(closes {len(best_coverage)} gaps, {len(uncovered_gaps)} remaining)")

    print(f"\n✅ OPTIMIZATION COMPLETE!")
    print(f"   Total training assignments needed: {len(selected_trainings)}")
    print(f"   All {len(all_gaps)} gaps covered")

    # Enrich with details
    for training in selected_trainings:
        eng_code = training['engineer_code']
        ride_code = training['ride_code']

        eng_info = engineer_details.get(eng_code, {})
        ride_info = rides_info.get(ride_code, {})

        training['engineer_name'] = eng_info.get('name', eng_code)
        training['engineer_role'] = eng_info.get('role', 'Unknown')
        training['engineer_rota'] = eng_info.get('rota_number', 0)
        training['ride_name'] = ride_info.get('name', ride_code)
        training['ride_type'] = ride_info.get('type', '?')
        training['team'] = team

    return selected_trainings


def analyze_training_plan(selected_trainings):
    """Analyze the optimized training plan"""
    print(f"\n{'='*80}")
    print(f"TRAINING PLAN ANALYSIS")
    print(f"{'='*80}")

    # Group by engineer
    engineer_trainings = defaultdict(list)
    for training in selected_trainings:
        eng_code = training['engineer_code']
        engineer_trainings[eng_code].append(training)

    print(f"\n📚 TRAINING ASSIGNMENTS BY ENGINEER:")
    print(f"   Total engineers requiring training: {len(engineer_trainings)}\n")

    engineer_summary = []

    for eng_code in sorted(engineer_trainings.keys()):
        trainings = engineer_trainings[eng_code]
        eng_name = trainings[0]['engineer_name']
        eng_role = trainings[0]['engineer_role']
        team = trainings[0]['team']

        total_gaps = sum(t['gaps_closed'] for t in trainings)
        rides = [t['ride_code'] for t in trainings]

        print(f"{eng_name} ({eng_role}) - Team {team}")
        print(f"   Rides to learn: {len(rides)} - Total gap instances: {total_gaps}")
        print(f"   Rides: {', '.join(rides)}")

        engineer_summary.append({
            'Team': team,
            'Engineer_Code': eng_code,
            'Engineer_Name': eng_name,
            'Engineer_Role': eng_role,
            'Rides_To_Learn': len(rides),
            'Total_Gaps_Closed': total_gaps,
            'Ride_List': ', '.join(rides)
        })

    # Group by ride
    ride_trainings = defaultdict(list)
    for training in selected_trainings:
        ride_code = training['ride_code']
        ride_trainings[ride_code].append(training)

    print(f"\n\n🎢 TRAINING ASSIGNMENTS BY RIDE:")
    print(f"   Total rides requiring training: {len(ride_trainings)}\n")

    for ride_code in sorted(ride_trainings.keys()):
        trainings = ride_trainings[ride_code]
        ride_name = trainings[0]['ride_name']
        ride_type = trainings[0]['ride_type']

        engineers = [(t['engineer_name'], t['gaps_closed']) for t in trainings]

        print(f"{ride_code} - {ride_name} (Type {ride_type})")
        print(f"   Engineers to train: {len(engineers)}")
        for eng_name, gaps in engineers:
            print(f"      • {eng_name} ({gaps} gap instances)")

    return engineer_summary


def export_results(all_results):
    """Export optimized training plan to CSV"""
    output_dir = Path('outputs/current')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export detailed training assignments
    all_trainings = []
    for result in all_results:
        all_trainings.extend(result['trainings'])

    if all_trainings:
        df = pd.DataFrame(all_trainings)
        df = df.sort_values(['team', 'iteration'])

        output_file = output_dir / 'late_shift_training_optimized.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Optimized training plan exported to: {output_file}")

    # Export engineer summaries
    all_summaries = []
    for result in all_results:
        all_summaries.extend(result['engineer_summary'])

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        df = df.sort_values(['Team', 'Rides_To_Learn'], ascending=[True, False])

        output_file = output_dir / 'late_shift_training_optimized_by_engineer.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Engineer training summary exported to: {output_file}")

    # Overall summary
    print(f"\n📊 OVERALL SUMMARY:")
    for result in all_results:
        team = result['team']
        total_trainings = len(result['trainings'])
        total_engineers = len(result['engineer_summary'])
        total_gaps = result['total_gaps']

        print(f"\n   Team {team}:")
        print(f"      Total gaps identified: {total_gaps}")
        print(f"      Training assignments needed: {total_trainings}")
        print(f"      Engineers requiring training: {total_engineers}")


def main():
    """Run optimized training analysis for all teams"""
    print("🎯 LATE SHIFT TRAINING OPTIMIZER - SET COVER APPROACH")
    print("="*80)
    print("Finding MINIMUM training assignments to achieve 100% coverage\n")

    all_results = []

    for team in [1, 2]:
        # Step 1: Identify all gaps
        all_gaps, gap_to_engineers = identify_all_gaps(team, max_weeks=18)

        # Step 2: Optimize training plan
        selected_trainings = greedy_set_cover_optimization(team, all_gaps, gap_to_engineers)

        # Step 3: Analyze plan
        engineer_summary = analyze_training_plan(selected_trainings)

        all_results.append({
            'team': team,
            'total_gaps': len(all_gaps),
            'trainings': selected_trainings,
            'engineer_summary': engineer_summary
        })

    # Export results
    export_results(all_results)

    print(f"\n{'='*80}")
    print("✅ Optimization complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
