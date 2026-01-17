# scripts/fixed_rota_parser.py

import json
import pandas as pd
from pathlib import Path

def parse_team_rota_csv(file_path, engineer_map):
    """
    Parse rota CSV that has format:
    Week X,Mon,Tues,Wed,Thurs,Fri,Sat,Sun
    Engineer 1,E,E,E,E,E,O,O
    Engineer 2,E,E,E,E,E,O,O
    ...
    """
    df = pd.read_csv(file_path)
    
    # The CSV structure appears to be different - let's handle it properly
    week_data = {}
    current_week = None
    
    for index, row in df.iterrows():
        first_col = str(row.iloc[0]).strip()
        
        # Check if this is a week header
        if first_col.startswith('Week'):
            current_week = first_col
            week_data[current_week] = {}
            continue
        
        # Check if this is an engineer row
        if first_col.startswith('Engineer'):
            engineer_label = first_col.strip()
            
            # Get the shifts (columns 1-7 for Mon-Sun)
            shifts = [str(x).strip() for x in row.iloc[1:8].tolist()]
            
            # Map engineer label to actual employee codes
            eng_codes = engineer_map.get(engineer_label)
            if eng_codes:
                if not isinstance(eng_codes, list):
                    eng_codes = [eng_codes]
                
                for code in eng_codes:
                    if current_week:
                        week_data[current_week][code] = shifts
    
    return week_data

def parse_week_block_csv(file_path, engineer_map):
    """
    Parse rota CSV with week-block structure - handles both header variations
    """
    # Try reading without header first to see raw structure
    df_raw = pd.read_csv(file_path, header=None)
    
    print(f"📊 CSV Analysis:")
    print(f"   Shape: {df_raw.shape}")
    print(f"   First few cells: {df_raw.iloc[0, 0]}, {df_raw.iloc[1, 0]}")
    
    # Determine if first row is "Week 1" (mechanical) or empty/header (electrical)
    first_cell = str(df_raw.iloc[0, 0]).strip()
    if first_cell.startswith('Week'):
        # Mechanical format: Week 1 is in first row
        df = df_raw
        print(f"   📋 Detected mechanical format (Week 1 in first row)")
    else:
        # Electrical format: Skip first row, Week 1 is in second row
        df = pd.read_csv(file_path)
        print(f"   📋 Detected electrical format (headers in first row)")
    
    week_data = {}
    current_week = None
    
    # Process each row
    for index, row in df.iterrows():
        first_col = str(row.iloc[0]).strip()
        
        # Skip empty rows
        if first_col == 'nan' or first_col == '' or pd.isna(first_col):
            continue
        
        # Check if this is a week header
        if first_col.startswith('Week'):
            current_week = first_col
            week_data[current_week] = {}
            print(f"   📅 Found {current_week}")
            continue
        
        # Check if this is an engineer row
        if first_col.startswith('Engineer'):
            if current_week is None:
                print(f"   ⚠️  Engineer row found before week header: {first_col}")
                continue
                
            engineer_label = first_col.strip()
            
            # Get the shifts (7 days)
            shifts = []
            for i in range(1, 8):  # Columns 1-7 for Mon-Sun
                if i < len(row):
                    shift = str(row.iloc[i]).strip()
                    # Handle various shift codes
                    if shift == 'nan' or pd.isna(shift):
                        shift = 'O'  # Convert NaN to O (Off)
                    shifts.append(shift)
                else:
                    shifts.append('O')  # Default to Off if missing
            
            # Validate shifts - M is valid for electrical teams (Mid shift 09:30-18:45)
            valid_shifts = all(s in ['E', 'L', 'M', 'O'] for s in shifts)
            if not valid_shifts:
                invalid_shifts = [s for s in shifts if s not in ['E', 'L', 'M', 'O']]
                print(f"   ⚠️  Invalid shifts for {engineer_label} in {current_week}: {invalid_shifts}")
                # Convert any invalid shifts to O (Off)
                shifts = ['O' if s not in ['E', 'L', 'M', 'O'] else s for s in shifts]
            
            # Map engineer label to actual employee codes
            eng_codes = engineer_map.get(engineer_label)
            if eng_codes:
                if not isinstance(eng_codes, list):
                    eng_codes = [eng_codes]
                
                for code in eng_codes:
                    week_data[current_week][code] = shifts
                    
                if len(week_data) <= 3:  # Show first 3 weeks only
                    print(f"   👤 {engineer_label} -> {eng_codes}: {shifts}")
            else:
                print(f"   ⚠️  No mapping found for: {engineer_label}")
    
    total_weeks = len(week_data)
    print(f"   ✅ Successfully parsed {total_weeks} weeks")
    
    # Show summary
    if week_data:
        weeks_list = sorted(week_data.keys(), key=lambda x: int(x.split()[-1]))
        first_week = weeks_list[0]
        last_week = weeks_list[-1]
        engineers_in_week = len(week_data[first_week])
        print(f"   📊 Range: {first_week} to {last_week}")
        print(f"   📊 Engineers per week: {engineers_in_week}")
    
    return week_data

def load_engineer_map_with_roles(engineer_json_path, role_filter):
    """Load engineer mapping filtered by role"""
    with open(engineer_json_path) as f:
        engineers = json.load(f)["engineers"]
    
    mapping = {}
    for eng in engineers:
        if not eng.get("rota_number") or not eng.get("role"):
            continue
        if eng["role"].lower() != role_filter.lower():
            continue
            
        # Create key that matches CSV format
        key = f"Engineer {eng['rota_number']}"
        
        # Handle multiple engineers per rota position
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(eng["employee_code"])
    
    print(f"Engineer mapping for {role_filter}:")
    for key, codes in mapping.items():
        print(f"  {key}: {codes}")
    
    return mapping

def debug_csv_structure(file_path):
    """Debug function to understand CSV structure"""
    print(f"\n=== Debugging {file_path} ===")
    
    # Try reading without headers first
    df_no_header = pd.read_csv(file_path, header=None)
    print(f"Shape without header: {df_no_header.shape}")
    print("First few rows without header:")
    print(df_no_header.head())
    
    # Try reading with headers
    df_with_header = pd.read_csv(file_path)
    print(f"\nShape with header: {df_with_header.shape}")
    print("Columns:", list(df_with_header.columns))
    print("First few rows with header:")
    print(df_with_header.head())
    
    return df_with_header

def save_json(data, output_path):
    """Save data to JSON file"""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")

def main():
    """Parse all team rota files"""
    
    # Define all team/role combinations with correct file structure
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
        print(f"\n{'='*60}")
        print(f"🔄 PARSING TEAM {team_config['team']} {team_config['role'].upper()}")
        print(f"{'='*60}")
        
        try:
            # Debug CSV structure
            print(f"📁 Reading: {team_config['csv_path']}")
            df = debug_csv_structure(team_config['csv_path'])
            
            # Load engineer mapping
            print(f"👥 Loading engineers from: {team_config['engineer_json']}")
            engineer_map = load_engineer_map_with_roles(
                team_config['engineer_json'], 
                team_config['role']
            )
            
            if not engineer_map:
                print(f"⚠️  No {team_config['role']} engineers found in team {team_config['team']}")
                continue
            
            # Parse the rota
            print(f"⚙️  Parsing rota data...")
            rota_data = parse_week_block_csv(team_config['csv_path'], engineer_map)
            
            if rota_data:
                save_json(rota_data, team_config['output_path'])
                print(f"✅ Successfully parsed Team {team_config['team']} {team_config['role']}")
                
                # Show summary
                total_weeks = len(rota_data)
                total_engineers = len(set().union(*[week_data.keys() for week_data in rota_data.values()]))
                print(f"📊 Summary: {total_weeks} weeks, {total_engineers} engineers")
            else:
                print(f"❌ Failed to parse Team {team_config['team']} {team_config['role']}")
                
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
        except Exception as e:
            print(f"❌ Error parsing Team {team_config['team']} {team_config['role']}: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 PARSING COMPLETE")
    print(f"{'='*60}")
    print("Ready to run: python3 scripts/rota_aware_competency_optimizer.py")

if __name__ == "__main__":
    main()
    
    # Run python3 scripts/rota_parser.py to execute and update team 1 and 2 parsed rota.json files 