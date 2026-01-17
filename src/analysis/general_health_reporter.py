"""
General Health Reporter
=======================

Computes progress towards MILP target qualifications using active current
qualifications from EngQual.csv and exports a single Health.csv suitable for
Power BI. Progress is measured strictly against the MILP-selected engineer →
qualification pairs (target), not any extra qualifications on other engineers.
"""

from pathlib import Path
from typing import Dict, Tuple, Set, List
import csv
import pandas as pd


class GeneralHealthReporter:
    """Compute and export health metrics against MILP target pairs."""

    def __init__(self, optimizer):
        self.optimizer = optimizer  # Has rides_info and ppms_by_type

    def _load_active_current_pairs(self) -> Set[Tuple[str, str]]:
        """Load current active (engineer_code, qualification_code) pairs from EngQual.csv.

        Mirrors the EngQual filtering used elsewhere: prefer 'expired' flag if present,
        otherwise use Expiration date; exclude temp disqualified and OUT OF SERVICE.
        """
        engqual_path = Path('data/raw/EngQual.csv')
        if not engqual_path.exists():
            return set()

        df = pd.read_csv(engqual_path)

        # Parse dates where needed
        if 'Qualification Start' in df.columns:
            df['Qualification Start'] = pd.to_datetime(df['Qualification Start'], errors='coerce')
        if 'Expiration' in df.columns:
            df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')

        # Active mask: prefer 'expired' if exists; else Expiration > now
        if 'expired' in df.columns:
            active_not_expired = (df['expired'].astype(str).str.lower() == 'false')
        else:
            # If missing dates, treat as expired to be conservative
            # Use tz-naive "now" to avoid tz-aware vs tz-naive comparison errors
            now = pd.Timestamp.utcnow().tz_localize(None)
            active_not_expired = df['Expiration'] > now

        temp_ok = (df.get('Temp Disqualified', '') != '+')
        not_ooo = (df.get('Employee Name', '') != 'OUT OF SERVICE')

        mask = active_not_expired & temp_ok & not_ooo
        active_df = df[mask].copy()

        pairs: Set[Tuple[str, str]] = set()
        if {'Employee Code', 'Qualification'}.issubset(active_df.columns):
            for emp, qual in zip(active_df['Employee Code'], active_df['Qualification']):
                if pd.notna(emp) and pd.notna(qual):
                    pairs.add((str(emp), str(qual)))
        return pairs

    def _build_target_pairs(self, matrices: Dict[int, Dict]) -> Tuple[Set[Tuple[str, str]], Dict[str, str], Dict[str, int]]:
        """Build target pairs from MILP matrices and supporting lookups.

        Returns:
            - target_pairs: set of (engineer_code, qualification_code)
            - engineer_role: engineer_code -> role ('electrical'/'mechanical')
            - engineer_team: engineer_code -> team (1/2)
        """
        target_pairs: Set[Tuple[str, str]] = set()
        engineer_role: Dict[str, str] = {}
        engineer_team: Dict[str, int] = {}

        for team, eng_map in matrices.items():
            for eng_code, eng_data in eng_map.items():
                role = eng_data.get('role')
                quals = eng_data.get('qualifications', [])
                if role:
                    engineer_role[eng_code] = role
                engineer_team[eng_code] = team
                for qual in quals:
                    target_pairs.add((eng_code, qual))

        return target_pairs, engineer_role, engineer_team

    def _ppm_type_from_qualification(self, qual_code: str) -> str:
        if '.2.D' in qual_code:
            return 'DAILY'
        if '.3.W' in qual_code:
            return 'WEEKLY'
        if '.4.M' in qual_code or '.4.Q' in qual_code:
            return 'MONTHLY'
        if '.5.' in qual_code:
            return 'REACTIVE'
        return 'UNKNOWN'

    def _ride_from_qualification(self, qual_code: str) -> str:
        return qual_code.split('.')[0] if '.' in qual_code else qual_code

    def _write_rows(self, output_path: Path, rows: List[Dict[str, object]]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            'level', 'team', 'role', 'ride_code', 'qualification_code', 'ppm_type',
            'target_pairs', 'satisfied_pairs', 'percent_complete'
        ]
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def compute_and_export_health(self, matrices: Dict[int, Dict], output_file: str = 'outputs/current/Health.csv') -> str:
        """Compute health metrics and export a single CSV with multi-level rows.

        Args:
            matrices: MILP qualification matrices {team: {eng_code: {...}}}
            output_file: target CSV path
        Returns:
            The output file path as string
        """
        target_pairs, engineer_role, engineer_team = self._build_target_pairs(matrices)
        current_pairs = self._load_active_current_pairs()

        def pct(num: int, den: int) -> float:
            return round((num / den) * 100.0, 1) if den > 0 else 0.0

        rows: List[Dict[str, object]] = []

        # Precompute indexes
        # By team
        teams = sorted(set(engineer_team.values()))

        # TEAM level
        for team in teams:
            T_team = {(e, q) for (e, q) in target_pairs if engineer_team.get(e) == team}
            C_team = T_team & current_pairs
            rows.append({
                'level': 'TEAM', 'team': team, 'role': '', 'ride_code': '', 'qualification_code': '',
                'ppm_type': '', 'target_pairs': len(T_team), 'satisfied_pairs': len(C_team),
                'percent_complete': pct(len(C_team), len(T_team))
            })

        # TEAM_ROLE level
        for team in teams:
            for role in ['electrical', 'mechanical']:
                T_tr = {(e, q) for (e, q) in target_pairs if engineer_team.get(e) == team and engineer_role.get(e) == role}
                C_tr = T_tr & current_pairs
                rows.append({
                    'level': 'TEAM_ROLE', 'team': team, 'role': role.title(), 'ride_code': '', 'qualification_code': '',
                    'ppm_type': '', 'target_pairs': len(T_tr), 'satisfied_pairs': len(C_tr),
                    'percent_complete': pct(len(C_tr), len(T_tr))
                })

        # RIDE level (per team, aggregated across roles)
        # Collect all ride codes present in target
        ride_codes = sorted({self._ride_from_qualification(q) for (_, q) in target_pairs})
        for team in teams:
            for ride in ride_codes:
                T_ride = {(e, q) for (e, q) in target_pairs
                          if engineer_team.get(e) == team and q.startswith(ride + '.')}
                if not T_ride:
                    continue
                C_ride = T_ride & current_pairs
                rows.append({
                    'level': 'RIDE', 'team': team, 'role': '', 'ride_code': ride, 'qualification_code': '',
                    'ppm_type': '', 'target_pairs': len(T_ride), 'satisfied_pairs': len(C_ride),
                    'percent_complete': pct(len(C_ride), len(T_ride))
                })

        # QUALIFICATION level (per team, aggregated across roles)
        qual_codes = sorted({q for (_, q) in target_pairs})
        for team in teams:
            for qual in qual_codes:
                T_qual = {(e, q) for (e, q) in target_pairs if engineer_team.get(e) == team and q == qual}
                if not T_qual:
                    continue
                C_qual = T_qual & current_pairs
                rows.append({
                    'level': 'QUALIFICATION', 'team': team, 'role': '', 'ride_code': self._ride_from_qualification(qual),
                    'qualification_code': qual, 'ppm_type': self._ppm_type_from_qualification(qual),
                    'target_pairs': len(T_qual), 'satisfied_pairs': len(C_qual),
                    'percent_complete': pct(len(C_qual), len(T_qual))
                })

        output_path = Path(output_file)
        self._write_rows(output_path, rows)
        return str(output_path)


