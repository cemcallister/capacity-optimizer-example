# Theme Park Capacity Optimizer

Production workforce optimization system solving the NP-hard problem of assigning maintenance engineers to theme park rides while guaranteeing 100% coverage of Planned Preventive Maintenance (PPM) schedules across rotating shift patterns.

## The Problem

Theme parks require daily, weekly, and monthly maintenance on every ride. Engineers work rotating shifts (9-week and 18-week cycles), have different qualifications, and can only perform maintenance when physically present. This creates a constrained assignment problem with ~10^50 possible combinations.

**Constraints:**
- 31 rides across 2 operational teams
- 62 engineers with ~180 unique certifications
- Daily PPMs must complete in 3-hour morning window
- Type A (complex) rides require exactly 2 qualified engineers each
- Workload must be evenly distributed

## Solution

**Mixed Integer Linear Programming (MILP)** with 36-week full rotation validation (LCM of shift cycles).

```python
# Objective: Fairness + Efficiency + Training Preservation
minimize(
    10 * (max_rides - min_rides) +    # Fairness (weighted)
    0.01 * total_assignments +         # Efficiency
    -0.5 * training_bias               # Preserve ongoing training
)
```

## Quick Start

```bash
pip install -r requirements.txt
python3 run_optimization.py      # Select Option 1 for MILP
python3 validate_qualifications.py
```

## Tech Stack

- **Python 3.8+** - Core implementation
- **PuLP** - MILP optimization with CBC solver
- **pandas/numpy** - Data processing
- **Power BI** - Dashboard integration (CSV exports)

## Architecture

```
EngQual.csv + PPM Schedules + Shift Rotas
                ↓
      PPMCapacityOptimizer (data loading)
                ↓
    MILPOptimizationDesigner (PuLP/CBC)
                ↓
      CoverageValidator (36-week test)
                ↓
         Power BI CSVs
```

**Key modules:**
- `src/analysis/milp_optimization_designer.py` - Core MILP formulation
- `src/analysis/ppm_capacity_optimizer.py` - Data loading engine
- `src/analysis/coverage_validator.py` - 36-week rotation validation

## Results

Achieves 100% coverage on all PPM types with mathematically optimal fairness distribution.
