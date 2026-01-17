# Workforce Capacity Optimizer

Training gap analysis using MILP to guarantee maintenance coverage.

## The Problem

Given a fixed workforce with fixed rotating shifts (union-negotiated, can't change), can we guarantee 100% maintenance coverage? If not, what's the minimum training investment to get there?

## What This Tool Does

- Models current workforce qualifications against maintenance requirements
- Validates coverage across all 252 days of the 36-week rotation cycle
- Identifies gaps - which PPMs can't be covered and when
- Recommends exactly who to train on what, prioritised by impact

## What It Doesn't Do

- Doesn't schedule shifts (those are fixed inputs)
- Doesn't assign engineers to tasks daily (that's operational)

## The Output

A training plan that guarantees 100% PPM coverage with minimum training investment.

```
outputs/current/specific_qualifications_needed.csv
```

Shows each engineer, their missing qualifications, and priority scores.

## How It Works

Mixed Integer Linear Programming optimizer using PuLP/CBC. Binary decision variables for engineer-ride assignments, hard constraints for coverage, multi-objective function:

```python
minimize(
    10 * (max_rides - min_rides) +    # Fairness (weighted 10x)
    0.01 * total_assignments +         # Efficiency
    -0.5 * training_bias               # Preserve ongoing training
)
```

## Key Features

- MILP formulation with 7,400+ constraints
- 36-week rotation cycle validation (LCM of 9/18 week shift patterns)
- Training gap analysis with prioritization
- Graceful degradation if solver times out

## Quick Start

```bash
pip install -r requirements.txt
python3 run_optimization.py
```

Select Option 2 for training optimization.

## Tech Stack

- Python 3.8+
- PuLP (MILP solver interface)
- CBC (Coin-or Branch and Cut solver)
- pandas

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
       Training Recommendations
```

## Design Decisions

- **Fairness weighted 10x over efficiency** - Union environment requires equal workload distribution
- **Type A rides = exactly 2 per engineer** - Specialists need depth, not breadth
- **36-week validation, not worst-case** - Catches edge cases where week 36 fails even if weeks 1-35 pass
- **Training sunk cost preservation** - Optimizer biases toward completing in-progress training rather than starting fresh
- **Rotas treated as immutable** - Union-negotiated shifts can't be changed, so optimize assignments around fixed patterns

## Results

Identifies minimum training required to achieve 100% coverage on daily, weekly, and monthly PPMs.
