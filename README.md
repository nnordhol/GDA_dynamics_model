# GDA Dynamics Model

ODE-based population dynamics model for gene duplication–amplification (GDA) mediated heteroresistance, accompanying the paper:

> **Gene amplification mediated heteroresistance delays fixation of stable resistance but rescues small populations under antibiotic stress**
> Johannes Kupke\*, Yuwen Fang\*, Fereshteh Ghazisaeedi, Dennis Hanke, Frank Schreiber, Karsten Tedin, Niclas Nordholt† & Marcus Fulde†

## Overview

The model simulates bacterial population dynamics under antibiotic selection, where cells can acquire gene duplication–amplifications (GDAs) that increase antibiotic resistance (via IC50) at a multiplicative fitness cost per copy. GDA copy number evolves through duplication and recombination (following Pettersson 2005), and point mutations give rise to stably resistant subpopulations.

The state space tracks cell densities for every combination of GDA copy number and genetic background:

| Background | Description |
|---|---|
| Wild-type (WT) | GDA dynamics active |
| *dacB* mutant | High-level resistance, low fitness cost |
| *ampD* mutant | High-level resistance, high fitness cost |

## Repository contents

| File | Description |
|---|---|
| `GDA_ODEvolution_final.py` | Model source code (`Genotype` and `Experiment` classes) |
| `Plots_paper_final.ipynb` | Jupyter notebook reproducing all paper figures and additional usage examples |
| `gda_df.xlsx` | Experimental GDA copy number data |
| `pap_data.xlsx` | Experimental population analysis profile (PAP) assay data |

## Requirements

- Python ≥ 3.9
- `numpy`
- `scipy`
- `matplotlib`
- `pandas` (notebook only)
- `seaborn` (notebook only)

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas seaborn openpyxl
```

## Quick start

```python
from GDA_ODEvolution_final import Genotype, Experiment

# Create a founding genotype
g = Genotype(
    mumax=1.7,          # maximum growth rate (h⁻¹)
    gda_cost=1e-3,      # multiplicative fitness cost per GDA copy
    gda_benefit=3,      # IC50 increase per GDA copy
    k_dup=6e-4,         # GDA duplication rate per division
    k_rec=0.2,          # recombination rate constant
    min_dr=-0.1,        # lower asymptote of dose–response curve
    hill=8,             # Hill coefficient
)

# Set up a serial-dilution evolution experiment at 32 µg/ml antibiotic
exp = Experiment([g], volume=10, sample_volume=0.1, gluc_start=40)
exp.set_antibiotic(32)
exp.conduct(20)  # run 20 serial dilution cycles

# Access results
import matplotlib.pyplot as plt
import numpy as np

plt.plot(exp.genotypes[0].ts / 24, exp.from_genos('n').sum(axis=1)[:-1])
plt.yscale('log')
plt.xlabel('Time (days)')
plt.ylabel('Total population size')
plt.show()
```

### Population analysis profile (PAP) assay

```python
exp.pap_assay(start_ab=1, end_ab=256)
```

### Comparing GDA⁺ and GDA⁻ strains

Disable GDA dynamics by setting `k_dup=0` and `k_rec=0`:

```python
g_no_gda = Genotype(mumax=1.7, gda_cost=1e-3, gda_benefit=3,
                    k_dup=0, k_rec=0, min_dr=-0.1, hill=8)
exp_no_gda = Experiment([g_no_gda], volume=10, sample_volume=0.1)
```

## Model description

### Fitness model

Each genotype has a Hill-type dose–response function:

```
fitness(ab) = min_dr + (max_dr − min_dr) / (1 + (ab / IC50)^hill)
```

IC50 scales linearly with GDA copy number (`IC50 = gda_benefit × gdas`). The multiplicative cost model (Pal & Andersson) reduces fitness by `(1 − gda_cost)^(gdas − 1)` for each additional GDA copy.

### GDA copy number dynamics

Copy number changes through two processes:
- **Duplication**: cells gain one copy at rate `k_dup` per division (wild-type background only)
- **Recombination**: cells lose copies at a copy-number-dependent rate following Pettersson (2005) Eqs. 1, 4–5

### Resource and population dynamics

Growth follows Monod kinetics. The full ODE system is integrated with `scipy.integrate.solve_ivp` for each serial dilution cycle.

## Key references

- Pettersson, M. E. *et al.* (2005) — recombination probability model
- Pal, C. & Andersson, D. I. — multiplicative fitness cost model for GDA burden

## Citation

If you use this model, please cite:

> Kupke J, Fang Y, Ghazisaeedi F, Hanke D, Schreiber F, Tedin K, Nordholt N & Fulde M. Gene amplification mediated heteroresistance delays fixation of stable resistance but rescues small populations under antibiotic stress. (2026)
