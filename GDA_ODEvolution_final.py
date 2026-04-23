#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDA_ODEvolution.py — ODE-based model for gene duplication amplifications (GDAs) accompanying paper by Kupke et al, 2026.

Models population dynamics under antibiotic selection pressure where cells can acquire
gene duplication-amplifications (GDAs) that increase antibiotic resistance (via IC50) at a
multiplicative fitness cost per GDA copy. Recombination dynamics follow Pettersson (2005).

Simulations track cell densities for each combination of (GDA copy number, genetic
background). Backgrounds are: wild-type (wt), dacB mutant, and ampD mutant, where the
latter two carry point mutations that confer high-level resistance but a fixed fitness cost.


Classes
-------
Genotype
    Represents a bacterial genotype with a defined number of GDA copies.
Experiment
    Manages a population of Genotype objects through serial dilution cycles,
    integrating the ODE system each cycle.

Key references
--------------
Pettersson, M. E. et al. (2005) — recombination probability model (Eqs 1, 4, 5).
Pal & Andersson — multiplicative fitness cost model for GDA burden.

Author: nnordholt
"""

from scipy.integrate import solve_ivp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import warnings
from copy import deepcopy


# ---------------------------------------------------------------------------
# Recombination functions (Pettersson 2005)
# ---------------------------------------------------------------------------

def v_rec(gda_n: int, k_rec: float) -> float:
    """Probability of recombination for a cell carrying gda_n GDA copies (Eq. 1).

    Parameters
    ----------
    gda_n : int
        Number of GDA copies in the cell.
    k_rec : float
        Recombination rate constant.

    Returns
    -------
    float
        Probability that a recombination event occurs at division.
    """
    return (k_rec * (gda_n - 1)) / (1 + k_rec * (gda_n - 1))


def recombine(k_rec: float, m: int, n: int) -> float:
    """Probability of producing n GDA copies from a cell with m copies upon recombination.

    Implements Pettersson (2005) Eqs 4–5.

    Parameters
    ----------
    k_rec : float
        Recombination rate constant (not used in this formula but kept for API
        consistency with the rest of the recombination module).
    m : int
        GDA copy number of the parent cell.
    n : int
        GDA copy number of the daughter cell after recombination.

    Returns
    -------
    float
        Probability p(n | m).
    """
    if n > 2 * m:
        return 0
    if n < m:
        p_n_m = n / m ** 2
    if n >= m:
        p_n_m = (2 * m - n) / m ** 2
    return p_n_m


def r_rec(ns: np.ndarray, gda_n: int, gda_max_m: int, k_rec: float, mus: np.ndarray) -> float:
    """Rate of change in cell number with gda_n copies due to recombination.

    Parameters
    ----------
    ns : np.ndarray
        Array of cell densities indexed by GDA copy number (length gda_max_m).
    gda_n : int
        Target GDA copy number.
    gda_max_m : int
        Maximum GDA copy number present in the population.
    k_rec : float
        Recombination rate constant.
    mus : np.ndarray
        Growth rates for each GDA class.

    Returns
    -------
    float
        Net rate of cells entering the gda_n class from recombination.
    """
    return sum(
        mus[m] * ns[m] * v_rec(m + 1, k_rec) * recombine(k_rec, m + 1, gda_n)
        for m in range(gda_max_m)
    )


def r_rec_factor(gda_n: int, gda_max_m: int, k_rec: float) -> float:
    """Pre-compute the recombination flux factor into GDA class gda_n (growth-rate-independent).

    Unlike r_rec, this function omits cell density and growth rate so the result
    can be used as a fixed coefficient that is multiplied by (n * mu) at runtime.

    Parameters
    ----------
    gda_n : int
        Target GDA copy number.
    gda_max_m : int
        Maximum GDA copy number considered.
    k_rec : float
        Recombination rate constant.

    Returns
    -------
    float
        Sum of recombination flux factors from all source classes into gda_n.
    """
    return sum(v_rec(m, k_rec) * recombine(k_rec, m, gda_n) for m in range(1, gda_max_m + 1))


def rec_matrix(gda_max_m: int, k_rec: float) -> np.ndarray:
    """Build the recombination transition matrix.

    Entry [m, n] gives the rate at which cells in GDA class m produce cells in
    class n per unit of (cell density × growth rate). The matrix is transposed so
    that a dot product with the (density × growth rate) vector directly yields the
    recombination flux into each GDA class.

    Parameters
    ----------
    gda_max_m : int
        Maximum GDA copy number (sets matrix dimension gda_max_m × gda_max_m).
    k_rec : float
        Recombination rate constant.

    Returns
    -------
    np.ndarray
        Shape (gda_max_m, gda_max_m) transition matrix (source classes as rows).
    """
    matrix = np.array(
        [[v_rec(m, k_rec) * recombine(k_rec, m, n) for m in range(1, gda_max_m + 1)]
         for n in range(1, gda_max_m + 1)]
    )
    return matrix.T


# ---------------------------------------------------------------------------
# Fitness / dose-response helper functions
# ---------------------------------------------------------------------------

def calc_ic50(gda: float, y: float = 0.5) -> float:
    """IC50 as a linear function of GDA copy number.

    Parameters
    ----------
    gda : float
        Number of GDA copies.
    y : float
        Benefit (IC50 increase) per GDA copy.

    Returns
    -------
    float
        IC50 value.
    """
    return gda * y


def calc_ic50_from_mic(mic: float = 1, min: float = -0.01, max: float = 1, p: float = 2) -> float:
    """Invert the dose-response MIC formula to obtain IC50.

    Parameters
    ----------
    mic : float
        Minimum inhibitory concentration.
    min : float
        Lower asymptote of the dose-response curve (must be < 0 for MIC to exist).
    max : float
        Upper asymptote of the dose-response curve.
    p : float
        Hill coefficient.

    Returns
    -------
    float
        IC50 corresponding to the given MIC.
    """
    ic50 = mic / (-max / min) ** (1 / p)
    return ic50


def calc_mic(gda: float = 1, y: float = 0.5, l: float = 0.007,
             min: float = -0.01, max: float = 1, p: float = 2) -> float:
    """Calculate the MIC for a cell with a given GDA copy number.

    Uses the multiplicative fitness cost formulation. The MIC is derived from the
    antibiotic concentration at which fitness (dose-response × cost term) equals zero.

    Parameters
    ----------
    gda : float
        Number of GDA copies.
    y : float
        IC50 benefit per GDA copy.
    l : float
        Multiplicative cost per GDA copy (fitness multiplied by (1-l)^(gda-1)).
    min : float
        Lower asymptote of the dose-response curve (must be < 0).
    max : float
        Upper asymptote of the dose-response curve.
    p : float
        Hill coefficient.

    Returns
    -------
    float
        Minimum inhibitory concentration.
    """
    ic50 = gda * y
    gda_burden = gda * l
    mic = ic50 * ((gda_burden - max) / (min - gda_burden)) ** (1 / p)
    return mic


def dose_resp(ab: float, min: float = -0.01, max: float = 1,
              ic50: float = 0.05, p: float = 2) -> float:
    """Hill-type dose-response function.

    Parameters
    ----------
    ab : float
        Antibiotic concentration.
    min : float
        Lower asymptote (minimum fitness, typically < 0).
    max : float
        Upper asymptote (maximum fitness at zero antibiotic).
    ic50 : float
        Concentration at which response equals min + (max - min) / 2.
    p : float
        Hill coefficient (steepness of the response curve).

    Returns
    -------
    float
        Fitness value at antibiotic concentration ab.
    """
    return min + (max - min) / (1 + (ab / ic50) ** p)


# ---------------------------------------------------------------------------
# Genotype class
# ---------------------------------------------------------------------------

class Genotype:
    """A bacterial genotype characterised by its GDA copy number and background.

    Parameters
    ----------
    mumax : float
        Maximum growth rate (h⁻¹).
    w : float
        Relative fitness (updated dynamically by w_antibiotic).
    e : float
        Yield coefficient (resource consumed per cell produced).
    km : float
        Monod half-saturation constant for the resource.
    n0 : float
        Initial cell density.
    gid : str
        Genotype identifier string.
    mr : float
        Point-mutation rate (mutations per cell per generation).
    gdas : int
        Initial GDA copy number.
    gda_benefit : float
        IC50 increase per GDA copy (y in IC50 = y * gdas).
    gda_cost : float
        Multiplicative fitness cost per GDA copy above 1; fitness is multiplied
        by (1 - gda_cost)^(gdas - 1).
    k_dup : float
        GDA duplication rate (probability of gaining one copy per division).
    k_rec : float
        Recombination rate constant (see Pettersson 2005, Eq. 1).
    min_dr : float
        Minimum of the dose-response curve (must be < 0 for MIC to be defined).
    max_dr : float
        Maximum of the dose-response curve (= 1 for wild-type, < 1 for mutants
        with a growth cost).
    hill : float
        Hill coefficient of the dose-response curve.
    mutant_cost : float
        Growth-rate cost for point-mutation backgrounds (reduces max_dr).
    mutant_ic50 : float
        IC50 for point-mutation backgrounds (overrides GDA-based IC50).
    mutant : str or False
        Mutation identifier ('dacb', 'ampd') or False for wild-type background.
    mutant_dict : dict or False
        Mapping of mutation names to (mutation_rate, cost, ic50) tuples.
        Defaults to empirically derived values for dacB and ampD.
    max_gdas : int
        Maximum GDA copy number tracked. If 0, inferred as ceil(1 / gda_cost).
    """

    def __init__(self, mumax=2, w=1, e=5e-9, km=0.25, n0=1, gid='WT', mr=3e-8,
                 gdas=1, gda_benefit=1.5, gda_cost=0.002,
                 k_dup=1e-4, k_rec=0.15, min_dr=-0.5, max_dr=1, hill=2,
                 mutant_cost=0.01, mutant_ic50=20, mutant=False,
                 mutant_dict=False, max_gdas=300):

        self.gid = gid
        self.mumax = mumax
        self.e = e
        self.km = km
        # Time-series attributes are stored as Python lists during simulation
        # and converted to numpy arrays by _finalize_arrays() at the end of
        # conduct() / conduct_steps(), so external code always sees numpy arrays.
        self.n = [n0]
        self.ts = []
        self.stepdy = []
        self.survivors = []
        self.mr = mr
        self.extinct = False
        self.emerged = 0
        self.gdas = gdas
        self.gda_benefit = gda_benefit
        self.gda_cost = gda_cost
        self.k_dup = k_dup
        self.k_rec = k_rec
        self.min_dr = min_dr
        self.max_dr = max_dr
        self.hill = hill
        self.mutant_cost = mutant_cost
        self.mutant_ic50 = mutant_ic50
        self.max_gdas = int(np.ceil(1 / self.gda_cost)) if max_gdas == 0 else max_gdas
        self.mutant = mutant
        if mutant_dict:
            self.mutant_dict = mutant_dict
        else:
            # Mutation rates proportional to gene size (dacB: 1434 bp, ampD: 564 bp).
            # Tuples: (mutation_rate, fitness_cost, IC50).
            self.mutant_dict = {
                'dacb': (3e-8 * (1434 / (564 + 1434)), 0.01, 350),
                'ampd': (3e-8 * (564 / (564 + 1434)), 0.18, 700),
            }
        self._calc_mic()
        self.w_antibiotic(0)  # Initialise fitness without antibiotic.

    def w_antibiotic(self, ab: float):
        """Compute and store fitness as a function of GDA copy number and antibiotic.

        Uses the multiplicative cost model (Pal & Andersson):
            fitness = dose_resp(ab) × (1 − gda_cost)^(gdas − 1)

        Parameters
        ----------
        ab : float
            Antibiotic concentration.

        Returns
        -------
        float or np.ndarray
            Fitness value(s); stored in self.w.
        """
        fitness = self._dose_resp(ab) * (1 - self.gda_cost) ** (self.gdas - 1)

        fitness = np.asarray(fitness)
        if fitness.ndim == 0:
            fitness = fitness[None]
            if fitness < 0:
                self.w = 0
                return 0
            else:
                self.w = fitness[0]
                return fitness[0]
        fitness = np.where((fitness < 0) | (np.isnan(fitness)), 0, fitness)
        self.w = fitness
        return fitness

    def update(self, ab: float) -> None:
        """Recompute MIC and fitness after a parameter change.

        Should be called after modifying gda_benefit, gda_cost, gdas, or
        any dose-response parameter.

        Parameters
        ----------
        ab : float
            Current antibiotic concentration.
        """
        self._calc_mic()
        self.w_antibiotic(ab)

    def growth(self, r: float) -> float:
        """Monod growth rate (per cell, independent of density).

        Parameters
        ----------
        r : float
            Resource (glucose) concentration.

        Returns
        -------
        float
            Growth rate: w × mumax × r / (r + km).
        """
        return self.w * self.mumax * r / (r + self.km)

    def _dose_resp(self, ab: float) -> float:
        """Evaluate the Hill-type dose-response curve for this genotype.

        For mutant backgrounds, max_dr is reduced by mutant_cost and ic50 is
        set to mutant_ic50 (overriding the GDA-based calculation).

        Parameters
        ----------
        ab : float
            Antibiotic concentration.

        Returns
        -------
        float
            Fitness value from the dose-response curve.
        """
        max_dr = self.max_dr
        if self.mutant:
            max_dr = 1 - self.mutant_cost
        return self.min_dr + (max_dr - self.min_dr) / (1 + (ab / self.ic50) ** self.hill)

    def _calc_mic(self, a: float = 1) -> float:
        """Compute and store IC50 and MIC for this genotype.

        IC50 scales linearly with GDA copy number. MIC is derived by finding the
        antibiotic concentration at which fitness reaches zero.

        Under the multiplicative fitness model, fitness = dose_resp(ab) × (1-c)^(n-1).
        Since (1-c)^(n-1) > 0, setting fitness = 0 reduces to dose_resp(MIC) = 0.
        With min_dr < 0 (drug can kill), this always has a solution. The cost
        factor cancels and MIC scales with GDA copy number only through IC50.

        Parameters
        ----------
        a : float
            Exponent for GDA copy number in the IC50 formula (default 1 = linear).

        Returns
        -------
        float
            Minimum inhibitory concentration (stored in self.mic).
        """
        if self.mutant:
            self.ic50 = self.mutant_ic50
        else:
            self.ic50 = self.gda_benefit * self.gdas ** a

        # MIC is defined as the concentration where dose_resp(MIC) = 0.
        # Under the multiplicative fitness model (fitness = dose_resp × c, c > 0),
        # the cost factor cancels and MIC depends only on IC50, min_dr, max_dr,
        # and hill — not on gda_cost. MIC still scales with GDA copy number via
        # IC50 = gda_benefit × gdas, which is the intended biological dependence.
        max_dr_eff = (1 - self.mutant_cost) if self.mutant else self.max_dr
        mic = self.ic50 * (-max_dr_eff / self.min_dr) ** (1 / self.hill)
        self.mic = 0 if type(mic) is complex else mic
        return self.mic

    def _calc_ic50_from_mic(self) -> float:
        """Invert the dose-response curve to recover IC50 from the stored MIC.

        Returns
        -------
        float
            IC50 value (also stored in self.ic50).
        """
        ic50 = self.mic / (-self.max_dr / self.min_dr) ** (1 / self.hill)
        self.ic50 = ic50
        return ic50

    def get_pars(self) -> list:
        """Return core ODE parameters as a list.

        Returns
        -------
        list
            [n, mumax, km, e]
        """
        return [self.n, self.mumax, self.km, self.e]



# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """Serial-dilution evolution experiment with ODE-based population dynamics.

    The state space is a vector of cell densities for each (GDA copy number,
    genetic background) combination, plus resource concentration. At the end of
    each cycle cells are diluted (or killed then diluted) to simulate serial
    passage.

    Parameters
    ----------
    genotypes : list of Genotype or None
        Founding genotype(s). If None, a single default Genotype() is used.
        The GDA copy-number range and mutant backgrounds are inferred from the
        first genotype.
    volume : float
        Total culture volume (ml).
    sample_volume : float
        Volume transferred at each dilution (ml).
    predilution : float
        Pre-dilution factor applied before the killing step (when killing=True).
    gluc_start : float
        Initial glucose concentration (µM or arbitrary units).
    t_span : list of float
        [t_start, t_end] for each ODE integration cycle (h).
    killing : bool
        If True, apply a killing step (using survival probability) before
        dilution. If False, dilute directly.
    poisson : bool
        If True, sample cell counts with Poisson noise at each dilution.
    antibiotic : float
        Antibiotic concentration applied to all genotypes.
    allow_extinction : bool
        If True, genotypes with fewer than 1 cell after dilution are set to 0.
    """

    def __init__(self, genotypes=None, volume=1, sample_volume=0.1, predilution=1,
                 gluc_start=40, t_span=None, killing=False, poisson=False,
                 antibiotic=0, allow_extinction=True):

        self.genotypes = [Genotype()] if genotypes is None else genotypes
        self.cycles = 0
        self.volume = volume
        self.sample_volume = sample_volume
        self.dilution = volume / sample_volume
        self.predilution = predilution
        self.gluc_start = gluc_start
        self.gluc = np.array([gluc_start])
        self.antibiotic = antibiotic
        self.t_span = [0, 24] if t_span is None else t_span
        self.ts = np.zeros((0, 1))
        self.poisson = poisson
        self.allow_extinction = allow_extinction
        self.initialize_genotypes()
        self.update_genotypes()
        self.sols = np.zeros((0, len(self.genotypes) + 1))
        self.killing = killing
        self.survivors = np.zeros((0, len(self.genotypes)))
        self.fractions = np.zeros((0, len(self.genotypes)))
        self.total = np.zeros((0, len(self.genotypes)))
        self.create_rec_vec()

    def update_genotypes(self) -> None:
        """Update MIC and fitness for all genotypes at the current antibiotic concentration.

        Also refreshes the cached per-background parameter arrays used in the ODE.
        """
        for g in self.genotypes:
            g.update(self.antibiotic)
        self._cache_geno_arrays()

    def _cache_geno_arrays(self) -> None:
        """Cache per-background numpy arrays of w, mumax, km, and e.

        Called after any fitness update so that pop_ivp_mut can compute growth
        rates with a single vectorized operation instead of a Python loop over
        all genotype objects.

        Stored as dicts keyed by background name (same keys as _backgrounds).
        """
        self._bg_w     = {}
        self._bg_mumax = {}
        self._bg_km    = {}
        self._bg_e     = {}
        for bg, genos in self._backgrounds.items():
            self._bg_w[bg]     = np.array([g.w     for g in genos])
            self._bg_mumax[bg] = np.array([g.mumax for g in genos])
            self._bg_km[bg]    = np.array([g.km    for g in genos])
            self._bg_e[bg]     = np.array([g.e     for g in genos])

    def create_rec_vec(self) -> None:
        """Pre-compute the recombination matrix and per-class recombination probabilities.

        rec_vec : public attribute, shape (max_gdas, max_gdas).
            Entry [m, n] is the flux from GDA class m into class n per unit
            of (cell density × growth rate).
        _rec_mat : internal attribute, rec_vec transposed.
            Stored in the orientation where row n gives the total incoming flux
            into GDA class n, so that _rec_mat @ (y * mu) directly yields the
            recombination flux vector without a runtime transpose.
        v_recs : recombination probability for each GDA class (length max_gdas).
        """
        g = self.genotypes[0]
        self.rec_vec = rec_matrix(self.max_gdas, g.k_rec)
        self._rec_mat = self.rec_vec.T
        self.v_recs = v_rec(np.array(range(1, self.max_gdas + 1)), g.k_rec)

    def set_antibiotic(self, a: float) -> None:
        """Set antibiotic concentration and update all genotype fitnesses.

        Parameters
        ----------
        a : float
            New antibiotic concentration.
        """
        self.antibiotic = a
        self.update_genotypes()

    def initialize_genotypes(self) -> None:
        """Expand the genotype list to cover all GDA copy numbers and mutant backgrounds.

        Creates one Genotype instance per (GDA copy number, background) combination.
        The GDA range is [1, max_gdas] and backgrounds are wt + all entries in
        mutant_dict of the founding genotype. All non-founding genotypes start at
        density 0.

        Also caches the background dict (_backgrounds) since the genotype list
        is static after initialisation.

        Notes
        -----
        This is called once at Experiment initialisation and is static — the
        maximum GDA number does not change during the experiment.
        """
        g = self.genotypes[0]
        max_gdas = max([g.max_gdas for g in self.genotypes])
        genotypes = []
        wts = [g] + [deepcopy(g) for i in np.arange(max_gdas - 1)]
        for i, w in enumerate(wts[1:]):
            w.gdas = i + 2
            w.n = [0]
        genotypes += wts
        for mutant, props in g.mutant_dict.items():
            muts = deepcopy(wts)
            for i, w in enumerate(muts):
                w.n = [0]
                w.mutant = mutant
                w.mutant_cost = props[1]
                w.mutant_ic50 = props[2]
            genotypes += muts
        self.genotypes = genotypes
        self.max_gdas = max_gdas

        # Cache background dict once — slices are valid for the lifetime of
        # this experiment (genotype list does not change except via add_genotypes,
        # which calls _rebuild_backgrounds).
        self._rebuild_backgrounds()

    def _rebuild_backgrounds(self) -> None:
        """Rebuild and cache the background→genotype-list mapping.

        Should be called whenever self.genotypes is modified (i.e. after
        initialize_genotypes and after add_genotypes).

        Caching the background dict here avoids rebuilding it on every ODE
        evaluation; it is only rebuilt when the genotype list changes.
        """
        bgs = {}
        bgs['wt'] = self.genotypes[:self.max_gdas]
        for i, mutant in enumerate(self.genotypes[0].mutant_dict):
            bgs[mutant] = self.genotypes[(i + 1) * self.max_gdas:(i + 2) * self.max_gdas]
        self._backgrounds = bgs

    def append_genos(self, geno_att: str, toappend) -> None:
        """Append values to a named attribute of each non-extinct genotype.

        For list attributes (n, ts, stepdy, survivors) values are extended
        in O(k) time. For numpy attributes (mutations, mutation_times) np.append
        is used as before.

        Parameters
        ----------
        geno_att : str
            Name of the Genotype attribute to append to.
        toappend : array-like
            Values to append (one element per non-extinct genotype, consumed in order).
        """
        if not hasattr(self.genotypes[0], geno_att):
            print(f'{geno_att} not an attribute of class Genotype().')
            return
        toappend = list(toappend)
        idx = 0
        for g in self.genotypes:
            if not g.extinct:
                val = toappend[idx]
                attr = g.__dict__[geno_att]
                if isinstance(attr, list):
                    g.__dict__[geno_att].extend(np.atleast_1d(val).tolist())
                else:
                    g.__dict__[geno_att] = np.append(attr, val)
                idx += 1

    def set_all_geno_att(self, geno_att: str, vals) -> None:
        """Set a named attribute to given value(s) for all genotypes.

        Parameters
        ----------
        geno_att : str
            Name of the Genotype attribute to set.
        vals : array-like
            Values to assign (one per genotype, or a single shared value).
        """
        if not hasattr(self.genotypes[0], geno_att):
            print(f'{geno_att} not an attribute of class Genotype().')
            return
        if len(self.genotypes) > 1:
            for i, g in enumerate([self.genotypes]):
                g.__dict__[geno_att] = vals[i] if len(vals) > 1 else vals
        else:
            g = self.genotypes[0]
            g.__dict__[geno_att] = vals[0]

    def from_genos(self, geno_att: str) -> np.ndarray:
        """Collect a named attribute from all genotypes into a 2D array.

        Parameters
        ----------
        geno_att : str
            Name of the Genotype attribute to retrieve.

        Returns
        -------
        np.ndarray
            Array of shape (time_points, n_genotypes), or prints an error if
            the attribute does not exist.
        """
        if not hasattr(self.genotypes[0], geno_att):
            print(f'{geno_att} not an attribute of class Genotype().')
            return
        return np.array([g.__dict__[geno_att] for g in self.genotypes]).T

    def get_wt(self) -> list:
        """Return all wild-type genotypes (GDA copies 1 … max_gdas)."""
        return self.genotypes[:self.max_gdas]

    def get_mutants(self) -> dict:
        """Return a dict mapping mutant name → list of Genotype objects."""
        mutants = {}
        for i, mutant in enumerate(self.genotypes[0].mutant_dict):
            mutants[mutant] = self.genotypes[(i + 1) * self.max_gdas:(i + 2) * self.max_gdas]
        return mutants

    def get_backgrounds(self) -> dict:
        """Return all genotype lists grouped by genetic background.

        Returns the cached background dict. To force a rebuild (e.g. after
        manually modifying self.genotypes), call _rebuild_backgrounds() first.

        Returns
        -------
        dict
            Keys: 'wt', 'dacb', 'ampd' (or whatever is in mutant_dict).
            Values: list of Genotype objects for each GDA class in that background.
        """
        return self._backgrounds

    def dilute(self, last_n=None) -> None:
        """Perform the dilution step at the end of a growth cycle.

        Samples a fraction (sample_volume / volume) of the population, resets
        glucose to gluc_start, and stores survivors.

        Parameters
        ----------
        last_n : np.ndarray or None
            Final cell densities to dilute from. If None, taken from the last
            recorded density of each non-extinct genotype.
        """
        if last_n is None:
            last_n = np.array([g.n[-1] for g in self.get_not_extinct()])
        self.append_genos('survivors', last_n)

        lam = last_n * self.sample_volume
        if self.poisson:
            sample = stats.poisson(lam).rvs()
            if type(sample) == int:
                sample = np.array([sample])
        else:
            sample = lam

        if self.allow_extinction:
            diluted = np.where(sample >= 1, sample / self.volume, 0)
        else:
            diluted = sample / self.volume

        self.append_genos('n', diluted)
        self.append_genos('ts', [self.ts[-1]] * len(self.genotypes))
        self.gluc = np.append(self.gluc, self.gluc_start)

    def kill(self) -> None:
        """Apply killing step (using survival probability) then dilute."""
        survivors = np.array(
            [(g.n[-1] / self.predilution) * g.survival for g in self.get_not_extinct()]
        )
        survivors = np.where(survivors * self.volume >= 1, survivors, 0)
        self.dilute(survivors)

    def get_not_extinct(self) -> list:
        """Return list of genotypes that have not gone extinct."""
        return [g for g in self.genotypes if not g.extinct]

    def create_ode_vec(self) -> np.ndarray:
        """Return the current cell-density vector for all genotypes."""
        return np.transpose([g.n[-1] for g in self.genotypes])

    def pop_ivp_mut(self, t: float, ys: np.ndarray) -> np.ndarray:
        """ODE right-hand side for the full population dynamics with mutations.

        State vector layout: [wt_1, …, wt_M, dacb_1, …, dacb_M, ampd_1, …, ampd_M, R]
        where M = max_gdas and R is the resource concentration.

        This implementation assumes exactly three genetic backgrounds (wt, dacb, ampd)
        as defined in the default mutant_dict. A custom mutant_dict with different keys
        or a different number of backgrounds will raise an error.

        GDA copy number dynamics per background include:
        - Monod growth (vectorized via cached w/mumax/km arrays)
        - GDA duplication (wt → wt+1 at rate k_dup, for wt background only)
        - Recombination (modelled via pre-computed _rec_mat matrix)
        - Point mutations from wt to mutant backgrounds

        Resource dynamics: dR/dt = -Σ_bg [ e_bg · Σ_n (dN_bg_n/dt) ]
        Each background uses its own yield coefficient e.

        Parameters
        ----------
        t : float
            Current time (passed by solve_ivp, unused directly).
        ys : np.ndarray
            Current state vector.

        Returns
        -------
        np.ndarray
            Derivative vector of the same shape as ys.
        """
        r = ys[-1]

        # Vectorized wt growth rates using cached arrays.
        mus_wt = np.maximum(
            self._bg_w['wt'] * self._bg_mumax['wt'] * r / (r + self._bg_km['wt']), 0
        )

        wt_ys, dacb_ys, ampd_ys = [
            ys[i - self.max_gdas:i] for i in range(self.max_gdas, len(ys), self.max_gdas)
        ]
        y_vec = [wt_ys, dacb_ys, ampd_ys]

        dydt = []
        dr_sum = 0.0  # accumulate resource draw per background separately

        for i, (mutant, genos) in enumerate(self._backgrounds.items()):
            # Total mutation rate out of wt into all mutant backgrounds (wt only).
            mr = sum([v[0] for v in genos[0].mutant_dict.values()]) if mutant == 'wt' else 0

            # Vectorized growth rates using cached arrays.
            mus = np.maximum(
                self._bg_w[mutant] * self._bg_mumax[mutant] * r / (r + self._bg_km[mutant]), 0
            )
            yms = y_vec[i]
            k_rec = genos[0].k_rec
            k_dup = genos[0].k_dup

            # Recombination flux via pre-transposed matrix multiply.
            recs = self._rec_mat @ (yms * mus)

            if mutant == 'wt':
                d1dt = mus[0] * yms[0] + recs[0] - mus[0] * yms[0] * (k_dup + mr)
                d2dt = (mus[1] * yms[1] + recs[1]
                        + mus[0] * yms[0] * k_dup
                        - mus[1] * yms[1] * (mr + self.v_recs[1]))
                dgdadt = [
                    mus[n] * yms[n] + recs[n] - mus[n] * yms[n] * (mr + self.v_recs[n])
                    for n in range(2, self.max_gdas)
                ]
            else:
                d1dt = (mus[0] * yms[0] + recs[0]
                        - mus[0] * yms[0] * k_dup
                        + mus_wt[0] * wt_ys[0] * genos[0].mutant_dict[mutant][0])
                d2dt = (mus[1] * yms[1] + recs[1]
                        + mus[0] * yms[0] * k_dup
                        - mus[1] * yms[1] * self.v_recs[1]
                        + mus_wt[1] * wt_ys[1] * genos[0].mutant_dict[mutant][0])
                dgdadt = [
                    mus[n] * yms[n] + recs[n]
                    + mus_wt[n] * wt_ys[n] * genos[n].mutant_dict[mutant][0]
                    - mus[n] * yms[n] * self.v_recs[n]
                    for n in range(2, self.max_gdas)
                ]

            bg_dydt = [d1dt, d2dt, *dgdadt]
            dydt += bg_dydt

            # Resource consumption per background using its own yield coefficient.
            dr_sum += np.dot(self._bg_e[mutant], bg_dydt)

        dydt = dydt + [-dr_sum]
        return np.array(dydt)

    def conduct_once(self, t_span=None) -> None:
        """Integrate the ODE system for one growth cycle.

        Results are appended to each genotype's history arrays (n, ts, stepdy)
        and to self.sols, self.gluc, self.ts.

        Parameters
        ----------
        t_span : list of float or None
            [t_start, t_end] for this integration. Defaults to self.t_span.
        """
        if t_span is None:
            t_span = self.t_span

        last_n = np.array([g.n[-1] for g in self.genotypes])
        y0 = np.append(last_n, self.gluc[-1])

        self.sol = solve_ivp(
            self.pop_ivp_mut, t_span, y0=y0,
            atol=1e-10,
            rtol=1e-9,
        )

        self.append_genos('n', self.sol.y[:-1, :])
        self.append_genos('ts', [(self.ts[-1] if self.ts.any() else 0) + self.sol.t] * (len(y0) - 1))
        self.append_genos('stepdy', self.sol.y[:-1, -1] - self.sol.y[:-1, 0])

        self.sols = np.append(self.sols, self.sol.y.T, axis=0)
        self.gluc = np.append(self.gluc, self.sol.y[-1])
        self.ts = np.append(self.ts, (self.ts[-1] if self.ts.any() else 0) + self.sol.t)
        self.days = self.ts / self.t_span[-1]

    def conduct(self, cycles: int = 1) -> None:
        """Run the experiment for a given number of serial dilution cycles.

        Each cycle: integrate ODE → kill (if enabled) or dilute → record.
        Summary statistics (fractions, average GDAs, fitness) are computed
        at the end.

        Parameters
        ----------
        cycles : int
            Number of cycles to run.
        """
        for c in range(cycles):
            self.conduct_once()
            self.cycles += 1
            self.kill() if self.killing else self.dilute()
        self._finalize_arrays()
        self.calc_fractions()
        self.calc_average_gdas()
        self.calc_background_fractions()
        self.calc_average_fitness()

    def conduct_steps(self, cycles: int = 15, steps: int = 25,
                      stop_after_first_extinct: bool = False) -> None:
        """Run the experiment with sub-cycle time steps.

        Each growth cycle is subdivided into `steps` equal intervals. Within
        each interval the ODE is integrated repeatedly until the population has
        stopped growing (net change ≤ 0), then the next interval begins.

        Parameters
        ----------
        cycles : int
            Number of serial dilution cycles.
        steps : int or None
            Number of sub-steps per cycle. If None, each cycle is a single
            integration over the full t_span.
        stop_after_first_extinct : bool
            If True, stop the experiment as soon as any genotype goes extinct.
        """
        if steps is None:
            t_spans = [self.t_span]
        else:
            l, step = np.linspace(self.t_span[0], self.t_span[1], steps,
                                  retstep=True, dtype='float64')
            t_spans = [[0, step]] * (steps - 1)

        for c in range(cycles):
            j = 0
            for t_span in t_spans:
                while (
                    any([g.stepdy[-1] > 0 if len(g.stepdy) > 0 else True
                         for g in self.get_not_extinct()])
                    or j == 0
                ) and (
                    self.ts.size == 0
                    or not np.isclose(self.ts[-1], self.t_span[1] * (self.cycles + 1))
                ):
                    self.conduct_once(t_span=t_span)
                    j += 1
                else:
                    self.append_genos('n', [g.n[-1] for g in self.get_not_extinct()])
                    self.append_genos('ts', [(self.cycles * self.t_span[-1]) + self.t_span[-1]]
                                      * len(self.get_not_extinct()))
                    self.append_genos('stepdy', [0] * len(self.get_not_extinct()))
                    self.ts = np.append(self.ts, [(self.cycles * self.t_span[-1]) + self.t_span[-1]])
                    break

            self.cycles += 1
            self.kill() if self.killing else self.dilute()
            if stop_after_first_extinct:
                if any([g.extinct for g in self.genotypes]):
                    break
        self._finalize_arrays()

    def _finalize_arrays(self) -> None:
        """Convert per-genotype list attributes to numpy arrays.

        Called at the end of conduct() and conduct_steps() so that external
        code always receives numpy arrays from n, ts, stepdy, and survivors,
        while the simulation itself uses Python lists for efficient appending.
        """
        for g in self.genotypes:
            g.n        = np.array(g.n)
            g.ts       = np.array(g.ts)
            g.stepdy   = np.array(g.stepdy)
            g.survivors = np.array(g.survivors)

    def calc_fractions(self) -> None:
        """Compute population fractions (relative abundance) at each time point.

        Notes
        -----
        If allow_extinction=True and the entire population reaches zero, the
        denominator np.sum(frac) is 0 and fractions become NaN. A warning is
        issued when this occurs. NaN fractions indicate population extinction;
        downstream analyses should handle this explicitly (e.g. by masking
        those time points). To avoid NaN fractions, use allow_extinction=False.
        """
        fractions = []
        for frac in self.from_genos('n'):
            total = np.sum(frac)
            if total == 0:
                warnings.warn(
                    'Total population is zero at one or more time points; '
                    'fractions will be NaN. Check allow_extinction setting.',
                    RuntimeWarning, stacklevel=2
                )
                fractions.append(frac / total)  # NaN by design — explicit, not silent
            else:
                fractions.append(frac / total)
        self.fractions = fractions

    def add_genotypes(self, genotypes) -> None:
        """Add new genotype objects to the experiment mid-run.

        Parameters
        ----------
        genotypes : iterable of Genotype
            New genotypes to add; their emergence time is recorded.
        """
        for genotype in genotypes:
            genotype.emerged = self.ts[-1]
            self.genotypes.append(genotype)
        self._rebuild_backgrounds()

    def plot(self, not_extinct: bool = True) -> None:
        """Plot cell density trajectories over time.

        Parameters
        ----------
        not_extinct : bool
            If True, only plot genotypes that have not gone extinct.
        """
        plt.figure()
        genos = self.get_not_extinct() if not_extinct else self.genotypes
        [plt.plot(np.array(g.ts) / 24, g.n[1:]) for g in genos]
        plt.yscale('log')

    def pap_assay(self, start_ab: float = 1, end_ab: float = 1024,
                  stepsize: float = 1, plot: bool = True) -> np.ndarray:
        """Simulate a population analysis profile (PAP) assay.

        Computes the fraction of the population surviving at each antibiotic
        concentration based on MIC values and current frequency distribution.

        Parameters
        ----------
        start_ab : float
            Lowest antibiotic concentration to test (log2 scale start).
        end_ab : float
            Highest antibiotic concentration to test.
        stepsize : float
            Step size in log2 units.
        plot : bool
            If True, plot the PAP curve.

        Returns
        -------
        np.ndarray
            Shape (n_concentrations, 2): [[concentration, surviving_fraction], …].
        """
        arr = np.array([self.from_genos('mic'), self.fractions[-1]])
        sorted_arr = arr[:, arr[0, :].argsort()]
        li = [
            (c, sum(sorted_arr[1, sorted_arr[0] >= c]))
            for c in 2 ** np.arange(np.log2(start_ab), np.log2(end_ab) + 1, stepsize)
        ]
        if plot:
            plt.plot(*zip(*li), '-x')
            plt.yscale('log')
            plt.xscale('log', base=2)
            plt.xlabel('Antibiotic [µg/ml]')
            plt.ylabel('Fraction of population')
        return np.array(li)

    def calc_average_gdas(self) -> None:
        """Compute the frequency-weighted average GDA copy number at each time point."""
        gf = self.from_genos('gdas')
        self.average_gdas = [sum(f * gf) for f in self.fractions]

    def calc_background_fractions(self) -> None:
        """Compute the total fraction in each genetic background at each time point."""
        bgs = len(self.get_backgrounds())
        self.background_fractions = [
            [np.sum(f) for f in np.split(fs, bgs)] for fs in self.fractions
        ]

    def calc_background(self) -> None:
        """Compute total cell density per genetic background at each time point."""
        bgs = len(self.get_backgrounds())
        self.background = [
            [np.sum(f) for f in np.split(fs, bgs)] for fs in self.from_genos('n')
        ]

    def calc_average_fitness(self) -> None:
        """Compute the frequency-weighted average fitness at each time point."""
        w = self.from_genos('w')
        self.average_fitness = [sum(f * w) for f in self.fractions]
