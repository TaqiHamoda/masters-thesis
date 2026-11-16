import csv
import gc
import math

import numpy as np
import pyceres

from typing import Dict, List, Tuple

from .dataset import Dataset

# Two-way absorption loss (20-log rule on voltage samples): 10^(-2 * alpha * r / 10)
# Its alpha derivative contributes (2 * ln(10) / 10) * r * loss per unit alpha
ALPHA_JAC = 2.0 * np.log(10.0) / 10.0


def chebyshev_basis(u: np.ndarray, degree: int) -> np.ndarray:
    """
    Returns the (len(u), degree + 1) Chebyshev basis matrix T_0..T_degree of the
    normalized pattern angle u = 2 * theta / max_angle - 1 in [-1, 1].

    Chebyshev is used instead of raw Taylor monomials because the monomial basis
    over a finite angle interval is severely ill-conditioned; for a given degree
    the reachable polynomials are identical.
    """
    u = np.asarray(u, dtype=np.float64)
    T = np.empty((u.shape[0], degree + 1), dtype=np.float64)

    T[:, 0] = 1.0
    if degree == 0:
        return T

    T[:, 1] = u
    for k in range(2, degree + 1):
        T[:, k] = 2.0 * u * T[:, k - 1] - T[:, k - 2]

    return T


class VertexObservationCost(pyceres.CostFunction):
    """
    Residuals for all observations of one mesh vertex (Lambertian model with a
    parametric beam pattern):

        res_i = I_i - R_v * cos(theta_i) * G_c[i](theta_i) * 10^(-2 * alpha * r_i / 10)

        G_c(theta) = exp( sum_k c_[c,k] * T_k(2 * theta / max_angle - 1) )

    One residual per observation of the vertex, batched into a single block so
    the number of ceres residual blocks scales with vertices, not observations.

    Parameter blocks:
        [reflectivity (1,), beam coefficients of each channel observed by this
         vertex ((degree + 1,) each), absorption coeff (1,)]
    """

    def __init__(self,
        bases: Tuple[np.ndarray, ...],
        start: int,
        size: int,
        degree: int,
        port_present: bool,
        stbd_present: bool,
    ):
        pyceres.CostFunction.__init__(self)

        beam_sizes = [degree + 1] * (int(port_present) + int(stbd_present))
        self.set_num_residuals(size)
        self.set_parameter_block_sizes([1] + beam_sizes + [1])

        self.bases = bases
        self.start = int(start)
        self.size = int(size)
        self.degree = int(degree)
        self.port_present = bool(port_present)
        self.stbd_present = bool(stbd_present)

    def _channel_slot(self, channel: int) -> int:
        """Index of the given channel's coefficient block in the parameter list."""
        if channel == 0:
            return 1
        return 1 + int(self.port_present)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        start, end = self.start, self.start + self.size

        reflectivity = parameters[0][0]
        alpha = parameters[-1][0]

        # Lambertian scattering factor from the known incidence angle
        scatter = self.bases[1][start:end]
        # Loss factor starts at 1.0 by definition (0 dB at zero range)
        loss = 10.0 ** (-2.0 * alpha * self.bases[2][start:end] / 10.0)

        u = self.bases[3][start:end]
        channels = self.bases[4][start:end]

        T = chebyshev_basis(u, self.degree)

        # Beam gains: log-gain is the pattern polynomial per channel
        log_gain = np.zeros(self.size, dtype=np.float64)
        for channel, present in ((0, self.port_present), (1, self.stbd_present)):
            if not present:
                continue

            coefficients = np.asarray(parameters[self._channel_slot(channel)], dtype=np.float64)
            mask = channels == channel
            if np.any(mask):
                log_gain[mask] = T[mask] @ coefficients
        gain = np.exp(log_gain)

        predicted = reflectivity * scatter * gain * loss

        res = self.bases[0][start:end] - predicted
        try:
            residuals[:] = res
        except TypeError:
            for i in range(len(res)):
                residuals[i] = res[i]

        if jacobians is not None:
            # d res / d R = -cos(theta) * G * L
            if jacobians[0] is not None:
                jacobians[0][:] = -scatter * gain * loss

            # d res_i / d c_k = -predicted_i * T_k(u_i), rows of the other
            # channel stay zero. Ceres stores per-block jacobians row-major:
            # entry [i * block_size + k].
            for channel, present in ((0, self.port_present), (1, self.stbd_present)):
                if not present:
                    continue
                slot = self._channel_slot(channel)
                jb = jacobians[slot]
                if jb is None:
                    continue
                mask = channels == channel
                block = np.zeros((self.size, self.degree + 1), dtype=np.float64)
                if np.any(mask):
                    block[mask, :] = -predicted[mask, None] * T[mask, :]
                jb[:] = block.ravel()
                del block

            # d res / d alpha = (2 * ln(10) / 10) * r * predicted
            if jacobians[-1] is not None:
                jacobians[-1][:] = ALPHA_JAC * self.bases[2][start:end] * predicted
        return True


class BeamPriorCost(pyceres.CostFunction):
    """
    Anchors the global scale gauge between reflectivity and the beam pattern.

    The constant Chebyshev coefficient c_0 is the pattern's log-scale: the
    observation model is invariant to (reflectivity -> R * s, c_0 -> c_0 + ln s),
    so c_0 = 0 pins gain patterns (and reflectivity) to a fixed global scale.

        res = c_0 * gauge_weight

    Parameter blocks: [beam pattern coefficients (degree + 1,)]
    """

    def __init__(self, degree: int, gauge_weight: float):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([degree + 1])

        self.degree = int(degree)
        self.gauge_weight = float(gauge_weight)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        residuals[0] = parameters[0][0] * self.gauge_weight

        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = 0.0
            jacobians[0][0] = self.gauge_weight
        return True


class BundleAdjustment:
    """
    Solves for per-vertex reflectivity (local) jointly with the parametric
    port/starboard beam patterns and the absorption coefficient (global) using
    redundant observations of each mesh vertex from multiple pings, headings,
    and ranges.

    Observation model per vertex hit (Lambertian):

        I(ping, bin) = R_v * cos(theta) * G_channel(theta) * 10^(-2 * alpha * r / 10)

    where theta is the incidence angle, r the slant range, I the raw waterfall
    intensity at the hit's (ping_idx, bin_idx) sample, and the beam gain
    G_channel is a degree-'beam_degree' Chebyshev polynomial in log space
    (2 * (degree + 1) global coefficients in total).

    Problem structure (scales to hundreds of millions of observations):
      - One residual block per vertex holding all of its observations.
      - The beam pattern is parametric, so each residual block references at
        most three global blocks: one coefficient vector per channel observed
        plus the absorption scalar.
      - Per-vertex reflectivity values live in one contiguous buffer; each
        ceres parameter block is a view into it.
      - solve() reorders the observation arrays by vertex in place, so after
        solving the obs_* attributes are in vertex-sorted order (their contents
        are unchanged and every downstream use is order independent). Only the
        arrays the decomposition stage needs are kept afterwards.
    """

    def __init__(self,
        dataset: Dataset,
        beam_degree: int = 6,
        max_angle: float = np.pi / 2,
        absorption_coeff: float = 0.1,
        max_iterations: int = 100,
        loss_delta: float = 1000.0,
        gauge_weight: float = 1.0,
        num_threads: int = 6,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.beam_degree = int(beam_degree)
        self.max_angle = float(max_angle)
        self.absorption_coeff = float(absorption_coeff)
        self.max_iterations = int(max_iterations)
        self.loss_delta = float(loss_delta)
        self.gauge_weight = float(gauge_weight)
        self.num_threads = int(num_threads)
        self.verbose = bool(verbose)

        self.load_observations()

    def _count_matches_rows(self, matches_files: list) -> int:
        """
        Counts newlines across all matches files in binary chunks to reserve an
        upper bound of observations without parsing a single row.
        """
        total = 0
        chunk_size = 1 << 20
        for matches_file in matches_files:
            with open(matches_file, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    total += chunk.count(b'\n')

        # Margin for header lines and files missing the trailing newline
        return total + len(matches_files) + 1

    def load_observations(self):
        matches_files = sorted(self.dataset.vertex_matches_dir.glob("*.csv"))

        waterfall = np.load(self.dataset.sonar_file)["data"]
        n_pings, n_bins = waterfall.shape

        # Ping index -> number of samples per channel (port | starboard split)
        num_samples_map: Dict[int, int] = {
            sss.ping_idx: sss.num_samples
            for sss in self.dataset.sonar.values()
        }

        # Reserve the exact memory needed up front instead of growing Python lists
        upper = self._count_matches_rows(matches_files)

        obs_vertex = np.zeros(upper, dtype=np.int64)
        obs_channel = np.zeros(upper, dtype=np.int8)
        obs_cos_incidence = np.zeros(upper, dtype=np.float64)
        obs_pattern_angle = np.zeros(upper, dtype=np.float32)
        obs_range = np.zeros(upper, dtype=np.float64)
        obs_ping_idx = np.zeros(upper, dtype=np.int32)
        obs_bin_idx = np.zeros(upper, dtype=np.int32)

        count = 0
        for matches_file in matches_files:
            # Stream row dicts instead of VertexHit lists: constructing a
            # VertexHit (Pose + AcousticHit) per row doubles the peak memory.
            with open(matches_file) as f:
                for row in csv.DictReader(f):
                    ping_idx = int(row["ping_idx"])
                    bin_idx = int(row["bin_idx"])
                    if ping_idx < 0 or ping_idx >= n_pings:
                        continue
                    if bin_idx < 0 or bin_idx >= n_bins:
                        continue

                    num_samples = num_samples_map.get(ping_idx)
                    if num_samples is None or num_samples <= 0:
                        continue

                    incidence_angle = float(row["incidence_angle"])

                    # Half-open/current observation angles can graze the limits
                    theta = min(max(incidence_angle, 0.0), self.max_angle)

                    obs_vertex[count] = int(row["vertex_idx"])
                    obs_channel[count] = 0 if bin_idx < num_samples else 1
                    obs_cos_incidence[count] = min(max(np.cos(theta), 0.0), 1.0)
                    # Normalized pattern angle of the Chebyshev basis
                    obs_pattern_angle[count] = np.float32(2.0 * theta / self.max_angle - 1.0)
                    obs_range[count] = float(row["distance"])
                    obs_ping_idx[count] = ping_idx
                    obs_bin_idx[count] = bin_idx
                    count += 1

        if count == 0:
            raise ValueError("No valid vertex observations found for bundle adjustment.")

        # Copy out the used prefix and release each oversized reserved buffer as
        # soon as its copy exists (instead of keeping all of them until return)
        self.obs_vertex, obs_vertex = obs_vertex[:count].copy(), None
        self.obs_channel, obs_channel = obs_channel[:count].copy(), None
        self.obs_cos_incidence, obs_cos_incidence = obs_cos_incidence[:count].copy(), None
        self.obs_pattern_angle, obs_pattern_angle = obs_pattern_angle[:count].copy(), None
        self.obs_range, obs_range = obs_range[:count].copy(), None
        self.obs_ping_idx, obs_ping_idx = obs_ping_idx[:count].copy(), None
        self.obs_bin_idx, obs_bin_idx = obs_bin_idx[:count].copy(), None
        del obs_vertex, obs_channel, obs_cos_incidence, obs_pattern_angle, obs_range, obs_ping_idx, obs_bin_idx

        # Gather the observed intensities in one vectorized pass, then release
        # the full waterfall (the largest allocation of this phase)
        self.obs_intensity = waterfall[self.obs_ping_idx, self.obs_bin_idx].astype(np.float64)
        del waterfall
        gc.collect()

        if self.verbose:
            print(f"Loaded {count} vertex observations")

    def initial_reflectivity(self,
        vertex_ids: np.ndarray,
        group_starts: np.ndarray,
        group_ends: np.ndarray,
    ) -> np.ndarray:
        """
        Reflectivity solved using the initial global parameters
        (G = 1, alpha = absorption_coeff) with the Lambertian factor divided out:

            R_v = mean over observations of I * 10^(2 * alpha_init * r / 10) / cos(theta)

        Must be called with the vertex-sorted observation arrays (solve Phase A)
        so per-vertex means reduce to contiguous segments. Near-grazing
        observations (cos(theta) -> 0) carry almost no signal about reflectivity,
        so the divisor is floored to avoid exploding the initialization.
        """
        compensation = 10.0 ** (2.0 * self.absorption_coeff * self.obs_range / 10.0)
        weighted = self.obs_intensity * compensation
        del compensation

        # Floor the Lambertian divisor: grazing hits have no signal to divide
        weighted /= np.clip(self.obs_cos_incidence, 1.0e-3, 1.0)

        sums = np.add.reduceat(weighted, group_starts)
        del weighted
        gc.collect()

        reflectivity = np.zeros(int(vertex_ids.max()) + 1, dtype=np.float64)
        reflectivity[vertex_ids] = sums / (group_ends - group_starts)
        del sums

        return reflectivity

    def solve(self):
        """
        Builds and solves the bundle adjustment problem.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, pyceres.SolverSummary]:
                Reflectivity array indexed by vertex_idx (zero for unobserved),
                port beam pattern coefficients, starboard beam pattern
                coefficients, absorption coeff, summary. Gains at any angle are
                exp(chebyshev_basis(2 * theta / max_angle - 1, beam_degree) @ coefficients).
        """
        problem = pyceres.Problem()

        # --- Phase 0: global parameter blocks ---
        # One (degree + 1)-sized coefficient block per channel, plus alpha
        beam_port_coeffs = np.zeros(self.beam_degree + 1, dtype=np.float64)
        beam_stbd_coeffs = np.zeros(self.beam_degree + 1, dtype=np.float64)
        problem.add_parameter_block(beam_port_coeffs, self.beam_degree + 1)
        problem.add_parameter_block(beam_stbd_coeffs, self.beam_degree + 1)

        alpha = np.array([self.absorption_coeff], dtype=np.float64)
        problem.add_parameter_block(alpha, 1)
        problem.set_parameter_lower_bound(alpha, 0, 0.0)

        # Anchor the reflectivity <-> beam pattern scale gauge (c_0 = 0)
        problem.add_residual_block(
            BeamPriorCost(self.beam_degree, self.gauge_weight), None, [beam_port_coeffs]
        )
        problem.add_residual_block(
            BeamPriorCost(self.beam_degree, self.gauge_weight), None, [beam_stbd_coeffs]
        )

        # --- Phase A: reorder the observation arrays by vertex ---
        # The sorted arrays replace the attributes in place so only one copy of
        # each is ever alive; the permutation indexes are released as soon as
        # they are consumed.
        order = np.argsort(self.obs_vertex, kind="stable")
        self.obs_vertex = self.obs_vertex[order]
        self.obs_intensity = self.obs_intensity[order]
        self.obs_cos_incidence = self.obs_cos_incidence[order]
        self.obs_pattern_angle = self.obs_pattern_angle[order]
        self.obs_range = self.obs_range[order]
        self.obs_channel = self.obs_channel[order]
        del order
        gc.collect()

        if self.verbose:
            print(f"Sorted observations by vertex")

        # --- Phase B: group boundaries (obs_vertex is sorted: no re-sort copy) ---
        boundaries = np.flatnonzero(self.obs_vertex[1:] > self.obs_vertex[:-1]) + 1
        group_starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
        del boundaries
        vertex_ids = self.obs_vertex[group_starts]
        group_ends = np.append(group_starts[1:], self.obs_vertex.size)
        gc.collect()

        # --- Phase C: initial reflectivity from the grouped observations ---
        init_reflectivity = self.initial_reflectivity(vertex_ids, group_starts, group_ends)

        # --- Phase D: build one residual block per vertex ---
        # Per-vertex reflectivity values live in a single contiguous buffer;
        # each ceres parameter block is a view into it.
        rv_buffer = init_reflectivity[vertex_ids]
        del init_reflectivity

        bases = (
            self.obs_intensity,
            self.obs_cos_incidence,
            self.obs_range,
            self.obs_pattern_angle,
            self.obs_channel,
        )

        n_port = n_stbd = 0
        for i, (start, end) in enumerate(zip(group_starts, group_ends)):
            # Loop temporaries (rv_block, ...) are rebound each iteration;
            # CPython's reference counting frees the previous iteration's
            # objects immediately.
            rv_block = rv_buffer[i:i + 1]
            problem.add_parameter_block(rv_block, 1)
            problem.set_parameter_lower_bound(rv_block, 0, 0.0)

            channels = self.obs_channel[start:end]
            port_present = bool(np.any(channels == 0))
            stbd_present = bool(np.any(channels == 1))
            n_port += port_present
            n_stbd += stbd_present

            cost = VertexObservationCost(
                bases, start, end - start, self.beam_degree,
                port_present, stbd_present,
            )

            # Ceres applies the loss to the squared norm of the whole residual
            # block: scale delta by sqrt(m) to keep the configured
            # per-observation robustness after batching.
            beam_blocks = []
            if port_present:
                beam_blocks.append(beam_port_coeffs)
            if stbd_present:
                beam_blocks.append(beam_stbd_coeffs)

            problem.add_residual_block(
                cost,
                pyceres.HuberLoss(self.loss_delta * math.sqrt(end - start)),
                [rv_block, *beam_blocks, alpha],
            )

        del bases
        gc.collect()

        if self.verbose:
            print(f"Built problem: {vertex_ids.size} vertex blocks "
                  f"({n_port} referencing port, {n_stbd} starboard), "
                  f"patterns of degree {self.beam_degree}")

        # --- Phase E: solve ---
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.max_num_iterations = self.max_iterations
        options.num_threads = self.num_threads
        options.minimizer_progress_to_stdout = self.verbose

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        if self.verbose:
            print(summary.BriefReport())
            print(f"Solve finished")

        # --- Phase F: collect results and release everything heavy ---
        reflectivity = np.zeros(int(self.obs_vertex[-1]) + 1, dtype=np.float64)
        reflectivity[vertex_ids] = rv_buffer

        # The beam pattern is its coefficient vector: gains at any angle are
        # exp(chebyshev_basis(u, beam_degree) @ coefficients)
        beam_port = beam_port_coeffs.copy()
        beam_stbd = beam_stbd_coeffs.copy()
        absorption = float(alpha[0])

        # Release the solver with its per-block structures, then the observation
        # columns the decomposition stage does not need (intensity and cosine are
        # solver inputs only; ping, bin, range, and vertex are kept).
        del problem, rv_buffer, beam_port_coeffs, beam_stbd_coeffs
        del self.obs_intensity, self.obs_cos_incidence, self.obs_pattern_angle, self.obs_channel
        gc.collect()

        if self.verbose:
            print(f"Collected results")

        return reflectivity, beam_port, beam_stbd, absorption, summary
