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


class VertexCost(pyceres.CostFunction):
    """
    Residual for one mesh vertex after analytically eliminating its reflectivity.

    Given fixed beam and absorption parameters, the optimal R_v is

        R_v* = Σ_k I_k * W_k / Σ_k W_k^2,   W_k = cos(theta_k) * G_k * L_k

    and the minimized cost is

        cost_v = Σ_k I_k^2 - (Σ_k I_k * W_k)^2 / Σ_k W_k^2.

    One scalar residual per vertex; parameter blocks are only the three global
    blocks: [beam_port_coeffs, beam_stbd_coeffs, alpha]. The cost function holds
    only the start/end indices into the BundleAdjustment's observation arrays
    and references to those arrays, avoiding copying observation data.

    Parameter blocks:
        [beam_port_coeffs (degree+1,), beam_stbd_coeffs (degree+1,), alpha (1,)]
    """

    def __init__(
        self,
        start: int,
        end: int,
        obs_intensity: np.ndarray,
        obs_cos_incidence: np.ndarray,
        obs_range: np.ndarray,
        obs_pattern_angle: np.ndarray,
        obs_channel: np.ndarray,
        degree: int,
        port_present: bool,
        stbd_present: bool,
    ):
        pyceres.CostFunction.__init__(self)

        self.set_num_residuals(1)

        beam_sizes = [degree + 1] * (int(port_present) + int(stbd_present))
        self.set_parameter_block_sizes(beam_sizes + [1])

        self.start = int(start)
        self.end = int(end)
        # Keep references to the global observation arrays — no data is copied
        self.obs_intensity = obs_intensity
        self.obs_cos_incidence = obs_cos_incidence
        self.obs_range = obs_range
        self.obs_pattern_angle = obs_pattern_angle
        self.obs_channel = obs_channel
        self.degree = int(degree)
        self.port_present = bool(port_present)
        self.stbd_present = bool(stbd_present)
        self.port_slot = 0
        self.stbd_slot = 1 if port_present else 0

    def _channel_slot(self, channel: int) -> int:
        if channel == 0:
            return self.port_slot
        return self.stbd_slot

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        beam_params = list(parameters[: -1])
        alpha_val = parameters[-1][0]

        # Slice the observation arrays for this vertex
        I = self.obs_intensity[self.start:self.end]
        cos_theta = self.obs_cos_incidence[self.start:self.end]
        r = self.obs_range[self.start:self.end]
        u = self.obs_pattern_angle[self.start:self.end]
        channels = self.obs_channel[self.start:self.end]

        n_obs = self.end - self.start

        # Lambertian + propagation loss: W_k = cos(theta_k) * exp(-ALPHA_JAC * alpha * r_k)
        W = cos_theta * np.exp(-ALPHA_JAC * alpha_val * r)

        # Evaluate Chebyshev basis once for all observations of this vertex
        T = chebyshev_basis(u, self.degree)

        # Accumulate per-vertex statistics
        sum_I2 = 0.0
        sum_IW = 0.0
        sum_W2 = 0.0

        # Pre-allocate arrays for Jacobian computation if needed
        if jacobians is not None:
            d_sum_IW_dalpha = 0.0
            d_sum_W2_dalpha = 0.0
            d_sum_IW_dbeam_port = np.zeros(self.degree + 1, dtype=np.float64)
            d_sum_W2_dbeam_port = np.zeros(self.degree + 1, dtype=np.float64)
            d_sum_IW_dbeam_stbd = np.zeros(self.degree + 1, dtype=np.float64)
            d_sum_W2_dbeam_stbd = np.zeros(self.degree + 1, dtype=np.float64)

        # Process each observation
        for local_idx in range(n_obs):
            channel = int(channels[local_idx])
            coeffs = beam_params[self._channel_slot(channel)]

            # Beam gain in log space: log_G = T[local_idx] @ coeffs
            G = np.exp(T[local_idx] @ coeffs)

            # Full weight: W_k * G
            WG = W[local_idx] * G

            sum_I2 += I[local_idx] ** 2
            sum_IW += I[local_idx] * WG
            sum_W2 += WG ** 2

            if jacobians is not None:
                # Derivative of WG w.r.t. alpha:
                # dW/dalpha = -ALPHA_JAC * r * W
                # dWG/dalpha = -ALPHA_JAC * r * WG
                dWG_dalpha = -ALPHA_JAC * r[local_idx] * WG
                d_sum_IW_dalpha += I[local_idx] * dWG_dalpha
                d_sum_W2_dalpha += 2 * WG * dWG_dalpha

                # Derivative of WG w.r.t. beam coefficients:
                # dG/dcoeffs[k] = G * T[local_idx, k]
                # dWG/dcoeffs[k] = W * dG/dcoeffs[k] = WG * T[local_idx, k]
                dWG_dbeam = WG * T[local_idx, :]
                if channel == 0:
                    d_sum_IW_dbeam_port += I[local_idx] * dWG_dbeam
                    d_sum_W2_dbeam_port += 2 * WG * dWG_dbeam
                else:
                    d_sum_IW_dbeam_stbd += I[local_idx] * dWG_dbeam
                    d_sum_W2_dbeam_stbd += 2 * WG * dWG_dbeam

        # Compute minimized cost residual
        # Clamp to avoid sqrt(negative) from numerical instability when sum_W2 is tiny
        if sum_W2 < 1e-15:
            # All WG effectively zero: residual = sqrt(sum I_k^2)
            residual = np.sqrt(sum_I2) if sum_I2 >= 0 else 0.0
        else:
            residual_value = sum_I2 - sum_IW ** 2 / sum_W2
            residual = np.sqrt(max(0.0, residual_value))

        residuals[0] = residual

        # If we clamped a negative residual_value to zero, the vertex contributes
        # no signal to the optimization, so zero out its Jacobian entirely.
        if residual == 0.0 and sum_W2 >= 1e-15:
            # residual_value was negative and got clamped to 0
            if jacobians is not None:
                for jb in jacobians:
                    if jb is not None:
                        jb[:] = 0.0
                return True

        if jacobians is not None:
            # Jacobian of residual w.r.t. parameters
            # residual = sqrt(sum_I2 - sum_IW^2 / sum_W2)
            # f = sum_I2 - sum_IW^2 / sum_W2
            # df/dp = -2*sum_IW*d(sum_IW)/dp/sum_W2 + sum_IW^2*d(sum_W2)/dp/sum_W2^2
            #       = (sum_IW/sum_W2) * (d(sum_W2)/dp/sum_W2 - 2*d(sum_IW)/dp/sum_IW)

            if sum_W2 < 1e-15 or residual < 1e-15:
                # Degenerate case (near-zero weights or residual): zero out Jacobian
                for jb in jacobians:
                    if jb is not None:
                        jb[:] = 0.0
                return True

            ratio = sum_IW / sum_W2
            inv_res = 1.0 / (2.0 * residual)

            # Process beam parameter blocks
            slot = 0
            for channel, present in ((0, self.port_present), (1, self.stbd_present)):
                if not present:
                    continue
                jb = jacobians[slot]
                if jb is not None:
                    if channel == 0:
                        d_sum_IW = d_sum_IW_dbeam_port
                        d_sum_W2 = d_sum_W2_dbeam_port
                    else:
                        d_sum_IW = d_sum_IW_dbeam_stbd
                        d_sum_W2 = d_sum_W2_dbeam_stbd
                    df_dbeam = ratio * (d_sum_W2 / sum_W2 - 2.0 * d_sum_IW / (sum_IW + 1e-30))
                    jb[:] = inv_res * df_dbeam
                slot += 1

            # Alpha parameter block
            jb_alpha = jacobians[-1]
            if jb_alpha is not None:
                df_dalpha = ratio * (d_sum_W2_dalpha / sum_W2 - 2.0 * d_sum_IW_dalpha / (sum_IW + 1e-30))
                jb_alpha[0] = inv_res * df_dalpha

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

    Optimization structure:
      - Vertex reflectivity is analytically eliminated: given fixed beam and alpha,
        the optimal R_v for each vertex is a closed-form weighted average.
      - Only 3 global parameter blocks are optimized: port beam coefficients,
        starboard beam coefficients, and the absorption scalar (total 15 or
        2*degree+3 parameters).
      - One residual block per vertex, returning the minimized cost after
        eliminating R_v (scalar). This keeps the Jacobian tiny: ~15 entries
        per vertex, independent of the number of observations.
      - Memory usage scales linearly with the number of observations, and the
        solver memory scales only with the 15 global parameters. Each VertexCost
        instance holds only indices and references to the global observation
        arrays; no observation data is copied.
    """

    def __init__(
        self,
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

                    try:
                        incidence_angle = float(row["incidence_angle"])
                        distance = float(row["distance"])
                        vertex_idx = int(row["vertex_idx"])
                    except (ValueError, TypeError):
                        continue

                    # Skip NaN/inf in geometry
                    if (math.isnan(incidence_angle) or math.isinf(incidence_angle) or
                            math.isnan(distance) or math.isinf(distance) or
                            distance <= 0):
                        continue

                    # Half-open/current observation angles can graze the limits
                    theta = min(max(incidence_angle, 0.0), self.max_angle)

                    obs_vertex[count] = vertex_idx
                    obs_channel[count] = 0 if bin_idx < num_samples else 1
                    obs_cos_incidence[count] = min(max(np.cos(theta), 0.0), 1.0)
                    # Normalized pattern angle of the Chebyshev basis
                    obs_pattern_angle[count] = np.float32(2.0 * theta / self.max_angle - 1.0)
                    obs_range[count] = distance
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
        self.obs_ping_idx = self.obs_ping_idx[order]
        self.obs_bin_idx = self.obs_bin_idx[order]
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

        # --- Phase C: build one residual block per vertex ---
        # Each vertex contributes one scalar residual (the minimized cost after
        # analytically eliminating R_v). Parameter blocks are only the global
        # beam and alpha blocks (3 total). Each cost function holds only
        # references to the global observation arrays plus start/end indices.
        n_port = n_stbd = 0
        for i, (start, end) in enumerate(zip(group_starts, group_ends)):
            channels = self.obs_channel[start:end]
            port_present = bool(np.any(channels == 0))
            stbd_present = bool(np.any(channels == 1))
            n_port += port_present
            n_stbd += stbd_present

            cost = VertexCost(
                start, end,
                self.obs_intensity,
                self.obs_cos_incidence,
                self.obs_range,
                self.obs_pattern_angle,
                self.obs_channel,
                self.beam_degree,
                port_present,
                stbd_present,
            )

            # Build parameter block list based on which channels are present
            beam_blocks = []
            if port_present:
                beam_blocks.append(beam_port_coeffs)
            if stbd_present:
                beam_blocks.append(beam_stbd_coeffs)

            problem.add_residual_block(
                cost,
                pyceres.HuberLoss(self.loss_delta),
                [*beam_blocks, alpha],
            )

        gc.collect()

        if self.verbose:
            print(f"Built problem: {vertex_ids.size} vertex blocks "
                  f"({n_port} referencing port, {n_stbd} starboard), "
                  f"patterns of degree {self.beam_degree}")

        # --- Phase D: solve ---
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        options.max_num_iterations = self.max_iterations
        options.num_threads = self.num_threads
        options.minimizer_progress_to_stdout = self.verbose

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        if self.verbose:
            print(summary.BriefReport())
            print(f"Solve finished")

        # --- Phase E: compute final reflectivity values ---
        # With optimized beam and alpha, compute R_v* = Σ I_k W_k / Σ W_k^2 for each vertex
        reflectivity = np.zeros(int(self.obs_vertex[-1]) + 1, dtype=np.float64)

        for i, (start, end) in enumerate(zip(group_starts, group_ends)):
            vid = int(self.obs_vertex[start])
            I = self.obs_intensity[start:end]
            cos_theta = self.obs_cos_incidence[start:end]
            r = self.obs_range[start:end]
            u = self.obs_pattern_angle[start:end]
            channels = self.obs_channel[start:end]

            # Compute W_k = cos(theta_k) * exp(-ALPHA_JAC * alpha * r_k)
            W = cos_theta * np.exp(-ALPHA_JAC * alpha[0] * r)

            # Evaluate beam gains
            T = chebyshev_basis(u, self.beam_degree)
            sum_IW = 0.0
            sum_W2 = 0.0

            for local_idx in range(end - start):
                channel = int(channels[local_idx])
                coeffs = beam_port_coeffs if channel == 0 else beam_stbd_coeffs
                log_G = T[local_idx] @ coeffs
                G = np.exp(log_G)
                WG = W[local_idx] * G
                sum_IW += I[local_idx] * WG
                sum_W2 += WG ** 2

            if sum_W2 > 1e-15:
                reflectivity[vid] = sum_IW / sum_W2
            else:
                reflectivity[vid] = 0.0

        # Release solver structures
        del problem, group_starts, group_ends, vertex_ids
        gc.collect()

        if self.verbose:
            print(f"Collected results")

        # The beam pattern is its coefficient vector: gains at any angle are
        # exp(chebyshev_basis(u, beam_degree) @ coefficients)
        beam_port = beam_port_coeffs.copy()
        beam_stbd = beam_stbd_coeffs.copy()
        absorption = float(alpha[0])

        return reflectivity, beam_port, beam_stbd, absorption, summary
