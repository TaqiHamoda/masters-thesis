import csv
import gc
import math

import numpy as np
import pyceres

from typing import Dict, List, Tuple

from .dataset import Dataset

# Two-way propagation loss.
# For VOLTAGE/amplitude samples with absorption alpha in dB/m:
#   linear_loss = 10^(-2 * alpha * r / 20) = exp(-alpha * r * ln(10) / 10)
# For POWER/intensity samples:
#   linear_loss = 10^(-2 * alpha * r / 10) = exp(-2 * alpha * r * ln(10) / 10)
# This code uses the POWER convention (ALPHA_JAC = 2*ln(10)/10) so the returned
# absorption coefficient is in dB/m for POWER samples. If your waterfall data
# is VOLTAGE, divide the returned alpha by 2 to get the correct dB/m value.
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
        obs_depression_angle: np.ndarray,
        obs_range: np.ndarray,
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
        self.obs_depression_angle = obs_depression_angle
        self.obs_range = obs_range
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
        u = self.obs_depression_angle[self.start:self.end]
        channels = self.obs_channel[self.start:self.end]

        n_obs = self.end - self.start

        # Evaluate Chebyshev basis once for all observations
        T_basis = chebyshev_basis(u, self.degree)

        # Precompute WG for all observations
        WG = np.empty(n_obs, dtype=np.float64)
        sum_W2 = 0.0

        for local_idx in range(n_obs):
            channel = int(channels[local_idx])
            coeffs = beam_params[self._channel_slot(channel)]
            log_G = T_basis[local_idx] @ coeffs
            G = np.exp(np.clip(log_G, -10.0, 10.0))
            WG_val = cos_theta[local_idx] * np.exp(-ALPHA_JAC * alpha_val * r[local_idx]) * G
            WG[local_idx] = WG_val
            sum_W2 += WG_val ** 2

        # Guard: skip vertex if all weights are near-zero (grazing angles or far range)
        if sum_W2 < 1e-15:
            residuals[0] = np.linalg.norm(I)  # sqrt(sum I_k^2)
            if jacobians is not None:
                for jb in jacobians:
                    if jb is not None:
                        jb[:] = 0.0
            return True

        # Optimal reflectivity and per-observation errors
        R_v_star = np.dot(I, WG) / sum_W2
        errors = I - R_v_star * WG

        # Residual: sqrt(sum of squared errors) -- numerically stable
        residual = np.linalg.norm(errors)
        residuals[0] = residual

        if jacobians is None:
            return True

        # Guard: zero gradient when residual is zero (perfect fit)
        if residual < 1e-15:
            for jb in jacobians:
                if jb is not None:
                    jb[:] = 0.0
            return True

        # Jacobians using verified envelope theorem formula:
        # \partial residual / \partial p = (1 / (2 * residual)) * \partial f / \partial p
        # where f = sum_k errors_k^2
        # \partial f / \partial p = -2 * R_v* * sum_k errors_k * \partial W_k / \partial p
        #
        # \partial W_k / \partial \beta_{c,j} = W_k * T_j(u_k)  for observations of channel c
        # \partial W_k / \partial \alpha      = -ALPHA_JAC * r_k * W_k

        inv_two_res = 1.0 / (2.0 * residual)
        factor = -2.0 * R_v_star * inv_two_res  # = -R_v* / residual

        # Zero all Jacobian blocks first
        for jb in jacobians:
            if jb is not None:
                jb[:] = 0.0

        # Alpha parameter block (last block)
        jb_alpha = jacobians[-1]
        if jb_alpha is not None:
            for local_idx in range(n_obs):
                dW_dalpha = -ALPHA_JAC * r[local_idx] * WG[local_idx]
                jb_alpha[0] += factor * errors[local_idx] * dW_dalpha

        # Beam coefficient blocks
        slot = 0
        for channel, present in ((0, self.port_present), (1, self.stbd_present)):
            if not present:
                continue
            jb = jacobians[slot]
            if jb is not None:
                for local_idx in range(n_obs):
                    if channels[local_idx] != channel:
                        continue
                    # dWG/d\beta_j = WG * T_j(u)
                    dWG_dbeta = WG[local_idx] * T_basis[local_idx, :]
                    jb[:] += factor * errors[local_idx] * dWG_dbeta
            slot += 1

        return True


class BeamPriorCost(pyceres.CostFunction):
    """
    Anchors the global scale gauge between reflectivity and the beam pattern.

    The constant Chebyshev coefficient c_0 is the pattern's log-scale: the
    observation model is invariant to (reflectivity -> R * s, G -> G / s),
    which in log-space means (R -> R * s, c_0 -> c_0 - ln s). To fix the gauge,
    we add a quadratic prior cost w * c_0^2, where w is gauge_weight^2.

    The residual returned is gauge_weight * c_0, so the actual cost added is
    (gauge_weight * c_0)^2 = gauge_weight^2 * c_0^2.

    Parameter blocks: [beam pattern coefficients (degree + 1,)]
    """

    def __init__(self, degree: int, gauge_weight: float):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([degree + 1])

        self.degree = int(degree)
        self.gauge_weight = float(gauge_weight)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        # residual = gauge_weight * c_0, so cost = (gauge_weight * c_0)^2
        c0 = parameters[0][0]
        residuals[0] = self.gauge_weight * c0

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
        # Note: if port and starboard for the same ping have different ping_idx values,
        # this works correctly. If they share ping_idx, only the last entry is kept.
        # The channel is then determined by bin_idx < num_samples.
        num_samples_map: Dict[int, int] = {}
        for sss in self.dataset.sonar.values():
            ping_idx = sss.ping_idx
            if ping_idx in num_samples_map:
                # Multiple entries for same ping_idx: keep the larger num_samples
                # (assumes port and starboard are stored separately)
                num_samples_map[ping_idx] = max(num_samples_map[ping_idx], sss.num_samples)
            else:
                num_samples_map[ping_idx] = sss.num_samples

        # Reserve the exact memory needed up front instead of growing Python lists
        upper = self._count_matches_rows(matches_files)

        obs_vertex = np.zeros(upper, dtype=np.int64)
        obs_channel = np.zeros(upper, dtype=np.int8)
        obs_cos_incidence = np.zeros(upper, dtype=np.float64)
        obs_depression_angle = np.zeros(upper, dtype=np.float64)
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
                        depression_angle = float(row["depression_angle"])
                        distance = float(row["distance"])
                        vertex_idx = int(row["vertex_idx"])
                    except (ValueError, TypeError):
                        continue

                    # Skip NaN/inf in geometry
                    if (math.isnan(incidence_angle) or math.isinf(incidence_angle) or
                            math.isnan(depression_angle) or math.isinf(depression_angle) or
                            math.isnan(distance) or math.isinf(distance) or
                            distance <= 0):
                        continue

                    # Skip out-of-range incidence angles: these are back-facing or invalid
                    # and would have W_k = cos(theta) * ... = 0, poisoning the residual
                    if incidence_angle < 0.0 or incidence_angle > self.max_angle:
                        continue
                    elif depression_angle < 0.0 or depression_angle > self.max_angle:
                        continue

                    obs_vertex[count] = vertex_idx
                    obs_channel[count] = 0 if bin_idx < num_samples else 1
                    obs_cos_incidence[count] = np.clip(np.cos(incidence_angle), 0.0, 1.0)
                    # Normalized pattern angle of the Chebyshev basis
                    obs_depression_angle[count] = np.float64(2.0 * np.clip(2.0 * depression_angle / np.pi, 0.0, 1.0) - 1.0)
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
        self.obs_depression_angle, obs_depression_angle = obs_depression_angle[:count].copy(), None
        self.obs_range, obs_range = obs_range[:count].copy(), None
        self.obs_ping_idx, obs_ping_idx = obs_ping_idx[:count].copy(), None
        self.obs_bin_idx, obs_bin_idx = obs_bin_idx[:count].copy(), None
        del obs_vertex, obs_channel, obs_cos_incidence, obs_depression_angle, obs_range, obs_ping_idx, obs_bin_idx

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

        # Bound beam coefficients to prevent overflow in exp(T @ coeffs).
        # Chebyshev polynomials T_k are bounded by |T_k(u)| <= 1 for u in [-1, 1].
        # With degree d, |T @ c| <= sum |c_i|. To keep |log_G| <= 3 (G in [0.05, 20]),
        # we need sum |c_i| <= 3.  For degree 6 (7 coeffs), bound each |c_i| <= 3/7 ≈ 0.43.
        # Use [-0.5, 0.5] for all coefficients for safety (sum |c_i| <= 3.5 for degree 6).
        coeff_bound = 0.5
        for i in range(self.beam_degree + 1):
            problem.set_parameter_lower_bound(beam_port_coeffs, i, -coeff_bound)
            problem.set_parameter_upper_bound(beam_port_coeffs, i, coeff_bound)
            problem.set_parameter_lower_bound(beam_stbd_coeffs, i, -coeff_bound)
            problem.set_parameter_upper_bound(beam_stbd_coeffs, i, coeff_bound)

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
        self.obs_depression_angle = self.obs_depression_angle[order]
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
                self.obs_depression_angle,
                self.obs_range,
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
        # Only 15 global parameters, no point/edge blocks: DENSE_QR is appropriate
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
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
        # Size by max observed vertex ID + 1. Vertices beyond this range or in gaps
        # will be zero (unobserved). If you need reflectivity for all mesh vertices,
        # the caller should pad this array to the mesh's vertex count.
        reflectivity = np.zeros(int(self.obs_vertex[-1]) + 1, dtype=np.float64)

        for i, (start, end) in enumerate(zip(group_starts, group_ends)):
            vid = int(self.obs_vertex[start])
            I = self.obs_intensity[start:end]
            cos_theta = self.obs_cos_incidence[start:end]
            r = self.obs_range[start:end]
            u = self.obs_depression_angle[start:end]
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
                # Clamp to match the cost function's safety limit
                G = np.exp(np.clip(log_G, -10.0, 10.0))
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
