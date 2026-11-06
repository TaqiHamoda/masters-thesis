import csv
import math

import numpy as np
import pyceres

from typing import Dict, List

from .dataset import Dataset

# Two-way absorption loss (20-log rule on voltage samples): 10^(-2 * alpha * r / 10)
# Its alpha derivative contributes (2 * ln(10) / 10) * r * loss per unit alpha
ALPHA_JAC = 2.0 * np.log(10.0) / 10.0


class VertexObservationCost(pyceres.CostFunction):
    """
    Residuals for all observations of one mesh vertex (Lambertian model):

        res_i = I_i - R_v * cos(theta_i) * G_c[i][k_i] * 10^(-2 * alpha * r_i / 10)

    One residual per observation of the vertex, batched into a single block so
    the number of ceres residual blocks scales with vertices, not observations.

    Parameter blocks:
        [reflectivity (1,), beam gain scalar per unique (channel, bin) observed
         by this vertex ((1,) each), absorption coeff (1,)]
    """

    def __init__(self,
        intensity: np.ndarray,
        cos_incidence: np.ndarray,
        slant_range: np.ndarray,
        beam_local_idx: np.ndarray,
    ):
        pyceres.CostFunction.__init__(self)

        m = len(intensity)
        n_beam_blocks = len(np.unique(beam_local_idx))
        self.set_num_residuals(m)
        self.set_parameter_block_sizes([1] + [1] * n_beam_blocks + [1])

        # Views into the vertex-grouped observation arrays
        self.intensity = intensity
        self.cos_incidence = cos_incidence
        self.slant_range = slant_range
        self.beam_local_idx = beam_local_idx

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        reflectivity = parameters[0][0]
        alpha = parameters[-1][0]

        n_beam_blocks = len(parameters) - 2
        gains = np.fromiter(
            (parameters[1 + j][0] for j in range(n_beam_blocks)),
            dtype=np.float64, count=n_beam_blocks,
        )

        # Lambertian scattering factor from the known incidence angle
        scatter = self.cos_incidence
        # Loss factor starts at 1.0 by definition (0 dB at zero range)
        loss = 10.0 ** (-2.0 * alpha * self.slant_range / 10.0)
        predicted = reflectivity * gains[self.beam_local_idx] * scatter * loss

        res = self.intensity - predicted
        try:
            residuals[:] = res
        except TypeError:
            for i in range(len(res)):
                residuals[i] = res[i]

        if jacobians is not None:
            # Common factor: d res / d R = -G[k] * cos(theta) * L
            if jacobians[0] is not None:
                jacobians[0][:] = gains[self.beam_local_idx] * scatter * loss * -1.0

            # d res_i / d G_j = -R * cos(theta_i) * loss_i for the obs of beam j
            if any(jb is not None for jb in jacobians[1:1 + n_beam_blocks]):
                base = scatter * loss * -1.0
                mask = self.beam_local_idx[:, None] == np.arange(n_beam_blocks)[None, :]
                weighted = base[:, None] * mask * reflectivity
                for j in range(n_beam_blocks):
                    jb = jacobians[1 + j]
                    if jb is not None:
                        jb[:] = weighted[:, j]

            # d res / d alpha = (2 * ln(10) / 10) * r * predicted
            if jacobians[-1] is not None:
                jacobians[-1][:] = ALPHA_JAC * self.slant_range * predicted
        return True


class BeamPriorCost(pyceres.CostFunction):
    """
    Anchors the global scale gauge between reflectivity and beam pattern:

        res = (mean(beam) - 1) * gauge_weight

    Parameter blocks: one (1,) scalar per beam gain sample of the channel.
    """

    def __init__(self, ang_samples: int, gauge_weight: float):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1] * ang_samples)

        self.ang_samples = int(ang_samples)
        self.gauge_weight = float(gauge_weight)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        gains = np.fromiter(
            (p[0] for p in parameters), dtype=np.float64, count=self.ang_samples
        )
        residuals[0] = (float(np.mean(gains)) - 1.0) * self.gauge_weight

        if jacobians is not None:
            for j in range(self.ang_samples):
                if jacobians[j] is not None:
                    jacobians[j][:] = self.gauge_weight / self.ang_samples
        return True


class BundleAdjustment:
    """
    Solves for per-vertex reflectivity (local) jointly with the port/starboard
    beam pattern gains and the absorption coefficient (global) using redundant
    observations of each mesh vertex from multiple pings, headings, and ranges.

    Observation model per vertex hit (Lambertian):

        I(ping, bin) = R_v * cos(theta) * G_channel[k(theta)] * 10^(-2 * alpha * r / 10)

    where theta is the incidence angle, r the slant range, and I the raw
    waterfall intensity at the hit's (ping_idx, bin_idx) sample.

    Problem structure (scales to hundreds of millions of observations):
      - One residual block per vertex holding all of its observations.
      - Beam gains are 2 * ang_samples scalar parameter blocks so each residual
        block references only the handful of gains it touches.
    """

    def __init__(self,
        dataset: Dataset,
        ang_samples: int = 360,
        max_angle: float = np.pi / 2,
        absorption_coeff: float = 0.1,
        max_iterations: int = 100,
        loss_delta: float = 1000.0,
        gauge_weight: float = 1.0,
        num_threads: int = 6,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.ang_samples = int(ang_samples)
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
        obs_beam_bin = np.zeros(upper, dtype=np.int16)
        obs_cos_incidence = np.zeros(upper, dtype=np.float64)
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

                    # Beam pattern lookup bin: theta in [0, max_angle) -> [0, ang_samples)
                    beam_bin = int(incidence_angle / self.max_angle * self.ang_samples)
                    beam_bin = min(max(beam_bin, 0), self.ang_samples - 1)

                    obs_vertex[count] = int(row["vertex_idx"])
                    obs_channel[count] = 0 if bin_idx < num_samples else 1
                    obs_beam_bin[count] = beam_bin
                    obs_cos_incidence[count] = min(max(math.cos(incidence_angle), 0.0), 1.0)
                    obs_range[count] = float(row["distance"])
                    obs_ping_idx[count] = ping_idx
                    obs_bin_idx[count] = bin_idx
                    count += 1

        if count == 0:
            raise ValueError("No valid vertex observations found for bundle adjustment.")

        # Copy out the used prefix so the oversized reserved buffers are freed
        self.obs_vertex = obs_vertex[:count].copy()
        self.obs_channel = obs_channel[:count].copy()
        self.obs_beam_bin = obs_beam_bin[:count].copy()
        self.obs_cos_incidence = obs_cos_incidence[:count].copy()
        self.obs_range = obs_range[:count].copy()
        self.obs_ping_idx = obs_ping_idx[:count].copy()
        self.obs_bin_idx = obs_bin_idx[:count].copy()

        # Gather the observed intensities in one vectorized pass
        self.obs_intensity = waterfall[self.obs_ping_idx, self.obs_bin_idx].astype(np.float64)

        if self.verbose:
            print(f"Loaded {count} vertex observations "
                  f"({np.unique(self.obs_vertex).size} vertices)")

    def initial_reflectivity(self) -> np.ndarray:
        """
        Reflectivity solved using the initial global parameters
        (G = 1, alpha = absorption_coeff) with the Lambertian factor
        divided out:

            R_v = mean over observations of I * 10^(2 * alpha_init * r / 10) / cos(theta)

        Near-grazing observations (cos(theta) -> 0) carry almost no signal
        about reflectivity, so the divisor is floored to avoid exploding
        the initialization.
        """
        compensation = 10.0 ** (2.0 * self.absorption_coeff * self.obs_range / 10.0)
        # Floor the Lambertian divisor: grazing hits have no signal to divide
        scatter = np.clip(self.obs_cos_incidence, 1.0e-3, 1.0)
        weighted = self.obs_intensity * compensation / scatter

        vertex_ids = np.unique(self.obs_vertex)
        sums = np.zeros(int(vertex_ids.max()) + 1, dtype=np.float64)
        counts = np.zeros(int(vertex_ids.max()) + 1, dtype=np.int64)
        np.add.at(sums, self.obs_vertex, weighted)
        np.add.at(counts, self.obs_vertex, 1)

        reflectivity = np.zeros(int(vertex_ids.max()) + 1, dtype=np.float64)
        observed = counts > 0
        reflectivity[observed] = sums[observed] / counts[observed]

        return reflectivity

    def solve(self):
        """
        Builds and solves the bundle adjustment problem.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, pyceres.SolverSummary]:
                Reflectivity array indexed by vertex_idx (zero for unobserved),
                port beam gains, starboard beam gains, absorption coeff, summary.
        """
        problem = pyceres.Problem()

        # --- Global parameter blocks: one scalar per beam gain, plus alpha ---
        beam_pool: List[np.ndarray] = []
        for _ in range(2 * self.ang_samples):
            block = np.array([1.0], dtype=np.float64)
            problem.add_parameter_block(block, 1)
            problem.set_parameter_lower_bound(block, 0, 0.0)
            beam_pool.append(block)

        alpha = np.array([self.absorption_coeff], dtype=np.float64)
        problem.add_parameter_block(alpha, 1)
        problem.set_parameter_lower_bound(alpha, 0, 0.0)

        # Pull the shared scale ambiguity between reflectivity and beam gains
        # towards beam patterns with unit mean (gauge fixed per channel).
        problem.add_residual_block(
            BeamPriorCost(self.ang_samples, self.gauge_weight),
            None,
            [beam_pool[c] for c in range(self.ang_samples)],
        )
        problem.add_residual_block(
            BeamPriorCost(self.ang_samples, self.gauge_weight),
            None,
            [beam_pool[self.ang_samples + c] for c in range(self.ang_samples)],
        )

        # --- Group the observations by vertex for the batched residual blocks ---
        order = np.argsort(self.obs_vertex, kind="stable")
        vertex_all = self.obs_vertex[order]
        intensity_all = self.obs_intensity[order]
        cos_all = self.obs_cos_incidence[order]
        range_all = self.obs_range[order]
        codes_all = self.obs_channel[order].astype(np.int64) * self.ang_samples \
            + self.obs_beam_bin[order].astype(np.int64)

        vertex_ids, group_starts = np.unique(vertex_all, return_index=True)
        group_ends = np.append(group_starts[1:], len(vertex_all))

        init_reflectivity = self.initial_reflectivity()

        # Ceres applies the loss to the squared norm of the whole residual
        # block, but loss_delta is configured per observation: scale each
        # block's delta by sqrt(m) so robustness semantics are preserved
        # after batching all observations of a vertex into one block.
        losses: List[pyceres.HuberLoss] = []

        rv_blocks: List[np.ndarray] = []
        n_beam_blocks = 0
        for v, start, end in zip(vertex_ids, group_starts, group_ends):
            rv_block = np.array([init_reflectivity[v]], dtype=np.float64)
            problem.add_parameter_block(rv_block, 1)
            problem.set_parameter_lower_bound(rv_block, 0, 0.0)
            rv_blocks.append(rv_block)

            # Unique (channel, beam bin) gains touched by this vertex
            uniq_codes, beam_local_idx = np.unique(codes_all[start:end], return_inverse=True)
            beam_blocks = [beam_pool[c] for c in uniq_codes]
            n_beam_blocks += len(beam_blocks)

            cost = VertexObservationCost(
                intensity=intensity_all[start:end],
                cos_incidence=cos_all[start:end],
                slant_range=range_all[start:end],
                beam_local_idx=beam_local_idx,
            )
            block_loss = pyceres.HuberLoss(self.loss_delta * math.sqrt(end - start))
            losses.append(block_loss)
            problem.add_residual_block(cost, block_loss, [rv_block, *beam_blocks, alpha])

        if self.verbose:
            print(f"Built problem: {len(rv_blocks)} vertex blocks, "
                  f"{n_beam_blocks} beam block references, 2 gauge priors")

        # --- Solve ---
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.max_num_iterations = self.max_iterations
        options.num_threads = self.num_threads
        options.minimizer_progress_to_stdout = self.verbose

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        if self.verbose:
            print(summary.BriefReport())

        # Collect the per-vertex solutions (zero for unobserved vertices)
        reflectivity = np.zeros(int(self.obs_vertex.max()) + 1, dtype=np.float64)
        reflectivity[vertex_ids] = np.fromiter(
            (block[0] for block in rv_blocks), dtype=np.float64, count=len(rv_blocks)
        )

        beam_port = np.fromiter(
            (beam_pool[c][0] for c in range(self.ang_samples)),
            dtype=np.float64, count=self.ang_samples,
        )
        beam_stbd = np.fromiter(
            (beam_pool[self.ang_samples + c][0] for c in range(self.ang_samples)),
            dtype=np.float64, count=self.ang_samples,
        )

        return reflectivity, beam_port, beam_stbd, float(alpha[0]), summary
