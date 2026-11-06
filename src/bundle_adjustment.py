import csv
import ctypes
import gc
import math
import os

import numpy as np
import pyceres

from typing import Dict, List, Tuple

from .dataset import Dataset

# Two-way absorption loss (20-log rule on voltage samples): 10^(-2 * alpha * r / 10)
# Its alpha derivative contributes (2 * ln(10) / 10) * r * loss per unit alpha
ALPHA_JAC = 2.0 * np.log(10.0) / 10.0


def collect_garbage():
    """
    Forces a full garbage collection pass and returns freed heap pages to the
    operating system (on glibc). Call this after deleting memory heavy variables.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # malloc_trim is a glibc extension; ignore elsewhere


def current_rss_mb() -> float:
    """Returns this process's resident set size in MB (or -1 if unavailable)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


class VertexObservationCost(pyceres.CostFunction):
    """
    Residuals for all observations of one mesh vertex (Lambertian model):

        res_i = I_i - R_v * cos(theta_i) * G_c[i][k_i] * 10^(-2 * alpha * r_i / 10)

    One residual per observation of the vertex, batched into a single block so
    the number of ceres residual blocks scales with vertices, not observations.

    To keep the per-block memory footprint small, no observation data is copied
    at construction: the cost holds a shared reference to the four canonical
    observation arrays (intensity, cos of incidence, slant range, beam-local
    index) plus the (start, size) span of this vertex's observations.

    Parameter blocks:
        [reflectivity (1,), beam gain scalar per unique (channel, bin) observed
         by this vertex ((1,) each), absorption coeff (1,)]
    """

    def __init__(self,
        bases: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        start: int,
        size: int,
        n_beam_blocks: int,
    ):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(size)
        self.set_parameter_block_sizes([1] + [1] * n_beam_blocks + [1])

        self.bases = bases
        self.start = int(start)
        self.size = int(size)
        self.n_beam_blocks = int(n_beam_blocks)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        start, end = self.start, self.start + self.size

        reflectivity = parameters[0][0]
        alpha = parameters[-1][0]

        g = self.n_beam_blocks
        gains = np.fromiter(
            (parameters[1 + j][0] for j in range(g)),
            dtype=np.float64, count=g,
        )

        # Lambertian scattering factor from the known incidence angle
        scatter = self.bases[1][start:end]
        # Loss factor starts at 1.0 by definition (0 dB at zero range)
        loss = 10.0 ** (-2.0 * alpha * self.bases[2][start:end] / 10.0)
        predicted = reflectivity * gains[self.bases[3][start:end]] * scatter * loss

        res = self.bases[0][start:end] - predicted
        try:
            residuals[:] = res
        except TypeError:
            for i in range(len(res)):
                residuals[i] = res[i]

        if jacobians is not None:
            # d res / d R = -G[k] * cos(theta) * L
            if jacobians[0] is not None:
                jacobians[0][:] = gains[self.bases[3][start:end]] * scatter * loss * -1.0

            # d res_i / d G_j = -R * cos(theta_i) * loss_i for the obs of beam j
            if any(jb is not None for jb in jacobians[1:1 + g]):
                beam_local = self.bases[3][start:end]
                base = scatter * loss * -1.0
                mask = beam_local[:, None] == np.arange(g)[None, :]
                weighted = base[:, None] * mask * reflectivity
                for j in range(g):
                    jb = jacobians[1 + j]
                    if jb is not None:
                        jb[:] = weighted[:, j]
                del beam_local, base, mask, weighted

            # d res / d alpha = (2 * ln(10) / 10) * r * predicted
            if jacobians[-1] is not None:
                jacobians[-1][:] = ALPHA_JAC * self.bases[2][start:end] * predicted
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
      - Per-vertex reflectivity values live in one contiguous buffer; each
        ceres parameter block is a view into it.
      - solve() reorders the observation arrays by vertex in place, so after
        solving the obs_* attributes are in vertex-sorted order (their contents
        are unchanged and every downstream use is order independent). Only the
        arrays the decomposition stage needs are kept afterwards.
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

        # Copy out the used prefix and release each oversized reserved buffer as
        # soon as its copy exists (instead of keeping all of them until return)
        self.obs_vertex, obs_vertex = obs_vertex[:count].copy(), None
        self.obs_channel, obs_channel = obs_channel[:count].copy(), None
        self.obs_beam_bin, obs_beam_bin = obs_beam_bin[:count].copy(), None
        self.obs_cos_incidence, obs_cos_incidence = obs_cos_incidence[:count].copy(), None
        self.obs_range, obs_range = obs_range[:count].copy(), None
        self.obs_ping_idx, obs_ping_idx = obs_ping_idx[:count].copy(), None
        self.obs_bin_idx, obs_bin_idx = obs_bin_idx[:count].copy(), None
        del obs_vertex, obs_channel, obs_beam_bin, obs_cos_incidence, obs_range, obs_ping_idx, obs_bin_idx

        # Gather the observed intensities in one vectorized pass, then release
        # the full waterfall (the largest allocation of this phase)
        self.obs_intensity = waterfall[self.obs_ping_idx, self.obs_bin_idx].astype(np.float64)
        del waterfall
        collect_garbage()

        if self.verbose:
            print(f"Loaded {count} vertex observations, RSS {current_rss_mb():.1f} MB")

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
        collect_garbage()

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
                port beam gains, starboard beam gains, absorption coeff, summary.
        """
        problem = pyceres.Problem()

        # --- Phase 0: global parameter blocks ---
        # One scalar block per beam gain sample (port codes 0..N-1, stbd N..2N-1)
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

        # --- Phase A: reorder the observation arrays by vertex ---
        # The sorted arrays replace the attributes in place so only one copy of
        # each is ever alive; the permutation indexes are released as soon as
        # they are consumed.
        order = np.argsort(self.obs_vertex, kind="stable")
        self.obs_vertex = self.obs_vertex[order]
        self.obs_intensity = self.obs_intensity[order]
        self.obs_cos_incidence = self.obs_cos_incidence[order]
        self.obs_range = self.obs_range[order]

        # Channel and beam bin only exist to build per-observation beam codes:
        # codes are int32 (2 * ang_samples fits easily) and the raw columns are
        # released immediately after.
        codes = self.obs_channel[order].astype(np.int32) * np.int32(self.ang_samples)
        codes += self.obs_beam_bin[order]
        del self.obs_channel, self.obs_beam_bin, order
        collect_garbage()

        if self.verbose:
            print(f"Sorted observations by vertex, RSS {current_rss_mb():.1f} MB")

        # --- Phase B: group boundaries (obs_vertex is sorted: no re-sort copy) ---
        boundaries = np.flatnonzero(self.obs_vertex[1:] > self.obs_vertex[:-1]) + 1
        group_starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
        del boundaries
        vertex_ids = self.obs_vertex[group_starts]
        group_ends = np.append(group_starts[1:], self.obs_vertex.size)
        collect_garbage()

        # --- Phase C: initial reflectivity from the grouped observations ---
        init_reflectivity = self.initial_reflectivity(vertex_ids, group_starts, group_ends)

        # --- Phase D: build one residual block per vertex ---
        # Per-vertex reflectivity values live in a single contiguous buffer;
        # each ceres parameter block is a view into it.
        rv_buffer = init_reflectivity[vertex_ids]
        del init_reflectivity

        # Beam-local indices are stored per observation in one shared array;
        # int16 suffices (bounded by the observations of a single vertex).
        beam_local = np.zeros(self.obs_vertex.size, dtype=np.int16)

        bases = (self.obs_intensity, self.obs_cos_incidence, self.obs_range, beam_local)

        n_beam_refs = 0
        for i, (start, end) in enumerate(zip(group_starts, group_ends)):
            # Loop temporaries (rv_block, uniq_codes, inverses, beam_blocks) are
            # rebound each iteration; CPython's reference counting frees the
            # previous iteration's objects immediately.
            rv_block = rv_buffer[i:i + 1]
            problem.add_parameter_block(rv_block, 1)
            problem.set_parameter_lower_bound(rv_block, 0, 0.0)

            # Unique (channel, beam bin) gains touched by this vertex
            uniq_codes, inverses = np.unique(codes[start:end], return_inverse=True)
            beam_local[start:end] = inverses.astype(np.int16)
            del inverses

            beam_blocks = [beam_pool[c] for c in uniq_codes]
            del uniq_codes
            n_beam_refs += len(beam_blocks)

            cost = VertexObservationCost(bases, start, end - start, len(beam_blocks))
            problem.add_residual_block(
                cost,
                # Ceres applies the loss to the squared norm of the whole
                # residual block: scale delta by sqrt(m) to keep the configured
                # per-observation robustness after batching.
                pyceres.HuberLoss(self.loss_delta * math.sqrt(end - start)),
                [rv_block, *beam_blocks, alpha],
            )

        del codes, beam_local, bases
        collect_garbage()

        if self.verbose:
            print(f"Built problem: {vertex_ids.size} vertex blocks, "
                  f"{n_beam_refs} beam block references, RSS {current_rss_mb():.1f} MB")

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
            print(f"Solve finished, RSS {current_rss_mb():.1f} MB")

        # --- Phase F: collect results and release everything heavy ---
        reflectivity = np.zeros(int(self.obs_vertex[-1]) + 1, dtype=np.float64)
        reflectivity[vertex_ids] = rv_buffer

        beam_port = np.fromiter(
            (beam_pool[c][0] for c in range(self.ang_samples)),
            dtype=np.float64, count=self.ang_samples,
        )
        beam_stbd = np.fromiter(
            (beam_pool[self.ang_samples + c][0] for c in range(self.ang_samples)),
            dtype=np.float64, count=self.ang_samples,
        )
        absorption = float(alpha[0])

        # Release the solver with its per-block structures, then the observation
        # columns the decomposition stage does not need (intensity and cosine are
        # solver inputs only; ping, bin, range, and vertex are kept).
        del problem, beam_pool, rv_buffer
        del self.obs_intensity, self.obs_cos_incidence
        collect_garbage()

        if self.verbose:
            print(f"Collected results, RSS {current_rss_mb():.1f} MB")

        return reflectivity, beam_port, beam_stbd, absorption, summary
