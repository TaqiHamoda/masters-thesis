import numpy as np
import pyceres

from typing import Dict

from .dataset import Dataset, VertexHit

# Two-way absorption loss (20-log rule on voltage samples): 10^(-2 * alpha * r / 10)
# Its alpha derivative contributes (2 * ln(10) / 10) * r * loss per unit alpha
ALPHA_JAC = 2.0 * np.log(10.0) / 10.0

PORT = 0
STBD = 1


class ObservationCost(pyceres.CostFunction):
    """
    Residual for one vertex observation (Lambertian model):

        res = I - R_v * cos(theta) * G_channel[k(theta)] * 10^(-2 * alpha * r / 10)

    Parameter blocks: [reflectivity (1,), beam gain channel (ang_samples,), absorption coeff (1,)]
    """

    def __init__(self,
        intensity: float,
        beam_bin: int,
        slant_range: float,
        cos_incidence: float,
        ang_samples: int,
    ):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, ang_samples, 1])

        self.intensity = float(intensity)
        self.beam_bin = int(beam_bin)
        self.slant_range = float(slant_range)
        self.cos_incidence = float(cos_incidence)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        reflectivity = parameters[0][0]
        beam_gain = parameters[1][self.beam_bin]
        alpha = parameters[2][0]

        # Lambertian scattering factor from the known incidence angle
        scatter = self.cos_incidence
        # Loss factor starts at 1.0 by definition (0 dB at zero range)
        loss = 10.0 ** (-2.0 * alpha * self.slant_range / 10.0)
        predicted = reflectivity * scatter * beam_gain * loss

        residuals[0] = self.intensity - predicted

        if jacobians is not None:
            # d res / d R = -G[k] * cos(theta) * L
            if jacobians[0] is not None:
                jacobians[0][0] = -beam_gain * scatter * loss
            # d res / d G[k] = -R * cos(theta) * L (rest of the beam row stays zero)
            if jacobians[1] is not None:
                jacobians[1][:] = 0.0
                jacobians[1][self.beam_bin] = -reflectivity * scatter * loss
            # d res / d alpha = (2 * ln(10) / 10) * r * predicted
            if jacobians[2] is not None:
                jacobians[2][0] = ALPHA_JAC * self.slant_range * predicted
        return True


class BeamPriorCost(pyceres.CostFunction):
    """
    Anchors the global scale gauge between reflectivity and beam pattern:

        res = (mean(beam) - 1) * gauge_weight

    Parameter blocks: [beam gain channel (ang_samples,)]
    """

    def __init__(self, ang_samples: int, gauge_weight: float):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([ang_samples])

        self.ang_samples = int(ang_samples)
        self.gauge_weight = float(gauge_weight)

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        residuals[0] = (float(np.mean(parameters[0])) - 1.0) * self.gauge_weight

        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = self.gauge_weight / self.ang_samples
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

    def load_observations(self):
        waterfall = np.load(self.dataset.sonar_file)["data"]
        n_pings, n_bins = waterfall.shape

        # Ping index -> number of samples per channel (port | starboard split)
        num_samples_map: Dict[int, int] = {
            sss.ping_idx: sss.num_samples
            for sss in self.dataset.sonar.values()
        }

        obs_vertex = []
        obs_channel = []
        obs_beam_bin = []
        obs_cos_incidence = []
        obs_range = []
        obs_intensity = []
        obs_ping_idx = []
        obs_bin_idx = []

        for matches_file in sorted(self.dataset.vertex_matches_dir.glob("*.csv")):
            for vertex in VertexHit.from_csv(matches_file):
                if vertex.hit.ping_idx < 0 or vertex.hit.ping_idx >= n_pings:
                    continue
                if vertex.hit.bin_idx < 0 or vertex.hit.bin_idx >= n_bins:
                    continue

                num_samples = num_samples_map.get(vertex.hit.ping_idx)
                if num_samples is None or num_samples <= 0:
                    continue

                channel = PORT if vertex.hit.bin_idx < num_samples else STBD

                # Beam pattern lookup bin: theta in [0, max_angle) -> [0, ang_samples)
                beam_bin = int(vertex.hit.incidence_angle / self.max_angle * self.ang_samples)
                beam_bin = min(max(beam_bin, 0), self.ang_samples - 1)

                cos_incidence = min(max(np.cos(vertex.hit.incidence_angle), 0.0), 1.0)

                obs_vertex.append(vertex.vertex_idx)
                obs_channel.append(channel)
                obs_beam_bin.append(beam_bin)
                obs_cos_incidence.append(cos_incidence)
                obs_range.append(vertex.hit.distance)
                obs_ping_idx.append(vertex.hit.ping_idx)
                obs_bin_idx.append(vertex.hit.bin_idx)
                # Gather only observed samples, kept as float64
                obs_intensity.append(float(waterfall[vertex.hit.ping_idx, vertex.hit.bin_idx]))

        if not obs_intensity:
            raise ValueError("No valid vertex observations found for bundle adjustment.")

        self.obs_vertex = np.array(obs_vertex, dtype=np.int64)
        self.obs_channel = np.array(obs_channel, dtype=np.int64)
        self.obs_beam_bin = np.array(obs_beam_bin, dtype=np.int64)
        self.obs_cos_incidence = np.array(obs_cos_incidence, dtype=np.float64)
        self.obs_range = np.array(obs_range, dtype=np.float64)
        self.obs_intensity = np.array(obs_intensity, dtype=np.float64)
        self.obs_ping_idx = np.array(obs_ping_idx, dtype=np.int64)
        self.obs_bin_idx = np.array(obs_bin_idx, dtype=np.int64)

        if self.verbose:
            print(f"Loaded {len(self.obs_intensity)} vertex observations "
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

        n_vertices = int(self.obs_vertex.max()) + 1
        sums = np.zeros(n_vertices, dtype=np.float64)
        counts = np.zeros(n_vertices, dtype=np.int64)
        np.add.at(sums, self.obs_vertex, weighted)
        np.add.at(counts, self.obs_vertex, 1)

        reflectivity = np.zeros(n_vertices, dtype=np.float64)
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

        # --- Global parameter blocks ---
        beam_port = np.ones(self.ang_samples, dtype=np.float64)
        beam_stbd = np.ones(self.ang_samples, dtype=np.float64)
        alpha = np.array([self.absorption_coeff], dtype=np.float64)

        problem.add_parameter_block(beam_port, self.ang_samples)
        problem.add_parameter_block(beam_stbd, self.ang_samples)
        problem.add_parameter_block(alpha, 1)

        problem.set_parameter_lower_bound(alpha, 0, 0.0)
        for i in range(self.ang_samples):
            problem.set_parameter_lower_bound(beam_port, i, 0.0)
            problem.set_parameter_lower_bound(beam_stbd, i, 0.0)

        # Pull the shared scale ambiguity between reflectivity and
        # beam gains towards beam patterns with unit mean.
        problem.add_residual_block(
            BeamPriorCost(self.ang_samples, self.gauge_weight), None, [beam_port]
        )
        problem.add_residual_block(
            BeamPriorCost(self.ang_samples, self.gauge_weight), None, [beam_stbd]
        )

        # --- Local (per-vertex) parameter blocks and observation residuals ---
        init_reflectivity = self.initial_reflectivity()
        loss = pyceres.HuberLoss(self.loss_delta)

        rv_blocks = {}
        for i in range(len(self.obs_intensity)):
            v = int(self.obs_vertex[i])
            rv_block = rv_blocks.get(v)
            if rv_block is None:
                rv_block = np.array([init_reflectivity[v]], dtype=np.float64)
                rv_blocks[v] = rv_block
                problem.add_parameter_block(rv_block, 1)
                problem.set_parameter_lower_bound(rv_block, 0, 0.0)

            cost = ObservationCost(
                intensity=self.obs_intensity[i],
                beam_bin=self.obs_beam_bin[i],
                slant_range=self.obs_range[i],
                cos_incidence=self.obs_cos_incidence[i],
                ang_samples=self.ang_samples,
            )

            beam_block = beam_port if self.obs_channel[i] == PORT else beam_stbd
            problem.add_residual_block(cost, loss, [rv_block, beam_block, alpha])

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
        n_vertices = int(self.obs_vertex.max()) + 1
        reflectivity = np.zeros(n_vertices, dtype=np.float64)
        for v, block in rv_blocks.items():
            reflectivity[v] = block[0]

        return reflectivity, beam_port, beam_stbd, float(alpha[0]), summary
