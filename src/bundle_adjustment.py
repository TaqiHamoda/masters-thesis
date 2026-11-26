import csv
import gc
import math

import numpy as np
from numpy.polynomial.chebyshev import chebvander
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

# Target observations per residual block. Large enough to amortize the
# Python/pyceres call overhead (~50 us/call), small enough that the
# per-chunk dense Jacobian (n x 21 doubles) stays in L2/L3 cache:
# 32768 * 21 * 8 B ~= 5.5 MB.
CHUNK_TARGET_OBS = 32768


# =====================================================================
# 1. Verbose diagnostics: depression-angle spread + observation counts
# =====================================================================

def per_vertex_iqr(u_sorted: np.ndarray, group_starts: np.ndarray,
                   group_lengths: np.ndarray) -> np.ndarray:
    """Per-vertex interquartile range of u, vectorized.

    u must be sorted by vertex AND by u within each vertex; use
    np.lexsort((u, vertex)) on the vertex-sorted observation arrays.
    Positional quartiles (floor at 25%, ceil at 75%); single-observation
    vertices get IQR = 0. u = 4*phi/pi - 1, so multiply the result by
    pi/4 to obtain radians of depression angle.
    """
    pos_lo = group_starts + ((group_lengths - 1) * 0.25).astype(np.int64)
    pos_hi = group_starts + np.ceil((group_lengths - 1) * 0.75).astype(np.int64)
    return u_sorted[pos_hi] - u_sorted[pos_lo]


def log_vertex_diagnostics(obs_vertex: np.ndarray,
                           obs_depression_angle: np.ndarray,
                           group_starts: np.ndarray,
                           group_lengths: np.ndarray,
                           stream=print) -> None:
    """Verbose-mode report: obs-count distribution and angle spread.

    Call after Phase B (arrays already sorted by vertex). Peak extra
    memory is one int64 permutation + one float64 copy of the angle
    array (~2.2 GB at 138M observations); it is released on return.
    """
    lengths = group_lengths.astype(np.int64)

    order = np.lexsort((obs_depression_angle, obs_vertex))
    u_sorted = obs_depression_angle[order]
    del order
    iqr_u = per_vertex_iqr(u_sorted, group_starts, lengths)
    del u_sorted

    # depression-angle spread in radians (u = 4*phi/pi - 1)
    iqr_phi = iqr_u * (np.pi / 4.0)
    multi = lengths > 1  # singletons have IQR = 0 by construction

    q = np.percentile
    stream("Vertex diagnostics:")
    stream(f"  observations/vertex: min {lengths.min()}, "
           f"p1 {q(lengths, 1):.0f}, median {np.median(lengths):.0f}, "
           f"p99 {q(lengths, 99):.0f}, max {lengths.max()}")
    stream(f"  single-observation vertices: {int((~multi).sum())} "
           f"({100.0 * (~multi).mean():.2f}%)")
    stream(f"  depression-angle IQR (rad), vertices with n>1: "
           f"median {np.median(iqr_phi[multi]):.4f}, "
           f"p25 {q(iqr_phi[multi], 25):.4f}, "
           f"p75 {q(iqr_phi[multi], 75):.4f}, "
           f"p99 {q(iqr_phi[multi], 99):.4f}")
    # Vertices with a large IQR are the ones that actually identify the
    # beam pattern; if the median IQR is small, most vertices are nearly
    # single-angle and the pattern is only weakly constrained (cf. the
    # identifiability discussion in the review canvas).


# =====================================================================
# 2+3+4. Chunked cost function with per-observation residuals and
#        nonnegative reflectivity
# =====================================================================

class VertexChunkCost(pyceres.CostFunction):
    """Per-observation least-squares cost for a contiguous chunk of vertices.

    The chunk holds the observations of several consecutive vertices
    (observation arrays are sorted by vertex). Reflectivity is still
    eliminated analytically per vertex, so the parameter blocks remain
    only the global ones:

        [beam_port_coeffs (d+1), beam_stbd_coeffs (d+1), alpha (1)]
        (drop the channel blocks that are absent from this chunk)

    Residual layout:
      per_observation=True  : one residual per observation, e_k = I_k -
          R_v* W_k. This is what makes Huber act per observation.
      per_observation=False : one residual per vertex, the RMS error
          sqrt(sum_k e_k^2 / n_v). Memory stays at the per-vertex level
          and Huber's delta becomes independent of the observation count;
          the objective then weights each vertex by its mean squared
          error (equal weight per seabed patch).

    Nonnegativity: R_v* = max(0, sum I_k W_k / sum W_k^2) is the exact
    1-D NNLS solution. For uint16-derived intensities R* >= 0 almost
    always holds anyway (unless noise drives sum(IW) < 0); the clip is
    insurance, and the Jacobian below is exact on the clipped side.

    seg_starts/seg_lengths are LOCAL to the chunk (seg_starts[0] == 0).
    """

    def __init__(self, start: int, end: int,
                 seg_starts: np.ndarray, seg_lengths: np.ndarray,
                 obs_intensity: np.ndarray,
                 obs_cos_incidence: np.ndarray,
                 obs_depression_angle: np.ndarray,
                 obs_range: np.ndarray,
                 obs_channel: np.ndarray,
                 degree: int,
                 port_present: bool,
                 stbd_present: bool,
                 per_observation: bool = True):
        pyceres.CostFunction.__init__(self)

        self.start = int(start)
        self.end = int(end)
        self.n = self.end - self.start
        self.V = len(seg_starts)
        self.seg_starts = np.asarray(seg_starts, dtype=np.int64)
        self.seg_lengths = np.asarray(seg_lengths, dtype=np.int64)
        # local observation -> local vertex index
        self.seg_ids = np.repeat(np.arange(self.V), self.seg_lengths)
        self.obs_intensity = obs_intensity            # references, not copies
        self.obs_cos_incidence = obs_cos_incidence
        self.obs_depression_angle = obs_depression_angle
        self.obs_range = obs_range
        self.obs_channel = obs_channel
        self.degree = int(degree)
        self.port_present = bool(port_present)
        self.stbd_present = bool(stbd_present)
        self.per_observation = bool(per_observation)

        if self.per_observation:
            self.set_num_residuals(self.n)
        else:
            self.set_num_residuals(self.V)
        beam_sizes = [degree + 1] * (int(port_present) + int(stbd_present))
        self.set_parameter_block_sizes(beam_sizes + [1])

        self.port_slot = 0
        self.stbd_slot = 1 if port_present else 0

    # -----------------------------------------------------------------

    def Evaluate(self, parameters, residuals, jacobians) -> bool:
        s0, s1 = self.start, self.end
        I = self.obs_intensity[s0:s1]
        cos_t = self.obs_cos_incidence[s0:s1]
        u = self.obs_depression_angle[s0:s1]
        r = self.obs_range[s0:s1]
        ch = self.obs_channel[s0:s1]
        n = self.n
        d = self.degree

        port_c = parameters[self.port_slot] if self.port_present else None
        stbd_c = parameters[self.stbd_slot] if self.stbd_present else None
        alpha_val = parameters[-1][0]

        T = chebvander(u, d)                                # (n, d+1)
        is_port = ch == 0

        # channel-matched log gains (clipped exactly like the old code)
        if self.port_present and self.stbd_present:
            log_g = np.where(is_port, T @ port_c, T @ stbd_c)
        elif self.port_present:
            log_g = T @ port_c
        else:
            log_g = T @ stbd_c
        G = np.exp(np.clip(log_g, -10.0, 10.0))
        W = cos_t * np.exp(-ALPHA_JAC * alpha_val * r) * G  # (n,)

        starts = self.seg_starts                           # already local
        S = np.add.reduceat(W * W, starts)                  # sum W^2     (V,)
        SIW = np.add.reduceat(I * W, starts)               # sum I W     (V,)
        SIRW = np.add.reduceat(I * W * r, starts)          # sum I r W   (V,)
        SRW2 = np.add.reduceat(W * W * r, starts)           # sum r W^2   (V,)

        # ---- per-vertex eliminated reflectivity, clipped nonnegative ----
        R_raw = np.zeros(self.V)
        np.divide(SIW, S, out=R_raw, where=S > 0)
        R = np.maximum(R_raw, 0.0)
        active = (S > 0) & (R_raw > 0.0)                   # interior of clip

        # ---- dR*/dp on active vertices (envelope theorem) ----
        dR_alpha = np.zeros(self.V)
        np.divide(-ALPHA_JAC * SIRW + 2.0 * ALPHA_JAC * R * SRW2, S,
                  out=dR_alpha, where=active)
        if self.port_present:
            IW_T_p = np.add.reduceat(
                (I * W)[:, None] * T * is_port[:, None], starts, axis=0)
            W2_T_p = np.add.reduceat(
                (W * W)[:, None] * T * is_port[:, None], starts, axis=0)
            dR_port = np.zeros((self.V, d + 1))
            np.divide(IW_T_p - 2.0 * R[:, None] * W2_T_p, S[:, None],
                      out=dR_port, where=active[:, None])
        if self.stbd_present:
            IW_T_s = np.add.reduceat(
                (I * W)[:, None] * T * (~is_port)[:, None], starts, axis=0)
            W2_T_s = np.add.reduceat(
                (W * W)[:, None] * T * (~is_port)[:, None], starts, axis=0)
            dR_stbd = np.zeros((self.V, d + 1))
            np.divide(IW_T_s - 2.0 * R[:, None] * W2_T_s, S[:, None],
                      out=dR_stbd, where=active[:, None])

        e = I - R[self.seg_ids] * W   # clipped/zero-weight rows -> e = I

        if not self.per_observation:
            self._evaluate_rms(e, W, r, R, active, dR_alpha,
                               dR_port if self.port_present else None,
                               dR_stbd if self.stbd_present else None,
                               T, is_port, residuals, jacobians)
            return True

        residuals[:] = e

        if jacobians is None:
            return True

        Rb = R[self.seg_ids]                                 # (n,)
        if self.port_present:
            Jp = np.where(is_port[:, None], -(Rb * W)[:, None] * T, 0.0)
            Jp -= W[:, None] * dR_port[self.seg_ids]     # envelope term: all rows
            jb = jacobians[self.port_slot]
            if jb is not None:
                jb[:] = Jp.reshape(-1)                  # row-major (n, d+1)
        if self.stbd_present:
            Js = np.where((~is_port)[:, None], -(Rb * W)[:, None] * T, 0.0)
            Js -= W[:, None] * dR_stbd[self.seg_ids]
            jb = jacobians[self.stbd_slot]
            if jb is not None:
                jb[:] = Js.reshape(-1)
        Ja = ALPHA_JAC * r * Rb * W - W * dR_alpha[self.seg_ids]
        jb = jacobians[-1]
        if jb is not None:
            jb[:] = Ja
        return True

    # -----------------------------------------------------------------

    def _evaluate_rms(self, e, W, r, R, active, dR_alpha, dR_port, dR_stbd,
                      T, is_port, residuals, jacobians) -> None:
        """One RMS residual per vertex: sqrt(sum e^2 / n_v).

        dr_v/dp = (1 / (n_v r_v)) * sum_k e_k de_k/dp on active vertices;
        clipped/zero-weight vertices have e = I constant in p, so their
        rows are exactly zero.
        """
        starts = self.seg_starts
        nv = self.seg_lengths.astype(np.float64)
        ssq = np.add.reduceat(e * e, starts)
        res = np.zeros(self.V)
        np.divide(ssq, nv, out=res)
        res = np.sqrt(res)
        residuals[:] = res

        if jacobians is None:
            return

        ok = res > 0  # perfect fits: zero row
        inv = np.zeros(self.V)
        np.divide(1.0, nv * res, out=inv, where=ok)

        eW = e * W
        EWsum = np.add.reduceat(eW, starts)               # sum e W      (V,)

        if self.port_present:
            # A_j(v) = sum_k e_k de_k/dbeta_j on active vertices
            #   de_k/dbeta_j = -R* W_k T_j (channel match) - W_k dR*/dbeta_j
            EWT_p = np.add.reduceat(
                (eW[:, None] * T * is_port[:, None]), starts, axis=0)
            A = -(R[:, None] * EWT_p + EWsum[:, None] * dR_port)
            A[~active] = 0.0
            jb = jacobians[self.port_slot]
            if jb is not None:
                jb[:] = ((A * inv[:, None]).reshape(-1))
        if self.stbd_present:
            EWT_s = np.add.reduceat(
                (eW[:, None] * T * (~is_port)[:, None]), starts, axis=0)
            A = -(R[:, None] * EWT_s + EWsum[:, None] * dR_stbd)
            A[~active] = 0.0
            jb = jacobians[self.stbd_slot]
            if jb is not None:
                jb[:] = ((A * inv[:, None]).reshape(-1))
        ErW = np.add.reduceat(e * r * W, starts)
        A = ALPHA_JAC * R * ErW - EWsum * dR_alpha
        A[~active] = 0.0
        jb = jacobians[-1]
        if jb is not None:
            jb[:] = A * inv


def build_chunks(group_starts: np.ndarray, group_ends: np.ndarray,
                 target_obs: int = CHUNK_TARGET_OBS):
    """Pack consecutive vertex groups into chunks of ~target_obs observations.

    A group is never split across chunks. Returns a list of
    (obs_start, obs_end, group_lo, group_hi) tuples; slice the group
    arrays with [group_lo:group_hi] to get the chunk's vertex groups.
    """
    V = len(group_starts)
    chunks = []
    g0 = 0
    while g0 < V:
        g1 = g0 + 1
        while g1 < V and (group_ends[g1 - 1] - group_starts[g0]) < target_obs:
            g1 += 1
        chunks.append((int(group_starts[g0]), int(group_ends[g1 - 1]), g0, g1))
        g0 = g1
    return chunks


# =====================================================================
# Phase E replacement: vectorized, nonnegative, instability-flagged
# =====================================================================

def extract_reflectivity(obs_intensity: np.ndarray,
                         obs_cos_incidence: np.ndarray,
                         obs_depression_angle: np.ndarray,
                         obs_range: np.ndarray,
                         obs_channel: np.ndarray,
                         group_starts: np.ndarray,
                         vertex_ids: np.ndarray,
                         beam_port_coeffs: np.ndarray,
                         beam_stbd_coeffs: np.ndarray,
                         alpha_val: float,
                         degree: int,
                         w2_floor: float = 1e-12,
                         min_correlation: float = 0.2,
                         verbose: bool = True):
    """Final R_v* for every vertex, in one vectorized pass.

    A vertex is marked unreliable and its reflectivity set to 0 when
    either
      (a) sum W^2 <= w2_floor: the weights are numerically zero (grazing
          angles, extreme range, or the fitted pattern G ~ 0 there), so
          R* = sum(IW)/sum(W^2) is pure amplification, or
      (b) the Cauchy-Schwarz cosine
              rho = sum(IW) / sqrt(sum(W^2) sum(I^2)) < min_correlation,
          i.e. the intensity vector is nearly orthogonal to the weight
          vector: the model explains almost none of the data, and R* is
          again amplifying noise. Mathematically R* sqrt(S) <= rms(I)
          always holds, so large R* with small rho is the signature of
          the 3e8-style blow-ups.

    Returns (reflectivity indexed by vertex id, stats dict).
    """
    T = chebvander(obs_depression_angle, degree)
    log_g = np.where(obs_channel == 0, T @ beam_port_coeffs,
                    T @ beam_stbd_coeffs)
    G = np.exp(np.clip(log_g, -10.0, 10.0))       # match the cost's clamp
    W = obs_cos_incidence * np.exp(-ALPHA_JAC * alpha_val * obs_range) * G

    S = np.add.reduceat(W * W, group_starts)       # (V,)
    SIW = np.add.reduceat(obs_intensity * W, group_starts)
    SI2 = np.add.reduceat(obs_intensity * obs_intensity, group_starts)

    V = len(group_starts)
    R_raw = np.zeros(V)
    np.divide(SIW, S, out=R_raw, where=S > 0)

    rho = np.zeros(V)
    denom = np.sqrt(S * SI2)
    np.divide(SIW, denom, out=rho, where=denom > 0)

    reliable = (S > w2_floor) & (rho >= min_correlation)
    R = np.where(reliable, np.maximum(R_raw, 0.0), 0.0)

    reflectivity = np.zeros(int(obs_vertex_max(vertex_ids)) + 1,
                            dtype=np.float64)
    reflectivity[vertex_ids] = R

    stats = {
        "n_vertices": V,
        "n_zero_weight": int((S <= w2_floor).sum()),
        "n_low_correlation": int(((S > w2_floor) & (rho < min_correlation)).sum()),
        "n_negative_clipped": int((R_raw < 0).sum()),
        "n_unreliable": int((~reliable).sum()),
    }
    if verbose:
        print("Reflectivity extraction:")
        print(f"  unreliable vertices: {stats['n_unreliable']}/{V} "
              f"({stats['n_zero_weight']} near-zero weight, "
              f"{stats['n_low_correlation']} low correlation), "
              f"{stats['n_negative_clipped']} clipped at R=0")
    return reflectivity, stats


def obs_vertex_max(vertex_ids: np.ndarray) -> int:
    return int(vertex_ids[-1]) if len(vertex_ids) else -1


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
        T_basis = chebvander(u, self.degree)

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

         # --- Phase C: build chunked per-observation residual blocks ---
        # One pyceres cost per ~32k observations (a vertex is never split),
        # amortizing the per-call Python overhead of 19M+ vertices down to
        # a few thousand calls. Residuals are per OBSERVATION, so Huber
        # acts on individual observation errors, and R_v* is clipped
        # nonnegative (exact 1-D NNLS).
        # per_observation=False switches to one RMS residual per vertex:
        # ~3.2 GB Jacobian instead of ~23 GB at 138M observations, and
        # Huber's delta becomes independent of the observation count.
        per_observation = True

        chunks = build_chunks(group_starts, group_ends, CHUNK_TARGET_OBS)
        huber = pyceres.HuberLoss(self.loss_delta)   # shared instance
        n_port = n_stbd = 0
        for cs, ce, g0, g1 in chunks:
            ch_slice = self.obs_channel[cs:ce]
            port_present = bool(np.any(ch_slice == 0))
            stbd_present = bool(np.any(ch_slice == 1))
            n_port += port_present
            n_stbd += stbd_present

            cost = VertexChunkCost(
                cs, ce,
                group_starts[g0:g1] - cs,                  # local group starts
                group_ends[g0:g1] - group_starts[g0:g1],   # group lengths
                self.obs_intensity,
                self.obs_cos_incidence,
                self.obs_depression_angle,
                self.obs_range,
                self.obs_channel,
                self.beam_degree,
                port_present,
                stbd_present,
                per_observation=per_observation,
            )

            blocks = ([beam_port_coeffs] if port_present else []) \
                     + ([beam_stbd_coeffs] if stbd_present else []) \
                     + [alpha]
            problem.add_residual_block(cost, huber, blocks)

        gc.collect()

        if self.verbose:
            print(f"Built problem: {vertex_ids.size} vertices in {len(chunks)} "
                  f"chunk blocks ({n_port} chunks referencing port, "
                  f"{n_stbd} starboard), degree {self.beam_degree}, "
                  f"{'per-observation' if per_observation else 'per-vertex RMS'} "
                  f"residuals")
            log_vertex_diagnostics(self.obs_vertex,
                                   self.obs_depression_angle,
                                   group_starts,
                                   group_ends - group_starts)

        # --- Phase D: solve ---
        options = pyceres.SolverOptions()
        # Only 15 global parameters, no point/edge blocks: DENSE_QR is appropriate
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.max_num_iterations = self.max_iterations
        options.num_threads = self.num_threads
        options.minimizer_progress_to_stdout = self.verbose
        options.function_tolerance = 1e-10
        options.gradient_tolerance = 1e-6
        options.parameter_tolerance = 1e-10
        options.initial_trust_region_radius = 10.0

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        cov_opts = pyceres.CovarianceOptions()
        cov_opts.apply_loss_function = True    # see caveat below
        # cov_opts.num_threads = 1               # Python cost functions: GIL-serialized anyway
        cov = pyceres.Covariance(cov_opts)
        pairs = [(beam_port_coeffs, beam_port_coeffs), (beam_stbd_coeffs, beam_stbd_coeffs),
                (beam_port_coeffs, beam_stbd_coeffs), (alpha, alpha),
                (beam_port_coeffs, alpha), (beam_stbd_coeffs, alpha)]
        if cov.compute(pairs, problem):
            C_pp = cov.get_covariance_block(beam_port_coeffs, beam_port_coeffs)  # (d+1, d+1)
            C_aa = cov.get_covariance_block(alpha, alpha)                        # (1, 1)
            C_pa = cov.get_covariance_block(beam_port_coeffs, alpha)             # (d+1, 1)
            # marginal standard errors:

            se_port = np.sqrt(np.diag(C_pp))
            se_alpha = np.sqrt(C_aa[0, 0])
            print(f"Covariance computed: SE port={se_port}, SE alpha={se_alpha}")

        if self.verbose:
            print(summary.BriefReport())
            print(f"Solve finished")

        # --- Phase E: compute final reflectivity values (vectorized) ---
        # R_v* = max(0, sum IW / sum W^2). Vertices whose weights are
        # numerically ~0 (grazing angles, extreme range, fitted G ~ 0) or whose
        # weight/intensity vectors are nearly uncorrelated get R = 0 and are
        # counted in the verbose report: this is the 3e8-max fix.
        reflectivity, refl_stats = extract_reflectivity(
            self.obs_intensity,
            self.obs_cos_incidence,
            self.obs_depression_angle,
            self.obs_range,
            self.obs_channel,
            group_starts,
            vertex_ids,
            beam_port_coeffs,
            beam_stbd_coeffs,
            float(alpha[0]),
            self.beam_degree,
            verbose=self.verbose,
        )

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

        delta = -0.5 * (beam_port[0] + beam_stbd[0])   # shift the MEAN c_0 to zero
        beam_port[0] += delta
        beam_stbd[0] += delta
        reflectivity *= np.exp(-delta)                 # compensates G *= exp(delta)

        return reflectivity, beam_port, beam_stbd, absorption, summary
