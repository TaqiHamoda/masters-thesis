import cv2
import numpy as np
from ..dataset import Dataset


def export_first_return(
    dataset: Dataset,
    bin_offset: int,
    k_size: int = 100,
    lookahead: int = 5,
    w_grad: float = 0.50,
    w_idx_dist: float = 0.30,
    w_nadir_dist: float = 0.20,
):
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    waterfall = cv2.imread(str(dataset.sonar_png), cv2.IMREAD_GRAYSCALE)
    waterfall = cv2.medianBlur(waterfall, 11)
    waterfall = clahe.apply(waterfall)

    first_returns = np.zeros((waterfall.shape[0], 2))

    middle = waterfall.shape[1] // 2
    bin_range = (middle - bin_offset, middle + bin_offset)

    gradient = np.zeros_like(waterfall, dtype=np.float32)
    gradient[:, :bin_range[0]] = cv2.Sobel(waterfall[:, :bin_range[0]], ddepth=cv2.CV_32F, dx=1, dy=0)
    gradient[:, bin_range[1]:] = np.flip(cv2.Sobel(np.flip(waterfall[:, bin_range[1]:], axis=1), ddepth=cv2.CV_32F, dx=1, dy=0), axis=1)
    gradient = np.clip(-1 * gradient, 0, None)

    waterfall = cv2.cvtColor(waterfall, cv2.COLOR_GRAY2BGR)

    start_idx = waterfall.shape[0] // 2

    port_idx = np.argmax(gradient[start_idx, :bin_range[0]])
    stbd_idx = np.argmax(gradient[start_idx, bin_range[1]:]) + bin_range[1]

    first_returns[start_idx, 0] = port_idx
    first_returns[start_idx, 1] = stbd_idx

    def calculate_cost(start_bin, end_bin, i, index, history):
        dists = np.arange(start_bin, end_bin, dtype=np.float32)

        prev_dists = np.abs(dists - index)
        anchor_dists = np.abs(dists - np.median(history))
        index_dists = normalize(np.power(0.6 * prev_dists + 0.4 * anchor_dists, 2))

        nadir_dists = normalize(np.abs(dists - middle))

        grads = 1 - normalize(gradient[i, start_bin:end_bin])
        diffs = 1 - normalize([
            np.power(np.mean(waterfall[i, j - lookahead:j, 0]) - np.mean(waterfall[i, j:j + lookahead, 0]), 2)
            for j in range(start_bin, end_bin)
        ])
        grad_dists = 0.5 * grads + 0.5 * diffs

        return w_grad * grad_dists + w_idx_dist * index_dists + w_nadir_dist * nadir_dists

    def detect_bottom_track(start, end, step):
        nonlocal port_idx, stbd_idx

        for i in range(start, end, step):
            if step > 0:
                start_ping = max(start_idx, i - k_size)
                history = first_returns[start_ping:i]
            else:
                start_ping = min(start_idx, i + k_size)
                history = first_returns[i + 1:start_ping + 1]

            start_bin = np.maximum(0, port_idx - k_size)
            end_bin = np.minimum(bin_range[0], port_idx + k_size)
            port_costs = calculate_cost(start_bin, end_bin, i, port_idx, history[:, 0])
            port_start = port_idx - start_bin

            start_bin = np.maximum(bin_range[1], stbd_idx - k_size)
            end_bin = np.minimum(gradient.shape[1], stbd_idx + k_size)
            stbd_costs = calculate_cost(start_bin, end_bin, i, stbd_idx, history[:, 1])
            stbd_start = stbd_idx - start_bin

            port_idx += np.argmin(port_costs) - port_start
            stbd_idx += np.argmin(stbd_costs) - stbd_start

            first_returns[i, 0] = port_idx
            first_returns[i, 1] = stbd_idx

            waterfall[i, port_idx - 5: port_idx + 5, 2] = 255
            waterfall[i, stbd_idx - 5: stbd_idx + 5, 2] = 255

    detect_bottom_track(start_idx - 1, -1, -1)
    detect_bottom_track(start_idx + 1, first_returns.shape[0], 1)

    # Find distance from nadir
    first_returns[:, 0] = middle - first_returns[:, 0]
    first_returns[:, 1] = middle + first_returns[:, 1]

    np.savez(dataset.first_return, data=np.flip(first_returns, axis=0))
    cv2.imwrite(str(dataset.first_return_png), waterfall)
