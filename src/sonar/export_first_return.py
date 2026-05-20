import cv2
import numpy as np

from ..dataset import Dataset


def export_first_return(
    dataset: Dataset,
    nadir_offset: int,
    radius: int = 100,
    w_grad: float = 0.50,
    w_nadir: float = 0.20,
    w_history: float = 0.30,
):
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    waterfall = cv2.imread(str(dataset.sonar_png), cv2.IMREAD_GRAYSCALE)
    waterfall = cv2.medianBlur(waterfall, 11)
    waterfall = clahe.apply(waterfall)

    first_returns = np.zeros((waterfall.shape[0], 2))

    middle = waterfall.shape[1] // 2
    bin_range = (middle - nadir_offset, middle + nadir_offset)

    gradient = np.zeros_like(waterfall, dtype=np.float32)
    for d in range(radius, waterfall.shape[1] - radius):
        gradient[:, d] = np.mean(waterfall[:, d - radius:d], axis=1) - np.mean(waterfall[:, d:d + radius], axis=1)
        gradient[:, d] = np.power(gradient[:, d], 2)

    start_idx = waterfall.shape[0] // 2

    port_idx = np.argmax(gradient[start_idx, :bin_range[0]])
    stbd_idx = np.argmax(gradient[start_idx, bin_range[1]:]) + bin_range[1]

    first_returns[start_idx, 0] = port_idx
    first_returns[start_idx, 1] = stbd_idx

    waterfall = cv2.cvtColor(waterfall, cv2.COLOR_GRAY2BGR)

    def calculate_cost(start_bin, end_bin, i, index, history):
        dists = np.arange(start_bin, end_bin, dtype=np.float32)

        prev_dists = np.abs(dists - index)
        anchor_dists = np.abs(dists - np.median(history))
        index_dists = normalize(np.power(0.6 * prev_dists + 0.4 * anchor_dists, 10))

        nadir_dists = normalize(np.power(dists - middle, 10))

        grad_dists = 1 - normalize(gradient[i, start_bin:end_bin])

        return w_grad * grad_dists + w_history * index_dists + w_nadir * nadir_dists

    def detect_bottom_track(start, end, step):
        nonlocal port_idx, stbd_idx

        for i in range(start, end, step):
            if step > 0:
                start_ping = max(start_idx, i - radius)
                history = first_returns[start_ping:i]
            else:
                start_ping = min(start_idx, i + radius)
                history = first_returns[i + 1:start_ping + 1]

            start_bin = np.maximum(0, port_idx - radius)
            end_bin = np.minimum(bin_range[0], port_idx + radius)
            port_costs = calculate_cost(start_bin, end_bin, i, port_idx, history[:, 0])
            port_start = port_idx - start_bin

            start_bin = np.maximum(bin_range[1], stbd_idx - radius)
            end_bin = np.minimum(waterfall.shape[1], stbd_idx + radius)
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
    first_returns[:, 1] = first_returns[:, 1] - middle

    np.savez(dataset.first_return, data=np.flip(first_returns, axis=0))
    cv2.imwrite(str(dataset.first_return_png), waterfall)
