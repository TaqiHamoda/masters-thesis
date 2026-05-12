import cv2
import numpy as np
from ..dataset import Dataset


def export_first_return(
    dataset: Dataset,
    bin_offset: int,
    k_size: int = 25,
    look_ahead: int = 10,
    w_grad: float = 0.60,
    w_intensity: float = 0.20,
    w_idx_dist: float = 0.10,
    w_nadir_dist: float = 0.10,
):
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

    waterfall = cv2.imread(str(dataset.sonar_png), cv2.IMREAD_GRAYSCALE)
    waterfall = cv2.flip(waterfall, 0)
    waterfall = cv2.medianBlur(cv2.medianBlur(cv2.medianBlur(waterfall, 3), 5), 7)

    first_returns = np.zeros((waterfall.shape[0], 2))

    middle = waterfall.shape[1] // 2
    bin_range = (middle - bin_offset, middle + bin_offset)

    gradient = np.zeros_like(waterfall, dtype=np.float32)
    gradient[:, :bin_range[0]] = cv2.Sobel(waterfall[:, :bin_range[0]], ddepth=cv2.CV_32F, dx=1, dy=0)
    gradient[:, bin_range[1]:] = np.flip(cv2.Sobel(np.flip(waterfall[:, bin_range[1]:], axis=1), ddepth=cv2.CV_32F, dx=1, dy=0), axis=1)
    gradient = np.clip(-1 * gradient, 0, None)

    port_idx = np.argmax(gradient[0, :bin_range[0]])
    stbd_idx = np.argmax(gradient[0, bin_range[1]:]) + bin_range[1]

    waterfall = cv2.cvtColor(waterfall, cv2.COLOR_GRAY2BGR)

    first_returns[0, 0] = port_idx
    first_returns[0, 1] = stbd_idx

    def calculate_cost(start_bin, end_bin, i, index):
        dists = np.arange(start_bin, end_bin)
        index_dists = normalize(np.abs(dists - index))
        nadir_dists = normalize(np.abs(dists - middle))
        grads = 1 - normalize(gradient[i, start_bin:end_bin])
        intensities = 1 - normalize(np.array([
            np.abs(np.mean(waterfall[i, idx + look_ahead, 0]) - np.mean(waterfall[i, idx - look_ahead, 0]))
            for idx in dists
        ]))

        return w_grad * grads + w_intensity * intensities + w_idx_dist * index_dists + w_nadir_dist * nadir_dists

    for i in range(1, first_returns.shape[0]):
        start_bin = np.maximum(0, port_idx - k_size)
        end_bin = np.minimum(bin_range[0], port_idx + k_size)
        costs = calculate_cost(start_bin, end_bin, i, port_idx)
        port_idx += np.argmin(costs) - (port_idx - start_bin)

        start_bin = np.maximum(bin_range[1], stbd_idx - k_size)
        end_bin = np.minimum(gradient.shape[1], stbd_idx + k_size)
        costs = calculate_cost(start_bin, end_bin, i, stbd_idx)
        stbd_idx += np.argmin(costs) - (stbd_idx - start_bin)

        first_returns[i, 0] = port_idx
        first_returns[i, 1] = stbd_idx

        waterfall[i, port_idx - 5: port_idx + 5, 2] = 255
        waterfall[i, stbd_idx - 5: stbd_idx + 5, 2] = 255

    cv2.imwrite("temp.png", waterfall)
    np.savez(dataset.sonar_first_returns, data=first_returns)

