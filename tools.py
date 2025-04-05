from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def estimate_blur_radius(image):
    # FIXME: make direction-aware
    min_intensity = image.mean(axis=1).min()
    return 5 if min_intensity < 160 else 3


def estimate_bins(image):
    # FIXME: make direction-aware
    return image.shape[0] // 8


def filter_rho_values(rho_values, theta_values, approx_theta, padding):
    return [rho for theta, rho in zip(theta_values, rho_values) if abs(float(theta) - approx_theta) < padding]


def custom_find_peaks(counts, **kwargs):
    peaks, props = find_peaks(counts, **kwargs)
    peaks = list(peaks)

    # Check left edge
    if counts[0] > counts[1]:
        peaks.insert(0, 0)

    # Check right edge
    if counts[-1] > counts[-2]:
        peaks.append(len(counts) - 1)

    return np.array(peaks), props


def draw_lines(image, lines):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.title('Lines detected')
    plt.imshow(image_rgb)
    plt.show()


def cut_image_with_horizontal_lines(image, lines_rho_theta, padding=5, saveto=Path('patches')):
    # Get height of image
    h, w = image.shape[:2]

    # Convert rho-theta lines to y-coordinates (only horizontal-ish lines)
    y_coords = []
    for rho, theta in lines_rho_theta:
        # Filter horizontal lines (theta close to 90 degrees or pi/2 radians)
        if np.abs(np.cos(theta)) < 1e-2:
            y = int(rho / np.sin(theta))
            y_coords.append(y)

    if not y_coords:
        return []

    # Sort and clamp y values to image dimensions
    y_coords = sorted(set(np.clip(y_coords, 0, h - 1)))

    # Add artificial bounds at top and bottom
    y_coords = [0] + y_coords + [h]

    slices = []
    for i in range(1, len(y_coords)):
        y1 = max(y_coords[i - 1] - padding, 0)
        y2 = min(y_coords[i] + padding, h)
        if y2 > y1:
            slice_img = image[y1:y2]
            slices.append(slice_img)

    # save the slices
    saveto.mkdir(exist_ok=True)
    for i, slice_img in enumerate(slices):
        cv2.imwrite(str(saveto / f'patch_{i}.jpg'), slice_img)

    return slices
