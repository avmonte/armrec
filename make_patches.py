#!/usr/bin/env python
# coding: utf-8

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

import tools


def process_lines(rho_values, theta_values, bins_estimate, approx_value=np.pi / 2, padding=0.05):
    _, insights = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of rho and theta values
    rho_values = tuple(map(abs, rho_values))
    insights[0].scatter(theta_values, rho_values, marker='.', color='blue')
    insights[0].set_title('Scatter plot of rho and theta values')
    insights[0].set_xlabel('theta (radians)'); insights[0].set_ylabel('rho (pixels)')

    # Filter rho values based on a chosen theta range
    filtered_rho_values = tools.filter_rho_values(rho_values, theta_values, approx_value, padding)

    # Histogram of rho values per theta
    counts, bins, _ = insights[1].hist(filtered_rho_values, bins=bins_estimate, color='blue')
    insights[1].set_title(f'Histogram of rho values where theta in {approx_value - padding, approx_value + padding}')
    insights[1].set_xlabel('rho'); insights[1].set_ylabel('Frequency')

    plt.show()

    peaks, _ = tools.custom_find_peaks(counts, height=1, prominence=0.1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    peak_locations = bin_centers[peaks]

    lines_filtered = list(zip(peak_locations, [approx_value] * len(peak_locations)))

    return lines_filtered


def main(args):
    # Load and preprocess the image
    original_image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    print(original_image.shape)
    image = cv2.bitwise_not(original_image)
    image = cv2.medianBlur(image, tools.estimate_blur_radius(image))  # TODO

    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.show()

    # Apply Hough Transform
    edges = cv2.Canny(image, 50, 150)
    lines_ = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    if lines_ is None:
        print("No lines detected")
        exit()

    rho_values, theta_values = zip(*[line[0] for line in lines_])

    # Filter and process lines
    lines_filtered = process_lines(rho_values, theta_values, bins_estimate=tools.estimate_bins(image))
    tools.draw_lines(original_image, lines_filtered)

    # Generate patches
    tools.cut_image_with_horizontal_lines(original_image, lines_filtered, padding=5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='example/N-002_bw.jpg', help='Input image path')

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
