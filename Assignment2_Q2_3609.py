# EG/2019/3609 - Jayakody D.S.
# Assignment 2 - Question 2

# Importing necessary libraries
import numpy as np
import cv2
from queue import Queue
import matplotlib.pyplot as plt

# Create a noisy image with two objects for demonstration
img_original = np.zeros((250, 250), dtype=np.uint8)

# Drawing a filled hexagon
hexagon_center = (75, 75)
hexagon_radius = 40
hexagon_angle = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points and close the loop
hexagon_points = np.stack((np.cos(hexagon_angle), np.sin(hexagon_angle)), axis=1) * hexagon_radius
hexagon_points += np.array(hexagon_center)
hexagon_points = hexagon_points.astype(np.int32)
cv2.fillPoly(img_original, [hexagon_points], color=200)

# Drawing a filled square
square_top_left = (150, 150)
square_size = 70
square_points = np.array([
    square_top_left,
    (square_top_left[0] + square_size, square_top_left[1]),
    (square_top_left[0] + square_size, square_top_left[1] + square_size),
    (square_top_left[0], square_top_left[1] + square_size)
])
cv2.fillPoly(img_original, [square_points], color=150)

# Background with noise
img_original[img_original == 0] = 50
noise = np.random.normal(0, 15, img_original.shape)
img_noise = img_original + noise
img_noise = np.clip(img_noise, 0, 255)

# Seeds for the regions (specifically chosen within the objects)
seeds = [(75, 75), (175, 175)]  # Roughly center of hexagon and center of square
pixel_range = 50  # Range for pixel values
mask = np.zeros(img_noise.shape, dtype=np.uint8)

def check_range(pixel, seed, img):
    return abs(int(img[pixel[0], pixel[1]]) - int(img[seed[0], seed[1]])) <= pixel_range

# Region growing based on the defined range and seeds
def region_growing(img, seeds, pixel_range):
    for seed in seeds:
        pix_queue = Queue()
        pix_queue.put(seed)
        while not pix_queue.empty():
            current_pixel = pix_queue.get()
            if mask[current_pixel] == 0 and check_range(current_pixel, seed, img):
                mask[current_pixel] = 255  # Mark as part of the region
                # Check 8-connected neighbors
                neighbors = [(current_pixel[0] + dx, current_pixel[1] + dy)
                             for dx in range(-1, 2) for dy in range(-1, 2) if not (dx == 0 and dy == 0)]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < img.shape[0] and 0 <= neighbor[1] < img.shape[1]:
                        pix_queue.put(neighbor)

region_growing(img_noise, seeds, pixel_range)

# Displaying results
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img_noise, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Greyscale Image with Noise")
axs[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("Foreground Selected Using Region Growing")
plt.show()

