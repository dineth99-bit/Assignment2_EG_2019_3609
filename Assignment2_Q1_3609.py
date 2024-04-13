# EG/2019/3609 - Jayakody D.S.
# Assignment 2 - Question 1


# Importing necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt


## Part 1 - Generating an image with 2 objects and a total of 3-pixel values ##
##############################################################################

# Creating a black image of size 250 x 250
image_with_shapes = np.zeros((250, 250), dtype=np.uint8)

# Drawing a filled hexagon with pixel value of 200
# Hexagon centered at (75, 75) with a radius of 40 pixels
radius_hexagon = 40
center_hexagon = (75, 75)
angles_hexagon = np.linspace(0, 2 * np.pi, 7)[:-1]
points_hexagon = np.stack((np.cos(angles_hexagon), np.sin(angles_hexagon)), axis=1) * radius_hexagon
points_hexagon += np.array(center_hexagon)
points_hexagon = points_hexagon.astype(np.int32)
cv2.fillPoly(image_with_shapes, [points_hexagon], color=200)

# Drawing a filled square with pixel value of 150
# Square top-left corner at (150, 150) with side length 70 pixels
size_square = 70
top_left_square = (150, 150)
points_square = np.array([
    top_left_square,
    (top_left_square[0] + size_square, top_left_square[1]),
    (top_left_square[0] + size_square, top_left_square[1] + size_square),
    (top_left_square[0], top_left_square[1] + size_square)
])
cv2.fillPoly(image_with_shapes, [points_square], color=150)

# Setting the background to pixel value
image_with_shapes[image_with_shapes == 0] = 50



## Part 2 - Adding Gaussian noise to the image ##
##############################################################################


# Adding Gaussian noise
noise_intensity = np.sqrt(500)
noise_gaussian = np.random.normal(0, noise_intensity, image_with_shapes.shape)
noisy_image = noise_gaussian + image_with_shapes
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Display original image and the image with Gaussian noise
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(image_with_shapes, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Original Image with Hexagon and Square")
axs[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("Image with Added Gaussian Noise")
plt.show()



## Part 3 - . Implementing and testing Otsu’s algorithm with the image ##
##############################################################################

# Implementing and testing Otsu's algorithm
histogram, bin_edges = np.histogram(noisy_image, bins=256, range=(0, 255))
probabilities = histogram / histogram.sum()
optimal_threshold = 0
max_variance = 0

# Iterate over all possible thresholds
for threshold in range(1, 256):
    foreground_prob = probabilities[:threshold].sum()
    background_prob = probabilities[threshold:].sum()
    if foreground_prob == 0 or background_prob == 0:
        continue
    foreground_mean = (probabilities[:threshold] * np.arange(threshold)).sum() / foreground_prob
    background_mean = (probabilities[threshold:] * np.arange(threshold, 256)).sum() / background_prob
    inter_class_variance = foreground_prob * background_prob * (foreground_mean - background_mean) ** 2
    if inter_class_variance > max_variance:
        optimal_threshold = threshold
        max_variance = inter_class_variance

# Apply threshold and display results
segmented_image = cv2.threshold(noisy_image, optimal_threshold, 255, cv2.THRESH_BINARY)[1]
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Noisy Image")
axs[1].imshow(segmented_image, cmap='gray', vmin=0, vmax=255)
axs[1].set_title(f"Segmented Image Using Otsu's Algorithm (Threshold = {optimal_threshold})")
plt.show()
