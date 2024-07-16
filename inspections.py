import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load your original image and segmented image
original_image = cv2.imread('C:/Users/rainlab/Downloads/baseball.v1i.coco-segmentation/val2/0qAHy8CBkyY42_mp4_frame38_jpg.rf.f994a2a7291e347a17355cb72dea2ff3.jpg')
segmented_image = cv2.imread('C:/Users/rainlab/Downloads/baseball.v1i.coco-segmentation/val2/0qAHy8CBkyY42_mp4_frame38_jpg.rf.f994a2a7291e347a17355cb72dea2ff3_mask.png', cv2.IMREAD_GRAYSCALE)


# Define colors for each segment (0, 1, 2)
segment_colors = {
    0: (0, 0, 0),     # Black (background)
    1: (255, 0, 0),   # Red (ball)
    2: (0, 255, 0)    # Green (zone)
}

# Create a new image where each segment value is mapped to a color
height, width = segmented_image.shape[:2]
overlay = np.zeros((height, width, 3), dtype=np.uint8)

for row in range(height):
    for col in range(width):
        segment_value = segmented_image[row, col]
        overlay[row, col] = segment_colors.get(segment_value, (255, 255, 255))  # Default to white for unknown values

# Resize overlay to match original image size (if necessary)
overlay_resized = cv2.resize(overlay, (original_image.shape[1], original_image.shape[0]))

# Blend the images together using alpha blending
alpha = 0.9  # Adjust transparency here
blended_image = cv2.addWeighted(original_image, 1 - alpha, overlay_resized, alpha, 0)

# Display the blended image
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


