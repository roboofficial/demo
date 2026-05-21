import cv2
import numpy as np

import matplotlib.pyplot as plt

def clean_and_reconstruct_edges(image_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize to ensure purely black and white pixels
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological Closing to bridge gaps in the boundaries
    # A 5x5 square kernel is large enough to bridge the small gaps in the lines
    kernel = np.ones((5, 5), np.uint8)
    closed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Contour detection to identify all distinct shapes (and noise)
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank black mask to draw the final result on
    final_clean_image = np.zeros_like(img)
    
    # 4. Filter out noise by area
    min_area = 100 # Adjust this threshold based on the specific image size
    
    for contour in contours:
        # Calculate the area of each shape
        area = cv2.contourArea(contour)
        
        # If the shape is large enough to be a real object (not noise)
        if area > min_area:
            # Draw it onto the clean mask with a solid thickness
            cv2.drawContours(final_clean_image, [contour], -1, 255, thickness=2)
            
    return final_clean_image

# Set the image path
img_path = '6-a.jpg.jpeg'

# Run the function
cleaned_edges = clean_and_reconstruct_edges(img_path)

# Load the original image for comparison plotting
original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original image
axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original Image (Noisy & Broken Edges)')
axes[0].axis('off')

# Plot the cleaned and reconstructed image
axes[1].imshow(cleaned_edges, cmap='gray')
axes[1].set_title('Cleaned & Reconstructed Edges')
axes[1].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()
