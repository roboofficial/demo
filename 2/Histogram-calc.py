import numpy as np
import cv2
import matplotlib.pyplot as plt

# The 8x7 matrix extracted from the uploaded image
img = np.array([
    [52,  55,  61,  59,  79,  61,  61],
    [62,  59,  55, 104,  94,  89,  71],
    [63,  65,  66, 113, 144, 104,  62],
    [64,  70,  70, 126, 154, 109,  63],
    [67,  73,  68, 106, 122,  88,  68],
    [68,  79,  60,  70,  77,  68,  75],
    [89,  65,  64,  58,  55,  61,  83],
    [70,  87,  69,  65,  73,  78,  90]
], dtype=np.uint8)

# 1. Built-in Function
img_eq_cv2 = cv2.equalizeHist(img)

# 2. Manual Algorithm
def manual_histogram_equalization(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # Calculate PDF and CDF
    pdf = hist / image.size
    cdf = pdf.cumsum()
    
    # Create the transformation mapping
    # Multiplying by 255 and rounding
    transformation_map = np.round(cdf * 255).astype(np.uint8)
    
    # Map the original image values to the equalized values
    img_eq_manual = transformation_map[image]
    return img_eq_manual

img_eq_manual = manual_histogram_equalization(img)

# Display Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image (8x7)")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Built-in cv2.equalizeHist")
plt.imshow(img_eq_cv2, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Manual Equalization")
plt.imshow(img_eq_manual, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()