import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_person_basic(image_path):
    img = cv2.imread(image_path)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding (Otsu's method)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morphological Closing (Fill holes in the mask)
    kernel = np.ones((15, 15), np.uint8)
    mask_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find the largest contour
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Draw clean mask and apply
    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    extracted = cv2.bitwise_and(img, img, mask=clean_mask)
    
    return extracted


def extract_suitcase_basic(image_path):
    img = cv2.imread(image_path)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Hard thresholding (Isolate very dark objects)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphological Opening (Erase thin floor lines)
    kernel = np.ones((7, 7), np.uint8)
    mask_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 4. Find the largest contour
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Draw clean mask and apply
    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    extracted = cv2.bitwise_and(img, img, mask=clean_mask)
    
    return extracted

# Run the functions
person = extract_person_basic('3-a.jpg.jpeg')
suitcase = extract_suitcase_basic('3-b.jpg.jpeg')

# Convert from BGR (OpenCV) to RGB (Matplotlib)
person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
suitcase_rgb = cv2.cvtColor(suitcase, cv2.COLOR_BGR2RGB)

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the extracted person
axes[0].imshow(person_rgb)
axes[0].set_title('Extracted Person')
axes[0].axis('off') # Hides the axis ticks/numbers

# Plot the extracted suitcase
axes[1].imshow(suitcase_rgb)
axes[1].set_title('Extracted Suitcase')
axes[1].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()