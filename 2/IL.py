import cv2
import matplotlib.pyplot as plt

def enhance_illumination_global_he(image_path):
    # 1. Read the image
    img = cv2.imread(image_path)
    
    # 2. Convert from BGR to LAB color space to protect colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 3. Split the channels into L, A, and B
    l_channel, a, b = cv2.split(lab)
    
    # 4. Apply Standard Global Histogram Equalization ONLY to the L channel
    equalized_l = cv2.equalizeHist(l_channel)
    
    # 5. Merge the enhanced L-channel back with the untouched A and B channels
    merged = cv2.merge((equalized_l, a, b))
    
    # 6. Convert back to standard BGR for viewing/saving
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

# Apply the function
enhanced_4a_global = enhance_illumination_global_he('4-a.jpg.jpeg')
enhanced_4b_global = enhance_illumination_global_he('4-b.jpg.jpeg')

# Convert from BGR (OpenCV) to RGB (Matplotlib)
enhanced_4a_rgb = cv2.cvtColor(enhanced_4a_global, cv2.COLOR_BGR2RGB)
enhanced_4b_rgb = cv2.cvtColor(enhanced_4b_global, cv2.COLOR_BGR2RGB)

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first enhanced image
axes[0].imshow(enhanced_4a_rgb)
axes[0].set_title('Enhanced Image (4-a)')
axes[0].axis('off') # Hides the axis ticks/numbers

# Plot the second enhanced image
axes[1].imshow(enhanced_4b_rgb)
axes[1].set_title('Enhanced Image (4-b)')
axes[1].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()
