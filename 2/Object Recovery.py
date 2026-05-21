import cv2
import numpy as np
import matplotlib.pyplot as plt


img_a = cv2.imread("1-a.jpg.jpeg", cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread("1-b.jpg.jpeg", cv2.IMREAD_GRAYSCALE)



# Step 1: Histogram Equalization
he_a = cv2.equalizeHist(img_a)

# Step 2: CLAHE (local adaptive contrast)
clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_a = clahe_engine.apply(img_a)

# Step 3: Gaussian Blur on CLAHE result (noise suppression)
blurred_a = cv2.GaussianBlur(clahe_a, (5, 5), 0)

# Step 4: Canny Edge Detection on raw image (for comparison)
edges_raw_a = cv2.Canny(cv2.GaussianBlur(img_a, (5, 5), 0), 30, 100)

# Step 5: Canny Edge Detection on CLAHE image (good result)
edges_clahe_a = cv2.Canny(blurred_a, 30, 100)

# Plot Image 1-a results
plt.figure(figsize=(22, 4))
plt.suptitle("Image 1-a  |  Pier & Figure in Fog", fontsize=13, fontweight="bold")

plt.subplot(1, 5, 1)
plt.imshow(img_a, cmap="gray")
plt.title("Original (foggy)")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(he_a, cmap="gray")
plt.title("Histogram Equalization\n(global stretch)")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(clahe_a, cmap="gray")
plt.title("CLAHE\n(local adaptive)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(edges_raw_a, cmap="gray")
plt.title("Canny on Original\n(mostly noise)")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(edges_clahe_a, cmap="gray")
plt.title("Canny after CLAHE\n(clean edges)")
plt.axis("off")

plt.tight_layout()
plt.savefig("1-a_result.jpg", dpi=120, bbox_inches="tight")
plt.show()

# ── IMAGE 1-b 

# Step 1: Histogram Equalization
he_b = cv2.equalizeHist(img_b)

# Step 2: CLAHE (local adaptive contrast)
clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_b = clahe_engine.apply(img_b)

# Step 3: Gaussian Blur on CLAHE result (noise suppression)
blurred_b = cv2.GaussianBlur(clahe_b, (5, 5), 0)

# Step 4: Canny Edge Detection on raw image (for comparison)
edges_raw_b = cv2.Canny(cv2.GaussianBlur(img_b, (5, 5), 0), 30, 100)

# Step 5: Canny Edge Detection on CLAHE image (good result)
edges_clahe_b = cv2.Canny(blurred_b, 30, 100)

# Plot Image 1-b results
plt.figure(figsize=(22, 4))
plt.suptitle("Image 1-b  |  Tree in Fog", fontsize=13, fontweight="bold")

plt.subplot(1, 5, 1)
plt.imshow(img_b, cmap="gray")
plt.title("Original (foggy)")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(he_b, cmap="gray")
plt.title("Histogram Equalization\n(global stretch)")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(clahe_b, cmap="gray")
plt.title("CLAHE\n(local adaptive)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(edges_raw_b, cmap="gray")
plt.title("Canny on Original\n(mostly noise)")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(edges_clahe_b, cmap="gray")
plt.title("Canny after CLAHE\n(clean edges)")
plt.axis("off")

plt.tight_layout()
plt.savefig("1-b_result.jpg", dpi=120, bbox_inches="tight")
plt.show()

print("Done! Results saved as 1-a_result.jpg and 1-b_result.jpg")