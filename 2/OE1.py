import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Load images 
img_a = cv2.imread("3-a.jpg.jpeg")
img_b = cv2.imread("3-b.jpg.jpeg")

img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)



gray_a   = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
h_a, w_a = gray_a.shape

# Step 1: Crop black border so it doesn't merge with dark objects
crop_a     = gray_a[30:h_a-20, 30:w_a-20]
crop_a_rgb = img_a_rgb[30:h_a-20, 30:w_a-20]
hc_a, wc_a = crop_a.shape

# Step 2: Threshold — dark objects become white
_, thresh_a = cv2.threshold(crop_a, 80, 255, cv2.THRESH_BINARY_INV)

# Step 3: Erosion — remove noise and thin stray connections
kernel_e_a = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded_a   = cv2.erode(thresh_a, kernel_e_a, iterations=1)

# Step 4: Dilation — restore object body back
dilated_a  = cv2.dilate(eroded_a, kernel_e_a, iterations=2)

# Step 5: Closing — fill holes inside the object mask
kernel_c_a = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed_a   = cv2.morphologyEx(dilated_a, 3, kernel_c_a)   # 3 = MORPH_CLOSE

# Step 6: Find contours and pick best — largest blob in left 60%, under 15% of image area
contours_a, _ = cv2.findContours(closed_a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
total_a = hc_a * wc_a
candidates_a = []
for cnt in contours_a:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if 5000 < area < total_a * 0.15 and x < wc_a * 0.6:
        candidates_a.append(cnt)

best_a = max(candidates_a, key=cv2.contourArea)

# Step 7: Fill mask and extract
mask_a = np.zeros(crop_a.shape, np.uint8)
cv2.drawContours(mask_a, [best_a], -1, 255, thickness=cv2.FILLED)
result_a = crop_a_rgb.copy()
result_a[mask_a == 0] = [255, 255, 255]

# ── Plot 3-a ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(22, 4))
plt.suptitle("3-a | Person — Crop → Threshold → Erosion → Dilation → Closing → Extract",
             fontsize=11, fontweight="bold")

plt.subplot(1, 6, 1)
plt.imshow(img_a_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 6, 2)
plt.imshow(thresh_a, cmap="gray")
plt.title("Threshold = 80\n(dark = white)")
plt.axis("off")

plt.subplot(1, 6, 3)
plt.imshow(eroded_a, cmap="gray")
plt.title("Erosion\n(strips noise)")
plt.axis("off")

plt.subplot(1, 6, 4)
plt.imshow(dilated_a, cmap="gray")
plt.title("Dilation\n(restores body)")
plt.axis("off")

plt.subplot(1, 6, 5)
plt.imshow(closed_a, cmap="gray")
plt.title("Closing\n(fills holes)")
plt.axis("off")

plt.subplot(1, 6, 6)
plt.imshow(result_a)
plt.title("Extracted Person\n(white bg)")
plt.axis("off")

plt.tight_layout()
plt.savefig("3-a_unified_result.jpg", dpi=120, bbox_inches="tight")
plt.show()
print("3-a done")

# IMAGE 3-b  |  Suitcase Extraction — SAME pipeline

gray_b   = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
h_b, w_b = gray_b.shape

# Step 1: Crop black border
crop_b     = gray_b[30:h_b-20, 30:w_b-20]
crop_b_rgb = img_b_rgb[30:h_b-20, 30:w_b-20]
hc_b, wc_b = crop_b.shape

# Step 2: Threshold
_, thresh_b = cv2.threshold(crop_b, 80, 255, cv2.THRESH_BINARY_INV)

# Step 3: Erosion
kernel_e_b = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded_b   = cv2.erode(thresh_b, kernel_e_b, iterations=1)

# Step 4: Dilation
dilated_b  = cv2.dilate(eroded_b, kernel_e_b, iterations=2)

# Step 5: Closing
kernel_c_b = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed_b   = cv2.morphologyEx(dilated_b, 3, kernel_c_b)   # 3 = MORPH_CLOSE

# Step 6: Find contours and pick best — same filter as 3-a
contours_b, _ = cv2.findContours(closed_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
total_b = hc_b * wc_b
candidates_b = []
for cnt in contours_b:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if 5000 < area < total_b * 0.15 and x < wc_b * 0.6:
        candidates_b.append(cnt)

best_b = max(candidates_b, key=cv2.contourArea)

# Step 7: Fill mask and extract
mask_b = np.zeros(crop_b.shape, np.uint8)
cv2.drawContours(mask_b, [best_b], -1, 255, thickness=cv2.FILLED)
result_b = crop_b_rgb.copy()
result_b[mask_b == 0] = [255, 255, 255]

# ── Plot 3-b ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(22, 4))
plt.suptitle("3-b | Suitcase — Crop → Threshold → Erosion → Dilation → Closing → Extract",
             fontsize=11, fontweight="bold")

plt.subplot(1, 6, 1)
plt.imshow(img_b_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 6, 2)
plt.imshow(thresh_b, cmap="gray")
plt.title("Threshold = 80\n(dark = white)")
plt.axis("off")

plt.subplot(1, 6, 3)
plt.imshow(eroded_b, cmap="gray")
plt.title("Erosion\n(strips noise)")
plt.axis("off")

plt.subplot(1, 6, 4)
plt.imshow(dilated_b, cmap="gray")
plt.title("Dilation\n(restores body)")
plt.axis("off")

plt.subplot(1, 6, 5)
plt.imshow(closed_b, cmap="gray")
plt.title("Closing\n(fills holes)")
plt.axis("off")

plt.subplot(1, 6, 6)
plt.imshow(result_b)
plt.title("Extracted Suitcase\n(white bg)")
plt.axis("off")

plt.tight_layout()
plt.savefig("3-b_unified_result.jpg", dpi=120, bbox_inches="tight")
plt.show()
print("3-b done")
