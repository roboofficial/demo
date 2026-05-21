import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images 
img_a = cv2.imread("3-a_jpg.jpeg")
img_b = cv2.imread("3-b_jpg.jpeg")

img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

# IMAGE 3-a  |  Person Extraction + Morphological Refinement

gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

# Step 1: Otsu Threshold
ret_a, thresh_a = cv2.threshold(gray_a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 2: EROSION — strip away thin stray edges and background noise touching the mask
kernel_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded_a = cv2.erode(thresh_a, kernel_a, iterations=2)

# Step 3: DILATION — grow mask back to recover fine details (hair, collar edges)
dilated_a = cv2.dilate(eroded_a, kernel_a, iterations=3)

# Step 4: CLOSING — seals any small holes inside the person mask (gaps in suit/face)
kernel_close_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
closed_a = cv2.morphologyEx(dilated_a, cv2.MORPH_CLOSE, kernel_close_a)

# Step 5: Largest contour on refined mask = person
contours_a, _ = cv2.findContours(closed_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_a = max(contours_a, key=cv2.contourArea)

mask_a = np.zeros(gray_a.shape, np.uint8)
cv2.drawContours(mask_a, [largest_a], -1, 255, thickness=cv2.FILLED)

# Step 6: Extract person on white background
result_a = img_a_rgb.copy()
result_a[mask_a == 0] = [255, 255, 255]

# Plot 3-a 
plt.figure(figsize=(24, 4))
plt.suptitle("3-a | Person — Otsu → Erosion → Dilation → Closing → Extract",
             fontsize=12, fontweight="bold")

plt.subplot(1, 6, 1)
plt.imshow(img_a_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 6, 2)
plt.imshow(thresh_a, cmap="gray")
plt.title(f"Otsu Threshold\nauto = {ret_a:.0f}")
plt.axis("off")

plt.subplot(1, 6, 3)
plt.imshow(eroded_a, cmap="gray")
plt.title("After Erosion\n(strips noise)")
plt.axis("off")

plt.subplot(1, 6, 4)
plt.imshow(dilated_a, cmap="gray")
plt.title("After Dilation\n(recovers edges)")
plt.axis("off")

plt.subplot(1, 6, 5)
plt.imshow(closed_a, cmap="gray")
plt.title("After Closing\n(fills inner holes)")
plt.axis("off")

plt.subplot(1, 6, 6)
plt.imshow(result_a)
plt.title("Extracted Person\n(white bg)")
plt.axis("off")

plt.tight_layout()
plt.savefig("3-a_morph_result.jpg", dpi=120, bbox_inches="tight")
plt.show()
print("3-a done")

# IMAGE 3-b  |  Suitcase Extraction + Morphological Refinement

gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# Step 1: Otsu Threshold
ret_b, thresh_b = cv2.threshold(gray_b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 2: EROSION — removes thin noise specks and breaks weak links to other dark objects
kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded_b = cv2.erode(thresh_b, kernel_b, iterations=2)

# Step 3: DILATION — restores suitcase body size lost during erosion
dilated_b = cv2.dilate(eroded_b, kernel_b, iterations=3)

# Step 4: CLOSING — fills holes inside suitcase (handle gap, zipper lines, label gap)
kernel_close_b = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
closed_b = cv2.morphologyEx(dilated_b, cv2.MORPH_CLOSE, kernel_close_b)

# Step 5: Largest contour in left half of frame = suitcase
contours_b, _ = cv2.findContours(closed_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
w_b = img_b.shape[1]
left_cnts = [c for c in contours_b if cv2.boundingRect(c)[0] < w_b * 0.55
             and cv2.contourArea(c) > 3000]
largest_b = max(left_cnts, key=cv2.contourArea)

mask_b = np.zeros(gray_b.shape, np.uint8)
cv2.drawContours(mask_b, [largest_b], -1, 255, thickness=cv2.FILLED)

# Step 6: Extract suitcase on white background
result_b = img_b_rgb.copy()
result_b[mask_b == 0] = [255, 255, 255]

# Plot 3-b 
plt.figure(figsize=(24, 4))
plt.suptitle("3-b | Suitcase — Otsu → Erosion → Dilation → Closing → Extract",
             fontsize=12, fontweight="bold")

plt.subplot(1, 6, 1)
plt.imshow(img_b_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 6, 2)
plt.imshow(thresh_b, cmap="gray")
plt.title(f"Otsu Threshold\nauto = {ret_b:.0f}")
plt.axis("off")

plt.subplot(1, 6, 3)
plt.imshow(eroded_b, cmap="gray")
plt.title("After Erosion\n(strips noise)")
plt.axis("off")

plt.subplot(1, 6, 4)
plt.imshow(dilated_b, cmap="gray")
plt.title("After Dilation\n(restores body)")
plt.axis("off")

plt.subplot(1, 6, 5)
plt.imshow(closed_b, cmap="gray")
plt.title("After Closing\n(fills handle gaps)")
plt.axis("off")

plt.subplot(1, 6, 6)
plt.imshow(result_b)
plt.title("Extracted Suitcase\n(white bg)")
plt.axis("off")

plt.tight_layout()
plt.savefig("3-b_morph_result.jpg", dpi=120, bbox_inches="tight")
plt.show()
print("3-b done")
