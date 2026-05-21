import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load all images ───────────────────────────────────────────
img_a = cv2.imread('5-a.jpg.jpeg', cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread('5-b.jpg.jpeg', cv2.IMREAD_GRAYSCALE)
img_c = cv2.imread('5-c.jpg.jpeg', cv2.IMREAD_GRAYSCALE)

# ── Unsharp Mask (Step 1) ─────────────────────────────────────
blur_a    = cv2.GaussianBlur(img_a, (0,0), 2.0)
sharp_a   = cv2.addWeighted(img_a, 3.5, blur_a, -2.5, 0)

blur_b    = cv2.GaussianBlur(img_b, (0,0), 2.0)
sharp_b   = cv2.addWeighted(img_b, 3.5, blur_b, -2.5, 0)

blur_c    = cv2.GaussianBlur(img_c, (0,0), 2.0)
sharp_c   = cv2.addWeighted(img_c, 3.5, blur_c, -2.5, 0)

# ── Adaptive Threshold (Step 2) ───────────────────────────────
thresh_a  = cv2.adaptiveThreshold(sharp_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
thresh_b  = cv2.adaptiveThreshold(sharp_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
thresh_c  = cv2.adaptiveThreshold(sharp_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Image Restoration — 5-a, 5-b, 5-c', fontsize=14, fontweight='bold')

axes[0][0].imshow(img_a,    cmap='gray');  axes[0][0].set_title('5-a Original');   axes[0][0].axis('off')
axes[0][1].imshow(sharp_a,  cmap='gray');  axes[0][1].set_title('5-a Sharpened');  axes[0][1].axis('off')
axes[0][2].imshow(thresh_a, cmap='gray');  axes[0][2].set_title('5-a Threshold');  axes[0][2].axis('off')

axes[1][0].imshow(img_b,    cmap='gray');  axes[1][0].set_title('5-b Original');   axes[1][0].axis('off')
axes[1][1].imshow(sharp_b,  cmap='gray');  axes[1][1].set_title('5-b Sharpened');  axes[1][1].axis('off')
axes[1][2].imshow(thresh_b, cmap='gray');  axes[1][2].set_title('5-b Threshold');  axes[1][2].axis('off')

axes[2][0].imshow(img_c,    cmap='gray');  axes[2][0].set_title('5-c Original');   axes[2][0].axis('off')
axes[2][1].imshow(sharp_c,  cmap='gray');  axes[2][1].set_title('5-c Sharpened');  axes[2][1].axis('off')
axes[2][2].imshow(thresh_c, cmap='gray');  axes[2][2].set_title('5-c Threshold');  axes[2][2].axis('off')

plt.tight_layout()
plt.savefig('output/5abc_no_loop.png', dpi=150, bbox_inches='tight')
plt.close()
print("Done")
