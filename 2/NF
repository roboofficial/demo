import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Load images ───────────────────────────────────────────────────────────────
img_a = cv2.imread("2-a.jpg.jpeg")          # Original clean Lena (color)
img_b = cv2.imread("2-b.jpg.jpeg", cv2.IMREAD_GRAYSCALE)  # Salt-only black image
img_c = cv2.imread("2-c.jpg.jpeg")          # Noisy Lena (color)

img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_c_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 2-a  |  Clean Reference — show histogram to confirm low noise
# ═══════════════════════════════════════════════════════════════════════════════
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 4))
plt.suptitle("Image 2-a  |  Original Clean Image (Reference)", fontsize=13, fontweight="bold")

plt.subplot(1, 3, 1)
plt.imshow(img_a_rgb)
plt.title("Original (Clean)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.plot(hist_a, color="gray")
plt.title("Histogram\n(smooth = low noise)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Count")

plt.subplot(1, 3, 3)
plt.text(0.1, 0.5,
    "Noise Type: NONE\n\n"
    "This is the clean reference.\n"
    "Histogram is smooth with\n"
    "no extreme spikes at 0 or 255.\n\n"
    "No filtering needed.",
    fontsize=11, verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4))
plt.axis("off")
plt.title("Diagnosis")

plt.tight_layout()
plt.savefig("2-a_result.jpg", dpi=120, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 2-b  |  Salt Noise (white dots on black background)
# ═══════════════════════════════════════════════════════════════════════════════

# Noise identification: count pixels at extreme values
total_pixels = img_b.shape[0] * img_b.shape[1]
salt_pixels   = np.sum(img_b == 255)
pepper_pixels = np.sum(img_b == 0)
noise_percent = (salt_pixels / total_pixels) * 100

hist_b = cv2.calcHist([img_b], [0], None, [256], [0, 256])

# Apply Median Filter (kernel size 3x3)
median_b3 = cv2.medianBlur(img_b, 3)

# Apply Median Filter (kernel size 5x5) for stronger cleaning
median_b5 = cv2.medianBlur(img_b, 5)

# Compare: Mean (Gaussian) filter — shows it smears noise instead of removing it
gaussian_b = cv2.GaussianBlur(img_b, (5, 5), 0)

plt.figure(figsize=(22, 4))
plt.suptitle("Image 2-b  |  Salt Noise (white specks on black)", fontsize=13, fontweight="bold")

plt.subplot(1, 5, 1)
plt.imshow(img_b, cmap="gray")
plt.title(f"Original\n(Salt noise ~{noise_percent:.1f}%)")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.plot(hist_b, color="black")
plt.title("Histogram\n(spike at 255 = salt)")
plt.xlabel("Intensity"); plt.ylabel("Count")

plt.subplot(1, 5, 3)
plt.imshow(gaussian_b, cmap="gray")
plt.title("Gaussian Blur 5x5\n(smears — NOT ideal)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(median_b3, cmap="gray")
plt.title("Median Filter 3x3\n(removes salt ✓)")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(median_b5, cmap="gray")
plt.title("Median Filter 5x5\n(stronger clean ✓)")
plt.axis("off")

plt.tight_layout()
plt.savefig("2-b_result.jpg", dpi=120, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 2-c  |  Salt & Pepper noise on Lena
# ═══════════════════════════════════════════════════════════════════════════════

gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

# Noise identification
total_c = gray_c.shape[0] * gray_c.shape[1]
salt_c   = np.sum(gray_c == 255)
pepper_c = np.sum(gray_c == 0)
sp_percent = ((salt_c + pepper_c) / total_c) * 100

hist_c = cv2.calcHist([gray_c], [0], None, [256], [0, 256])

# Optimal: Median Filter 3x3 on each channel separately (color image)
b, g, r = cv2.split(img_c)
b_med = cv2.medianBlur(b, 3)
g_med = cv2.medianBlur(g, 3)
r_med = cv2.medianBlur(r, 3)
median_c = cv2.merge([b_med, g_med, r_med])
median_c_rgb = cv2.cvtColor(median_c, cv2.COLOR_BGR2RGB)

# Stronger pass: 5x5
b5 = cv2.medianBlur(b, 5)
g5 = cv2.medianBlur(g, 5)
r5 = cv2.medianBlur(r, 5)
median_c5 = cv2.merge([b5, g5, r5])
median_c5_rgb = cv2.cvtColor(median_c5, cv2.COLOR_BGR2RGB)

# Gaussian for comparison (shows blurring artifacts)
gaussian_c = cv2.GaussianBlur(img_c, (5, 5), 0)
gaussian_c_rgb = cv2.cvtColor(gaussian_c, cv2.COLOR_BGR2RGB)

# Difference image: shows exactly what was removed
diff = cv2.absdiff(gray_c, cv2.cvtColor(median_c, cv2.COLOR_BGR2GRAY))

plt.figure(figsize=(22, 4))
plt.suptitle("Image 2-c  |  Salt & Pepper Noise on Lena", fontsize=13, fontweight="bold")

plt.subplot(1, 5, 1)
plt.imshow(img_c_rgb)
plt.title(f"Original Noisy\n(S&P ~{sp_percent:.1f}%)")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(img_a_rgb)
plt.title("Clean Reference\n(2-a)")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(gaussian_c_rgb)
plt.title("Gaussian 5x5\n(blurs — NOT ideal)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(median_c_rgb)
plt.title("Median 3x3\n(clean result ✓)")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(diff, cmap="hot")
plt.title("Difference Map\n(noise removed, hot=more)")
plt.axis("off")

plt.tight_layout()
plt.savefig("2-c_result.jpg", dpi=120, bbox_inches="tight")
plt.show()

# ── Summary printout ──────────────────────────────────────────────────────────
print("=" * 55)
print("NOISE SUMMARY")
print("=" * 55)
print(f"2-a : NO noise  — clean reference image")
print(f"2-b : SALT noise only  — {noise_percent:.2f}% white pixels")
print(f"2-c : SALT & PEPPER    — {sp_percent:.2f}% extreme pixels")
print("-" * 55)
print("Optimal Filter : Median (3x3 or 5x5)")
print("Why            : Replaces pixel with neighborhood median.")
print("                 Outliers (0 or 255) are ignored entirely.")
print("                 Gaussian would SMEAR noise, not remove it.")
print("=" * 55)
print("Saved: 2-a_result.jpg, 2-b_result.jpg, 2-c_result.jpg")
