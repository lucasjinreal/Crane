import time
import cv2
import numpy as np
import kornia_rs as K
import matplotlib.pyplot as plt
import urllib.request
import os

# --- 1. Setup: Download a HIGH RES Image (Robust Method) ---
image_path = "high_res_test.jpg"

# List of high-res URLs to try (Wiki Commons Originals are more stable than thumbs)
urls_to_try = [
    # URL 1: Mount Everest (High Res ~3800px wide)
    "https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
    # URL 2: Lake Mapourika (High Res ~4000px wide)
    "https://upload.wikimedia.org/wikipedia/commons/2/23/Lake_mapourika_NZ.jpeg",
    # URL 3: Fallback (Standard 1080p+ if others fail)
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
]

def download_image(dest_path, url_list):
    if os.path.exists(dest_path):
        print("Image already exists on disk.")
        return True
    
    for url in url_list:
        print(f"Trying to download from: {url} ...")
        try:
            # User-agent is often required for these requests to succeed
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download successful!")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            continue
    return False

if not download_image(image_path, urls_to_try):
    raise FileNotFoundError("All download attempts failed. Please manually upload an image named 'high_res_test.jpg'")

# --- 2. Load Image ---
cv2.setNumThreads(1)

# Load as BGR -> RGB
img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError("Downloaded file is not a valid image.")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("-" * 60)
print(f"Input Image Resolution: {img.shape[1]}x{img.shape[0]} (WxH)")
print("-" * 60)

# --- 3. Visualization Function ---
def show_comparison(original, img_cv, img_kr, target_size):
    """Plots the resized versions side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # OpenCV Result
    axes[0].imshow(img_cv)
    axes[0].set_title(f"OpenCV ({target_size[0]}x{target_size[1]})")
    axes[0].axis('off')

    # Kornia-rs Result
    axes[1].imshow(img_kr)
    axes[1].set_title(f"Kornia-rs ({target_size[0]}x{target_size[1]})")
    axes[1].axis('off')

    plt.suptitle(f"Visual Check: Downscaling from {original.shape[1]}x{original.shape[0]} to {target_size}", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- 4. Run Single Sample & Visualize ---
viz_target = (512, 512) # H, W
h, w = viz_target

# OpenCV uses (W, H)
out_cv = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
# Kornia-rs uses (H, W)
out_kr = K.resize(img, (h, w), interpolation="bilinear")

print("Generating visual sample...")
show_comparison(img, out_cv, out_kr, viz_target)

# --- 5. Benchmarking Section ---
sizes = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

iters = 100
warmup = 10

def bench_opencv(img, size):
    h, w = size
    for _ in range(warmup):
        cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    t0 = time.perf_counter()
    for _ in range(iters):
        cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return (time.perf_counter() - t0) / iters

def bench_kornia(img, size):
    h, w = size
    for _ in range(warmup):
        K.resize(img, (h, w), interpolation="bilinear")

    t0 = time.perf_counter()
    for _ in range(iters):
        K.resize(img, (h, w), interpolation="bilinear")
    return (time.perf_counter() - t0) / iters

print(f"\nStarting Benchmark (Iterations: {iters}, Warmup: {warmup})")
print("-" * 60)

for size in sizes:
    t_cv = bench_opencv(img, size)
    t_kr = bench_kornia(img, size)
    
    # Calculate speedup (how many times faster/slower Kornia is compared to OpenCV)
    speedup = t_kr / t_cv
    
    print(
        f"Target: {size[0]:4d}x{size[1]:4d} | "
        f"OpenCV: {t_cv*1000:6.2f} ms | "
        f"kornia-rs: {t_kr*1000:6.2f} ms | "
        f"Speedup (CV/KR): {speedup:5.2f}x"
    )