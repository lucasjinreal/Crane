import time
import cv2
import numpy as np
import kornia_rs as K

cv2.setNumThreads(1)

sizes = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

img = cv2.imread("data/images/test_ocr_page2.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

iters = 200
warmup = 20


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


print(f"Iterations: {iters}, Warmup: {warmup}")
print("-" * 60)

for size in sizes:
    t_cv = bench_opencv(img, size)
    t_kr = bench_kornia(img, size)

    print(
        f"{size[0]:4d}x{size[1]:4d} | "
        f"OpenCV: {t_cv*1000:6.2f} ms | "
        f"kornia-rs: {t_kr*1000:6.2f} ms | "
        f"ratio: {t_kr / t_cv:5.2f}x"
    )
