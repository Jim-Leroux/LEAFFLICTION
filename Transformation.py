import sys

import cv2

import argparse

import matplotlib.pyplot as plt

from pathlib import Path

from plantcv import plantcv as pcv


def do_transformations(image_path):
    img, path, filename = pcv.readimage(str(image_path))

    # 1. Gaussian Blur
    blur = pcv.gaussian_blur(img, ksize=(5, 5))

    # 2. Masking
    gray_a = pcv.rgb2gray_lab(rgb_img=blur, channel='a')
    mask = pcv.threshold.binary(gray_img=gray_a,
                                threshold=125, object_type='dark')
    mask = pcv.fill(bin_img=mask, size=200)

    # 3. ROI (Image détourée)
    roi = pcv.apply_mask(img=img, mask=mask, mask_color='white')

    # 4. Analysis (Contours)
    analysis_img = roi.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(analysis_img, contours, -1, (0, 0, 255), 3)

    # 5. Landmarks (Points de repère)
    landmarks_img = roi.copy()
    try:
        top_v, bottom_v, center_v = pcv.homology.vertical(img=roi, mask=mask)

        pt_top = (int(top_v[0]), int(top_v[1]))
        pt_bottom = (int(bottom_v[0]), int(bottom_v[1]))
        pt_center = (int(center_v[0]), int(center_v[1]))

        cv2.line(landmarks_img, pt_top, pt_bottom, (255, 0, 0), 2)
        cv2.circle(landmarks_img, pt_top, 8, (0, 0, 255), -1)
        cv2.circle(landmarks_img, pt_bottom, 8, (0, 0, 255), -1)
        cv2.circle(landmarks_img, pt_center, 8, (0, 255, 0), -1)
    except Exception:
        pass

    # 6. Pseudocolor (Heatmap)
    heatmap = cv2.applyColorMap(gray_a, cv2.COLORMAP_JET)
    heatmap = pcv.apply_mask(img=heatmap, mask=mask, mask_color='black')

    return [img, blur, mask, analysis_img, landmarks_img, heatmap]


def show_transformations(src_path):
    if not src_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        sys.exit("Error : The file is not an image")

    results = do_transformations(src_path)

    names = [
        "Original", "Gaussian Blur", "Mask", "Analysis (Contours)",
        "Landmarks", "Pseudocolor (Heatmap)"]

    plt.figure(figsize=(15, 10))

    for i, img_data in enumerate(results):
        plt.subplot(2, 3, i + 1)
        plt.title(names[i])
        plt.axis('off')

        if len(img_data.shape) == 2:
            plt.imshow(img_data, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


def save_transformations(src_path, dst_path):
    try:
        results = do_transformations(src_path)
        names = ["original", "blur", "mask", "analysis",
                 "landmarks", "pseudocolor"]

        sub_directory_path = Path(dst_path) / src_path.stem
        sub_directory_path.mkdir(parents=True, exist_ok=True)

        for img, name in zip(results, names):
            save_name = sub_directory_path / f"{name}.jpg"
            cv2.imwrite(str(save_name), img)
    except Exception as e:
        print(f"Failed to process {src_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Transformation Tool")
    parser.add_argument("-src", "--source",
                        required=True, help="Path to image or directory")
    parser.add_argument("-dst", "--destination",
                        help="Path to save results (required for directories)")

    args = parser.parse_args()
    src_path = Path(args.source)

    if not src_path.exists():
        sys.exit("Error: Path does not exist.")

    if src_path.is_file():
        show_transformations(src_path)

    elif src_path.is_dir():
        if not args.destination:
            sys.exit("Error : -dst is required when -src is a directory.")

        dst_path = Path(args.destination)

        ext = ['.jpg', '.jpeg', '.png']

        images = [f for f in src_path.rglob('*') if f.suffix.lower() in ext]

        if not images:
            sys.exit("Error : No valid images found.")

        for img_path in images:
            save_transformations(img_path, dst_path)
