import cv2
import numpy as np
import glob
import argparse
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser("Camera intrinsic calibration (pinhole model)")
    ap.add_argument("--images", default="C:\\Users\\adjip\\Documents\\python\\ASO-IPM\\src\\data\\calibration_images_640x480\\*.png",
                    help="Glob pattern OR directory containing calibration images. If a directory is provided, '*.jpg' will be appended.")
    ap.add_argument("--inner-corners", type=int, nargs=2, default=[7,9],
                    metavar=("COLUMNS","ROWS"),
                    help="Number of INNER checkerboard corners (cols rows)")
    ap.add_argument("--square-mm", type=float, default=20.0,
                    help="Square size in millimeters")
    ap.add_argument("--out", default="calibration/camera_intrinsics640x480.npz",
                    help="Output .npz file to save intrinsics")
    ap.add_argument("--show", action="store_true",
                    help="Show corner detections (slow)")
    return ap.parse_args()

def main():
    args = parse_args()
    pattern_size = tuple(args.inner_corners)  # (cols, rows)
    square_size_m = args.square_mm / 1000.0   # convert mm -> meters

    # Allow user to pass a directory (with or without trailing slash); auto-expand to standard extensions
    img_arg = args.images
    p = Path(img_arg)
    if p.exists() and p.is_dir():
        # Build a list of common image extensions
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
        image_paths = []
        for e in exts:
            image_paths.extend(sorted(glob.glob(str(p / e))))
    else:
        # Treat as glob pattern directly
        image_paths = sorted(glob.glob(img_arg))
    if not image_paths:
        raise SystemExit(f"No images found. Interpreted '{img_arg}' as {'directory' if (p.exists() and p.is_dir()) else 'pattern'} with supported extensions.")

    print(f"[info] Using {len(image_paths)} images")
    print(f"[info] Inner corners (cols, rows): {pattern_size}")
    print(f"[info] Square size: {args.square_mm} mm")

    objpoints = []  # 3D points in board coordinates
    imgpoints = []  # 2D points in image pixels

    # Prepare a single object grid (Z=0 plane)
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m  # scale by square size in meters

    # Sub-pixel refinement criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-3)

    used = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Could not read {path}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        if not ret:
            print(f"[warn] Corners NOT found: {path}")
            continue

        # Refine to subpixel
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

        if args.show:
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
            cv2.imshow("corners", img)
            cv2.waitKey(60)

    if args.show:
        cv2.destroyAllWindows()

    if used < 5:
        raise SystemExit(f"Only {used} usable images found; need more for a stable calibration.")

    print(f"[info] Successfully detected corners in {used} images")

    image_shape = cv2.imread(image_paths[0]).shape[:2][::-1]  # (W,H)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    # Compute mean reprojection error
    total_err = 0.0
    for i, objp_i in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += err
    mean_err = total_err / len(objpoints)

    print(f"[result] Reprojection error: {mean_err:.4f} px")
    print("[result] Camera matrix K:\n", K)
    print("[result] Distortion coefficients:", dist.ravel())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=K,
        dist=dist,
        reprojection_error=mean_err,
        pattern_size=np.array(pattern_size),
        square_size_m=square_size_m,
        image_size=np.array(image_shape)
    )
    print(f"[saved] {out_path}")

    # Provide a quick undistort preview (first image)
    preview = cv2.imread(image_paths[0])
    h, w = preview.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    und = cv2.undistort(preview, K, dist, None, newK)
    cv2.imwrite(str(out_path.parent / "undistort_preview640x480.jpg"), und)
    print(f"[saved] {out_path.parent / 'undistort_preview640x480.jpg'} (preview)")

if __name__ == "__main__":
    main()