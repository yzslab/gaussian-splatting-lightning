import numpy as np
import cv2


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    # fx
    K[0, 0] = float(fx)
    # fy
    K[1, 1] = float(fy)
    # cx
    K[0, 2] = float(cx)
    # cy
    K[1, 2] = float(cy)

    return K


def perspective_undistort(
        img: np.ndarray,  # in [H, W, C]
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: np.ndarray,  # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
):
    h, w = img.shape[:2]

    K = build_intrinsic_matrix(fx=fx, fy=fy, cx=cx, cy=cy)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))

    dst = cv2.undistort(img, K, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return newcameramtx, dst


def fisheye_undistort(
        img: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: np.ndarray  # [k1, k2, k3, k4]
):
    img_dim = img.shape[:2][::-1]

    K = build_intrinsic_matrix(fx=fx, fy=fy, cx=cx, cy=cy)

    D = dist

    scaled_K = K
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K,
        D,
        img_dim,
        np.eye(3),
        balance=0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K,
        D,
        np.eye(3),
        new_K,
        img_dim,
        cv2.CV_32FC1,
    )
    undist_image = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return new_K, undist_image
