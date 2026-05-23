import cv2
import numpy as np


class ArUcoProcessor:
    def __init__(self, dictionary_id=cv2.aruco.DICT_6X6_250, marker_length=0.05):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.marker_length = marker_length

        # 临时相机内参（需要根据实际摄像头标定）
        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def estimate_pose(self, corners):
        if corners is None or len(corners) == 0:
            return None, None
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )
        return rvecs, tvecs

    def draw_cube_on_roi(frame, roi_center, roi_size):
        # 根据 roi 中心点计算立方体在图像上的 2D 投影（简单假 3D）
        cx, cy = roi_center
        half = roi_size // 4  # 立方体边长取框选区域宽高的较小者的 1/4
        points_2d = np.array([
            [cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half],  # 底面
            [cx - half, cy - half - 20], [cx + half, cy - half - 20], [cx + half, cy + half - 20],
            [cx - half, cy + half - 20]
        ], dtype=int)
        # 绘制线框（类似立方体）
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.line(frame, tuple(points_2d[i + 4]), tuple(points_2d[(i + 1) % 4 + 4]), (0, 255, 0), 2)
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i + 4]), (0, 255, 0), 2)