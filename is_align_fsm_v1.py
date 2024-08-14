import cv2
import numpy as np

from this_pixel_dist_class import RedBallDetector
from pre_processing import preprocess_img # 승재오빠 전처리 코드에서 모든 카메라 받아서 사용해야함

class dongjak:
    def __init__(self, actual_diameter_meters=0.04267, focal_length=403, neck_angle=60): # focal_length는 라파카메라 기준으로 다시 해야함
        self.actual_diameter_meters = actual_diameter_meters
        self.actual_area_meters = np.pi * (self.actual_diameter_meters / 2) ** 2
        self.focal_length = focal_length
        self.neck_angle = neck_angle
        self.neck_angle_radians = np.radians(self.neck_angle)

cap = cv2.VideoCapture('./alignment_case1.mov')

# [영상처리] 9분할 화면 중 가운데에 있는지 확인 및 픽셀 거리 측정

# [제어] 왼쪽 게걸음 1step

# [제어] 직진 3step + 왼쪽 방향 보정 1step 반복

# [제어] 시계 방향 90도 턴

# [제어] 고개 내려서 공 보기 -> 없다면 공 찾기 단계로 돌아감

# [영상처리] 고개 내려진 상태에서 공 있는지 확인(이번엔 8번째 칸에 있는지도 확인 + 픽셀 임계 이상인지 확인)

# [제어] 앞으로 3step

# [제어] 고개 최우측으로(90도) 회전

# [영상처리] 노란색 홀 인식 + align 여부 판단

# [제어] 타+1


def run()
