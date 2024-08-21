## 빨간 공 탐지 && 빨간 공까지의 거리 계산 update :: 240808

import cv2
import numpy as np

# 모든 길이 단위는 m 기준
actual_diameter_meters = 0.04267  # 공의 직경(골프공은 원래 42.67mm(0.04267m)의 직경임) # 0.05m에서 수정함
actual_area_meters = np.pi * (actual_diameter_meters / 2) ** 2  # 공의 면적(직경으로 계산한 값)
focal_length = 403  # 초점 거리 값 # 캘리 매트릭스에서 ((fx+fy)/2)로 평균한 결과 # 509.0532269132008(새거) 403.01356509306765(구)
neck_angle = 60  # 임시 목 각도 (degree)

# 빨간 공 탐지 및 거리 추정 함수
def detect_and_calculate_distance(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170, 100, 150])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 컨투어 중심 계산
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 공의 위치를 3x3(9분할) 화면 중 어떤 곳에 위치하는지 확인
        frame_height, frame_width = frame.shape[:2]
        segment_width = frame_width // 3
        segment_height = frame_height // 3

        case_number = None
        if cX < segment_width and cY < segment_height:
            case_number = 1
        elif segment_width <= cX < 2 * segment_width and cY < segment_height:
            case_number = 2
        elif cX >= 2 * segment_width and cY < segment_height:
            case_number = 3
        elif cX < segment_width and segment_height <= cY < 2 * segment_height:
            case_number = 4
        elif segment_width <= cX < 2 * segment_width and segment_height <= cY < 2 * segment_height:
            case_number = 5
        elif cX >= 2 * segment_width and segment_height <= cY < 2 * segment_height:
            case_number = 6
        elif cX < segment_width and cY >= 2 * segment_height:
            case_number = 7
        elif segment_width <= cX < 2 * segment_width and cY >= 2 * segment_height:
            case_number = 8
        elif cX >= 2 * segment_width and cY >= 2 * segment_height:
            case_number = 9

        # 최소한의 원 그리기
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour) # 가장 큰 외곽선을 포함하는 최소 외접원
        center = (int(cX), int(cY))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)

        # 중심점 표시는 내가 볼라고
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, "Center", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 면적 계산
        pixel_area = np.pi * (radius ** 2)

        if pixel_area == 0:
            return frame, None, None, None, None, case_number

        # 거리 계산
        distance = focal_length * np.sqrt(actual_area_meters / pixel_area)

        # walk_dist 계산
        neck_angle_radians = np.radians(neck_angle)  # degree to radian
        walk_dist = distance * np.cos(np.pi / 2 - neck_angle_radians)

        # 화면 하단 중심에서 공 중심까지 선 그리기
        bottom_center = (frame_width // 2, frame_height)
        cv2.line(frame, bottom_center, center, (255, 255, 0), 2)

        # 각도 계산
        angle = np.arctan2(center[1] - bottom_center[1], center[0] - bottom_center[0])  # 두 값의 비율로 각도 계산하는 함수임
        angle_degrees = np.degrees(angle)
        cv2.putText(frame, f"Angle: {angle_degrees:.2f} degrees", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, distance, center, pixel_area, walk_dist, case_number

    return frame, None, None, None, None, None

cap = cv2.VideoCapture(1)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('default resolution width {} height {}'.format(width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, distance, center, pixel_area, walk_dist, case_number = detect_and_calculate_distance(frame)
    if distance:
        cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Area: {pixel_area:.2f} pixels^2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Walk Dist: {walk_dist:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Case: {case_number}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
