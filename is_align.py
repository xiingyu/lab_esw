import cv2
import numpy as np

# 공과 관련된 설정
actual_diameter_meters = 0.04267
actual_area_meters = np.pi * (actual_diameter_meters / 2) ** 2
focal_length = 403
neck_angle = 60

# 공 탐지 및 거리 계산
def detect_and_calculate_distance(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170, 100, 150])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(cX), int(cY))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)

        pixel_area = np.pi * (radius ** 2)
        if pixel_area == 0:
            return frame, None, None, None, "No Case"

        distance = focal_length * np.sqrt(actual_area_meters / pixel_area)

        return frame, distance, center, pixel_area, None

    return frame, None, None, None, "No Case"

# 정렬 상태 확인
def check_alignment(red_center, yellow_center):
    if red_center and yellow_center:
        deviation = abs(yellow_center[0] - red_center[0])

        if deviation < 10:
            return "Aligned"
        else:
            return f"Deviation: {deviation} px"

    return "No Yellow Hole Detected"

# 방향 및 샷 강도 결정
def decide_action_based_on_alignment(situation, distance, red_center, yellow_center, frame):
    if situation == "Aligned":
        if distance < 1.0:  # 예를 들어, 1미터 내외의 거리라면
            return "Take Shot", frame
        else:
            return "Move Forward", frame
    elif "Deviation" in situation:
        deviation_value = int(situation.split(":")[1].strip().split()[0])
        if yellow_center[0] > red_center[0]:
            return f"Turn Left by {deviation_value}px", frame
        else:
            return f"Turn Right by {deviation_value}px", frame
    else:
        return "Search for Yellow Hole", frame

cap = cv2.VideoCapture('./re_alignment_case1.MP4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, distance, red_center, pixel_area, case_number = detect_and_calculate_distance(frame)

    # 노란색 홀 탐지
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([50, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    yellow_center = None
    if yellow_contours:
        largest_yellow_contour = max(yellow_contours, key=cv2.contourArea)
        M = cv2.moments(largest_yellow_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            yellow_center = (cX, cY)
            cv2.circle(frame, (cX, cY), 10, (0, 255, 255), -1)

    # 공의 중심에 맞춰 회색 중심선 그리기
    if red_center:
        gray_line_x = red_center[0]
        cv2.line(frame, (gray_line_x, 0), (gray_line_x, frame.shape[0]), (128, 128, 128), 2)

    # 가상선과 노란색 홀의 정렬 상태 확인
    situation = check_alignment(red_center, yellow_center)

    # 방향 및 샷 강도 결정
    action, frame = decide_action_based_on_alignment(situation, distance, red_center, yellow_center, frame)

    # 결과 출력
    cv2.putText(frame, situation, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, action, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if distance:
        cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
