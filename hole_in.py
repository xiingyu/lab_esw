import cv2
import numpy as np
import math


# 이미지 전처리 함수
def preprocess_image(frame):
    # 가우시안 블러를 적용하여 노이즈를 줄입니다.
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    # 이미지를 HSV 색상 공간으로 변환합니다.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv  # 블러 이미지와 HSV 이미지를 반환합니다.


# HSV 색상 공간에서 특정 색상을 마스크로 생성하는 함수
def create_color_mask(hsv, lower_color, upper_color):
    # 주어진 색상 범위에 해당하는 마스크를 생성합니다.
    return cv2.inRange(hsv, lower_color, upper_color)


# 빨간색 골프공 마스크 생성 함수
def create_red_mask(hsv):
    # 빨간색의 두 가지 HSV 범위 설정 (빨간색은 HSV 색상 공간에서 두 개의 범위를 가짐)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 두 범위의 마스크를 생성하고 합칩니다.
    mask1 = create_color_mask(hsv, lower_red1, upper_red1)
    mask2 = create_color_mask(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    return red_mask


# 빨간색 마스크 보강 함수
def enhance_red_mask(red_mask):
    # 빨간색 마스크에서 컨투어를 찾습니다.
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        print("Not enough contours found to enhance mask!")
        return red_mask  # 컨투어가 충분하지 않으면 원본 마스크 반환

    # 가장 큰 컨투어를 선택합니다.
    largest_contour = max(contours, key=cv2.contourArea)

    # 컨투어의 최소 외접 원을 찾습니다.
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 새로운 마스크를 생성하고 원을 그립니다.
    new_red_mask = np.zeros_like(red_mask)
    cv2.circle(new_red_mask, center, radius, 255, thickness=cv2.FILLED)

    return new_red_mask


# 노란색 홀 마스크 생성 함수
def create_yellow_mask(hsv):
    # 노란색의 HSV 범위 설정
    lower_yellow = np.array([18, 80, 80])
    upper_yellow = np.array([36, 230, 230])
    # 주어진 색상 범위에 해당하는 마스크를 생성합니다.
    return create_color_mask(hsv, lower_yellow, upper_yellow)


# 삼각형 검출 함수
def detect_triangles(mask):
    # 삼각형을 검출할 마스크에서 컨투어를 찾습니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_mask = np.zeros_like(mask)
    shape_detected = False

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 3 or len(approx) == 4 or len(approx) == 5 or len(approx) == 6 or len(approx) == 7 :
            cv2.drawContours(triangle_mask, [approx], -1, 255, thickness=cv2.FILLED)
            shape_detected = True

    if shape_detected:
        print("FLAG DETECT SUCCESS")
    else:
        print("FLAG DETECT FAIL")

    return triangle_mask


# 원 검출 함수
def detect_circles(yellow_mask, triangle_mask):
    # 삼각형 마스크를 뺀 후 원 마스크를 생성합니다.
    circle_mask = cv2.subtract(yellow_mask, triangle_mask)
    return circle_mask


# 원 마스크 보강 함수
def enhance_circle_mask(circle_mask):
    # 원 마스크에서 컨투어를 찾습니다.
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        print("Not enough contours found to enhance mask!")
        return circle_mask  # 컨투어가 충분하지 않으면 원본 마스크 반환

    # 가장 큰 컨투어를 선택합니다.
    largest_contour = max(contours, key=cv2.contourArea)

    # 컨투어의 최소 외접 원을 찾습니다.
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 새로운 마스크를 생성하고 원을 그립니다.
    new_circle_mask = np.zeros_like(circle_mask)
    cv2.circle(new_circle_mask, center, radius, 255, thickness=cv2.FILLED)

    return new_circle_mask


# 골프공이 노란색 홀 내부에 있는지 확인하는 함수
def is_ball_in_hole(ball_mask, hole_mask):
    # 빨간색 골프공 마스크와 노란색 홀 마스크에서 컨투어를 찾습니다.
    contours_ball, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_hole, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_ball) == 0 or len(contours_hole) == 0:
        return False, None, None

    # 가장 큰 컨투어를 선택합니다.
    golf_ball = max(contours_ball, key=cv2.contourArea)
    hole = max(contours_hole, key=cv2.contourArea)

    # 골프공의 중심을 계산합니다.
    M_ball = cv2.moments(golf_ball)
    if M_ball['m00'] == 0:
        return False, None, None
    cX_ball = int(M_ball['m10'] / M_ball['m00'])
    cY_ball = int(M_ball['m01'] / M_ball['m00'])

    # 홀의 중심과 반지름을 계산합니다.
    M_hole = cv2.moments(hole)
    if M_hole['m00'] == 0:
        return False, None, None
    cX_hole = int(M_hole['m10'] / M_hole['m00'])
    cY_hole = int(M_hole['m01'] / M_hole['m00'])
    radius_hole = int(np.sqrt(cv2.contourArea(hole) / np.pi))

    # 골프공의 중심이 홀의 중심에서 반지름 이내에 있는지 확인합니다.
    distance = np.sqrt((cX_ball - cX_hole) ** 2 + (cY_ball - cY_hole) ** 2)
    return distance <= radius_hole, (cX_ball, cY_ball), (cX_hole, cY_hole)


# 컨투어를 원본 이미지에 그리는 함수
def draw_contours_on_image(image, red_mask, yellow_mask, ball_center, hole_center, result_text):
    # 빨간색 골프공과 노란색 홀의 컨투어를 찾습니다.
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어를 이미지에 그립니다.
    cv2.drawContours(image, contours_red, -1, (0, 0, 0), 2)
    cv2.drawContours(image, contours_yellow, -1, (0, 0, 0), 2)

    # 골프공의 중심을 초록색으로 표시합니다.
    if ball_center is not None:
        cv2.circle(image, ball_center, 5, (0, 255, 0), -1)

    # 홀의 중심을 파란색으로 표시합니다.
    if hole_center is not None:
        cv2.circle(image, hole_center, 5, (255, 0, 0), -1)

    # 결과 텍스트를 이미지에 추가합니다.
    cv2.putText(image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)


# 메인 함수
def main():
    # 웹캠 비디오 캡처를 시작합니다.
    cap = cv2.VideoCapture(1)
    previous_state = None

    while True:
        # 웹캠에서 프레임을 읽어옵니다.
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지를 전처리합니다.
        hsv = preprocess_image(frame)

        # 빨간색 골프공 마스크를 생성합니다.
        red_mask = create_red_mask(hsv)
        # 노란색 홀 마스크를 생성합니다.
        yellow_mask = create_yellow_mask(hsv)
        # 삼각형을 검출합니다.
        yellow_triangle_mask = detect_triangles(yellow_mask)
        # 삼각형을 제외한 원을 검출합니다.
        yellow_circle_mask = detect_circles(yellow_mask, yellow_triangle_mask)

        # 빨간색 마스크를 보강합니다.
        new_red_mask = enhance_red_mask(red_mask)
        # 원 마스크를 보강합니다.
        new_circle_mask = enhance_circle_mask(yellow_circle_mask)

        # 골프공이 노란색 홀 내부에 있는지 확인합니다.
        ball_in_hole, ball_center, hole_center = is_ball_in_hole(new_red_mask, new_circle_mask)

        # 결과 텍스트 설정
        result_text = "O" if ball_in_hole else "X"
        # 원본 이미지에 결과를 그립니다.
        draw_contours_on_image(frame, new_red_mask, yellow_circle_mask, ball_center, hole_center, result_text)

        # 결과 이미지를 표시합니다.
        cv2.imshow('Original Image', frame)
        cv2.imshow('Red Mask', new_red_mask)
        cv2.imshow('Yellow Mask', yellow_mask)
        cv2.imshow('Detected Circle Mask', new_circle_mask)
        cv2.imshow('Detected Triangle Mask', yellow_triangle_mask)

        # 'q' 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 캡처를 종료하고 모든 윈도우를 닫습니다.
    cap.release()
    cv2.destroyAllWindows()


# 메인 함수 실행
if __name__ == "__main__":
    main()
