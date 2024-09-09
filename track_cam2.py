import numpy as np
import cv2
from time import sleep

def preprocess(frame):
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred

def get_min_max_hsv(hsv, mask):
    # 마스크에서 유효한 픽셀 추출
    points = cv2.findNonZero(mask)
    if points is not None:
        hsv_values = hsv[points[:, 0, 1], points[:, 0, 0]]
        min_hsv = hsv_values.min(axis=0)
        max_hsv = hsv_values.max(axis=0)
        return min_hsv, max_hsv
    else:
        return None, None

def detect(frame):
    # 전처리
    preprocessed_frame = preprocess(frame)
    
    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)

    # 색 범위 설정  
    green_lower = np.array([75, 120, 120])  # 대회 트랙
    green_upper = np.array([90, 255, 255])
    
    brown_lower = np.array([60, 0, 170])   # 벙커
    brown_upper = np.array([140, 35, 255])
    
    red_lower1 = np.array([0, 50, 140])     # 대회 공 (첫 번째 빨간색 범위)
    red_upper1 = np.array([15, 200, 255])
    
    red_lower2 = np.array([170, 80, 140])   # 대회 공 (두 번째 빨간색 범위)
    red_upper2 = np.array([180, 160, 255])
  
    yellow_lower = np.array([20, 120, 165]) # 깃발, 화살표
    yellow_upper = np.array([40, 255, 255])

    # 마스크 생성
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 컨투어 찾기
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 최종 마스크 초기화
    final_mask = np.zeros_like(green_mask)

    for contour in contours:
        # Convex Hull 계산
        hull = cv2.convexHull(contour)

        # Convex Hull을 따라 마스크 생성
        contour_mask = np.zeros_like(green_mask)
        cv2.drawContours(contour_mask, [hull], -1, 255, -1)

        # 초록색 컨투어 내에서 갈색 영역 추출
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_in_green_mask = cv2.bitwise_and(brown_mask, contour_mask)
        
        # 초록색 컨투어 내에서 첫 번째 빨간색 영역 추출
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_in_green_mask1 = cv2.bitwise_and(red_mask1, contour_mask)

        # 초록색 컨투어 내에서 두 번째 빨간색 영역 추출
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_in_green_mask2 = cv2.bitwise_and(red_mask2, contour_mask)

        # 최종 마스크에 초록색 마스크 추가
        final_mask = cv2.bitwise_or(final_mask, green_mask)
        final_mask = cv2.bitwise_or(final_mask, brown_in_green_mask)
        final_mask = cv2.bitwise_or(final_mask, red_in_green_mask1)
        final_mask = cv2.bitwise_or(final_mask, red_in_green_mask2)

    # 노란색 마스크를 최종 마스크에 추가
    final_mask = cv2.bitwise_or(final_mask, yellow_mask)
    
    # 모폴로지 연산으로 경계 다듬기
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 최종 마스크 블러링
    final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

    # 최종 마스크를 사용하여 원본 이미지에서 해당 영역만 남기기
    result = cv2.bitwise_and(frame, frame, mask=final_mask)

    # 결과 출력
    cv2.imshow('Result', result)
    
    # HSV 값들을 npy 파일로 저장
    np.save('green_hsv_range.npy', np.array([green_lower, green_upper]))
    np.save('brown_hsv_range.npy', np.array([brown_lower, brown_upper]))
    np.save('red_hsv_range1.npy', np.array([red_lower1, red_upper1]))
    np.save('red_hsv_range2.npy', np.array([red_lower2, red_upper2]))
    np.save('yellow_hsv_range.npy', np.array([yellow_lower, yellow_upper]))

    sleep(1)

    return result

def main():
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        processed_frame = detect(frame)

        # 키 입력을 1ms 기다리고, key가 'q'이면 break
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

