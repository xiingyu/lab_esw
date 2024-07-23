import cv2
import numpy as np
import math

def get_arrow_angle(thresh_image):

    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None, None  # Return all None if no contours found

    largest_contour = max(contours, key=cv2.contourArea)
    
    points = largest_contour.reshape(-1, 2)

    if points.shape[0] < 2:
        return None, None, None, None, None

    cov_matrix = np.cov(points, rowvar=False)

    if cov_matrix.shape != (2, 2):
        return None, None, None, None, None

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    largest_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    angle = math.degrees(math.atan2(largest_eigenvector[1], largest_eigenvector[0]))
    angle = (angle + 360) % 360

    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    return angle, (cx, cy), largest_contour, largest_eigenvector, points

def draw_arrow_and_camera_direction(frame, angle_arrow, angle_camera, largest_contour):
    length = 100
    cx, cy = largest_contour.mean(axis=0)[0]
    pt1 = (int(cx), int(cy))
    pt2_arrow = (int(cx + length * math.cos(math.radians(angle_arrow))),
                 int(cy - length * math.sin(math.radians(angle_arrow))))
    cv2.line(frame, pt1, pt2_arrow, (0, 255, 0), 2)

    pt2_camera = (int(cx + length * math.cos(math.radians(angle_camera))),
                  int(cy - length * math.sin(math.radians(angle_camera))))
    cv2.line(frame, pt1, pt2_camera, (255, 0, 0), 2)

    angle_between = angle_arrow - angle_camera

    cv2.putText(frame, f"Angle between arrow and camera: {angle_between:.2f} degrees", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return angle_between

if __name__ == "__main__":
    cap = cv2.VideoCapture(4)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera ERROR")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 45, 45])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        angle_arrow, centroid, largest_contour, largest_eigenvector, points = get_arrow_angle(mask)

        if angle_arrow is not None and centroid is not None and largest_contour is not None and largest_eigenvector is not None and points is not None:
            cx, cy = centroid
            angle_camera = 0  # Replace with actual camera angle calculation
            angle_between = draw_arrow_and_camera_direction(frame, angle_arrow, angle_camera, largest_contour)
            print("Angle :", angle_between)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
