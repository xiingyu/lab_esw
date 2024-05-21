import cv2
import numpy as np

class ArrowDetector:
    def __init__(self):
        self.hsv_lower_yellow = None
        self.hsv_upper_yellow = None

    def select_yellow(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = self.img_color[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
            h = hsv_color[0][0][0]

            # Define the range for yellow color in HSV
            self.hsv_lower_yellow = np.array([h - 10, 100, 100])
            self.hsv_upper_yellow = np.array([h + 10, 255, 255])

    def preprocess(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_img, self.hsv_lower_yellow, self.hsv_upper_yellow)
        return yellow_mask

    def find_tip(self, points, convex_hull):
        length = len(points)
        indices = np.setdiff1d(range(length), convex_hull)

        for i in range(2):
            j = indices[i] + 2
            if j > length - 1:
                j = length - j
            if np.all(points[j] == points[indices[i - 1] - 2]):
                return tuple(points[j])

    def is_arrow(self, img):
        yellow_mask = self.preprocess(img)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)

            if 15 > sides > 3 and sides + 2 == len(approx):
                arrow_tip = self.find_tip(approx[:, 0, :], hull.squeeze())
                if arrow_tip:
                    # Draw triangle
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

                    # Draw red dots at triangle vertices
                    for point in approx:
                        cv2.circle(img, tuple(point[0]), 3, (0, 0, 255), cv2.FILLED)

                    return True, arrow_tip

        return False, [False, False]

    def detect_arrow(self):
        cap = cv2.VideoCapture(1)
        cv2.namedWindow('Select Yellow')
        cv2.setMouseCallback('Select Yellow', self.select_yellow)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot open camera")
                break

            self.img_color = cv2.resize(frame, (640, 480))

            cv2.imshow('Select Yellow', self.img_color)

            if cv2.waitKey(1) & 0xFF == 27:
                if self.hsv_lower_yellow is not None and self.hsv_upper_yellow is not None:
                    break

        cv2.destroyAllWindows()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot open camera")
                break

            arrow_detected, arrow_tip = self.is_arrow(frame)

            if arrow_detected:
                # Draw a red dot at the arrow tip
                cv2.circle(frame, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
                cv2.imshow('Arrow Detection', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    detector = ArrowDetector()
    detector.detect_arrow()
