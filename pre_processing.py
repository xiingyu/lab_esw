import cv2

class preprocess_img:
    def __init__(self, camera_index=1):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise ValueError(f"Camera at index {camera_index} could not be opened.")
        
    def run(self):
        while True:
            image = self.direction_detector.get_image()
            if image is None:
                break