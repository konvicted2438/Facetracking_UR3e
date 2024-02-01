import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import imutils
import numpy as np

class SimpleKalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement = 0.3
        self.error_covariance = 1.2

    def predict(self):
        # Prediction step: Here we're assuming a simple model with no control input
        self.error_covariance += self.process_variance
        return self.estimated_measurement

    def update(self, measurement):
        # Kalman gain
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)

        # Update step
        self.estimated_measurement += kalman_gain * (measurement - self.estimated_measurement)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance
        return self.estimated_measurement
    

class SimpleBandpassFilter:
    def __init__(self, low_cutoff, high_cutoff, dt):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.dt = dt  # Time step
        self.low_pass_filtered = 0
        self.high_pass_filtered = 0
        self.previous_raw = 0

    def update(self, new_value):
        # Low-pass filter
        alpha_low = self.dt / (self.dt + 1/(2*np.pi*self.low_cutoff))
        self.low_pass_filtered = alpha_low * new_value + (1 - alpha_low) * self.low_pass_filtered

        # High-pass filter
        alpha_high = 1/(2*np.pi*self.high_cutoff*self.dt + 1)
        self.high_pass_filtered = alpha_high * (self.high_pass_filtered + new_value - self.previous_raw)
        self.previous_raw = new_value

        # The final output is the high-pass filtered signal of the low-pass filtered signal
        return self.high_pass_filtered


class FaceTracking:

    def __init__(self):
        # Face detection setup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

        # Face mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.my_drawing_specs = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

        # Video settings
        self.video_resolution = (1280, 720)
        self.video_midpoint = (int(self.video_resolution[0] / 2), int(self.video_resolution[1] / 2))
        self.video_asp_ratio = self.video_resolution[0] / self.video_resolution[1]

        # Landmarks for the mouth
        self.LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        self.NOSE_LANDMARKS = [4, 5, 6, 197, 195, 5, 4, 51, 195, 197, 6, 168, 197, 168, 6, 98, 97, 2]
        self.LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 473]
        self.RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 468]
        self.LEFT_EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
        self.RIGHT_EYEBROW_LANDMARKS = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        self.FACE_OVAL_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 127, 162, 21, 54, 103, 67, 109, 1]
        self.Face_orita_marks = [1, 152, 33, 263, 61, 291] # point order: nose tip, chin, left eye left cornor, right eye right conor, left mouth, right mouth
        
        #karman filter set-up
        self.kalman_filter_b1 = SimpleKalmanFilter(0.5, 1.2)  # process variance, measurement variance
        self.kalman_filter_b2 = SimpleKalmanFilter(0.5, 1.2)
        self.kalman_filter_b3 = SimpleKalmanFilter(0.5, 1.2)
        self.kalman_filter_b4 = SimpleKalmanFilter(0.5, 1.2)
        self.bp_filter_b1 = SimpleBandpassFilter(0.1, 0.1, 1/30)  # Example values for cutoff frequencies and dt
        self.bp_filter_b2 = SimpleBandpassFilter(0.1, 0.1, 1/30)
        self.bp_filter_b3 = SimpleBandpassFilter(0.1, 0.1, 1/30)
        self.bp_filter_b4 = SimpleBandpassFilter(0.1, 0.1, 1/30)

        #self.kalman_filters = {key: self.kalman_filter() for key in self.Face_orita_marks.values()}

        
        self.landmark_positions = []
        self.image_points = []
        self.Face_orita_marks = {
                'nose_tip': 1,
                'chin': 152,
                'left_eye_left_corner': 33,
                'right_eye_right_corner': 263,
                'left_mouth_corner': 61,
                'right_mouth_corner': 291
                }
        self.focal_length = self.video_resolution[1]
        self.center = (self.video_resolution[1] / 2, self.video_resolution[0] / 2)
        self.camera_matrix = np.array(
                    [[self.focal_length, 0, self.center[0]],
                     [0, self.focal_length, self.center[1]],
                     [0, 0, 1]], dtype="double"
                    )
        self.model_points = np.array([
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0.0, -330.0, -65.0),  # Chin
                    (-225.0, 170.0, -135.0),  # Left eye left corner
                    (225.0, 170.0, -135.0),  # Right eye right corne
                    (-150.0, -150.0, -125.0),  # Left Mouth corner
                    (150.0, -150.0, -125.0)  # Right mouth corner
                    ])
        self.dist_coeffs = np.zeros((4, 1))# Assuming no lens distortion




    def _process_frame(self, frame, process_type):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame based on type
        if process_type == 'detection':
            return self.face_detector.process(rgb_frame)
        elif process_type == 'mesh':
            return self.face_mesh.process(rgb_frame)
        
    def _draw_landmarks(self, frame, face_landmarks, landmarks_list):
        specific_landmarks = landmark_pb2.NormalizedLandmarkList()
        specific_landmarks.landmark.extend([face_landmarks.landmark[id] for id in landmarks_list])
        #specific_landmarks = mp.solutions.drawing_utils.NormalizedLandmarkList()
        #specific_landmarks.landmark.extend([face_landmarks.landmark[id] for id in landmarks_list])
        mp.solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=specific_landmarks,
            connections=[],
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1)
        )
        #print(specific_landmarks)
    
    def face_center_detection(self, frame):
        results = self._process_frame(frame, 'detection')
        frame_height, frame_width, _ = frame.shape
        face_center = False
        if results.detections:
            for face in results.detections:
                # Processing and drawing logic
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
                countt = 0
                for p in key_points_coords:
                    countt += 1
                    # Drawing the midpoint circle and line, if needed
                    cv2.circle(frame, p, 4, (255, 255, 255), 2)
                    cv2.circle(frame, p, 2, (0, 0, 0), -1)
                    if countt == 3:
                        face_center = p
        return frame,face_center
    
    def track_specific_features(self, frame, features_to_track):
        #print(features_to_track)
        face_landmarks = False
        results = self._process_frame(frame, 'mesh')
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for feature in features_to_track:
                    if feature == 'lips':
                        self._draw_landmarks(frame, face_landmarks, self.LIP_LANDMARKS)
                    elif feature == 'nose':
                        self._draw_landmarks(frame, face_landmarks, self.NOSE_LANDMARKS)
                    elif feature == 'left_eye':
                        self._draw_landmarks(frame, face_landmarks, self.LEFT_EYE_LANDMARKS)
                    elif feature == 'right_eye':
                        self._draw_landmarks(frame, face_landmarks, self.RIGHT_EYE_LANDMARKS)
                    elif feature == 'left_eyebrow':
                        self._draw_landmarks(frame, face_landmarks, self.LEFT_EYEBROW_LANDMARKS)
                    elif feature == 'right_eyebrow':
                        self._draw_landmarks(frame, face_landmarks, self.RIGHT_EYEBROW_LANDMARKS)
                    elif feature == 'face_shape':
                        self._draw_landmarks(frame, face_landmarks, self.FACE_OVAL_LANDMARKS)
        return frame,face_landmarks
    
    def face_landmark(self, frame):
        results = self._process_frame(frame, 'mesh')
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Drawing landmarks
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=[],
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
        return frame
    @staticmethod
    def draw_line(frame, a, b, color=(255, 255, 0)):
        cv2.line(frame, a, b, color, 10)

    def face_orientation(self, frame):
        results = self._process_frame(frame, 'mesh') 
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
            # Convert string keys to their corresponding integer indices
                landmark_indices = [self.Face_orita_marks[key] for key in self.Face_orita_marks]

            # Call _draw_landmarks with the correct integer indices
                self._draw_landmarks(frame, face_landmarks, landmark_indices)
                
            # Initialize an array for storing image points
                self.image_points = np.array([
                    [face_landmarks.landmark[self.Face_orita_marks['nose_tip']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['nose_tip']].y * frame.shape[0]],
                    [face_landmarks.landmark[self.Face_orita_marks['chin']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['chin']].y * frame.shape[0]],
                    [face_landmarks.landmark[self.Face_orita_marks['left_eye_left_corner']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['left_eye_left_corner']].y * frame.shape[0]],
                    [face_landmarks.landmark[self.Face_orita_marks['right_eye_right_corner']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['right_eye_right_corner']].y * frame.shape[0]],
                    [face_landmarks.landmark[self.Face_orita_marks['left_mouth_corner']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['left_mouth_corner']].y * frame.shape[0]],
                    [face_landmarks.landmark[self.Face_orita_marks['right_mouth_corner']].x * frame.shape[1],
                    face_landmarks.landmark[self.Face_orita_marks['right_mouth_corner']].y * frame.shape[0]]
                    ], dtype="double")
                
            success, rotation_vector, translation_vector = cv2.solvePnP(
                        self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs
                        )
            self.image_points = np.array(self.image_points, dtype='double')
                #success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs)
            (b1, jacobian) = cv2.projectPoints(np.array([(150.0, 0.0, 0.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
            (b2, jacobian) = cv2.projectPoints(np.array([(0.0, 150.0, 0.0)]), rotation_vector,
                                           translation_vector, self.camera_matrix, self.dist_coeffs)
            (b3, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 150.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
            (b4, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
            
            #karman filter for the position track
            predicted_b1 = self.kalman_filter_b1.predict()
            predicted_b2 = self.kalman_filter_b2.predict()
            predicted_b3 = self.kalman_filter_b3.predict()
            predicted_b4 = self.kalman_filter_b4.predict()

            #filtered_b1 = self.bp_filter_b1.update(b1[0][0])
            #filtered_b2 = self.bp_filter_b2.update(b1[0][0])
            #filtered_b3 = self.bp_filter_b3.update(b1[0][0])
            #filtered_b4 = self.bp_filter_b4.update(b1[0][0])


            filtered_b1 = self.kalman_filter_b1.update(b1[0][0])
            filtered_b2 = self.kalman_filter_b2.update(b2[0][0])
            filtered_b3 = self.kalman_filter_b3.update(b3[0][0])
            filtered_b4 = self.kalman_filter_b4.update(b4[0][0])
            # Convert the filtered points to integer for drawing
            filtered_b1 = (int(filtered_b1[0]), int(filtered_b1[1]))
            filtered_b2 = (int(filtered_b2[0]), int(filtered_b2[1]))
            filtered_b3 = (int(filtered_b3[0]), int(filtered_b3[1]))
            filtered_b4 = (int(filtered_b4[0]), int(filtered_b4[1]))

            #print("4 points:",filtered_b1,filtered_b2,filtered_b3,filtered_b4)
            
           
            self.draw_line(frame, filtered_b4, filtered_b1)
            self.draw_line(frame, filtered_b4, filtered_b2)
            self.draw_line(frame, filtered_b4, filtered_b3)

            #print("Image points:", self.image_points)

        return frame, self.image_points,filtered_b1,filtered_b2,filtered_b3,filtered_b4







'''
        frame_height, frame_width, _ = frame.shape
        face_center = False
        if results.detections:
            for face in results.detections:
                # Processing and drawing logic
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
                countt = 0
                for p in key_points_coords:
                    countt += 1
                    # Drawing the midpoint circle and line, if needed
                    cv2.circle(frame, p, 4, (255, 255, 255), 2)
                    cv2.circle(frame, p, 2, (0, 0, 0), -1)
                    if countt == 3:
                        face_center = p









        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_landmarks(frame, face_landmarks, self.Face_orita_marks)

                for id, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    self.landmark_positions[id] = np.array([x, y], dtype='double')
                    self.image_points.append(self.landmark_positions[id])

                    image_points = np.array([[landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                        for id, landmark in enumerate(face_landmarks.landmark)
                        if id in self.some_landmark_ids  # You need to specify which landmarks to use
                        ], dtype='float64')

                if len(image_points) >= 4:
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        self.model_points, image_points, self.camera_matrix, self.dist_coeffs
                        )
                self.image_points = np.array(self.image_points, dtype='double')
                #success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs)
                (b1, jacobian) = cv2.projectPoints(np.array([(270.0, 0.0, 0.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
                (b2, jacobian) = cv2.projectPoints(np.array([(0.0, 270.0, 0.0)]), rotation_vector,
                                           translation_vector, self.camera_matrix, self.dist_coeffs)
                (b3, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 270.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
                (b4, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
                b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
                b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
                b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
                b4 = (int(b4[0][0][0]), int(b4[0][0][1]))
                self.draw_line(frame, b4, b1)
                self.draw_line(frame, b4, b2)
                self.draw_line(frame, b4, b3)
        return frame
'''
                    
                        


    

    
