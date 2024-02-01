import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(1)
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
video_resolution = (1280, 720)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0]/2),
                  int(video_resolution[1]/2))
video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color = (255, 255, 255), thickness = 1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1,refine_landmarks = True,min_detection_confidence = 0.5,min_tracking_confidence = 0.5)

MOUTH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]


def facedetection(frame):

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    results = face_detector.process(rgb_frame)

    face_center = False

    frame_height, frame_width, c = frame.shape
    if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)
                cv2.rectangle(frame, face_react, color=(255, 255, 255), thickness=2)
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
                countt = 0
                for p in key_points_coords:
                    countt += 1
                    cv2.circle(frame, p, 4, (255, 255, 255), 2)
                    cv2.circle(frame, p, 2, (0, 0, 0), -1)
                    if countt == 3:
                        face_center = p
            cv2.circle(frame,video_midpoint,4,(100,100,100), 3)

            cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
    #face_center = (int(xmin + width / 2), int(startY + (endY - startY) / 2))
    '''
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10

    face_center = (int(xmin + width / 2), int(startY + (endY - startY) / 2))
    position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
    face_centers.append(position_from_center)
    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
    cv2.putText(frame, text, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #cv2.putText(frame, str(position_from_center), face_center, 0, 1, (0, 200, 0))
    cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
    cv2.circle(frame, face_center, 4, (0, 200, 0), 3)
    '''
    return frame,face_center

def face_landmark(frame):
    results = face_mesh.process(frame)
    #print(results.multi_face_landmarksq)
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            
        mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = my_drawing_specs
#                 .get_default_face_mesh_contours_style()
            )
        
    return image

def face_point_tracking(frame):
    results = face_mesh.process(frame)
    for face_landmarks in results.multi_face_landmarks:
        for id in MOUTH_LANDMARKS:
            lm = face_landmarks.landmark[id]

        mouth_landmarks = landmark_pb2.NormalizedLandmarkList()
        mouth_landmarks.landmark.extend([face_landmarks.landmark[id] for id in MOUTH_LANDMARKS])
        mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=mouth_landmarks,
                        connections=[],
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
    return image



#with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detector:
frame_counter = 0
fonts = cv2.FONT_HERSHEY_PLAIN
start_time = time.time()
    
    
while True:
    #frame_counter += 1
    ret, image = cap.read()
    #print(ret)
    if ret == True:
        detected_frame, facep_centre = facedetection(image)
        #detected_frame = face_point_tracking(image)
    else:
         pass
    #fps = frame_counter / (time.time() - start_time)
    #cv2.putText(frame,f"FPS: {fps:.2f}",(30, 30),cv2.FONT_HERSHEY_DUPLEX,0.7,(0, 255, 255),2,)
    cv2.imshow("frame", detected_frame)
    print(facep_centre)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 