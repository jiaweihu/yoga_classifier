from display_landmark import pose_landmark_display
from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np

imageFile = "data/yoga/surya/pan/Samasthiti-0.JPG"
imageName = "MountainPose-Samasthiti"

imageFile = "data/yoga/surya/pan/ekam-1.JPG"
imageName = "UpwardTree-ekam"

imageFile = "data/yoga/surya/pan/dve-2.JPG"
imageName = "StandingForwardFold-dve"

imageFile = "data/yoga/surya/pan/trini-3.JPG"
imageName = "HalfStanding-trini"

imageFile = "data/yoga/surya/pan/catvari-4.JPG"
imageName = "FourLimbedStaffPose-catvari"

imageFile = "data/yoga/surya/pan/panca-5.JPG"
imageName = "UpwardFacingDog-panca"

imageFile = "data/yoga/surya/pan/sat-6.JPG"
imageName = "DownwardFacingDog-sat"

input_frame = cv2.imread(imageFile)
input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

poseDisplay = pose_landmark_display()

pose_tracker = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
result = pose_tracker.process(image=input_frame)
pose_landmarks = result.pose_landmarks

if pose_landmarks is not None:
    # Get landmarks.
    frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                for lmk in pose_landmarks.landmark], dtype=np.float32)
    poseDisplay(input_frame, pose_landmarks, imageName)

    cv2.imshow('Test-Frame', input_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




