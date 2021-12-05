# https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR
# https://google.github.io/mediapipe/solutions/pose_classification.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tqdm

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from pose_body_embedder import FullBodyPoseEmbedder
from pose_classifier import PoseClassifier
from pose_smoothing import EMADictSmoothing
from display_landmark import pose_landmark_display
from text_tool import surya_name
import cv2
import pandas as pd

# from BlazePose.tools.image_tool import imageWrite

poseName = ["MountainPose-Samasthiti",
            "UpwardTree-ekam",
            "StandingForwardFold-dve",
            "HalfStanding-trini",
            "FourLimbedStaffPose-catvari",
            "UpwardFacingDog-panca",
            "DownwardFacingDog-sat",
            "HalfStandingForwardFold-sapta",
            "StandingForwardFold-astau",
            "UpwardTree-nava"]

poseShortName = ["Samasthiti", "Ekam",
            "Dve",
            "Trini",
            "Catvari",
            "Panca",
            "Sat", "Sapta", "Astau", "Nava"]

# initialise data of lists.
data = {'Pose':poseName, 'PoseShortName':poseShortName,
        'Number':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
 
# Create DataFrame
df = pd.DataFrame(data)

video_path = 'data/yoga/surya/video/original/surya_a.MP4'     #pan.mp4'
out_video_file = 'data/yoga/surya/video/result/surya_a.mp4'
pose_samples_csv_folder = 'data/yoga/surya/csv'

video_cap = cv2.VideoCapture(video_path)

landmark_display = pose_landmark_display()
name = surya_name()

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# iw = imageWrite()

pose_tracker = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()


# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.

pose_classifier = PoseClassifier(
    pose_samples_csv_folder=pose_samples_csv_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Open output video.
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter(out_video_file, fourcc, 15, (video_width, video_height))

frame_idx = 0

with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while True:
        success, input_frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()

        '''
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
        '''

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                        for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)



            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)
  
            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)

            ypos = 50
            for actionName in pose_classification_filtered:
                actionValue = pose_classification_filtered[actionName]
                if (actionValue > 3):
                    # cv2.putText(output_frame, suryaName + ": " + str(actionValue) ,(50,ypos), font, 1, (0,0,255), 1, cv2.LINE_AA)
                    suryaName = name(actionName)
                    suryaNameInUse = suryaName
                    ypos += 25

                    # display pose landmarks
                    landmark_display(output_frame, pose_landmarks, suryaName)

                    if (df.at[poseName.index("FourLimbedStaffPose-catvari"),'Number'] > 0):
                        # print(df.at[poseName.index("FourLimbedStaffPose-catvari"),'Number'])
                        if (suryaName == "HalfStanding-trini"):
                            suryaNameInUse = "HalfStandingForwardFold-sapta"
                            #df.at[poseName.index("HalfStandingForwardFold-sapta"),'Number'] += 1
                        elif (suryaName == "StandingForwardFold-dve"):
                            suryaNameInUse = "StandingForwardFold-astau"
                        elif (suryaName == "UpwardTree-ekam"):
                            suryaNameInUse = "UpwardTree-nava"
                        else:
                            suryaNameInUse = suryaName
                    
                    
                    df.at[poseName.index(suryaNameInUse),'Number'] += 1

            
            xpos = 100
            ypos = 50
            font = cv2.FONT_HERSHEY_COMPLEX  
            cv2.putText(output_frame, "Number of frames per pose", (xpos,ypos), font, 0.8, (204, 0, 0), 1, cv2.LINE_AA)
            ypos += 28
            xpos += 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            suryaNameInUse = suryaNameInUse.split("-")[1]
            for ind in df.index:
                poseNameInUse = df['PoseShortName'][ind]
                color = (255, 255, 204)  
                if (suryaNameInUse.lower() == poseNameInUse.lower()):
                    color = (0, 255, 204)
                    colorRed = (0, 0, 205)
                    output_frame = cv2.rectangle(output_frame, (xpos,ypos-20), (xpos+180,ypos+5), colorRed, 2)
                cv2.putText(output_frame, poseNameInUse + ":" + str(df['Number'][ind]) ,(xpos,ypos), font, 0.8, color, 1, cv2.LINE_AA)

                ypos += 28


        # Display the resulting frame
        cv2.imshow('Sun-Solution', output_frame)
            
        #if frame_idx % 3 == 0:
        #for i in range(5): 
        out_video.write(output_frame)

        #if frame_idx % 30 == 0:
        #    iw.writeImage("data/yoga/surya/result", output_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break

        frame_idx += 1
        pbar.update()

video_cap.release()
out_video.release()

