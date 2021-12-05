import numpy as np
import cv2

class pose_landmark_display(object):

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
            ]

        self.lineColour = (255, 255, 204)
        self.cycleColour = (224, 224, 224)
        self.cycleColour = (76, 153, 0)

    def drawLine(self, image, point1, point2):
        cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), self.lineColour, 2)
        cv2.circle(self.image, (int(point1[0]), int(point1[1])), 4, self.cycleColour, 2)
        cv2.circle(self.image, (int(point2[0]), int(point2[1])), 4, self.cycleColour, 2)

    #def displayaSamasthiti(self, image, landmarks):
    def displaySamasthiti(self):
        right_shoulder = self.landmarks[self._landmark_names.index('right_shoulder')]
        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_shoulder, right_ankle)

    def displayUpwardTree(self):
        right_wrist = self.landmarks[self._landmark_names.index('right_wrist')]
        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_wrist, right_ankle)

    def displayStandingForwardFold(self):
        right_wrist = self.landmarks[self._landmark_names.index('right_wrist')]
        right_elbow = self.landmarks[self._landmark_names.index('right_elbow')]
        self.drawLine(self.image, right_wrist, right_elbow)

        right_hip = self.landmarks[self._landmark_names.index('right_hip')]
        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_hip, right_ankle)

        right_shoulder = self.landmarks[self._landmark_names.index('right_shoulder')]
        self.drawLine(self.image, right_shoulder, right_elbow)

    def displayHalfStanding(self):
        right_hip = self.landmarks[self._landmark_names.index('right_hip')]
        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_hip, right_ankle)

        right_shoulder = self.landmarks[self._landmark_names.index('right_shoulder')]
        self.drawLine(self.image, right_shoulder, right_ankle)

        self.drawLine(self.image, right_shoulder, right_hip)

    def displayUpwardFacingDog(self):
        right_shoulder = self.landmarks[self._landmark_names.index('right_shoulder')]
        right_ear = self.landmarks[self._landmark_names.index('right_ear')]
        self.drawLine(self.image, right_ear, right_shoulder)

        right_wrist = self.landmarks[self._landmark_names.index('right_wrist')]
        self.drawLine(self.image, right_wrist, right_shoulder)

        right_hip = self.landmarks[self._landmark_names.index('right_hip')]
        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_hip, right_ankle)

        self.drawLine(self.image, right_shoulder, right_hip)

    def displayDownwardFacingDog(self):
        right_wrist = self.landmarks[self._landmark_names.index('right_wrist')]
        right_hip = self.landmarks[self._landmark_names.index('right_hip')]
        self.drawLine(self.image, right_wrist, right_hip)

        right_ankle = self.landmarks[self._landmark_names.index('right_ankle')]
        self.drawLine(self.image, right_hip, right_ankle)




    def d(self):
        print('Default function')

    funcDict = {
            "MountainPose-Samasthiti": displaySamasthiti,
            "UpwardTree-ekam": displayUpwardTree,
            "StandingForwardFold-dve": displayStandingForwardFold,
            "HalfStanding-trini": displayHalfStanding,
            "FourLimbedStaffPose-catvari": displaySamasthiti,
            "UpwardFacingDog-panca": displayUpwardFacingDog,
            "DownwardFacingDog-sat": displayDownwardFacingDog
        }

    def __call__(self, image, landmarks, surnaName):
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        self.image = image
        self.landmarks = landmarks

        '''
        for i, name in enumerate(self._landmark_names):
            #for lmk in landmarks:
            lmk = landmarks[i]
            pointx = lmk[0]
            pointy = lmk[1]

            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(image, str(i), (int(pointx), int(pointy)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(image, (int(pointx), int(pointy)), 2, self.cycleColour, 1)
        '''

        f = lambda self, x : self.funcDict.get(x, lambda x : self.d())(self)
        f(self, surnaName)

        #if (surnaName == "MountainPose-Samasthiti"):
        #    self.displayaSamasthiti(image, landmarks)



