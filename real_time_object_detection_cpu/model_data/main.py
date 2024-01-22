import os
import cv2
import numpy as np
import time

class Detector:
    def __init__(self, configPath, modelPath, classesPath):
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.2,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0, '__Background__')
        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[
            cv2.CAP_PROP_FRAME_WIDTH, 1920,
            cv2.CAP_PROP_FRAME_HEIGHT, 1080
        ])

        if (cap.isOpened()==False):
            print("Error opening file...")
            return
        
        (successs, image) = cap.read()

        while successs:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]

                    displayText = "{}:{:.4f}".format(classLabel, classConfidence)

                    x,y,w,h = bbox

                    if(classLabel == 'cell phone'):
                        cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,255,255), thickness=1)
                        cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, cv2.COLOR_BAYER_RG2RGB_EA, 2) 
                        newW = w // 2
                        newH = h // 2
                        cv2.line(image, (960, 1080), (x+newW, y+newH), cv2.COLOR_BAYER_BG2RGB, 5)
            cv2.imshow("Result", image)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("c"):
                cv2.imwrite('capture.jpg', image)

            (success, image) = cap.read() 
        cv2.destroyAllWindows()




def main():
    modelPath = os.path.join("real_time_object_detection_cpu", "model_data", "frozen_inference_graph.pb")
    configPath = os.path.join("real_time_object_detection_cpu", "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    classesPath = os.path.join("real_time_object_detection_cpu", "model_data", "coco.names")
    detector = Detector(configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()