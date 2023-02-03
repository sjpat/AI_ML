import cv2
import mediapipe as mp
import numpy as np
import time

class poseDetector():
    def __init__(self, static_image_mode = False, 
                       model_complexity = 1,
                       smooth_landmarks = True,
                       min_detection_confidence = 0.5,
                       min_tracking_confidence = 0.5):
        self.mode = static_image_mode
        self.complexity = model_complexity
        self.smooth = smooth_landmarks
        self.detectionCon = min_tracking_confidence
        self.trackCon = min_tracking_confidence
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode = self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            self.lmList.append([id,cx,cy])
            if draw:
                cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        '''
        Function to find angle between three points
        img: image 
        p1: landmark point 1
        p2: landmark point 2
        p3: landmark point 3
        '''
        
        # Get landmarks
        _,x1,y1 = self.lmList[p1]
        _,x2,y2 = self.lmList[p2]
        _,x3,y3 = self.lmList[p3]
        
        # Calculate angles
        angle = math.degrees(math.atan2((y3-y2,x3-x2) - 
                                        math.atan2(y1-y2,x1-x2)))
        if angle < 0:
            angle += 360
                                        
        if draw:
            cv2.line(img, (x1,y1),(x2,y2), (255,255,255),3)
            cv2.line(img, (x3,y3),(x2,y2), (255,255,255),3)
            cv2.circle(img, (x1,y1), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x3,y3), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (255,0,0), 2)
            cv2.circle(img, (x2,y2), 15, (255,0,0), 2)
            cv2.circle(img, (x3,y3), 15, (255,0,0), 2)
            draw_text(img,str(int(angle)),pos=(x2 - 45,y2 + 45),
                      text_color=(255,0,255))
    
def draw_rectangle(img, rect_start,rect_end, corner_width, box_color):
    '''
    Function to draw rectangle on image
    img: image
    rect_start: coordinates for where rectangle starts
    rect_end: coordinates for where rectangle ends
    corner_width: width of corners
    box_color: box color 
    '''
    x1,y1 = rect_start
    x2,y2 = rect_end
    w = corner_width
    
    # Draw filled rectangle
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)
    
    # Draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img

def draw_text(
    img,
    text,
    width=8,
    font= cv2.FONT_HERSHEY_PLAIN,
    pos=(0,0),
    font_scale=1,
    font_thickness=2,
    text_color=(0,255,0),
    text_color_bg=(0,0,0),
    box_offset=(20,10)
):
    '''
    Function to draw text on image
    '''
    offset = box_offset
    x,y = pos
    text_size,_ = cv2.getTextSize(text,font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_star = tuple(p - o for m, n, o in zip((x + text_w, y + text_h), offset, (25,0)))
    
    img = draw_rectangle(img, rec_start, rec_end, width, text_color_bg)
    
    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), 
        font,
        font_scale,
        text_color, 
        font_thickness,
        cv2.LINE_AA,
    )
    
    return text_size
         
def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0
    
    for i in range(start, end+1,8):
        cv2.circle(frame, (lm_coord[0],i+pix_step),2,line_color,-1, lineType=cv2.LINE_AA)
        
    return frame

def main():
    cap = cv2.VideoCapture('PoseVids/Bicep_curl/biceps curl_20.mp4')
    pTime=0
    detector = poseDetector()
    
    while True:
        success,img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) !=0:
            print(lmList[14])
            cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)
        draw_text(
            img,
            str(int(fps)),
            pos=(70,50),
            text_color=(255,0,0)
        )
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()