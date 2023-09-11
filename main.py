import cv2
import numpy as np

from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.3

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold_min=25):
    # Compute the absolute difference between the background and the current frame
    diff = cv2.absdiff(background.astype('uint8'), frame)
    
    # Apply a threshold to create a binary image
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        # Assuming the largest external contour in ROI is the hand
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):
    # Convex Hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)

    # Find the top, bottom, left, and right points of the convex hull
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # Calculate the centroid of the hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # Calculate distances from centroid to the top, bottom, left, and right points
    distance = pairwise.euclidean_distances(np.array([[cX, cY]]), Y=[left, right, top, bottom])[0]
    

    
    # Find the maximum distance (maximum finger extension)
    max_distance = distance.max()

    # Calculate the radius of a circle around the hand
    radius = int(0.7 * max_distance)
    circumfrence = (2.2 * np.pi * radius)

    # Create a circular region of interest (ROI) mask
    circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
    cv2.circle(circular_roi, (cX, cY), radius, 255, 5)

    # Apply the circular mask to the thresholded image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours in the circular ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    # Iterate over detected contours
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Check if the contour is below the wrist and its length is below a certain limit
        out_of_wrist = (cY + (cY * 0.25)) > (y + h)
        limit_points = ((circumfrence * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count

cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cam.read()
    
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    
    
    if num_frames < 60:
        calc_accum_avg(gray,accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        
        hand = segment(gray)
        
        if hand is not None:
            
            thresholded , hand_segment = hand
            
            # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),2)
            
            fingers = count_fingers(thresholded,hand_segment)
            
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()