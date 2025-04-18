
import cv2 
import numpy as np
import math 
import matplotlib.pyplot as plt 
import Board
import PID
import time

# setting 
detect_color = 'green' 
size = (640, 480)
# setup constant value 
square_length = 3
image_center_distance = 20
map_param = 0.037528


range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

color_range = {
    'red':[(0,151,100), (255,255,255)],
    'green':[(0,0,0), (255,115,255)],
    'blue':[(0,0,0), (255,255,110)],
    'black':[(0,0,0), (56,255,255)],
    'white':[(193,0,0), (255,250,255)],
}

x_dis = 500
x_pid = PID.PID(P=0.1, I=0.00, D=0.008)
    
# initial position
def initMove():
    Board.setPWMServoPulse(1, 500, 800)
    Board.setPWMServoPulse(2, 500, 800)
    time.sleep(1.5)


def run(img):
    img_h, img_w = img.shape[:2]
    # copy image 

    ##shape function (height, width, rgb).  [480, 640,3]
    img_h, img_w = img.shape[:2] 

    size = (img.shape[1],img.shape[0])

    cv2.line(img, (0, int(img_h / 2)), (img_w, int(img_h / 2)), (0, 0, 200), 1)
    cv2.line(img, (int(img_w / 2), 0), (int(img_w / 2), img_h), (0, 0, 200), 1)

    frame_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)  # convert from RGB to LAB


    #  bitwise manipulation finding the maximum contour
    frame_mask = cv2.inRange(frame_lab, color_range[detect_color][0], color_range[detect_color][1])  # perform bitwise operations on the original image and the mask
    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # Morphological opening
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # Morphological closing
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # find contours
    areaMaxContour, area_max = getAreaMaxContour(contours)  # find max contour
    (x,y,w,h)=cv2.boundingRect(areaMaxContour)
    x_center=x+w/2
    y_center=y+h/2
    if area_max > 2500:  # find max area
        
        rect = cv2.minAreaRect(areaMaxContour)
        box = np.int0(cv2.boxPoints(rect))

        roi = getROI(box) #get ROI area
        img_centerx, img_centery = getCenter(rect, roi, size, square_length)  # get centre coordinate
        print("the central point of box is : ",img_centerx, img_centery)
        world_x, world_y = convertCoordinate(img_centerx, img_centery, size) #convert to real life coordinate
        print('In real world : ',world_x, world_y)


        cv2.drawContours(frame_resize, [box], -1, range_rgb[detect_color], 2)
        cv2.putText(frame_resize, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_rgb[detect_color], 1) #draw centre point
        

    return frame_resize,x_center,y_center


# find the contour with max area
# input list of contours to be compared as parameter
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  
        contour_area_temp = math.fabs(cv2.contourArea(c))  # calculate contour area
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 300:  # contour is valid only if it is larger than 300
                area_max_contour = c

    return area_max_contour, contour_area_max  #return max


# get object's ROI (range of interesting)
# return the extreme points from the four vertices obtained by cv2.boxPoints(rect)
def getROI(box):
    '''[ [ 92 153]
         [240 153]
         [240 318]
         [ 92 318]]'''

    x_min = min(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    x_max = max(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    y_min = min(box[0, 1], box[1, 1], box[2, 1], box[3, 1])
    y_max = max(box[0, 1], box[1, 1], box[2, 1], box[3, 1])

    return (x_min, x_max, y_min, y_max)


def getCenter(rect, roi, size, square_length):
    x_min, x_max, y_min, y_max = roi
    #based on the center coordinates of the wooden block, 
    #select the vertex closest to the image center as the 
    #reference point for calculating the precise center.
    if rect[0][0] >= size[0]/2:
        x = x_max 
    else:
        x = x_min
    if rect[0][1] >= size[1]/2:
        y = y_max
    else:
        y = y_min

    #calculate length of diagnol
    square_l = square_length/math.cos(math.pi/4)

    #convert length to pixel length
    square_l = world2pixel(square_l, size)

    #calculate the center point based on the rotation angle of the wooden block
    dx = abs(math.cos(math.radians(45 - abs(rect[2]))))
    dy = abs(math.sin(math.radians(45 + abs(rect[2]))))
    if rect[0][0] >= size[0] / 2:
        x = round(x - (square_l/2) * dx, 2)
    else:
        x = round(x + (square_l/2) * dx, 2)
    if rect[0][1] >= size[1] / 2:
        y = round(y - (square_l/2) * dy, 2)
    else:
        y = round(y + (square_l/2) * dy, 2)

    return  x, y

#convert the pixel coordinates of the shape to the coordinate system of the robotic arm
#input coordinate and resolution
def convertCoordinate(x, y, size):
    x = leMap(x, 0, size[0], 0, 640)
    x = x - 320
    x_ = np.round(x * map_param, 2)

    y = leMap(y, 0, size[1], 0, 480)
    y = 240 - y
    y_ = np.round(y * map_param + image_center_distance, 2)

    return x_, y_


#convert real life length to pixel length
#input coordinate and resolution
def world2pixel(l, size):
    l_ = round(l/map_param, 2)

    l_ = leMap(l_, 0, 640, 0, size[0])

    return l_

#mapping
def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def update(SetPoint,feedback_value,last_time,last_error,ITerm):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        kp = 0.1
        ki = 0
        kd = 0.008
        error = SetPoint - feedback_value
        
        current_time = time.time()
	
        delta_time = current_time - last_time
        delta_error = error - last_error
        print('dt:',delta_time)

        
        PTerm = kp * error
        ITerm += error * delta_time
        DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
        last_time = current_time
        last_error = error
        output = PTerm + (ki * ITerm) + (kd * DTerm)
       


        return last_time,last_error,output,ITerm

            
if __name__ == '__main__':
    # If the input is the camera, pass 0 instead of the video file name
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(-1)
    last_time = time.time()
    last_time1= time.time()
    last_error = 0
    last_error1 = 0
    x_dis = 500
    y_dis = 800
    ITerm=0
    ITerm1=0
    Board.setBusServoPulse(4,800,500)
    time.sleep(0.5)
    Board.setBusServoPulse(6,500,500)
    time.sleep(0.5)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    dx=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            rect_frame,x_center,y_center = run(frame)
            cv2.imshow('Frame',rect_frame)
            img_w= frame.shape[1]
            img_h= frame.shape[0]

            if x_dis!=0:
                SetPoint = img_w / 2.0  # set
                feedback = x_center
                last_time, last_error, output, ITerm = update(SetPoint,feedback,last_time,last_error,ITerm)  # 当前
                dx = output
                x_dis += int(dx)  # output
                x_dis = 0 if x_dis < 0 else x_dis
                x_dis = 1000 if x_dis > 1000 else x_dis
                Board.setBusServoPulse(6,x_dis,50)
                time.sleep(0.05)
            if y_dis!=0:
                SetPoint1 = img_h / 2.0  # set
                feedback1 = y_center
                last_time1, last_error1, output1, ITerm1 = update(SetPoint1,feedback1,last_time1,last_error1,ITerm1)  # 当前
                dy = output1
                y_dis -= int(dy)  # output
                y_dis = 0 if y_dis < 0 else y_dis
                y_dis = 1000 if y_dis > 1000 else y_dis
            Board.setBusServoPulse(4,y_dis,50)
            time.sleep(0.05)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()