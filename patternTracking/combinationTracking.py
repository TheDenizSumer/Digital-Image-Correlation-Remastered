# Using a combination of colour isloation and tracking clusters and then using pixel groups instead of colour blobs to track the individual points

import numpy as np
import cv2
import os
import math 
import copy
import time
from scipy import stats

# video = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\videos\deform_stick.MOV"
# lower = np.array([0, 0, 0]) 
# upper = np.array([9, 255,  76])
# frame_number = 590
# DOTSHAPE = (2, 33)

# video = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\videos\deform_purple.mov"
# lower = np.array([ 9, 161, 38] ) 
# upper = np.array([ 62, 255,  83])
# DOTSHAPE = (9, 3)
# frame_number = 0

video = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\videos\deform_vice2.mov"
lower = np.array([ 0, 0, 21] ) 
upper = np.array([ 179, 62,  102])
DOTSHAPE = (18, 16)
frame_number = 0

def getColorMask(img):
    #deform_purple
    '''
    lower = np.array([ 9, 161, 38] ) 
    upper = np.array([ 62, 255,  83])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    '''
    #Deform_stick.MOV
    #lower = np.array([0, 0, 0]) 
    #upper = np.array([5, 255,  109])
    # lower = np.array([0, 0, 0]) 
    # upper = np.array([9, 255,  76])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    #cv2.imshow("out", out)
    #cv2.imshow("Input", img)
    
    return out

#8/23/23 :
# [ 33 224  66] [  3 204  26] [ 63 244 106]

def dilate(out):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(out,kernel,iterations=3)
    #cv2.imshow(dilated)
    return dilated

def get_contours(dilated_image):
    contours,_ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    return contours

def distance(x, y, px, py):
    xdiff = px-x
    ydiff = py-y
    csq = xdiff**2 + ydiff**2
    return math.sqrt(csq)

def find_dot_center_otsu(image_path):
    """
    Find the center of a black dot using Otsu's thresholding method.
    This automatically finds the optimal threshold between two color groups.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold_value, binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    print(f"Auto-calculated threshold: {threshold_value}")
    
    y_coords, x_coords = np.where(binary == 255)
    
    if len(x_coords) > 0:
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        return (center_x, center_y), binary
    else:
        return None, binary


# Extracting video and getting to the specific starting frame
cap = cv2.VideoCapture(video)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")


# Set the position to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
success, frame = cap.read()

untouched_frame = copy.deepcopy(frame)

#Get the rough positions of the starting dots throught color masking
contours = get_contours(dilate(getColorMask(frame))) 
def display_contours_with_rectangles(frame, contours):
    for cnt in contours:
        if len(cnt) == 5:
            x, y, w, h, keep = cnt  # returns top-left x,y and width,height
        else:
            x, y, w, h = cnt
            keep = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if keep else (255, 0, 0), 4)  # green box, thickness=2

def processCountours(contours, prev=None): # prev would be dots array
    #contours = [[int(cnt[0]), int(cnt[1]), int(cnt[2]*2), int(cnt[3]*2), True] for cnt in contours]
    #contours = [[int(cnt[0]-cnt[2]*.5), int(cnt[1]-cnt[3]*.5), int(cnt[2]*2), int(cnt[3]*2), True] for cnt in contours]
    contours = [[int(cnt[0]), int(cnt[1]), int(cnt[2]*1.2), int(cnt[3]*1.2), True] for cnt in contours]
    # remove bounding boxes within other bounding boxes
    for cnt in contours:
        x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
        for other in contours:
            if other == cnt:
                continue
            ox, oy, ow, oh = other[0], other[1], other[2], other[3]
            if x > ox and y > oy and (x + w) < (ox + ow) and (y + h) < (oy + oh):
                cnt[4] = False  # Mark for removal
                break
    contours = [cnt for cnt in contours if cnt[4]]
    if prev is None:
        #interactive selection of dots to keep and remove function
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # left mouse click
                print(f"Clicked at: ({x}, {y})")
                x , y = int(x * 4), int(y * 4)
                print("Transformed to full size:", (x, y))
                # Identifies which rectangle was clicked
                found = False
                for i, cnt in enumerate(contours):
                    bl, br, ul, ur = (cnt[0], cnt[1]), (cnt[0]+cnt[2], cnt[1]), (cnt[0], cnt[1]+cnt[3]), (cnt[0]+cnt[2], cnt[1]+cnt[3])
                    if bl[0] - 15 <= x <= br[0] + 15 and bl[1] - 15 <= y <= ul[1] + 15:
                        contours[i][4] = not contours[i][4]  # Toggle the keep status
                        print(f"Rectangle at ({cnt[0]}, {cnt[1]}) toggled to {'keep' if cnt[4] else 'discard'}")
                        found = True
                        break
                if found:
                    display_contours_with_rectangles(frame, contours)
                    resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
                    cv2.imshow('Bounding Rectangles', resized_frame)
                else:
                    print("Clicked outside any rectangle.")
                    pickEscape = False
                    cv2.destroyAllWindows()
                
        display_contours_with_rectangles(frame, contours)
        
        resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
        print(frame.shape)
        cv2.imshow('Bounding Rectangles', resized_frame)
        cv2.setMouseCallback('Bounding Rectangles', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        contours = [cnt[:4] for cnt in contours if cnt[4]]
        if SSuserInput != 'y' and input("Save selection? (y/n): ") == 'y':
            with open("clickLog.txt", "w") as f:
                for cnt in contours:
                    f.write(f"{cnt}\n")
    else:
        centers = returnAllDotCenters(prev)
        # if DOTSHAPE[0] * DOTSHAPE[1] != len(contours):
        #     print(f"Warning: Expected {DOTSHAPE[0] * DOTSHAPE[1]} contours, but found {len(contours)}. Skipping frame.")
        #     return prev.flatten().tolist(), True
        newContours = []
        distances = []
        for center in centers:
            cx, cy = center
            closest_cnt = None
            closest_dist = float('inf')
            for cnt in contours:
                x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
                cnt_cx, cnt_cy = x + w // 2, y + h // 2
                dist = distance(cx, cy, cnt_cx, cnt_cy)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_cnt = cnt
            if closest_cnt is not None and closest_cnt not in newContours:
                newContours.append(closest_cnt[:4])  # Keep this contour
                distances.append(closest_dist)
        pass
        contours = newContours
    return contours, False

#SSuserInput = 'y'
SSuserInput = input("Use saved selection? (y/n): ")

if SSuserInput == 'y':
    with open("clickLog.txt", "r") as f:
        saved_contours = f.readlines()
    contours = [[int(x) for x in line.strip()[1:-1].split(", ")] for line in saved_contours]
else:
    contours = processCountours(contours)


print(f"Kept {len(contours)} contours after selection.")

# if SSuserInput == 'y':
#     print("Final contours:")
#     display_contours_with_rectangles(frame, contours)
#     resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
    
#     cv2.imshow('Bounding Rectangles', resized_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Tracking dots more precisely within the rough contours found previously
# This class keeps track of each dot's position and image snippet
class Snippet:
    def __init__(self, initContourDetails):
        x, y, w, h = initContourDetails
        initial_pos = (x + w // 2, y + h // 2)
        self.positions = []  # List of (x, y) tuples
        self.bounding_box = (x, y, w, h)  # Store bounding box for reference
        self.snippet = None  # Placeholder for image snippet
        self.id = 0
        self.center = initial_pos
    def update_position(self, new_pos):
        self.positions.append(new_pos)
    def find_dot_center_otsu(self):
        img_gray = cv2.cvtColor(self.snippet, cv2.COLOR_BGR2GRAY)
        threshold_value, binary = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        y_coords, x_coords = np.where(binary == 255)
        
        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            self.center = (self.bounding_box[0] + round(center_x), self.bounding_box[1] + round(center_y))
            
        return binary
        

    def extract_snippet(self, img):   
        x, y, w, h = self.bounding_box 
        y1, y2 = y, y + h
        x1, x2 = x, x + w
        self.snippet = img[y1:y2, x1:x2]
        return self.snippet

def returnAllDotCenters(dotArray):
    points = dotArray.flatten()
    centers = []
    for pnt in points:
        centers.append(pnt.center)
    return centers

points = []

centeriodImage = copy.deepcopy(untouched_frame)


def initialContoursToDots(contours, prev=None):
    points = []
    for cnt in contours:
        points.append(Snippet(cnt[:4]))
        points[-1].extract_snippet(frame)
        points[-1].find_dot_center_otsu()
        #cv2.circle(centeriodImage, points[-1].center,  4, (0, 255, 0), -1)

    xValues = np.array([points[i].center[0] for i in range(len(points))])
    yValues = np.array([points[i].center[1] for i in range(len(points))])

    #sort by y first, then x
    xValuesSortedIndices = np.argsort(xValues)
    yValuesSortedIndices = np.argsort(yValues)

    biggestXDiff = 0
    biggestYDiff = 0
    for i in range(0, DOTSHAPE[1]-1, DOTSHAPE[0]):
        xDiff = xValuesSortedIndices[i:i+DOTSHAPE[0]].max() - xValuesSortedIndices[i:i+DOTSHAPE[0]].min()
        biggestXDiff = max(biggestXDiff, xDiff)
    for i in range(DOTSHAPE[0]-1):
        yDiff = yValuesSortedIndices[i:i+DOTSHAPE[1]].max() - yValuesSortedIndices[i:i+DOTSHAPE[1]].min()
        biggestYDiff = max(biggestYDiff, yDiff)

    print("Biggest X diff:", biggestXDiff)
    print("Biggest Y diff:", biggestYDiff)

    topLeftPnt = None
    bottomRightPnt = None


    for pnt in points:
        x, y = pnt.center
        if bottomRightPnt is None or (x + y) > (bottomRightPnt.center[0] + bottomRightPnt.center[1]):
            bottomRightPnt = pnt

    dots = np.empty(DOTSHAPE, dtype=object)


    dots[DOTSHAPE[0]-1, DOTSHAPE[1]-1] = bottomRightPnt
    dots[DOTSHAPE[0]-1, DOTSHAPE[1]-1].id = DOTSHAPE[0] * DOTSHAPE[1] - 1
    while None in dots:
        for i in range(DOTSHAPE[0]):
            topLeftPnt = bottomRightPnt
            for pnt in points:
                x, y = pnt.center
                if (not np.any(dots == pnt)) and (topLeftPnt is None or (x + y) < (topLeftPnt.center[0] + topLeftPnt.center[1])):
                    topLeftPnt = pnt
            dots[i, 0] = topLeftPnt
            dots[i, 0].id = i*DOTSHAPE[1]
            
            for j in range(1, DOTSHAPE[1]):
                if dots[i, j] is not None:
                    continue
                neighborPoint = bottomRightPnt
                for pnt in points:
                    x, y = pnt.center
                    refX, refY = dots[i, j-1].center
                    if x > refX and x < neighborPoint.center[0] + biggestXDiff and abs(y - refY) < biggestYDiff + 5:
                        neighborPoint = pnt
                dots[i, j] = neighborPoint
                dots[i, j].id = i*DOTSHAPE[1] + j
        # pop every neighbor point out of the points lsit and when going to a new row, take the point with the smallest x value and use it as the first refernce point (set it to be the first entry in that row)
    return xDiff, yDiff, dots

pxdiff, pydiff, firstFrameDots = initialContoursToDots(contours)
# firstFrameDots = returnAllDotCenters(firstFrameDots)
# for dot in firstFrameDots:
#     cv2.circle(centeriodImage, dot,  4, (0, 255, 0), -1)
# resized_frame = cv2.resize(centeriodImage, (size[0] // 4, size[1] // 4))
# cv2.imshow("Original with Center", resized_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("Final contours:")
# display_contours_with_rectangles(frame, contours)
# resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))

# cv2.imshow('Bounding Rectangles', resizesd_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def contoursToDots(contours, prevDots):
    contours = [[int(cnt[0]), int(cnt[1]), int(cnt[2]*1.2), int(cnt[3]*1.2), True] for cnt in contours]
    for cnt in contours:
        x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
        for other in contours:
            if other == cnt:
                continue
            ox, oy, ow, oh = other[0], other[1], other[2], other[3]
            if x > ox and y > oy and (x + w) < (ox + ow) and (y + h) < (oy + oh):
                cnt[4] = False  # Mark for removal
                break
    contours = [cnt for cnt in contours if cnt[4]]
    if len(contours) < DOTSHAPE[0] * DOTSHAPE[1]:
        print(f"Warning: Expected {DOTSHAPE[0] * DOTSHAPE[1]} contours, but found {len(contours)}. Skipping frame.")
        return prevDots, True, []
    
    newDots = np.empty(DOTSHAPE, dtype=object)
    distances = []
    for dot in prevDots.flatten():
        cx, cy = dot.center
        closest_cnt = None
        closest_dist = float('inf')
        for cnt in contours:
            x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
            cnt_cx, cnt_cy = x + w // 2, y + h // 2
            dist = distance(cx, cy, cnt_cx, cnt_cy)
            if dist < closest_dist:
                closest_dist = dist
                closest_cnt = cnt
        if closest_cnt is not None:
            newDot = Snippet(closest_cnt[:4])
            newDot.id = dot.id
            newDot.extract_snippet(frame)
            newDot.find_dot_center_otsu()
            newDots[dot.id // DOTSHAPE[1], dot.id % DOTSHAPE[1]] = newDot
            contours.remove(closest_cnt)
            distances.append(closest_dist)

    data = np.array(distances)

    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > 6]  # "3" means 3 standard deviations from the mean

    print("Outliers:", outliers)
    skipFrame = False
    for outlier in outliers:
        if outlier > pxdiff + 10 or outlier > pydiff + 10:
            skipFrame = True
            break
    return newDots, skipFrame, outliers



dotsInFrame = [copy.deepcopy(firstFrameDots)]
paused = False
id = 0
while cap.isOpened():
    id += 1
    print(id)
    if not paused:
        ret, frame = cap.read()
    if id == 130:
        pass
    contours = get_contours(dilate(getColorMask(frame)))
    #contours, skipFrame = processCountours(contours, dotsInFrame[-1])
    display_contours_with_rectangles(frame, contours)
    resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
    cv2.imshow(f"contours", resized_frame)
    #xDiff, yDiff, newDots = contoursToDots(contours)
    newDots, skipFrame, outliers = contoursToDots(contours, dotsInFrame[-1])
    # if xDiff > pxdiff + 10 or yDiff > pydiff + 10:
    #     print("Skipping frame due to large dot movement")
    #     #paused = True
    #     quit()
    #     continue
    if skipFrame:
        print("Skipping frame due to large dot movement")
        resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
        cv2.imshow("Video", resized_frame)
        time.sleep(3)
        #paused = True
        #quit()
        continue
    # Processing each dot snippet in the current frame
    dotsInFrame.append(newDots)
    for dot in dotsInFrame[-1].flatten():
        cv2.circle(frame, dot.center,  10, (0, 255, 0), -1)
        cv2.putText(frame, str(dot.id), (dot.center[0]-10, dot.center[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    
    resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
    cv2.imshow("Video", resized_frame)
    #time.sleep(.5)
        # Press 'q' to quit
    if len(outliers) > 0:
        time.sleep(4)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# # finding all of the squares made by the dots
# rows, cols = DOTSHAPE

# squares = []
# for r in range(rows - 1):
#     for c in range(cols - 1):
#         square = [dots[r, c], dots[r, c+1], dots[r+1, c], dots[r+1, c+1]]
#         squares.append(square)

# squaresIDs = []
# for square in squares:
#     SquareIDs = [point.id for point in square]
#     squaresIDs.append(SquareIDs)

# print(squaresIDs)



# resized_frame = cv2.resize(centeriodImage, (size[0] // 4, size[1] // 4))
# cv2.imshow("Original with Center", resized_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# '''
#     if ret == True:
#         #cv2.imshow('Frame',frame)
#         frame = cv2.resize(frame, (960, 540))
#         #if len(get_contours(dilate(getColorMask(frame)))) == 27:
#         images.append(frame)
#         frames += 1
#         #if cv2.waitKey(25) & 0xFF == ord('q'):
#         #    break
#         #print(frames)
#     else: 
#         break
# if len(images) != frames:
#     print(f'Images: {len(images)}')
#     print(f'Frames: {frames}')

# print(len(images))
# information = []

# for image in images:
#     contours = get_contours(dilate(getColorMask(image)))   
#     #new sticker coords
#     info = []
#     cont = []
#     init_points = 0
#     for cnt in contours:
#         x,y,w,h=cnt
#         cX, cY = x+int(w/2), y+int(h/2)
#         #cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
#         #cv2.putText(image, str(X), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#         cont.append([init_points, cX, cY])
#         init_points += 1
    
#     if information != []:
#         i = -1
#         try:
#             while len(information[i]) != 27:  
#                 i -= 1
#         except:
#             i = -1
#         prev_cont = information[i]
#         #prev_cont = information[-1]
#         for point in cont:
#             contX = point[1]
#             contY = point[2]
#             smallest_n, smallest_dist = prev_cont[0][0], distance(contX, contY, prev_cont[0][1], prev_cont[0][2])
#             for prev_point in prev_cont:
#                 dist = distance(contX, contY, prev_point[1], prev_point[2])
#                 if dist < smallest_dist:
#                     smallest_n, smallest_dist = prev_point[0], dist #distance calculated here
#             info.append([smallest_n, contX, contY])
#         information.append(info)
#     else:
#         information.append(cont)
    
# #result = cv2.VideoWriter('computed_colormasking.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
# xx= 0
# for info in information:
#     for point in info:
#         cv2.circle(images[xx], (point[1], point[2]), 2, (0, 0, 255), -1)
#         cv2.putText(images[xx], str(point[0]), (point[1]-9, point[2]-9),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#     cv2.imshow("hope this works",images[xx])
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#     #result.write(images[xx])
    
#     xx += 1

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #python C:\Users\deniz\Desktop\computer_2\DigitSoftware\template_matching\main.py'''