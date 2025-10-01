import cv2
import numpy as np

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
boundries = []
def blend(list_images): # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output

list_images = []
cap = cv2.VideoCapture('deform_purple.MOV')
while (cap.isOpened()):
    
    ret, frame = cap.read()
    if ret: 
        list_images.append(frame)
    else:
        break

output = blend(list_images)

def callback(event):
    boundries.append([event.xdata, event.ydata])
    print(event.xdata, event.ydata)
fig, ax = plt.subplots()
fig.canvas.callbacks.connect('button_press_event', callback)
print("Pick the top left boundry")
print("Pick the bottom right boundry")
_ = plt.imshow(output)
_ = plt.show()
if len(boundries) == 2:
    plt.close()
print("yello")











image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
Nodes = []

# mouse callback function

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global count
        if len(Nodes) == 0:
            print("Pick the bottom right boundry")
        print("New Node:", x, y)
        Nodes.append([len(Nodes), x, y])
        count += 1
        print(count)




import sys
import time
cap = cv2.VideoCapture('deform_purple.MOV')
ret, frame = cap.read()
image_src = frame  # pick.py my.png
image_src = cv2.resize(image_src, (1000, 750))
count = 0
print("Pick the top left boundry")
cv2.imshow("bgr", image_src)
cv2.setMouseCallback('bgr', pick_color)

## NEW ##
cv2.waitKey(0)
  
# closing all open windows 
cv2.destroyAllWindows()
print(Nodes)
