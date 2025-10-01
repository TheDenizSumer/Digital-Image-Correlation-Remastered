#contour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib

#DefGrad
import math
from sympy import symbols, diff
import numpy as np

#main
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math
from statistics import mode


def loadingBar(percent, length):
        percent = float(percent)
        length = float(length)
        numOfSpaces = int(str(percent/(100/length)).split('.')[0])
        numofnoSpaces = length - numOfSpaces
        x = 0
        shadedSpaces = []
        nonShadedSpaces = []
        while x != numOfSpaces:
            x = x + 1
            shadedSpaces.append(' ')

        x = 0
        while x != numofnoSpaces:
            x = x + 1
            nonShadedSpaces.append(' ')

        shadedSpaces = ''.join(shadedSpaces)
        nonShadedSpaces = ''.join(nonShadedSpaces)
        progressBar = '|' + '\x1b[0;31;41m' + shadedSpaces + '\x1b[0m'+ nonShadedSpaces +'|' + str(percent) + '%'
        return progressBar

def Calc_centroid(p1, p2, p3, p4):
    x = (p1[0]+p2[0]+p3[0]+p4[0])/4
    y = (p1[1]+p2[1]+p3[1]+p4[1])/4
    return x, y

def F(transposed, origin):
    a1_2 = math.sqrt(((origin[1][0] - origin[0][0]) ** 2) + ((origin[1][1] - origin[0][1]) ** 2)) / 2
    a4_3 = math.sqrt(((origin[2][0] - origin[3][0]) ** 2) + ((origin[2][1] - origin[3][1]) ** 2)) / 2
    b4_1 = math.sqrt(((origin[3][1] - origin[0][1]) ** 2) + ((origin[3][0] - origin[0][0]) ** 2)) / 2
    b2_3 = math.sqrt(((origin[2][1] - origin[1][1]) ** 2) + ((origin[2][0] - origin[1][0]) ** 2)) / 2

    x, y = symbols('x y', real=True)

    u = (transposed[0][0]-origin[0][0])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][0]-origin[1][0])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    v = (transposed[0][1]-origin[0][1])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][1]-origin[1][1])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][1]-origin[2][1])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][1]-origin[3][1])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    #u(x, y) = function of x & y returns relative x displacment in element
    #v(x, y) = function of x & y returns relative y displacment in element
    #tx, ty = Calc_centroid(transposed[0], transposed[1], transposed[2], transposed[3])
    tx, ty = Calc_centroid(origin[0], origin[1], origin[2], origin[3])

    xx = float(diff(u, x).replace(y, ty))
    xy = float(diff(u, y).replace(x, tx))
    yx = float(diff(v, x).replace(y, ty))
    yy = float(diff(v, y).replace(x, tx))

    #F = np.matrix([[xx, xy], [yx, yy]])
    F = [[xx, xy], [yx, yy]]
    return F, [tx, ty]

def getColorMask(img):
        #Purple Band
        #lower = np.array([ 9, 161, 38] )
        #upper = np.array([ 62, 255,  83])
        #Ruler
        lower = np.array([ 0, 0, 0] )
        upper = np.array([ 179, 255,  64])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out  = cv2.inRange(hsv, lower, upper)
        return out

def dilate(out):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(out,kernel,iterations=3)
    return dilated

def get_contours(dilated_image):
    contours,_ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    cont = []
    for cnt in contours:
        x,y,w,h=cnt
        cX, cY = x+int(w/2), -(y+int(h/2))
        cont.append([cX, cY])
    return cont

def distance(x, y, px, py):
    xdiff = px-x
    ydiff = py-y
    csq = xdiff**2 + ydiff**2
    return math.sqrt(csq)



def calc_coordinates(video, frame_cap=0, remove_frames=True):
    cap = cv2.VideoCapture(video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        raise "Error opening video stream or file"

    #adding frames of the video to an array called images
    images = []
    frames = 0
    countnum = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            images.append(frame)
            frames += 1
            countnum.append(len(get_contours(dilate(getColorMask(frame)))))
            if frame_cap != 0 and frames == frame_cap:
                break
        else:
            break
    if len(images) == frames:
        print('Images_Processed')
    else:
        print(f'Images: {len(images)}')
        print(f'Frames: {frames}')
        raise f"Frames returned and frames displayed aren't the same. (Images Captured{len(images)} Frames Displayed{frames})"

    #processing images to extract coordinates and countour information    

    Nodes = []
    def pick_color(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("New Node:", x, y)
            Nodes.append([len(Nodes) + 1, x, y])

    cap = cv2.VideoCapture('deform_purple.MOV')
    ret, frame = cap.read()
    image_src = frame  # pick.py my.png
    image_src = cv2.resize(image_src, (1000, 750))
    dot_num = len(Nodes)
    cv2.imshow("bgr",image_src)
    ## NEW ##
    cv2.setMouseCallback('bgr', pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    information = [Nodes]

    for frame in range(1, len(images)):
        contours = get_contours(dilate(getColorMask(images[frame])))
        #format: node - id, x, y
        New_Nodes = []
        for oldNode in information[frame - 1]:
            dist_min = distance(contours[0][1], contours[0][2], oldNode[1], oldNode[2])
            smallestPotNode = contours[0]
            for potNode in contours:
                if distance(potNode[1], potNode[2], oldNode[1], oldNode[2]) < dist_min:
                    smallestPotNode = potNode
            New_Nodes.append(smallestPotNode)
        information.append(New_Nodes)
    information = information[1:]
            
            


    

    print(dot_num)
    return information, images, frame_width, frame_height

def elements(T, NT, squares, direction=None):
    def_element = []
    centroids = []
    for element in squares:
        #et = [T[element[0]], T[element[1]], T[element[2]], T[element[3]]]
        #e = [NT[element[0]], NT[element[1]], NT[element[2]], NT[element[3]]]
        et = [T[element[0]][1:], T[element[1]][1:], T[element[2]][1:], T[element[3]][1:]]

        e = [NT[element[0]][1:], NT[element[1]][1:], NT[element[2]][1:], NT[element[3]][1:]]

        x, y = F(et, e)
        def_element.append(x)
        centroids.append(y)
    return def_element, centroids

#coordinates, images = calc_coordinates('deform_purple.mov', frame_cap=135)
coordinates, images, frame_width, frame_height = calc_coordinates('deform_purple.mov')
#frame cutoff = 135

squares = [
    [21, 22, 24, 25],
    [22, 23, 26, 24],
    [18, 19, 22, 21],
    [19, 20, 23, 22],
    [15, 16, 19, 18],
    [16, 17, 20, 19],
    [12, 13, 16, 15],
    [13, 14, 17, 16],
    [10, 9, 13, 12],
    [9, 11, 14, 13],
    [6, 7, 9, 10],
    [7, 8, 11, 9],
    [3, 4, 7, 6],
    [4, 5, 8, 7],
    [1, 0, 4, 3],
    [0, 2, 5, 4]
    ]

Elements = [] # frames, e1 e2 e3 e4 e5... e27, [xx xy], [yx yy]
Centroids = [] # frames, e1 e2 e3 e4 e5... e27, x y

for frame in range(1, len(coordinates)-7):
#for frame in range(3):
    print(loadingBar(int(frame/len(coordinates)*100), 100))
    element, centroid = elements(coordinates[frame], coordinates[frame-1], squares)
    #element = elementS[frame]
    if frame != 1:
      for i in range(len(element)):
        for x in range(2):
          for y in range(2):
            element[i][x][y] = element[i][x][y] + Elements[-1][i][x][y]
    Elements.append(element)
    Centroids.append(centroid)
thingi = []
for i in Elements:
    print(i[0][1][1])

master = []
for i in Elements:
    for x in i:
        master.append(round(x[1][1], 6))

print()
#smallest_strain = np.percentile(master, 80)
#largest_strain = np.percentile(master, 20)
smallest_strain = min(master)
largest_strain = max(master)
#smallest_strain = -0.028964
#largest_strain = 0.027823
print(smallest_strain, largest_strain)




import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
frame_width, frame_height = 1920, 1440
Levels = np.arange(smallest_strain, largest_strain, 0.0025)
for iter in range(len(Elements)):
    print(iter)
    x = []
    y = []
    z = []
    for i in Centroids[iter]:
        x.append(i[0])
    for i in Centroids[iter]:
        y.append(frame_height+i[1])
    for i in Elements[iter]:
        z.append(i[1][1])



    # Creating a grid for interpolation
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolating z values for the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Plotting contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xi, yi, zi, levels=Levels, cmap='viridis')  # Adjust levels for contour density, cmap for colormap
    plt.colorbar(contour, label='Elevation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)
    #plt.xlim(850, 1000)
    #plt.ylim(500, 1150)
    plt.title('Contour Plot from 3D Coordinates')
    plt.savefig(f"eachFrame/frame{iter}.png")


import cv2
import numpy as np
import glob
img_array = []
for x in range(len(Elements)):
   filename = f'eachFrame/frame{x}.png'
   img = cv2.imread(filename)
   height, width, layers = img.shape
   size = (width,height)
   img_array.append(img)

out = cv2.VideoWriter('video10.avi',cv2.VideoWriter_fourcc(*'DIVX'), 14, size)

for i in range(len(img_array)):
   out.write(img_array[i])
out.release()