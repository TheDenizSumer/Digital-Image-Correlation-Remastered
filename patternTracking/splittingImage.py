from dotPlacer import DotPlacer
import numpy as np
import cv2

VIDEOPATH = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\videos\benchvice2.mov"
subsectionSize = 70 #in pixels

cap = cv2.VideoCapture(VIDEOPATH)
ret, frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

#dotPickerUI = input("Would you like to use saved dot positions? (y/n): ").strip().lower()
dotPickerUI = 'y'
if dotPickerUI == 'y':
    with open(r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\savedDots.txt", 'r') as f:
        lines = f.readlines()
        coordinates = []
        for line in lines:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
else:
    placer = DotPlacer(frame, dot_radius=10)
    placer.show()

    coordinates = placer.get_coordinates()
    with open(r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\savedDots.txt", 'w') as f:
        for x, y in coordinates:
            f.write(f"{x},{y}\n")
print("\nFinal dot coordinates:")
for i, (x, y) in enumerate(coordinates, 1):
    print(f"Dot {i}: ({x:.2f}, {y:.2f})")

# Finding tl, tr, br, bl
tl, tr, br, bl = None, None, None, None
lowest_sum = float('inf')
highest_sum = float('-inf')
for dot in coordinates:
    if dot[0] + dot[1] < lowest_sum:
        lowest_sum = dot[0] + dot[1]
        tl = dot
    if dot[0] + dot[1] > highest_sum:
        highest_sum = dot[0] + dot[1]
        br = dot
coordinates.remove(tl)
coordinates.remove(br)
if coordinates[0][0] < coordinates[1][0]:
    tr = coordinates[1]
    bl = coordinates[0]
else:
    tr = coordinates[0]
    bl = coordinates[1]

lowestTop = max([tr, tl], key=lambda x: x[1])[1]
highestBottom = min([bl, br], key=lambda x: x[1])[1]
rightMostLeft = max([tl, bl], key=lambda x: x[0])[0]
leftMostRight = min([tr, br], key=lambda x: x[0])[0]
print(f"Top Left: {tl}, Top Right: {tr}, Bottom Right: {br}, Bottom Left: {bl}")
print(f"Lowest Top: {lowestTop}, Highest Bottom: {highestBottom}, Right Most Left: {rightMostLeft}, Left Most Right: {leftMostRight}")
cv2.circle(frame, (int(tl[0]), int(tl[1])), 5, (255, 0, 0), -1)
cv2.circle(frame, (int(tr[0]), int(tr[1])), 5, (0, 255, 0), -1)
cv2.circle(frame, (int(br[0]), int(br[1])), 5, (0, 0, 255), -1)
cv2.circle(frame, (int(bl[0]), int(bl[1])), 5, (0, 255, 255), -1)
tr, tl, bl, br = (int(leftMostRight), int(lowestTop)), (int(rightMostLeft), int(lowestTop)), (int(rightMostLeft), int(highestBottom)), (int(leftMostRight), int(highestBottom))
cv2.rectangle(frame, (int(leftMostRight), int(lowestTop)), (int(rightMostLeft), int(highestBottom)), (0, 255, 0), 2)
cv2.circle(frame, (int(tl[0]), int(tl[1])), 5, (255, 0, 0), -1)
cv2.circle(frame, (int(tr[0]), int(tr[1])), 5, (0, 255, 0), -1)
cv2.circle(frame, (int(br[0]), int(br[1])), 5, (0, 0, 255), -1)
cv2.circle(frame, (int(bl[0]), int(bl[1])), 5, (0, 255, 255), -1)

# cv2.imshow("Image with Rectangle", frame)
# cv2.waitKey(0)

# Each rectangle is repersented as a tuple of it's top left corner and bottom right corner
# intervalx = tl[0] - tr[0] // subsectionSize
# intervaly = tl[0] - bl[0] // subsectionSize

xValues = [x for x in range(tl[0], tr[0], subsectionSize)]
if tr[0] - xValues[-1] > int(2*(subsectionSize // 3)):
    #xValues = xValues[:-1]
    xValues.append(br[0])
yValues = [x for x in range(tl[1], bl[1], subsectionSize)]
if bl[1] - yValues[-1] > int(2*(subsectionSize // 3)):
    #yValues = yValues[:-1]
    yValues.append(bl[1])

rectangles = []
for i in range(len(xValues) -1):
    for j in range(len(yValues)-1):
        cv2.rectangle(frame, (xValues[i], yValues[j]), (xValues[i+1], yValues[j+1]), (255, 0, 0), 1)
        rectangles.append( ((xValues[i], yValues[j]), (xValues[i+1], yValues[j+1])) )
resized_frame = cv2.resize(frame, (size[0] // 2, size[1] // 2))
cv2.imshow("Image with Rectangle", resized_frame)
cv2.waitKey(0)