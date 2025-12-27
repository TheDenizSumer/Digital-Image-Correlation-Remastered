import cv2
import numpy as np

video_path = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\videos\deform_purple.mov"

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
frame_number = 0 # 590 for stick

# Set the position to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    resized_frame = cv2.resize(frame, (size[0] // 4, size[1] // 4))
    cv2.imshow('Frame',resized_frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames gjygyugyughygjh
cv2.destroyAllWindows()