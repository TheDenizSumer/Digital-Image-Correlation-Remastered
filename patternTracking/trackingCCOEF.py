import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, find_objects

ref_path = r"patternTracking\rulerImage.jpg"
defm_path = r"patternTracking\rulerSnippet.jpg"

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper: subpixel peak refinement
# -----------------------------
def subpixel_peak(res, max_loc):
    x, y = max_loc
    if 0 < x < res.shape[1]-1 and 0 < y < res.shape[0]-1:
        window = res[y-1:y+2, x-1:x+2]

        dx = 0.5 * (window[1,2] - window[1,0])
        dy = 0.5 * (window[2,1] - window[0,1])
        dxx = window[1,2] - 2*window[1,1] + window[1,0]
        dyy = window[2,1] - 2*window[1,1] + window[0,1]
        dxy = 0.25 * (window[2,2] - window[2,0] - window[0,2] + window[0,0])

        A = np.array([[dxx, dxy],
                      [dxy, dyy]])
        b = -np.array([dx, dy])
        try:
            offset = np.linalg.solve(A, b)
            return x + offset[0], y + offset[1]
        except np.linalg.LinAlgError:
            return float(x), float(y)
    return float(x), float(y)

# -----------------------------
# Load full image and screenshot
# -----------------------------
full_image_color = cv2.imread(ref_path)
screenshot_color = cv2.imread(defm_path)

# Convert to grayscale
full_image = cv2.cvtColor(full_image_color, cv2.COLOR_BGR2GRAY)
screenshot = cv2.cvtColor(screenshot_color, cv2.COLOR_BGR2GRAY)

# Optional: enhance contrast
full_image = cv2.equalizeHist(full_image)
screenshot = cv2.equalizeHist(screenshot)

# -----------------------------
# Template matching
# -----------------------------
res = cv2.matchTemplate(full_image, screenshot, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)

# Integer-pixel match
int_match = max_loc
# Subpixel refinement
sub_match = subpixel_peak(res, max_loc)

print(f"Integer match at {int_match}")
print(f"Subpixel match at {sub_match}")

# -----------------------------
# Visualization
# -----------------------------
h, w = screenshot.shape
img_disp = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)

# Draw integer match (red rectangle)
cv2.rectangle(img_disp, int_match, (int_match[0]+w, int_match[1]+h), (0,0,255), 2)
# Draw subpixel match (green circle)
cv2.circle(img_disp, (int(round(sub_match[0])), int(round(sub_match[1]))), radius=5, color=(0,255,0), thickness=2)

plt.figure(figsize=(8,6))
plt.title("Red=integer match, Green=subpixel refined match")
plt.imshow(img_disp[..., ::-1])
plt.show()
