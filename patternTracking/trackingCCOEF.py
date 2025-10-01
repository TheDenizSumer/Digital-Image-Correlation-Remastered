import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
patch_size = 32    # size of square patch
search_radius = 10 # search window around each patch center
step = 40          # spacing between subset centers

# --- Load frames ---
ref_path = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\image.png"
defm_path = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\streched.png"

ref_color = cv2.imread(ref_path)  # Loads in BGR by default

ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
cv2.imshow("ref", ref)
defm_color = cv2.imread(defm_path)
defm = cv2.cvtColor(defm_color, cv2.COLOR_BGR2GRAY)

h, w = ref.shape

# --- Grid of patch centers ---
ys = np.arange(patch_size//2, h - patch_size//2, step)
xs = np.arange(patch_size//2, w - patch_size//2, step)

displacements = []

for y in ys:
    for x in xs:
        # Extract reference patch
        patch = ref[y - patch_size//2:y + patch_size//2,
                    x - patch_size//2:x + patch_size//2]

        # Define search window in deformed frame
        y1 = max(y - search_radius - patch_size//2, 0)
        y2 = min(y + search_radius + patch_size//2, h)
        x1 = max(x - search_radius - patch_size//2, 0)
        x2 = min(x + search_radius + patch_size//2, w)

        search_area = defm[y1:y2, x1:x2]

        # Match template
        res = cv2.matchTemplate(search_area, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Compute displacement (dx, dy)
        dx = (max_loc[0] + patch_size//2 + x1) - x
        dy = (max_loc[1] + patch_size//2 + y1) - y

        displacements.append((x, y, dx, dy))

# --- Visualization ---
fig, ax = plt.subplots()
ax.imshow(defm, cmap="gray")

for (x, y, dx, dy) in displacements:
    ax.arrow(x, y, dx, dy, color="red", head_width=3, head_length=3)

ax.set_title("2D Deformation Field (matchTemplate)")
plt.show()
