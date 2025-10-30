import cv2
import numpy as np
from sklearn.cluster import KMeans


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


# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\rulerSnippet.jpg"
    
    print("Method 1: K-means Clustering")
    center_kmeans, mask_kmeans = find_dot_center_kmeans(image_path)
    if center_kmeans:
        print(f"Dot center: ({center_kmeans[0]:.2f}, {center_kmeans[1]:.2f})")
    
    print("\nMethod 2: Otsu's Thresholding")
    center_otsu, mask_otsu = find_dot_center_otsu(image_path)
    if center_otsu:
        print(f"Dot center: ({center_otsu[0]:.2f}, {center_otsu[1]:.2f})")
    
    print("\nMethod 3: Histogram Analysis")
    center_hist, mask_hist = find_dot_center_histogram(image_path)
    if center_hist:
        print(f"Dot center: ({center_hist[0]:.2f}, {center_hist[1]:.2f})")
    
    # Optional: Visualize results
    img = cv2.imread(image_path)
    if center_kmeans:
        cv2.circle(img, (int(center_kmeans[0]), int(center_kmeans[1])), 
                   2, (0, 255, 0), -1)
        cv2.imshow("Original with CenterK", img)
        cv2.imshow("Binary MaskK", mask_kmeans)
    if center_otsu:
        cv2.circle(img, (int(center_otsu[0]), int(center_otsu[1])), 
                   2, (0, 255, 0), -1)
        cv2.imshow("Original with CenterO", img)
        cv2.imshow("Binary MaskO", mask_otsu)
    if center_hist:
        cv2.circle(img, (int(center_hist[0]), int(center_hist[1])), 
                   2, (0, 255, 0), -1)
        cv2.imshow("Original with CenterH", img)
        cv2.imshow("Binary MaskH", mask_hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()