import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

class DotPlacer:
    def __init__(self, image_path, dot_radius=5):
        """
        Initialize the dot placer with an image.
        
        Args:
            image_path: Path to the image file
            dot_radius: Radius for detecting clicks near dots (in pixels)
        """
        if type(image_path) == str:
            self.image = mpimg.imread(image_path)
        else:
            self.image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        self.dot_radius = dot_radius
        self.dots = []  # List to store dot coordinates as (x, y) tuples
        self.dot_plots = []  # List to store plot objects
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.image)
        self.ax.set_title('Click to place dots, click on dots to remove them')
        
        # Connect the click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        
        # Check if click is near an existing dot
        dot_index = self.find_nearby_dot(x, y)
        
        if dot_index is not None:
            # Remove the dot
            self.remove_dot(dot_index)
        else:
            # Add a new dot
            self.add_dot(x, y)
            
        self.fig.canvas.draw()
    
    def find_nearby_dot(self, x, y):
        """
        Find if there's a dot near the clicked position.
        
        Returns:
            Index of the nearby dot, or None if no dot is nearby
        """
        for i, (dot_x, dot_y) in enumerate(self.dots):
            distance = np.sqrt((x - dot_x)**2 + (y - dot_y)**2)
            if distance <= self.dot_radius:
                return i
        return None
    
    def add_dot(self, x, y):
        """Add a new dot at the specified coordinates."""
        self.dots.append((x, y))
        dot_plot = self.ax.plot(x, y, 'ro', markersize=2)[0]
        self.dot_plots.append(dot_plot)
        print(f"Dot added at ({x:.2f}, {y:.2f})")
        print(f"Total dots: {len(self.dots)}")
    
    def remove_dot(self, index):
        """Remove a dot at the specified index."""
        removed_dot = self.dots.pop(index)
        dot_plot = self.dot_plots.pop(index)
        dot_plot.remove()
        print(f"Dot removed from ({removed_dot[0]:.2f}, {removed_dot[1]:.2f})")
        print(f"Total dots: {len(self.dots)}")
    
    def get_coordinates(self):
        """
        Get all current dot coordinates.
        
        Returns:
            List of (x, y) tuples
        """
        return self.dots.copy()
    
    def get_coordinates_array(self):
        """
        Get all current dot coordinates as a numpy array.
        
        Returns:
            Numpy array of shape (n, 2) where n is the number of dots
        """
        return np.array(self.dots) if self.dots else np.array([]).reshape(0, 2)
    
    def show(self):
        """Display the interactive plot."""
        plt.show()


# Example usage
if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\deniz\Coding\Digital-Image-Correlation-Remastered\patternTracking\rulerImage.jpg")
    #placer = DotPlacer('image.png', dot_radius=10)
    placer = DotPlacer(image, dot_radius=10)
    placer.show()

    coordinates = placer.get_coordinates()
    print("\nFinal dot coordinates:")
    for i, (x, y) in enumerate(coordinates, 1):
        print(f"Dot {i}: ({x:.2f}, {y:.2f})")
    
    # Or get as numpy array
    coords_array = placer.get_coordinates_array()
    print(f"\nCoordinates as numpy array:\n{coords_array}")