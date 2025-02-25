1. Shape Detection using OpenCV
This project detects and classifies basic geometric shapes in an image using OpenCV. It processes an image, identifies contours, and determines the shape based on the number of edges.

How It Works
Loads an image and converts it to grayscale.
Applies Gaussian blur and Canny edge detection.
Finds contours and filters small ones based on an area threshold.
Approximates contours to identify the number of edges.
Classifies detected shapes (Triangle, Square, Rectangle, Rhombus, Circle).
Draws contours and labels the shapes with their detected names and areas.

Output
Prints detected shapes and their areas in the console.
Displays the image with labeled shapes.


2. Limitations of the Canny Edge Detector:
Too Sensitive to Noise - If an image has a lot of small details or grainy areas, Canny might mistake that noise for real edges, even after smoothing the image.
Hard to Set the Right Parameters - Canny needs two threshold values to decide what counts as an edge. If you pick values that are too low, you get too many unwanted edges. If they’re too high, you might miss important ones.
Edges Can Break Apart - In images with shadows or uneven lighting, Canny might detect parts of an edge but miss others, causing broken or incomplete shapes.
Trouble with Subtle Edges - If an object blends softly into the background or has a lot of tiny textures, Canny might either miss the edges or confuse texture patterns with object boundaries.

Better Alternatives:
Laplacian of Gaussian (LoG) - This method smooths the image to reduce noise, then looks for areas where brightness changes suddenly (edges).
Sobel Edge Detector - Sobel looks at how brightness changes in both horizontal and vertical directions. 
AI-Powered Edge Detection - There are modern tools that use artificial intelligence to learn what edges look like in complex scenes. These can handle things like shadows, overlapping objects, and even faint edges — but they need more computing power and training data.
