# AutomaticThresholding
 PyCharm project for automatic thresholding of IntoTheDeep samples.  

The alpha release this project focuses on determining the HSV color
thresholding (cv2.inRange) parameters for a single FTC IntoTheDeep
sample. The project currently runs on Windows 11 using images
previously captured by a Limelight 3A camera.

The project could be ported to the Limelight itself as a custom
script. For this to happen, the code would need to be merged into
a single Python file, the HSV parameters would be encoded
into the Limelight output format - an array of doubles - and all
file writing and display of intermediate results would be
eliminated since the Limelight does not support these operations.

The source file AutomaticColorThresholding.py makes use of a
Pantone color card and ArUco marker recognition as described in
the online article on the website pyimagesearch.com Automatic
color correction with OpenCV and Python by Adrian Rosebrock on
February 15, 2021.

The alpha release contains a number of comments and questions
marked with ##**TODO that point in the direction of future
releases. However, the code successfully thresholds the three
images of single samples - blue, red, and yellow.
