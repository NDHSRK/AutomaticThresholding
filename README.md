# AutomaticThresholding
 PyCharm project for the automatic thresholding of IntoTheDeep samples.  

The alpha release this project focuses on determining the HSV color
thresholding (cv2.inRange) parameters for a single FTC IntoTheDeep
sample. The project currently runs in PyCharm on Windows 11 using images
previously captured by a Limelight 3A camera.

The project could be ported to the Limelight itself as a custom
script. For this to happen, the code would need to be merged into
a single Python file, the HSV parameters would be encoded
into the Limelight output format - an array of doubles - and all
file writing and display of intermediate results would be
eliminated since the Limelight does not support these operations.

The Alpha release of the current project is based on
the online article on the website pyimagesearch.com "Automatic
color correction with OpenCV and Python" by Adrian Rosebrock on
February 15, 2021. The goal of the article and accompanying code
is to match the color histogram of an input image of a Pantone
color card with that of a reference color card. In the current
project, the source file AutomaticColorThresholding.py makes use
of a Pantone color card and performs ArUco marker recognition
but then diverges from the pyimagesearch project. Instead, we
get the location of the aperture at the center of the Pantone
card and extract a sub-image which we then process to determine
the low and high HSV hue values.

In the Alpha release the use of the Pantone card is a convenience.
For IntoTheDeep samples it would be better to create a custom
template (on a piece of card stock, for example) with its own
ArUco markers and an aperture large enough to expose the full
size of a sample.

The Alpha release contains a number of comments and questions
marked with ##**TODO that point in the direction of future
releases. However, the code successfully thresholds the three
images of single samples - blue, red, and yellow.
