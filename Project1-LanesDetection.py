#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
import math
import os
from scipy.optimize import curve_fit

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
        
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    def lineEqu(x, a, b):
        return a * x + b
    leftLane = []
    leftLane.append([])
    leftLane.append([])
    rightLane = []
    rightLane.append([])
    rightLane.append([])
    for line in lines:
        for x1,y1,x2,y2 in line:
            #Emad: positive slope --> right lane, negative slope: left lane
            #Emad: Also, avoid taking lines from left part to distort right part and vice versa
            if (  (( (y2-y1) / (x2-x1) ) > 0) and
                (x1 > 500) and  (x2 > 500)):
                rightLane[0].append(x1);
                rightLane[0].append(x2);
                rightLane[1].append(y1);
                rightLane[1].append(y2);
            elif (  (( (y2-y1) / (x2-x1) ) < 0) and
                (x1 < 500) and  (x2 < 500)):
                leftLane[0].append(x1);
                leftLane[0].append(x2);
                leftLane[1].append(y1);
                leftLane[1].append(y2);
    if  (rightLane[0] and rightLane[1]):
        coeffRight = np.polyfit(rightLane[0], rightLane[1], 1)
        p = np.poly1d(coeffRight)
        cv2.line(img, (520, int(p(520))), (910, int(p(910))), color, thickness)
    if  (leftLane[0] and leftLane[1]):
        coeffLeft = np.polyfit(leftLane[0], leftLane[1], 1)
        p = np.poly1d(coeffLeft)
        cv2.line(img, (125, int(p(125))), (450, int(p(450))), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

#reading in an image
#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
#print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

##### Start of my code #####
file_list = os.listdir("test_images/")
for file_name in file_list:
    
    # Read in and grayscale the image
    image = mpimg.imread('test_images/'+file_name)
    gray = grayscale(image)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    #plt.imshow(edges, cmap='gray')
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(125,imshape[0]),(450, 320), (520, 320), (910,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges, cmap='gray')
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25 #minimum number of pixels making up a line
    max_line_gap = 15    # maximum gap in pixels between connectable line segments
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on a blank image
    #line_image = np.copy(image)*0 # creating a blank to draw lines on
    #draw_lines(line_image,lines, (255,0,0),10)
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    # Draw the lines on the edge image
    lines_edges = weighted_img(lines, color_edges)
    
    #plt.imshow(lines_edges)
    # Uncomment the following code if you are running the code locally and wish to save the image
    mpimg.imsave('test_images/Result_'+file_name, lines_edges)