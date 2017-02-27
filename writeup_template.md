#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

Firts Converts the image to grayScale
Defeine the parameter for the canny edge detction
the defines the vertices for the poligon used as mask for th eregon of interest
DEfine th ehough parameters for the transformation
Obtain the lines detected for the Hough 
Create a olor imaeg to combine whit the original 
Save the image.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the image contains curves 
on the land lane the dectector cant detect very vell the lines.
Another is the distortion of images by the camera.




###3. Suggest possible improvements to your pipeline

A possible improvement would be to cobine whit other techniques to identify lane lines.
And Terrains to improve the acuracity of the software.

