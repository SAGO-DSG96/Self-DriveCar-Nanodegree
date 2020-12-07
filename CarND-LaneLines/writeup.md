# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/step1.png "Grayscale"
[image2]: ./test_images_output/step2.png "BlurredImage"
[image3]: ./test_images_output/step3.png "CannyEdges"
[image4]: ./test_images_output/step4.png "RegionInterest"
[image5]: ./test_images_output/step5.png "HoughSpace"
[image6]: ./test_images_output/step6.png "FinalImage"

---

## Reflection

## 1. Describe pipeline

My pipeline consisted of 6 steps after read the image where with help of Python and OpenCV I  analytical pipeline that can be used to automate lane line detection in image and movie files. This report reflects some lessons learned.

### Step 1

First, I converted the images previus read to grayscale, using function 
``` 
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
Color lines images are withe and yellow converting images in to grayscale will help as we will have white lines.

![alt text][image1]

### Step 2

Second using Gaussian blurring over the grayscaled image helps to reduce high-frequency/noisy aspects in an image. 
``` 
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
Blurring images will help to our algorithm to detect edge detections providing an a better chance at capturing only lower-frequency contours of objects in the field of view.

![alt text][image2]


### Step 3

Third, using Canny function will find edges of objects in the image.
``` 
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```
![alt text][image3]


### Step 4

Fourth, trough setting up a region masking we will delimeter an area of interest as previus step we found the edge of all objects instead of lane lines.

``` 
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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
```
![alt text][image4]

After some test, I found parameters that works with the region of interest


### Step 5
Fifth, trough Hough function it is possible to group points that belong to the edges of possible figures through a voting procedure on a set of parameterized figures.
In a 2D Hough space, points represent lines through a 2D Euclidean space. Each line in a 2D Euclidean space can be represented by two pieces of information, rho and theta.

-   rho is the shortest distance of the line to origin, (0,0)
-   theta is the angle of the line connecting those two points hough space

``` 
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```

![alt text][image5]


### Step 6
Sixth, trough weighted_img function we will draw left and right lane lines by averaging/extrapolating the lines obtained obtained in [STEP5] we will joined result from [STEP5] and original image and get an awesome result.

``` 
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```
![alt text][image6]

To see the results from [images, click here.](https://github.com/SAGO-DSG96/Self-DriveCar-Nanodegree/tree/master/CarND-LaneLines/test_images_output)

### 2. Identify potential shortcomings with your current pipeline


- One potential shortcoming would be what would happen when some roadway are segmented like in the challenge the pipeline must be faster to alternate to another one and get same results as other test. 

- Another shortcoming are from hardware, what happens if the camera are in other position or if camera len recieve the sunlight direcly.

- Another shortcoming is that in any video faces traffic jam, this will have an impact in our algorithm.


### 3. Suggest possible improvements to your pipeline

- A possible improvement would be to adapt pipeline to be faster enough to face different roadways.
- Another potential improvement could be test in traffic jams.
