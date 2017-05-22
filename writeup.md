#Vehicle Detection and Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading the writeup!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 19th code cell of the IPython notebook. The HOG feature extraction happens inside the `extract_features()` function when `get_hog_features()` is called on the image. This `get_hog_feature()` can be found in the 4th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and saw how the Linear SVM performed for each combination and settled on values where I got satisfactory results in terms of validation accuracy of the SVC. This process can be found in 18th and 19th code cells. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the 19th code cell of the IPython notebook using a `train_and_validate()` helper function as defined in the 10th code cell. Feature vectors for the training data contain color histogram features, spatial features and HOG features obtained by calling `extract_features()` on paths of training data as seen in the 19th code cell. 

`train_and_validate()` uses `sklearn`'s `LinearSVC()` classifier. Here, the function takes feature vectors and labels as inputs. Then the datapoints are shuffled and split into train and test sets before training happens. Then the test accuracy is calculated  by evaluating the performance of the trained classifer on the test set datapoints. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search for cars is implemented in the `find_cars()` function at the 11th code cell of the IPython notebook. 

First, depending on the scale used, the image is resized to allow different sized search windows. Then the image is split into its costituent color channels and the individual channel HOG features for the entire image is calculated. After the parameters used for sliding window search is defined, each iteration over all the windows to be searched returns a window which can be classified as cars or non-cars using the trained classifier. To extract color features and spatial features of the window patch, `color_hist()` and `bin_spatial()` is used. For HOG features, we subsample the individual color channel HOG features of the entire image and stack them together. This is better than using `get_hog_features()` on all of our windows since we have overlapping of the sliding windows. This helps us minimize the processing time for each image.  

Below is an image on which `find_cars()` function was called using various scales.

![][image3]

The parameters that could be tuned were `scale`, `ystart`, `ystop` given as inputs and `cells_per_step` defined inside the helper function `find_cars()`. The `ystart` and `ystop` is determined by looking at our test images and selecting portions of the image where we intend to find cars (and eliminating portions like sky that won't contain cars). The values for `scale` were determined by experimenting with values starting from 1 to 3 and the best performing combination of three different scales were used for the final pipeline. Also, the `cells_per_step` that determines the overlap of windows was initially set to 2 which meant an overlap of 75% given the window size of 64 with 8x8 cells. Since this gave me satisactory results, I ended up using `cells_per_step` as 2.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Further optimization invloved removing of false positives and multiple detections obtained by using different scales. To do this, I generated a heatmap of detections and used a threshold such that it limited the detections to only true positives. 

Here are some example images where the optimized classification pipeline was used:

![][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/5t69oVhwUl8)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are six frames and their corresponding heatmaps:

![][image5]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![][image6]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Initially, I approached the problem of manual feature modeling. After experimenting on using various combinations of HOG features, spatial features and color histogram features, I ended up with a feature space that performed well on my classification problem involving detection of cars. The majority of my time was spent in trying out the various combination of parameters that defined the feature space. 

My pipeline uses a linear SVM classifier and hence it might fail in places involving false positives (in this case, misclassification of portions of images having shadows, rails etc as cars) and false negatives like any other machine learning algorithm trained on limited data. This can be solved by using more training data and using better classifiers. However, the major problem with my pipeline is the proessing time which is around 2 seconds per frame. This is a very big issue since the application involoved requires real time detection. Although some performance gain can be made by further optimizing the code, it still is not possible to obtain real time detection using traditional computer vision techniques. Hence, a more robust method would be to use deep neural network methods such as YOLO or SSD that work near real time and would be ideal for our application.

---