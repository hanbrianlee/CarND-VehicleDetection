**Vehicle Detection Project**

**By Brian 'HAN UL' LEE**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_iamges/carsnoncars.png
[image2]: ./output_images/hogexplore.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/pipeline.jpg
[video1]: ./result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First of all, I read in all the .png images (which are read in to 0~1 range using matplotlib's imread function, RGB order) in the `data` folder using glob module. And then, I used the function `extract_features()` to extract HOG and/or color histogram and/or spatial bins features. The `extract_features()` function (lines 43-95) is defined in the 2nd code cell of the IPython notebook `project.ipynb`.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
![cars explore][image1]

Here is an example using the `YUV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![hog explore][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I haven't messed with cells_per_block as I didn't think the normalization of cells within blocks to seriously impact vehicle detection. For orientations, I thought that it makes sense to have the number a multiple of 4 so that every direction is covered symmetrically. I went with 12 as I wanted more resolution of the direction arrows than the 9 that was used during udacity lectures. Hypothetically, using higher resolution would result in being able to differentiate between cars and car-looking-like objects, but using too high resolution may not aid in vehicle detection, because if images in certain selected patches are pixelated, then cars in that patch may not satisfy the high resolution details that SVC linear fit has fit the data to. As for pixels_per_cell, with orientation 12 chosen, going with 8x8 really put a heavy burden on my computer. The memory usage was going over 2GB and memory swapping was occurring and the feature extraction kept failing. I thought that even with 16x16, which will have significantlly less features extracted per training data, the dominant gradients the pixels will vote for can still characterize the overall gradient within the cell well. However, if I had a stronger platform with more memory and processing power, I would like to try 8x8 and see how much better my results get.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After reading in all the cars first and then appending noncars data, I creating an array `y` of 1's and 0's according to the indexes of the array `X` that contains both cars and noncars. Hence I had my training data X and y prepared. Then, using `sklearn.cross_validation import train_test_split` module, I randomly selected training data and test data (80% training 20% test). 

As for color features, I used 64 bins for histogram which I thought was sufficient. 32 seemed to small to capture 256 different intensities to differentiate between many different images. I also used spatial bin size 32x32 to help aid in identifying very obvious matches that HOG and histogram could miss on. However, i would say overall, spatial bin is probably the most useless feature of all 3 used, since it is so scale-dependent, viewing-angle dependent, and window-location-dependent.

I trained a linear SVM using `LinearSVC()` `from sklearn.svm import LinearSVC`. I opted to not play around with its parameters, as I thought my data has enough features (4368 per image) that overfitting would not likely occur. Accuracy on test set was 98.73% which was good enough for me as I would use filtering algorithm to filter out false positives later in my code.

### Sliding Window Search / Subsampling

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At first, I tried using udacity lecture's given `sliding_windows` and `search_windows` function. However, I quickly found out that using this method, the HOG feature extraction was taking almost an hour with my surface pro 1 (generation 1, 4GB RAM, Intel i5-3317U CPU 1.7GHZ). Hence, I had to transition to subsampling method. Udacity's given code did not have if clauses to properly turning on/off different features, so I implemented that. And also there were some errors regarding array sizes and formats, so I had to correct them using reshape(1,-1) method. 

I implemented my own function called `find_cars()` which takes HOG features once and subsamples within the result for different search windows. Color histograms and spatial bins, if used, were processed in "sliding windows" fashion. 

My strategy of choosing scales was based on the assumption that this vehicle detection algorithm were to work on highways where the opposing traffic is seggraged by barrier in the center lane. With this assumption, the left side of each frame could mostly be ignored. This was also kind of necessary due to the limitaiton of my computer's computing power, as adding smaller scales all across the frame significantly added to the processing time. So the strategy was to use multiple scales (1.0, 1.5, 2.0, 3.0) spread out in multiple overlapping/non-overlapping regions in the frame. For example, in the lower half of the frame, we know that the cars will appear bigger than the cars way ahead closer to the y=400 pixels region. Hence, more scale 3.0 windows were utilized in such region. As for areas where cars will be smaller, more scale 1.0 or 1.5 windows were concentrated in the appropriate regions. A total of 143 various windows were used. Also, instead of using overlapping factor, i used `cells_per_step` method to dictate how much of overlaps the windows will have in both x and y directions. `cells_per_step = 2` would be equivalent to 50% overlap and `cells_per_step=1.5` would be equivalent to 75% overlap. I have also put in extra effort to ensure that the windows which lie close to the right-edge of the frame do lie as close to the right limit as possible so that newly emerging cars can be captured better. Also, I adjusted the `y_start_stop` parameter so that cars near y=400 can be properly captured by the classifier prediction. The windows I used are summarized in the image below:

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![windows][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried a multitude of HOG parameter adjustments as stated above in HOG section 2. But one other thing that affected the performance of my classifier was the color space the iamges were converted to. I have tried at first using RGB, but its performance was quite slow. I transitioned to using YUV and YCrCb and tried to see the difference in performance between them. I personally have not seen great difference between YUV and YCrCB in terms of accuracy, however, YUV was the fastest for me. I opted to use all 3-channels, as each channel would differently characterize different colors and I wanted to capture the most gradients as possible. This however took a toll on the speed of my pipeline. The total number of features used were 4368. I thought of applying the kind of pre-processing applied in the Advanced Lane Finding project to both the training data and video frames, before feeding the images to HOG feature extraction part, but I did not have enough time to experiment. In theory, this pre-processing would help significantly my classifier to avoid predicting certain shadowed image patches as cars.

After windows were detected, heatmap and label were utilized to combine windows.

Here are some resulting example images from my pipeline:

![pipeline][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The issue I ran into was frames including lots of shadows were being misjudged for having more cars in the frame than there really were. In order to filter out these false-positives, I had to employ a moving window of 20 frames of data (rotated using pop and append method) which were then used for thresholding. This way, even if some consecutive frames each contained 2~3 falsely identified rectangles in the same part of the images, if I set the threshold to 10, these will not be included in the final output. `rolling_heatmap_size` and `rolling_heatmap_threshold` in the first code cell are parameters that can be tweaked to adjust this false-positive filterig moving window scheme. 


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried using CLAHE equalization on training data images, thinking that some of the images which appear very dark would not do much to aid in developing a good SVM classifier. Of course, in order to generalize for darker hours of the day, these darker images would be necessary, but I thought the frames from darker hours of the day could also go through the CLAHE pre-processing. But for some reason, when I tried CLAHE, almost every window in my search regions would be classified as "containing car". I would have to do further research on why this happened, but it could be because CLAHE put the colors off in a way that when the stream jpg frames come in from real video, the colors do not match and mess up SVC classifier's prediction. This, however, could just be because I mis-aligned my ranges (0-1 vs 0-255) for images. I will look further into this later.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

