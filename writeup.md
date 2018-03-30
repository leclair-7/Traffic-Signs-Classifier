# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./datapoint_cardinality.jpg "Dataset Class Representation"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./GermanTrafficSigns/placeholder.png "01speedlimit30"
[image5]: ./GermanTrafficSigns/placeholder.png "14Stop sign"
[image6]: ./GermanTrafficSigns/placeholder.png "25Roadwork"
[image7]: ./GermanTrafficSigns/placeholder.png "28childrenCrossing"
[image8]: ./GermanTrafficSigns/placeholder.png "31animalCrossing"

---
### Writeup / README

Here is a link to my [project code](https://github.com/leclair-7/Traffic-Signs-Classifier/blob/master/Traffic-Signs-Classifier.ipynb)

### Data Set Summary & Exploration

#### Summary of the Data Set

The dataset contains more than 50,000 images of traffic signs each of which are in 1 of 43 classes. Each sign instance is unique within the dataset.

I used python built-ins to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410 
* The size of test set is 12630 
* Each datapoint, a traffic sign image, has shape = (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory Visualization

Here is an exploratory visualization of the data set. It is a scatter plot that shows how the dataset is unbalanced. Many of the classification labels have less than 500 examples in the training data.

![alt text][image0]


### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step (box 4), I shuffled the training data. The data was shuffled because the order in which the data samples are presented to the algorithm while training influences the models training. With that in mind I shuffled the datasets prior to training to eliminate bias from the order of data rows (images). Shuffling is done in the preprocessing section for elaboration purposes. It is done before each epoch during training.

Secondly, I converted the images to grayscale (box 5) because the extra color channels being processed added little (if any) accuracy to classification performance on the test set while adding significant runtime to training. In box 6, I normalized the data so the mean is zero. This makes each feature contribute the same amount to classification performance/training. We don't want classification to have a bias towards a certain part of the image when classifying.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, Valid Padding, outputs 28x28x64 	|
| Max pooling			| 2x2 stride, outputs 14x14x64					|
| ReLU  				| 												|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x128	|
| Max pooling			| 2x2 stride,  outputs 5x5x128					|
| ReLU	      			| 												|
| Flatten				| flatten to a 1D vector 3200 outputs			|
| Fully connected		| 400 outputs 									|
| Dropout      			| keep probability = .5							|
| ReLU	      			| 												|
| Fully connected		| 84 outputs 									|
| Dropout      			| keep probability = .5							|
| ReLU	      			| 												|
| Fully connected		| 43 outputs 									|


#### 3. Model Training Explanation

To train the model, I used the tensorflow supplied Adam Optimizer function which is a widely adopted alternative to stochastic gradient descent. The learning rate was 0.0005. It was changed in the end of testing from .001. As a result the number of epochs was increased from 10 to 20 because at learning rate = .001, the loss was still decreasing with 10 epochs. Dropout was used on the fully connected layers. It was experimentally found to be .5 (.4 and .7 were not as accurate on the test set). The batch size was decided to be 256. It seemed as though a higher learning rate would merely reduce the training time however, there seems to be a possibility that the model may get in a local minimum at various places with the higher batch size (above 512).

#### 4. Approach Taken to Making the Final Model

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.5%
* test set accuracy of 95.4%

An iterative approach was chosen beginning with the LeNet Architecture:

The LeNet architecture was chosen to begin with because it has proven to do well on image classification tasks. Since we are classifying images it was seen to be a good starting architecture.

 In this classification task, after changing the last fully connected layer to output 43 classes, it classified the test set with accuracy = 88%. The initial architecture seemed to not be complex enough for this classification task (as opposed to too many parameters which would overfit). Various adjustments were made to increase accuracy mostly by trial and error. I started with adding dropout with a probability of keeping the parameter = .7 before the ReLU layers. Then I added another fully connected layer which also increased accuracy. I noticed that at later epochs, the loss would oscillate and the model accuracy was increasing meaning the model was overfitting. I then decreased the dropout to .5. The accuracy and loss seemed to jump at later epochs which led me to test changing the learning rate. This was changed to .0005 and I increased the number of epochs to capture the point at which the loss on the validation set ceases to decrease. Lastly out of curiosity, I modified the convolutional layers to have a higher depths (64 then 128 on the first two convolutional layers respectively). The performance on the test set improved which led me to using it. 


### Test a Model on New Images

#### 1. Testing the Model on 5 German Traffic Signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30   		| Speed Limit 30 								| 
| Stop Sign    			| Stop Sign										|
| Road work				| Road Work										|
| Children Crossing 	| General Caution					 			|
| Wild Animals Crossing	| Wild Animals Crossing							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 80%

#### 3. Discussion of the Classification Performance on the 5 New Images

The code for making predictions on my final model is located on cell 32 of the Ipython notebook.

Image #1

For the first image, the model is certain that this is a Speed limit (30km/h) which is correct. I was very surprised by this result because there is background noise in the image of different colors and a vertical white line behind the sigh. It may be the lower level features of the circle (of the sign) and the circle in the 0 of '30' being prominently displayed in the image. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)							| 
| .000    				| Speed limit (20km/h)							|
| .000					| Speed limit (50km/h)							|
| .000	      			| Speed limit (70km/h) 							|
| .000	      			| Right-of-way at the next intersection		 	|

Image #2

For the second image, the model is relatively certain that this is a stop sign (probability of 0.92), which is correct. The noise around the sign seems to slightly obfuscate the classifiers confidence that it is a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .929         			| Stop sign   									| 
| .045     				| Turn left ahead 								|
| .008    				| Speed limit (30km/h)							|
| .003					| Speed limit (60km/h)							|
| .003				    | Road Work		     							|

Image #3

For the third image, the model is certain that this is a stop sign (probability of 0.6), and the image does contain a stop sign. It seems to be able to ignore the trees in the background from the original image which means the features of road work sign are strongly represented as weights in the network. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Road Work   									| 
| .000     				| Bumpy road									|
| .000					| Bicycles crossing								|
| .000	      			| Beware of ice/snow			 				|
| .000				    | Dangerous curve to the right 					|

Image #4

For the fourth image, the model is unsure that this is a general caution sign (.397 is less than a coin toss), which is incorrect because it is children crossing. Upon looking at the image after it was scaled down to 32x32, the picture in the sign resembles an upside down exclamation point which has similar features to the General caution sign. This is intriguing and it leads me to think that maybe there should be another convolutional layer because the abstraction of the '!' of a General Caution sign's relation with up and down (horizontal line under the '!') is not encoded in my network. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .397         			| General caution 								| 
| .199     				| Road work 									|
| .107					| Speed limit (70km/h)							|
| .104	      			| Speed limit (30km/h)			 				|
| .072				    | Bicycles crossing								|

Image #5

For the fifth image, the model is certain the sign is Wild Animals Crossing which is correct. This is probably due to the simplicity of the image. The image that the model classified was a large image scaled down to 32x32. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Wild animals crossing							| 
| .000     				| Road Work										|
| .000					| Double curve									|
| .000	      			| Right-of-way at the next intersection			|
| .000				    | Bicycles Crossing								|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

