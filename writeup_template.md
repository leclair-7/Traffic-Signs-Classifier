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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale because the extra color channels being processed add little (if any) to classification performance while adding significant runtime to training.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Normalize the data so the mean is zero. This allows the model to converge sooner. Also most machine learning models work better with data values with absolute values between 0 and 1.

We shuffled the data because the order in which the data samples are presented to the algorithm influences the models training. With that in mind we shuffle the datasets prior to training to eliminate the bias that the order in which the dataset is presented may have. For example (very simplified), if the first few hundred data points are all the points of a certain class then the model would have an initial bias toward that class. Then each backprobagation run would move the neuron weights from the values away from that class.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an tensorflow supplied Adam Optimizer function which is a widely adopted alternative to stochastic gradient descent. The learning rate was 0.0005. It was changed in the end of testing from .001. As a result the number of epochs was increased from 10 to 20 because the loss was still decreasing with 10 epochs. Dropout was used at various points as described in the architecture, was experimentally found to be .5 (.4 and .7 were not as accurate on the test set). The batch size was decided to be 256. It seemed as though a higher learning rate would merely reduce the training time however, there was a possibility of the model getting in a local minimum at various places with the higher batch size.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.5%
* test set accuracy of 95.4%
If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
The LeNet architecture was chosen to begin with because it has proven to be robust for image classification tasks. Since we are classifying different images of similar size it was decided that that is a good starting architecture.

 In this classification task, after changing the last fully connected layer to output 43 classes, it classified the test set with accuracy = 88%. The initial architecure seemed to not be complex enough for this classification task (as opposed to too many parameters which would overfit). Various adjustments were made to increase accuracy. I started with adding dropout with a probability of keeping the parameter = .7 before the ReLU layers. Then I added another fully connected layer which also increased accuracy. I noticed that at later epochs, the loss would oscillate and the model accuracy was increasing meaning the model was overfitting. I then decreased the dropout to .5. The accuracy and loss seemed to jump at later epochs which led me to test changing the learning rate. This was changed to .0005 and I increased the number of epochs to capture the point at which the loss on the validation set ceases to decrease. Lastly out of curiosity, I modified the convolutional layers to have a higher depths (64 then 128 on the first two convolutional layers respectively). 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


