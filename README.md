# Dog-Breed-Classification
# I. Definition

### Sections
- Project Overview
- Problem Statement
- Metrics
- Data Exploration
- Data Visualization
- Data Preprocessing
- Implementation
- Refinement
- Model Evaluation and Validation
- Justification
- Reflection
- Improvement

### Project Overview

In this project we will solve the problem of classifying dog breeds, it is an image classification problem and we use the convolutional network architecture to solve this problem. O dataset contém informações sobre 133 classes de cachorros, e é nos provido pela Udacity.

---
### Problem Statement

Given an image, we need to know a priori if it contains a dog, in this case, we will have to classify the dog's race among the 133 possible classes, we will use the convolutional network architecture, widely used for image classification problems, and therefore it fits perfectly in the problem we are facing.

### Metrics
 
As we are facing a multiclassification problem, there are some possible metrics to use as accuracy, precision, recall, F1-score, ROC, AUC ...

We are going to use:
- Accuracy: In multilabel classification, this method computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

- Precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. 

We are going to use accuracy because it is the simplest metric and gives an overview of how the model is performing.

In cases with unbalanced data, precision is a better metric than accuracy. And in our case, there are classes that contain a higher frequency than other classes, so precision will be a good metric here.

# II. Analysis
### Data Exploration

First let's explore our data, seeing our number of classes, total images and total data in each subset of training, validation and testing.
![image](https://user-images.githubusercontent.com/71555983/148663842-534a7ac6-51ba-4b40-aca4-8d9d2bd92cdc.png)
![image](https://user-images.githubusercontent.com/71555983/148663856-d3ad4d3f-0e9b-44d6-aa63-f65a31f7aa83.png)

### Data Visualization

In this part we'll take a look at our data to better understand what kind of image we're dealing with.
![image](https://user-images.githubusercontent.com/71555983/148663865-7a924201-aa78-4be0-aced-7b35935dc93f.png)

# III. Methodology
### Data Preprocessing



When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!

We rescale the images by dividing every pixel in every image by 255.

![image](https://user-images.githubusercontent.com/71555983/148663905-ff556fe7-e744-44c7-b8eb-1ab99eddb819.png)

### Implementation and Refinement
### Algorithms and Techniques
#### CNN
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

#### Transfer Learning
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

In this step we will implement some models, some being from 0 and others using transfer learning and combining an own architecture together with transfer learning.

![image](https://user-images.githubusercontent.com/71555983/148664004-088369b7-d153-4b82-a1b9-04571f5ec8d1.png)


### Refinement

Now that we've implemented a network from scratch and are using transferning, let's use the two techniques together to achieve even better results.
![image](https://user-images.githubusercontent.com/71555983/148663977-dc1e921f-94ad-4c8e-b4c0-ac4ef7b2f052.png)

# IV. Results
### Model Evaluation and Validation
Now let's use our test dataset to see how well we did, and we'll also use screenshots from my local computer to see how the network actually performs on any data used.

![image](https://user-images.githubusercontent.com/71555983/148664090-22892e3a-c609-4e1e-8a7c-8901d0e28a98.png)

Of the 6 images he only missed one, the model did better than expected since in the tests it only resulted in 70% accuracy.

![image](https://user-images.githubusercontent.com/71555983/148664117-2c29c441-78d7-44cb-b6dd-8aa5cfbba46e.png)


### Justification
Compared to the model without transfer learning, the model with transferlearning and some dense layers achieved surprising results.
While with the transfer learning only model, we had achieved an extremely low result of 49%
![image](https://user-images.githubusercontent.com/71555983/148664241-6a61e1d0-ff3b-4090-854d-a3aea822b16f.png)
Adding some dense layers, we had an increase to 72%
![Uploading image.png…]()

# V. Conclusion
### Reflection
First we reading the data, is a very simple task, since the exploration and visualization varies a lot based on the data, with images it is a simpler task sometimes.

However, the preprocessing part is a little more complex, there are several transformations that can be done on an image so that the model understands it well.

The implementation of the model is something that can be tiring since we need to test different architectures, like from scratch, with transferlearning, or a mixture of these.

The evaluation part of the model already becomes something simpler once we had already defined which metrics we would use and why.

It is interesting to see that due to the large amount of classes and little data, the model does not perform well during training, but thanks to transfer learning, even with a small amount of data, we were able to achieve reasonable metrics.

It is also very difficult to decide how many layers to choose, if we choose many layers, the accuracy increases, but the model becomes heavier and slower, perhaps this varies with the problem.

### Improvement
1-More data
Provide a larger dataset with more variant images of dogs, such as dressed dogs

2-Data augmentation
Use data augmentation techniques to make the model more accurate

3- Deeper architecture
We use a very simple network, maybe one with more Dense layers could help more to detect the patterns of each class


# VI. Web App Running
To run de web Application you need create a env and install the following dependencies:

Step - 1

open terminal and run the command:
- python -m venv env

Step - 2

Inside the env, open terminal and run de command:
- git clone https://github.com/ChrystianGreen2/Dog-Breed-Classification.git

Step - 3

Inside the env, run de command:
- env/Scripts/activate.ps1 (windows)
- source venv/bin/activate (linux)

Step - 4

install the dependencies:
- pip install streamlit
- pip install opencv-python
- pip install keras==2.0.3
- pip install tensorflow

Step - 5

Run the command:
- streamlit run app.py

Files
-----
~~~~~~~~~~~~
- dog_breed_classifier_notebook.ipynb -> The notebook
- app.py -> the web app

~~~~~~~~~~~~
