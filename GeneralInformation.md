### General Information

As you look at the notebooks I wanted to give you an idea of how they are structured and the general approach & information for each section/topic.  The notebooks follow the same basic structure and the approach I use is similar to most machine learning workflows.  Some notebooks will have more or less details depending on the type of solution.  The number of notebooks used will vary from 1 to multiple.  For example, classifying Cats and Dogs is a single notebook, classifying & locating Steel Defects (Kaggle) has three.  

These are the broad topics that are woven into the notebooks:

- Understand the problem and conduct research
- Examine and understand data
- Build an input pipeline
- Build  model and select optimizer, last-layer activation & loss function
- Train the model
- Validate the model and repeat any steps
- For Kaggle, submit results and repeat any steps


### Understand the problem and conduct research
 These are some things to think about as you start understanding a concept or new AI problem.  Sometimes I put this information at the top of the notebook, it helps me get a handle on the type of problem and challenges.  I use the Internet to increase my understanding and to do research.  (articles, discussion groups, Stack Overflow, etc.)  Research is a must before you jump in and start coding.  

 Some of the key questions you need to think about are:

**What new things will you have to learn?**
- If this is something you have done before, then you can usually jump right in.  If there are new things then you need to allow time for learning. For example, if the model is going to use a U-Net and you have never trained one before, then you need to spend time just learning what it is and the pros/cons of using that type of model.
- Besides models, this could cover different libraries or Kaggle specific requirements.

**Is this a multi-class or multi-label classification?**
- In Multi-Class classification there are more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Each sample is assigned to one and only one label: a fruit can be either an apple or an orange.

- In Multi-Label classification, each sample has a set of target labels. A comment might be threats, obscenity, insults, and identity-based hate at the same time or none of these.

**What environment(s) will you use?**
- For something simple, like Cats and Dogs, you could be able to use your laptop.
- For something more complex, like a Kaggle competition, you will need to use Kaggle's environment and maybe another environment like Google Collab.
- Data - number of images and size of each image.
- Within the environments, there could be library version differences.  Not a great comfort, sad but true, Amazon, Microsoft, Google and Kaggle do not keep their library versions in-sync...

### Examine and understand data
Datasets you get from public sources are usually well documented and have a common structure.  These usually have been used for many research efforts so they can basically be dropped in and used.  For Kaggle, you have to analyze the images and then modify your pipeline to accommodate for training.  I have a notebook in the **”0-EducationAndEnvironment”** subdirectory that I use for Kaggle competitions but can be used for any image problem.  (size, class coverage, color or greyscale, etc)  In the pipeline you usually need to resize the images so now is a good time to understand what happens if you need to enlarge, shrink or crop them.

https://datasetsearch.research.google.com/
https://dimensionless.in/deep-learning-data-sets-for-every-data-scientist/
http://moments.csail.mit.edu/

### Build an input pipeline
This is all of the work you need to do to prepare the images to be used for training.  Sometimes you can do any resizing & augmentation as part of the training pipeline.  Or you might need to do some resizing & augmentation on the images and store the results that then are used for training.

This is usually where you also take care of unbalanced classes and decide how to split your data.  You might need to create new images from the training set or find other public sources.  You might need just a training & validation split, or you might need training, validation and test split.  How your data is split and how that impacts the class balance is something to think about now before you start your training runs...

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator https://keras.io/preprocessing/image/#flow_from_dataframe

**Resizing**

There is a notebook in the **”0-EducationAndEnvironment”** subdirectory that you can use to experiment with different resizing approaches.  There are many opinions on how to resize.  I start with using the tf.image.resize default for initial training.  

But here are some things to think about:

interpolation => "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos." The list of filters is ordered from lowest to highest quality, but the higher quality options tend to be a bit slower.

BICUBIC is probably the best option for general use (the default is NEAREST, but a modern computer can run BICUBIC pretty quickly and the quality improvement is quite noticeable).


### Build model and select optimizer, last-layer activation & loss function
 **Build model**

The key decision is if you are going to train a model from scratch or use a pre-trained model for your base model.  My suggestion is that whenever possible, use a pre-trained model.  It is very satisfying to train a model, but time and effort are easier if you start with a base model and then add your specific layers.

https://www.tensorflow.org/api_docs/python/tf/keras/applications

This link for segmentation example:
https://github.com/mrgloom/awesome-semantic-segmentation

**Select Optimizer**

Basic info:

https://keras.io/optimizers/
https://www.kaggle.com/residentmario/keras-optimizers
http://ruder.io/optimizing-gradient-descent/ https://towardsdatascience.com/7-practical-deep-learning-tips-97a9f514100e

**Last-layer Activation and Loss Function**

https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

Multi-class classification use softmax activation function in the output layer. The probability of each class is dependent on the other classes. As the probability of one class increases, the probability of the other class decreases.

The softmax activation function is not appropriate in Multi-label classification because it has more than one label for a single text. The probabilities are independent of each other. Here we use the sigmoid activation function. This will predict the probability for each class independently.

| Problem Type | Last-Layer Activation | Loss Function | Example |
| --- | -- | ---- | ---|
| Binary classification | sigmoid | binary_crossentropy | <ul><li>Dog vs cat</li></ul> |
| Multi-class, single-label classification | softmax | categorical_crossentropy | <ul><li>MNIST has 10 classes single label (one prediction is one digit)</li><li>Dog breeds</li><li>Kaggle Proteins</li></ul>|
| Single-class, Multi-label classification | softmax | binary_crossentropy | <ul><li>U-Net classifications (Kaggle Salt Location, Ship Location, Steel Defects Location)</li><li>Kaggle Diabetic Retinopathy Detection</li></ul>|

This table is very helpful as you frame the type of problem and desired outcomes.


### Train the model
**Hyperparemeters**

https://en.wikipedia.org/wiki/Hyperparameter_optimization
https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/

**Clallbacks**

https://medium.com/singlestone/keras-callbacks-monitor-and-improve-your-deep-learning-205a8a27e91c
https://keras.io/callbacks/

### Validate the model and repeat any steps

This is where you can spend 50% of your time to analyze results and loop back to retrain.  Sometimes this is trail and error, other times you can see where the model improved which leads you to focus more on certain areas.  Also, this is where understanding what has been done in other training efforts or something similar can be very helpful.


### For Kaggle, submit results and repeat any steps

Similar to validation, but focused on the results you get back from Kaggle.  The reason I separated this from general validation is that sometimes the competition has testing data that does not exactly mirror training data.  Which means you might have to analyze Test images and compare against Training images.  Sometimes you might also want to pull some Test images forward into the Training pipeline.  

Usually, Kaggle keeps a secret Test set of images that are only used when you commit your code.  This keeps actual images for the final score hidden.  So there can be labeled public Training images, unlabeled public Testing images and unknown final Testing images mainly used for final score.
