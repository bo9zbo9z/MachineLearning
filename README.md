### Overview

This repository contains notebooks and libraries that I have used for understanding basic machine learning concepts and for Kaggle competitions.  When I have seen examples, they are usually created by different people, so trying to understand the code can be a challenge.  These notebooks follow a common flow and reuse concepts & methods so the hope is that if you go through one, going through the others will be easier.  I decided to move my code to github to give back to the community.

This is not an over-all series of AI notebooks, but mainly around models to categorize images.  The code is good for exploration and research, it is not intended for any real-life or production usage.  It is based on TensorFlow 2.x, Python 3.x and the normal imports: numpy, pandas, mathplot, CV2, sklearn, etc.  The main environment used is Google Colab, but I also use Kaggle's environment and my own laptop running Atom, Jupyter in a TensorFlow 2.x environment.  

I also use TensorFlow Datasets instead of Keras Generators or custom coding loading images.  TF Datasets uses graphs and I strived to use standard libraries instead of custom coding or using other image augmentation libraries.  There is a notebook in the "0-EducationAndEnvironment" subdirectory just devoted to playing with & understanding how augmentations work with TF 2.x.

 I am not a pure Python developer, so I apologize for those who will read my code and say “What the heck is he doing...”  Some of my methods are too long, I had my reasons, and I only have one class which I use to wrap global parameters.  The prior versions of my code had several classes but when I migrated, I took a hard look at why I used classes and if they actually helped understand the concepts or were they adding complexity.  Since this is for learning AI concepts, I simplified and removed the classes.  There are different types of problems to solve, so I focused on the core pieces and not the abstractions into classes.  (Focusing on the core concepts actually made the code simpler.)

This will not teach you how to setup an environment, to learn TensorFlow or to learn Python, there are many articles that cover those topics.  The code does have many comments, explanations and README's, but they are also not designed as blog entries or completed articles.  

The code was mainly developed on Google Colab, but should work in different environments, like AWS or MS, with modifications to the file paths.  The Kaggle examples can run for a long time on Google & Kaggle environments and consume memory & disk, so you might get warnings.

The subdirectories are prefixed to help with the type and level of example.  Most subdirectories contain a README file that has more detail and background on the notebooks.  The code has basic comments on purpose and structure.  Here is a list of the prefixes:

- **“0-<>”** are beginning/entry level
- **“1-<>”** are the next level up, but still basic
- **“2-<>”** are all of the Kaggle competitions
- **“9-<>”** are the libraries.

I'd recommend to start with all of the notebooks in the “0” subdirectories.  Have fun!!
