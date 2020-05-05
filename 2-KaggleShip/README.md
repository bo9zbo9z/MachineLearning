### Kaggle Competition Ship Identification

The notebooks are easy to understand and should give you an idea of what it takes to train a model using TensorFlow 2.x.  I included submitting the testing solution file.

The changes I made from my old notebooks were:

- Migrate to TensorFlow 2.x and use Datasets for pipeline.
- Used native tf.image methods
- Included any of my custom library code so notebook is a bit cluttered.  (Using your own libraries with Kaggle environment is a bit of a pain.)

There are two notebooks.  One builds the model and the other one creates the submission file.  Kaggle limits your run-time and file sizes, so I've found separating into multiple notebooks is easier.  You should be able to run these notebooks on Kaggle or in your own environment with very minor changes.

The accuracy was around 75%, so good results, but needs much more tuning for improvement.

Because of the size of the training data, I used the Kaggle environment for this work.

To obtain top results you need to train using larger files, and, honestly, I do not have the hardware or time.
