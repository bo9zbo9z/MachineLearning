### Kaggle Competition Salt Detection

The notebooks obtain accuracy around 80% and should give you an idea of what it takes to train this type of model.  I did not include submitting the testing solution file.  If you want to create it, should be ease to modify code.

My prior notebooks obtained slightly lower results, but took MUCH longer to train and evaluate.  The changes I made from my old notebooks were:

- Migrate to TensorFlow 2.x and use Datasets for pipeline.
- Swapped in some great work by Yauhen Babakhin for loss and accuracy.  (https://github.com/ybabakhin/kaggle_salt_bes_phalanx)  He won the Kaggle Salt Detection competition and shared his code and approach.
- Used tf.image.adjust_gamma with gamma based on over-all image mean.  In the prior notebook, I used blur and some other methods, but to use them would required tf.py_function.  One goal I had was to use native TensorFlow methods and not custom using tf.py_function.  Found that gamma worked pretty well.

There are two notebooks, one trains from scratch and the other one trains from a pre-trained network based on resnet.  

The resnet network is from Pavel Yakubovshiy, (https://github.com/qubvel/segmentation_models)  I had to migrate pieces of his code to TF 2.x along with some other small changes.  Instead of including his code and my changes, I have included the model weights that can be used for training other than this competition.  The model is named segmodel-224-c1-V01.h5.  You need to compile this model to use it.  The pre-trained model reduced training time and gave improved accuracy.  Because the model is (224, 224) there was some loss of accuracy when converted to (101, 101).

The accuracy and specific details are contained in each notebook.  The training parameters and pipeline were kept the same between notebooks.

I used Google Colab for all of this work.
