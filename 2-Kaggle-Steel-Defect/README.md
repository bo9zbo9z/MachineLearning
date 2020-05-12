### Kaggle Competition Steel Defects

This was a fun competition.  When you looked at the classes it felt like that this was going to be a multi-model approach.  One model to classify an image as having a defect or not, and another model that would handle the segmentation. (location and type of defect)

So, there are 3 main notebooks.

- Classification
- Segmentation
- Merge models for a final result

I also played with using sparse labeling, it was a fun exercise, but I did not end up using it.

The resnet network is from Pavel Yakubovshiy, (https://github.com/qubvel/segmentation_models)  I had to migrate pieces of code to TF 2.x along with some other small changes.  Instead of including his code and my changes, I have included the model weights that can be used for training other than this competition.  They are all labeled “segmodel.....” with the size and number of classes.  For example, segmodel-256-800-c4 has size 256x800 with 4 classes.

I used Google Colab for all of this work.
