### Kaggle Competition Identifying  Cell Proteins

When Kaggle hosts a competition, there are usually a few things that you have to figure out.  For this one, size of the files, class balance and small proteins were the main issues.  

I tried to use the different stains to help segment the classes to help with training, did not help.  I also tried to use component analysis to help isolate the proteins, did not help.  A good friend of mine is a microbiologist and I asked him for help classifying the proteins.  (You should always use your experts.)  He gave me some good information about where the proteins are located, but there are not really firm rules about if protein x is present, then you can be sure that protein y is not present.

I did develop a model that was very good identifying a protein that was within the cell nucleus.  But, did not really help solve the over-all problem.

This was the only competition that I did not have a submission.  It was a little frustrating....For example, in order to deal with the small proteins you need as large of files as possible.  They had additional files, but they were over 250G and not practical for the environments I was using.  To score well you needed your own servers.  I did learn a lot as I was researching options, but in the end, I created a training notebook that did ok, but never would have placed.

I do not have my own environment, I rely on Google Colab and Kaggle.  I used Kaggle environment for all of this work.
