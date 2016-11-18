# The Freiburg Groceries Dataset
The Freiburg Groceries Dataset consists of 5000 256x256 RGB images of 25 food classes. Examples for each class can be found below. The paper can be found [here](http://ais.informatik.uni-freiburg.de/publications/papers/jund16groceries.pdf) and the dataset [here](http://www2.informatik.uni-freiburg.de/~eitel/freiburg_groceries_dataset.html).

## Example images:
![Example images](figures/examples.png?raw=true "Example Images")
## Download the Dataset and Setup the Evaluation
First, clone the repository and navigate to the src directory: <br>
`[user@machine folder] git clone https://github.com/PhilJd/freiburg_groceries_dataset.git` <br>
`[user@machine folder] cd freiburg_groceries_dataset/src` <br>

You can download the dataset with python3: <br>
`[user@machine src] python download_dataset.py`

Then, edit `settings.py` and specify the path to your caffe installation,
the path to your cuda installation and the gpu that should be used for training.

To install the evluation software the following libraries are required: caffe, cuda, boost, python3, numpy.
The evaluation software is partly written in C++. To clone the repo and build the evluation run <br>
`[user@machine src] python install.py`<br>
This also downloads the bvlc_reference model we use for finetuning if necessary. Make sure you are 
in the src directory, as all paths are relative from there.

## Train
You can start training with <br>
`[user@machine folder] python train.py` <br>
This creates the lmdbs, trains the 5 splits and evaluates them on the corresponding test set. This includes
computing the accuracy for each class and producing a confusion matrix.
It also links the misclassified images for each class and names them to contain
the class they were confused with.

## Some Examples for all Classes
![Class overview images](figures/class_overview.png?raw=true "Class Overview Images")