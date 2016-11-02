# The Awesome Dataset
Include text from abstract that describes the dataset
The paper can be found here: link

## Example images:
![Example images](figures/examples.png?raw=true "Example Images")
## Download the Dataset and Setup the Evaluation
You can download the dataset with
python download_dataset.py

Edit `settings.py` and specify the path to your caffe installation,
your cuda path and the gpu to use.

Required libraries for the installation: caffe, cuda, boost, python3, numpy
The evaluation is partly written in C++ and can be built with 
'python install.py'
This also downloads the bvlc_reference model we use for finetuning. Make sure you are 
in the src directory, as all paths are relative from there.

## Train
Then you can start training with <br>
`python train.py` <br>
This trains the 5 splits and evaluates them on the corresponding test set. This includes
computing the accuracy for each class and producing a confusion matrix.
It also links the misclassified images for each class and names them to contain
the class they were confused with.

## Baseline Results


## Examples for Misclassified Packages