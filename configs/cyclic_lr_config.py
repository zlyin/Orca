## import packages
import os


## initialize the list of class label names
CLASSNAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
	"frog", "horse", "ship", "truck"]
OUTPUT = "./tutorial_outputs/cyclic_lr_decay/"


## define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
#CLR_METHODS = ["triangular", "triangular2", "exp_range"]
CLR_METHODS = ["exp_range"]
NUM_EPOCHS = 96 

