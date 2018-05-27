# Room classifier - Feed Foward Neural Network
This code implements a simple feed forward neural network with two hidden layers created using Microsoft CNTK API. It is used to classify the list of objects to one of five categories (room types): Bathroom, Bedroom, Kitchen, Living Room,  Office.

### Input:
- features _(Data/object_result.dat)_
- labels _(Data/types_result.dat)_

_**features**_ is a matrix of size  _NxM_ where N is a number of samples and M is the number of features (objects, i.e. _lamp_, _table_) whereas _**labels**_ is a vector of size _Nx1_ and contains _labels_ or numbers representing particular room type.

### Output:
- K classification models to be used in the web application (each model is created for each fold)
- Confusion matrix for K-fold cross valuation

### Dependencies
- matplotlib 1.5.3
- seaborn 0.8.1
- numpy 1.14
- cntk 2.3.0
- easydict 1.6
- Pillow 4.3.0
- utils 0.9.0
- PyYAML 3.12


### Use
1. To allow the program to execute you might be asked to set the environment variable _KMP_DUPLICATE_LIB_OK_ to _TRUE_ by:
 `export KMP_DUPLICATE_LIB_OK=TRUE`
3. Run the script with  `python3 FFNN_classificator.py`
