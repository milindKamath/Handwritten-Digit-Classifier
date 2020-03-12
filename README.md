# HandwrittenDigits

## Requirments
The required libraries are included in the requirements.txt. 

## Pretrained

To run the pretrained model and precomputed testing vectors run the command:

[[Python]] digitClassifier.py 3 [training file] [path to training data] [ground truth file] 
              [testing file] [training junk file] [path to junk data] [ground truth file] [testing junk file]

To run the pretrained mode with new testing vectors run the command:

[[Python]] digitClassifier.py 2 [training file] [path to training data] [ground truth file] 
              [testing file] [training junk file] [path to junk data] [ground truth file] [testing junk file]

## Train a new model

To train a new model run the command:

[[Python]] digitClassifier.py 1 [training file] [path to training data] [ground truth file]
              [testing file] [training junk file] [path to junk data] [ground truth file] [testing junk file]