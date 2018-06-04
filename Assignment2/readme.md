# Assignment2

    Experiment 1 :  Classification Performance Comparison Between Neural Network & Other Machine Learning Algorithm
    Experiment 2 :  Regression Performance Comparison Between Neural Network & Other Machine Learning Algorithm
   
## 1. Problem Setup
1. Classification
    - Neural Network 
    - Neural Network + Dropout
    - K-NN Classifier
    - Dicision Tree

2. Regression
    - Neural Network
    - Neural Network + L2 Regularization
    - Linear Regression
    - Linear Regression with L1 Regulalization
    - Linear Regression with L2 Regulalization 
    - K-NN Regression


## Use Keras Custum R2-Metric [[1]](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019)
```python
# custom R2-score metrics for keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
```



## 2. Experiment Result 
## 2.1 Experiment 1 :  Classification Performance
|Method                 |Train-Accuracy|Test-Accuracy|Duration(s)|
|:----------------------|:------------:|:-----------:|:---------:|
|Neural Network         |0.9695        |0.7756       |10.2058    |
|Neural Network + Dropout |0.9707      |0.8535       |7.9830     |
|Decision Tree          |0.968         |0.687       |30.8302     |
|KNN-Classifier         |too slow      |too slow   |too slow     |

## 2.1 Experiment 2 :  Regression Performance
|Method                 |Train-R2-Score|Test-R2-Score|Duration(s)|
|:----------------------|:------------:|:-----------:|:---------:|
|Neural Network         |0.9316        |0.8067       |2.8083     |
|Neural Network + L2    |0.9707        |0.8535       |7.9830     |
|Linear Regression(LR)  |0.7400        |0.7214       |0.0022     |
|LR + L1 (Alpha = 0.05) |0.7369        |0.7291       |0.0052     |
|LR + L1 (Alpha = 0.5)  |0.7399        |0.7223       |-          |
|LR + L1 (Alpha = 1)    |0.7387        |0.7270       |-          |
|LR + L1 (Alpha = 10)   |0.7268        |0.7357       |-          |
|LR + L1 (Alpha = 100)  |0.7087        |0.7259       |-          |
|LR + L2 (Alpha = 0.0001)|0.7400       |0.7214       |0.0079     |
|LR + L2 (Alpha = 0.01) |0.7395        |0.7262       |0.0025     |
|LR + L2 (Alpha = 1)    |0.6717        |0.6898       |0.0020     |
|KNN Regressor (k = 2)  |0.8464        |0.4323       |0.0007     |
|KNN Regressor (k = 4)  |0.7544        |0.5270       |0.0005     |
|KNN Regressor (k = 5)  |0.7063        |0.5559       |0.0028     |
|KNN Regressor (k = 7)  |0.6473        |0.5349       |0.0011     |


References  
[1] https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019