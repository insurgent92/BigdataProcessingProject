# Assignment1

    Experiment 1 :  Comparison of Different Depth of Networks
    Experiment 2 :  Comparison of Different Width of Layers
    Experiment 3 :  Comparison of Different Loss Functions
    Experiment 4 :  Comparison of Different Activation Functions


## 1. Problem Setup
1. Default Layer Depth : 3
2. Default Layer Width : 512
3. Default Loss Function : categorical_crossentropy [(More detail)](https://keras.io/losses/#categorical_crossentropy)
4. Default Activation Function : ReLU [(More detail)](https://keras.io/activations/#relu)
5. MNIST Dataset
6. Fixed Optimizer = 'rmsprop' (and Default parameters [(see)](https://keras.io/optimizers/#rmsprop))
7. Fixed epochs = '10'
8. Fixed batch_size = '512'
9. Fixed validation_split = '0.1'
10. \# of validation set = 6000
11. \# of training set = 54000



## 2. Experiment Result 
## 2.1 Experiment 1 : Comparison of Different Depth of Networks

|Models |Depth|Train Duration(s)| test_acc | test_loss |
|:-----:|:---:|:---------------:|:--------:|:---------:|
|Model 1| 1   |21.0873          |0.9807    |0.0630     |
|Model 2| 3   |30.3002          |0.9827    |0.0867     |
|Model 3| 4   |29.8534          |0.9814    |0.0767     |
|Model 4| 5   |35.9210          |0.9759    |0.1064     |
|Model 5| 9   |34.4833          |0.9799    |0.0907     |

## 2.1 Experiment 2 : Comparison of Different Width of Layers

|Models |Width|Train Duration(s)| test_acc | test_loss |
|:-----:|:---:|:---------------:|:--------:|:---------:|
|Model 1| 64  |19.5668          |0.9684    |0.1093     |
|Model 2| 128 |20.2792          |0.9790    |0.0716     |
|Model 3| 256 |16.8353          |0.9775    |0.0933     |
|Model 4| 512 |18.9405          |0.9825    |0.0818     |
|Model 5| 1024|33.9161          |0.9824    |0.0899     |


## 2.1 Experiment 3 : Comparison of Different Loss Functions
|Models  |lossfunc                       |Train Duration(s)| test_acc | test_loss  |
|:------:|:-----------------------------:|:---------------:|:--------:|:----------:|
|Model 1 |mean_squared_error             |27.2979          |0.9774    |0.0036      |
|Model 2 |mean_absolute_error            |23.5080          |0.9744    |0.0052      |
|Model 3 |mean_absolute_percentage_error |24.7757          |0.9707    |2938308.6908|
|Model 4 |mean_squared_logarithmic_error |24.8919          |0.9776    |0.0017      |
|Model 5 |squared_hinge                  |27.1572          |0.9719    |0.9028      |
|Model 6 |hinge                          |49.2027          |0.9730    |0.0535      |
|Model 7 |categorical_hinge              |33.9161          |0.9824    |0.0899      |
|Model 8 |logcosh                        |33.9161          |0.9824    |0.0899      |
|Model 9 |categorical_crossentropy       |33.9161          |0.9824    |0.0899      |
|Model 10|sparse_categorical_crossentropy|33.9161          |0.9824    |0.0899      |
|Model 11|binary_crossentropy            |33.9161          |0.9824    |0.0899      |
|Model 12|kullback_leibler_divergence    |33.9161          |0.9824    |0.0899      |
|Model 13|poisson                        |33.9161          |0.9824    |0.0899      |
|Model 14|cosine_proximity               |33.9161          |0.9824    |0.0899      |
## 2.1 Experiment 4 : Comparison of Different Activation Functions

## 3. Conclusion 



