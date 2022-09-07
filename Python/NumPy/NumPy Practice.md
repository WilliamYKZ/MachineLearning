# NumPy Practice

## Matrix Operations

Matrix operation will make your code way more efficient for two reason:

1. **Single Processor Speed-Ups**: Using matrix operations can help your processor plan much better for cache placements, and data fetching. Mosts numberical computation libraries exploit features of efficient cache useage, SIMD features, etc. This can yield enormous speed-ups using the same computational resources. 



2. **Multi-Processor Speed-Ups**: Matrix operations are extremely parallelizable. This has a signigicant impact on hardware acceleration; you could have a parallelizable code without even knowing much about parallelization, as most numberical computation libraries try to automatically exploit as much CPU parallelization as appropriate. 



3. **Spending Less Time Executing High-Level Commands and Out-sourcing the Heavy-Lifting to Lower-Level Backends**: The high-level languages tend to have expensive commands, as they priortize coding convinience over efficiency. However, utilizing matrix operations outsources most of the code's heavy-lifiting to lower-level languages without the user even knowing about it. For instance, NUmpy runs most operations on a precompiled C backend. This is also correct about other numberical evaluation libraries such as the automatic differentiation libraries. 



4. **Portability**: This will make your code smaller, more understandable, and therefore less prone to have bugs. 



## The Question

We have a collection of data on whether a patient has diabetes, (https://www.kaggle.com/uciml/pima-indians-diabetes-database/data). 

Input/Output: This data has a set of attributes of patients, and a categorical variable telling whether the patient is diabetic or not. 

Missing Data: For several attributes in this data set, a value of 0 may indicate a missing value of the variable. 

Final Goal: We want to build a classifier that can predict whether a patient has diabetes or not. To do this, we will train multiple kinds of models, and will be handing the missing data with different approaches for each method. 



We focus on "Clucose" level variable in the data. Normally, a healthy person should never have blood Glucose levels of more than 1400mg/liter, since a healthy pancreas should be able to control the glucose level by injecting proper amounts of insulin. 



## Task 1

Write a function ```simple_pred_vec``` that takes a 1-d array of glucose levels $\mathbf{g}$ and a threshold $\theta$ as input, and applies the following prediction rule for patient $i$:

- Predict 1, if the patients glucose level $g_i$ is equal or larger than the threshold. (if $g_i \ge \theta$)
- Otherwise predict 0



```python
def simple_pred_vec(g, theta):
  
  out = g >= theta
  
  return out
```





## Task 2

Using the `simple_pred_vec` function that we just wrote, write a new function `simple_pred` function that takes a pandas data frame `df` and threshold `theta` as input, and produces a prediction numpy array `pred.` 

The dataframe `df` has a column `Glucose` which indicated the blood glucose levels, and a column `Outcome` which indicated whether the patient is disbetic or not. You should extract the `Glucose` column from the dataframe and use it for thresholding and prediction. 

1. If you like to have the column `des_col` of a pandas dataframe `df` as a numpy array, then `df['des_col'].values` my be helpful.

```python
def simple_pred(df, theta):
  
  pred = simple_pred_vec(df['Glucose'].values.reshape(1, -1), theta)
  
  return pred
```





## Task 3

Using the `simple_pred` function that you previously wrote, write a new function `simple_acc` function that takes a pandas dataframe `df` and threshold `theta` as input, predicts the `Outcome` label, and returns the accuracy `acc` of the predictor. 