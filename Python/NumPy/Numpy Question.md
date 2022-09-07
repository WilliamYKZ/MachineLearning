# Numpy Question

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```



## Task: Floating point operation

1. it will hang your notebook since we should not try check for ecact equivalence (==) with floats. we can meansure approximate equivalence. 

   ```python
   fp = 0
   while fp != 1:
       fp = fp +0.1
   ```



2. Also, be careful about adding very small increments to a large-magnitude number. if you continuously add a small amount to a large float, you will notice that the float eventually stops changing at all.

   ```python
   x = 10.0**200 # 10 to the power of 200
   y = x + 1
   if x == y:
       print("x is still the same as y")
   
   # x is still the same as y
   ```



3. In python some time the floating point number x,y,z. (x  + y) + z != x + (y  + z)

   ```python
   x = 498
   y = 1e100
   z = -1e100
   
   #the small magnitude of 498 relative to 1e100 makes the 498 disappear in x+y:
   
   assert (x + y) + z != x + (y + z), "Incorrect counter-example: LHS equals RHS"
   print("Good counter-example!")
   ```



4. Also in python sometime the floating point number a,b,c a * (b + c) != a * b + a * c

   ```python
   a = 100
   b = 0.1
   c = 0.2
   assert a * (b + c) != a * b + a * c, "Incorrect counter-example: LHS equals RHS"
   print("Good counter-example!")
   ```



5. There are not equivalent in Python's floating point representation.

   ```python
   if .3 != .1 * 3:
       print("Floating point numbers are weird.")
   ```





## Task: Basic Operations

1. Some basic operations

   ```python
   a_mul_b = a * b
   a_plus_b = a + b
   a_minus_b = a - b
   a_div_b = a / b
   a_pow_b = a ** b
   ```



## Task: Element-wise Array Functions

Question: Can we write this fucntion using python.
$$
f\left(a_{i}\right)=\log \left(\frac{a_{i}}{10-a_{i}}\right)+\sin \left(a_{i}\right)
$$

1. Numpy offers comprehensive mathematical functions.

   ```python
   a = np.array([1,2,3])
   f = np.log(a/(10-a)) + np.sin(a)
   print (f)
   ```



## Task: Left multiply a matrix with a diagonal matrix

Description: Assume a numpy array $A$ with shape $(n,m)$ is given as well as a numpy array $a$ with shape $(n,)$. we want to find a matrix $B$ such that. 
$$
B=\left[\begin{array}{cccc}
a_{1} & 0 & \cdots & 0 \\
0 & a_{2} & \ldots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0 & 0 & \ldots & a_{n}
\end{array}\right] \times A
$$
where $a_1,....,a_n$ are the elements in $a$.

write a function `dial_left_mult` which takes the two arrays $a$ and $A$ as above and returns the matrix $B$. 

1. Solution 1

   ```python
   def diag_left_mult(a,A):
       
       a_diag = np.diag(a)
       
       return a_diag @ A
    
   a = np.array([1,2])
   A = np.array([[1,2],[3,4]])
   print (A)
   diag_left_mult(a,A)
   ```

   **Note that we have used the `@` operator for matrix multiplication.**



2. Solution 2

   ```python
   def diag_left_mult(a,A):
       return a.reshape(-1, 1) * A
     
   a = np.array([1,2])
   A = np.array([[1,2],[3,4]])
   diag_left_mult(a,A)
   ```

   The -1 argument with reshape will act as a wildcard for the appropriate number of missing dimensions. In this case, `a.reshape(-1, 1)` is the same as `a.reshape(2, 1)`.



## Task: Geometric Mean of an array

Description: Assume that you have a numpy array $a$ with shape $(d,)$, how can we find the geometric mean of the element in $a$? Remember that if the elements of $a$ are $(a_1,...,a_n)$, the geometric mean is $\sqrt[n]{a_1\times ...\times a_n}$.

Write a function it input a numpy array $a$ with shape $(d,)$. Do not assume anything about $d$ other than it is a positive integer. Also, do not assume anything about the element of $a$ other than they are positive values.

Output: a single value, which is the geometric mean of the elements in $a$.

```python
def geom_mean(a):
  d = a.shape[0]
  prod = np.prod(a)
  return prod**(1/d)

a = np.array([1,2,3])
geom_mean(a)
```



### `np.shape`

```python
import numpy as np

x = np.array([ [67, 63, 87],
             [77, 69, 59],
             [85, 87, 99],
             [68, 92, 78],
             [63, 89, 93]  ])
print(np.shape(x))

# it will return (6,3)
```





## Task: adjusting the elements in a matrix

Description: Given a numpy array $A$ with shape $(n,m)$, we want to generate another matrix $B$ with the same shape such that.
$$
B_{i, j}=A_{i, j} * i / j \quad 1 \leq i \leq n, 1 \leq j \leq m
$$
Example: if 
$$
A=\left[\begin{array}{lll}
3 & 2 & 1 \\
6 & 5 & 4
\end{array}\right]
$$
then
$$
B=\left[\begin{array}{lll}
3 \times 1 / 1 & 2 \times 1 / 2 & 1 \times 1 / 3 \\
6 \times 2 / 1 & 5 \times 2 / 2 & 4 \times 2 / 3
\end{array}\right]=\left[\begin{array}{ccc}
3 & 1 & 0.33 \\
12 & 5 & 2.66
\end{array}\right]
$$
Write a function `matrix_manipulate_1` which takes a numpy array $A$ as above, and outputs a numpy array $B$ as above.

```python
def matrix_manipulate_1(A):
  (n,m) = A.shape
  row_multipiers = (1 + np.arange(n)).reshape(-1,1)
  column_divisors = (1+np.arange(m)).reshape(1,-1).astype(np.float64)
  
  B1 = A * row_multipliers
  B = B1 / column_divisors
  return B

A = np.array([[3,2,1],[6,5,4]])
matrix_manipulate_1(A)
```





## Task: NumPy basic indexing

1. Basic indexing involves selecting specific elements in arrays and slicing arrays.



2. Advance indexing involves using indexing arrays to select the elements that you want.  



### Example

Select the top 5 elements of the list $x$ in a single line of code. Return a Numpy array of shape (5,), containing these elements. 



```python
import numpy as np

np.random.seed(1)
A = np.arange(100)
np.random.shuffle(A)
print (A)

def top_five(x):
    
    output = np.sort(x)[-5:]
    print (output)
    return output

assert np.all(np.sort(top_five(A)) == np.array([95,96,97,98,99])), 'Incorrect!'
```

1. Python **sort()** method sorts the list ascending by default. 
2. Python **Shuffle()** method takes a sequence, like a list, and reorganize the order of the items. 



## Task: NumPy advance indexing (integer indexing)

Description: Shuffle an input dataset $X$ (NumPy array of shape (N,28,28)) and the labels $y$  (Numpy array of shape(N,)). The outputs should be NUmpy arrays with the same shape as the inputs. 

```python
import numpy as np

np.random.seed(1)
A = np.random.randn(100, 28, 28)
b = np.random.randint(0, 10, size=100)

def shuffle_dataset(X, y):
    # your code here
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    
    shuffled_X = X[order]
    shuffled_y = y[order]
    
    return shuffled_X, shuffled_y

shuffled_A, shuffled_b = shuffle_dataset(A, b)
assert (shuffled_A.argmax() == 44718) and (shuffled_b.argmax() == 5), 'Incorrect!'

print (shuffled_A)
print (shuffled_b)
```





## Task: Adjacent Summations

Description: Assume that $a$ is a numpy array with shape (n,). Construct a numpy array $b$ with the same shape such that. 

```python
def sum_adjacent(a):
    b = np.zeros_like(a)
    print (b)
    b[:-1] += a[1:]
    print (b)
    b[1:] += a[:-1]
    print (b)
    return b

a = np.array([1,2,1,1,0])
print(sum_adjacent(a))
```





## Task: Subtracting geometric mean from the rows of a matrix 

Description: Assume that we have a numpy array $A$ with shape $(n,m)$ consisting of positive entries. We want to subtract all the elements in row of $A$ by the geometric mean of the elements in that row. For instance, If we have 
$$
A=\left[\begin{array}{lll}
a_{1,1} & a_{1,2} & a_{1,3} \\
a_{2,1} & a_{2,2} & a_{2,3}
\end{array}\right]
$$
Then we want to calculate the output matrix $B$ where
$$
\boldsymbol{B}=\left[\begin{array}{lll}
a_{1,1}-g_1 & a_{1,2}-g_1 & a_{1,3}-g_1 \\
a_{2,1}-g_2 & a_{2,2}-g_2 & a_{2,3}-g_2
\end{array}\right]
$$
Where
$$
g_1=\sqrt[3]{a_{1,1} a_{1,2} a_{1,3}} \quad g_2=\sqrt[3]{a_{2,1} a_{2,2} a_{2,3}}
$$
are the geometric means of the elemnts in each row of $A$.



```python
def subtract_geom_mean(A):
    print (A)
    m = A.shape[1] # the number of columns
    print (m)
    g = np.exp(np.mean(np.log(A),axis=1))
    print (g)
    g = g.reshape(2,1)
    print (g)
    return A - g

A = np.array([[1,2,3],[4,5,6]])
subtract_geom_mean(A)
```





## Task: Arg Sort

Description: Sorting is an important technique for anyone getting into machine learning and Python. Imagine a common scenario where a problem requires an array to be sorted based on another array, such as ordering flavors of ice cream based on a score from least to greatest. Here are the ice cream and their scores, in no particular order. 

```python
def rank_ice_cream(names, scores):
    # your code here
    print (names)
    print (scores)
    ranking = names[np.argsort(scores)]
    
    print (ranking)
    
    return ranking

scores = np.array([7, 8, 6, 3, 4, 10])
names = np.array(['Vanilla', 'Chocolate', 'Strawberry', 'Chocolate chip', 'Double chocolate', 'Green tea'])

ice_cream_ranking = rank_ice_cream(names, scores)

#print("This is the ranking of ice cream from lowest to highest score!")
#print(ice_cream_ranking)
```



**Different Between sort and argsort**

`sort()` will returns the array with the sorted values of the original array

`argsort()` will return the index of the sorted values in the original array.

Example: If the original array be `[3,1,2]`

`sort()` will give `[1,2,3]`

`argsort()` will give `[1,2,0]`





## Task: Manipulating Tensor Data

Let's start with some raw data in the form of a list of lists and convert that to a numpy array. We will then do various operations on the resulting tensor. 

```python
raw_data = [[5.3, 3.1, 1, 7, 8.3], [3, 5, 6.3, 4, 45], [99, 1, 101.2, 2., 0.2], [0., 0, 1., 22, 44.]]
data = np.array(raw_data).astype(np.float32)

print(type(data))   
print(data.dtype)  
print(data)
```



### Example 1

Double the values stored in even-number row (0,2,4,6)

```python
data[::2, :] *= 2
print(data)
```



### Example 2

Reverse the sequences in odd-numbered columns (1,3,5,7)

```python
# :: -1 is used to reverse a sequence 
data[:, 1::2] = data[:, 1::2][::-1, :]
print(data)
```



### Example 3

Add a new axis to the tensor

```python
# There are two way to do this, once is using None
# Other is using np.newaxis

print(data.shape)
data_shape_expand_none = data[None]
data_shape_expand_newaxis = data[np.newaxis]
print(data_shape_expand_none.shape)
print(data_shape_expand_newaxis.shape)
print (data_shape_expand_none)
print (data_shape_expand_newaxis)
```

And you can also add a new axis along any dimension. 

```python
print(data.shape)
data_shape_expand_none = data[:, None]
data_shape_expand_newaxis = data[:, np.newaxis]
print(data_shape_expand_none.shape)
print(data_shape_expand_newaxis.shape)

print (data_shape_expand_none)
print (data_shape_expand_newaxis)
```

quickly change the dimension. 

```python
one_d_arr = np.array([3.0, 4.0])
print(one_d_arr.shape)
three_d_arr = np.atleast_3d(one_d_arr)
print(three_d_arr.shape)

print (three_d_arr)
```



### Example 4

Permute the axes of a tensor

```python
data_p = data[:, np.newaxis]
print("Shape of tensor before transpose {}".format(data_p.shape))
data_p = np.transpose(data_p, (2,0,1))
print("Shape of tensor after transpose {}".format(data_p.shape))
```





## Task Array Stack

How can we stack two arrays $a$ and $b$ vertically?

```python
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
```



### Method 1

```python
c = np.concatenate([a, b], axis=0)
print (a)
print (b)
print (c)
```



### Mathod 2

```python
c = np.vstack([a, b])
print (a)
print (b)
print (c)
```



### Method 3

```python
c = np.r_[a, b]
print (a)
print (b)
print (c)
```





How can we stack two arrays $a$ and $b$ horizontally?

```python
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
```



### Method 1

```python
c = np.concatenate([a, b], axis=1)
print (a)
print (b)
print (c)
```



### Method 2

```python
c = np.hstack([a, b])
print (a)
print (b)
print (c)
```



### Method 3

```python
c = np.c_[a, b]
print (a)
print (b)
print (c)
```







## Tricks: useeful commands in NumPy

1. Matrix Multiply: `a.dot(b)` is equivalent to `a @ b`

```python
import numpy as np

a = np.random.rand(3, 2)
b = np.random.rand(2, 3)
assert np.any(a.dot(b) == a @ b)

print (a)
print (b)

print (a.dot(b))
print (a @ b)
```



2. Matrix Transpose: `a.transpose()` is equivalent to `a.T`

```python
print (a.T)
print (a.transpose())
```



