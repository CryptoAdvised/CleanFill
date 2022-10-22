# CleanFill
ClearFill is a python library you can use to fill NaN value in a matrix using various predictions techniques. This is useful in the context of collaborative filtering. It can be used to predict items rating in the context of recommendation engine. This code will fill NaN (Not A Number) with some predicted value according to the desired method of prediction. 

Alot of time, part of the data is left unfilled. This is frustating as sometime you are force to dump alot of data because of missing values. But if you use a library like CleanFill, you can avoid having to remove this potentially useful data.

It's a simple data transformation tool.

# How it works
ClearFill take in a numpy array matrix containing NaN and fill them with estimated value. For a demonstration simply look at test.py

# Available prediction methode
- Linear regression
- Nearest value
- Slope One (Fastest)
- Weighted Slope One
- Bipolar Slope One

# Installation
pip install CleanFill

# Depedencies
You'll need numpy and scipy installed in your venv to run this library.

# Exemple for NaN as value
```
import numpy as np
from cleanfill import cleanfill as cf



nan = np.NaN
my_data = np.array([[7, nan, 8, 7],
                    [6, 5, nan, 2],
                    [nan, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, nan, 2, 1]])


print(cf.fill_linear(my_data))
print(cf.fill_slope_one(my_data))
print(cf.fill_weighted_slope_one(my_data))
print(cf.fill_bipolar_slope_one(my_data))
```

# Exemple for 0 as value
```
import numpy as np
from cleanfill import cleanfill as cf


my_data2 = np.array([[7, 0, 8, 7],
                    [6, 5, 0, 2],
                    [0, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, 0, 2, 1]])


my_data2 = cf.ZeroToNaN(my_data2)

print(cf.fill_linear(my_data2))
print(cf.fill_slope_one(my_data2))
print(cf.fill_weighted_slope_one(my_data2))
print(cf.fill_bipolar_slope_one(my_data2))
