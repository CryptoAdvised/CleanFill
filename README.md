# CleanFill
ClearFill is a python library you can use to fill NaN value in a matrix using various predictions techniques. This is useful in the context of collaborative filtering. It can be used to predict items rating in the context of recommendation engine. This code will fill NaN (Not A Number) with some predicted value according to the desired method of prediction. 

Alot of time, part of the data is left unfilled. This is frustating as sometime you are force to dump alot of data because of missing values. But if you use a library like CleanFill, you can avoid having to remove this potentially useful data.

It's a simple data transformation tool.

# How it works
ClearFill take in a numpy array matrix containing NaN and fill them with estimated value. For a demonstration simply look at test.py

# Available prediction methode for filling data
- Linear regression
- Nearest value
- Slope One (Fastest)
- Weighted Slope One
- Bipolar Slope One
- Means

# Available filling tools
- NaNtoZero
- ZeroToNaN

# Installation
pip install CleanFill

# Depedencies
You'll need numpy, scipy and pandas installed in your venv to run this library.

# Exemple for NaN as value with numpy array
```
import numpy as np
from cleanfill import fill



nan = np.NaN
my_data = np.array([[7, nan, 8, 7],
                    [6, 5, nan, 2],
                    [nan, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, nan, 2, 1]])


print(fill.linear(my_data))
print(fill.nearest(my_data))
print(fill.slope_one(my_data))
print(fill.weighted_slope_one(my_data))
print(fill.bipolar_slope_one(my_data))
print(fill.means(my_data)
```

# Exemple for 0 as value
```
import numpy as np
from cleanfill import fill


my_data2 = np.array([[7, 0, 8, 7],
                    [6, 5, 0, 2],
                    [0, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, 0, 2, 1]])


my_data2 = CleanFill.ZeroToNaN(my_data2)

print(fill.linear(my_data2))
print(fill.nearest(my_data))
print(fill.slope_one(my_data2))
print(fill.weighted_slope_one(my_data2))
print(fill.bipolar_slope_one(my_data2))
print(fill.means(my_data2))
```

# Exemple for NaN as value with pandas dataframe
```
import numpy as np
import pandas as pd
from cleanfill import fill

d={'name': ['hello', 'mello', 'yellow', 'pink'],
   'number': [6., 4., np.nan, 8.],
   'number2': [7., np.nan, 9., 9.],
   'number3': [np.nan, 5., 9., 10.],
   'number4': [8., np.nan, 7., 5.],
   'number5': [8., 6., np.nan, 5.],
   'number6': [3., 6., 9., np.nan],
   'number7': [np.nan, 2., 10., 1.],
   'number7': [2., 10., np.nan, 3.],
   'number7': [1., 2., 3., np.nan],
   'number7': [8., np.nan, 9., 9.]
   }

df=pd.DataFrame(data=d)

print(fill.linear(df))
print(fill.nearest(df))
print(fill.slope_one(df))
print(fill.weighted_slope_one(df))
print(fill.bipolar_slope_one(df))
print(fill.means(df))
```

# Exemple for 0 as value with pandas dataframe
```
import numpy as np
import pandas as pd
from cleanfill import fill

d={'name': ['hello', 'mello', 'yellow', 'pink'],
   'number': [6., 4., 0, 8.],
   'number2': [7., 0, 9., 9.],
   'number3': [0, 5., 9., 10.],
   'number4': [8., 0, 7., 5.],
   'number5': [8., 6., 0, 5.],
   'number6': [3., 6., 9., 0],
   'number7': [0, 2., 10., 1.],
   'number7': [2., 10., 0, 3.],
   'number7': [1., 2., 3., 0],
   'number7': [8., 0, 9., 9.]
   }

df=pd.DataFrame(data=d)

df=fill.ZeroToNaN(df)

print(fill.linear(df))
print(fill.nearest(df))
print(fill.slope_one(df))
print(fill.weighted_slope_one(df))
print(fill.bipolar_slope_one(df))
print(fill.means(df))
```
