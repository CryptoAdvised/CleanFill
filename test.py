import numpy as np
from CleanFill import CleanFill


nan = np.NaN
my_data = np.array([[7, nan, 8, 7],
                    [6, 5, nan, 2],
                    [nan, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, nan, 2, 1]])


print(CleanFill.fill_linear(my_data))
print(CleanFill.fill_slope_one(my_data))
print(CleanFill.fill_weighted_slope_one(my_data))
print(CleanFill.fill_bipolar_slope_one(my_data))



my_data2 = np.array([[7, 0, 8, 7],
                    [6, 5, 0, 2],
                    [0, 2, 2, 5],
                    [1, 3, 4, 1],
                    [2, 0, 2, 1]])

my_data2 = CleanFill.ZeroToNaN(my_data2)

print(CleanFill.fill_linear(my_data2))
print(CleanFill.fill_slope_one(my_data2))
print(CleanFill.fill_weighted_slope_one(my_data2))
print(CleanFill.fill_bipolar_slope_one(my_data2))