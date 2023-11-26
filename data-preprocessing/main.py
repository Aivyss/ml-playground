# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer  # import SimpleImputer class in scikit-learn

# import dataset
dataset: pd.DataFrame = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  # [row][column]
y = dataset.iloc[:, -1].values

print(x)
print(y)

# take care of missing data (no data -> mean value (but, it's not the absolute rule))
imputer: SimpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

