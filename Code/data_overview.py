import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr


# Read the dataset 
filepath= r'C:\Users\Hp\Downloads\insurance_dataset.csv'
data = pd.read_csv(filepath)

data = data.sample(frac=0.01, random_state=0)  # Use 1% of the data


# Display the first few rows of the DataFrame
data.head()
