
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix


!pip install -q kaggle
from google.colab import files
files.upload()
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download galaxyh/kdd-cup-1999-data
!unzip kdd-cup-1999-data.zip
print(os.listdir("/content"))
# Load the dataset
df = pd.read_csv("/content/kddcup.data_10_percent_corrected", header=None)
print("Dataset loaded successfully!")
# Display first few rows
print(df.head())
