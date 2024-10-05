# Importing necessary libraries
import os
import datetime

import yfinance as yf
import pandas_datareader.data as web

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb

from joblib import Parallel, delayed

from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

