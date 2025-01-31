import os 
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn 

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
