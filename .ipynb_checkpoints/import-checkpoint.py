# quick_train_test.py
import os, sys
cipath = "/scratch/work/lupig1/apps/dmrgpy/"
sys.path.append(cipath + "/src")

import numpy as np
import matplotlib.pyplot as plt
import helper as hp
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib

import numpy as np
import pandas as pd
from dmrgpy import fermionchain
import matplotlib.pyplot as plt
