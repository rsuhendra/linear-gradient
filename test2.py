import os
import sys
import time
import numpy as np
from numpy.fft import fft, rfft
import scipy
import matplotlib.pyplot as plt
from functions_tracking import *
from functions_plotting import *
from functions_plotting2 import *

nperseg = 256
detrend = False
mode = 'arithmetic'

# ignacious('WT3', nperseg = nperseg, detrend = detrend, mode = mode)
# ignacious('WT2', nperseg = nperseg, detrend = detrend, mode = mode)
psd('WT3', nperseg = nperseg, detrend = detrend, mode = mode)

acf('WT')

