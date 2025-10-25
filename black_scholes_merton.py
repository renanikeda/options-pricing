from brownian_motion import geometric_brownian_motion
from matplotlib import pyplot as plt
import numpy as np


risk_free_rate = 0.07
t, GBM = geometric_brownian_motion(r=risk_free_rate, M=10)
