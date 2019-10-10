import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from PIL import Image
import ImageTools.index_generator as ig


class Exportable():
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs
    
    def exportColorImage(self, filename, **kwargs):
        pass
    
    def exportMonoImage(self, filename, **kwargs):
        pass
    
    