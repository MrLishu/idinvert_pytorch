import os
import sys
import io
import bz2
import numpy as np
from PIL import Image
import IPython.display
import scipy.ndimage
from utils.editor import manipulate
from utils.inverter import StyleGANInverter
from models.helper import build_generator


inverter = StyleGANInverter('styleganinv_ffhq256',
                            learning_rate=0.01,
                            iteration=100,
                            reconstruction_loss_weight=1.0,
                            perceptual_loss_weight=5e-5,
                            regularization_loss_weight=0)


