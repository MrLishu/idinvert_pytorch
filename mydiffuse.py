import os
import numpy as np
import matplotlib.pyplot as plt
from utils.inverter import StyleGANInverter
from models.face_landmark_detector import FaceLandmarkDetector

IMAGE_SIZE = 64

inverter = StyleGANInverter('styleganinv_ffhq256', learning_rate=0.01, iteration=100,
                            reconstruction_loss_weight=1.0, perceptual_loss_weight=5e-5, regularization_loss_weight=0)
generator = inverter.G
resolution = inverter.G.resolution


def align(image_name):
    face_landmark_detector = FaceLandmarkDetector(resolution)
    face_infos = face_landmark_detector.detect(os.path.join('images', image_name))[0]
    image = face_landmark_detector.align(face_infos)
    return image


target_image_name = ['wtt.jpg', 'cyy.jpg']
target_images = [align(name) for name in target_image_name]

context_image_names = ['000002.png', '000008.png', '000018.png', '000019.png']
context_images = [align(name) for name in context_image_names]

wtt, cyy = target_images

target_image_code = [inverter.easy_invert(image, 1)[0] for image in target_images]
context_image_code = [inverter.easy_invert(image, 1)[0] for image in target_images]


exit(100)
