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


# target_image_name = ['wtt.jpg', 'cyy.jpg']
# target_images = [align(name) for name in target_image_name]
#
# context_image_names = ['000002.png', '000008.png', '000018.png', '000019.png']
# context_images = [align(name) for name in context_image_names]
#
# wtt, cyy = target_images
#
# target_image_code = [inverter.easy_invert(image, 1)[0] for image in target_images]
# target_image_code = [inverter.easy_invert(image, 1)[0] for image in target_images]

step = 5

source_image_name = 'wtt.jpg'
target_image_name = 'cyy.jpg'

source_image = align(source_image_name)
target_image = align(target_image_name)

source_image_code = inverter.easy_invert(source_image, 1)[0]
target_image_code = inverter.easy_invert(target_image, 1)[0]

linspace = np.linspace(0, 1, step).reshape(-1, 1, 1).astype(np.float32)
inter_codes = (1 - linspace) * source_image_code + linspace * target_image_code
inter_images = generator.easy_synthesize(inter_codes, latent_space_type='wp')['image']

plt.figure(figsize=(19.2, 10.8))
plt.axis("off")
plt.title(f"Interpolated images between {source_image_name.split('.')[0]} and {target_image_name.split('.')[0]}")
plt.imshow(np.concatenate(inter_images, axis=1))
plt.show()

exit(100)
