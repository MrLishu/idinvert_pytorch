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


source_image_name = 'female.png'
source_image = align(source_image_name)
source_image_code = inverter.easy_invert(source_image, 1)[0]

layer_id = 'all'
layer_dict = {'all': list(range(generator.net.num_layers)), 'low': [0, 1], 'mid': [2, 3, 4, 5], 'high': list(range(6, 14))}
layers = layer_dict[layer_id]

weights = []
for layer in layers:
    weights.append(generator.net.synthesis.__getattr__(f'layer{layer}').epilogue.style_mod.dense.fc.weight.T.cpu().detach().numpy())
weight = np.concatenate(weights, axis=1).astype(np.float32)
weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

boundary = eigen_vectors[0]
new_codes = source_image_code.repeat(9, axis=0).reshape(9, generator.net.num_layers, -1)
new_codes[:, layers, :] += boundary.reshape(1, 1, -1) * np.linspace(-3, 3, 9, dtype=np.float32).reshape(-1, 1, 1)
new_images = generator.easy_synthesize(new_codes, latent_space_type='wp')['image']

plt.figure(figsize=(19.2, 10.8))
plt.axis("off")
plt.title(f"Images")
plt.imshow(np.concatenate(new_images, axis=1))
plt.show()

exit(100)
