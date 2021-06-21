import os
import dlib
import numpy as np
from PIL import Image
import scipy.ndimage
from utils.editor import manipulate
from utils.inverter import StyleGANInverter
from models.helper import build_generator


MODEL_DIR = os.path.join('models', 'pretrain')
LANDMARK_MODEL_NAME = 'shape_predictor_68_face_landmarks.dat'
LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, LANDMARK_MODEL_NAME)

class FaceLandmarkDetector(object):
  def __init__(self, align_size=256, enable_padding=True):
    # Download models if needed.
    self.face_detector = dlib.get_frontal_face_detector()
    self.landmark_detector = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    self.align_size = align_size
    self.enable_padding = enable_padding

  def detect(self, image_path):
    results = []

    # image_ = np.array(image)
    images = dlib.load_rgb_image(image_path)
    # Face detection (1 means to upsample the image for 1 time.)
    bboxes = self.face_detector(images, 1)
    # Landmark detection
    for bbox in bboxes:
      landmarks = []
      for point in self.landmark_detector(images, bbox).parts():
        landmarks.append((point.x, point.y))
      results.append({
          'image_path': image_path,
          'bbox': (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
          'landmarks': landmarks,
      })
    return results

  def align(self, face_info):
    img = Image.open(face_info['image_path'])

    landmarks = np.array(face_info['landmarks'])
    eye_left = np.mean(landmarks[36: 42], axis=0)
    eye_right = np.mean(landmarks[42: 48], axis=0)
    eye_middle = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_middle = (landmarks[48] + landmarks[54]) / 2
    eye_to_mouth = mouth_middle - eye_middle

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_middle + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / self.align_size * 0.5))
    if shrink > 1:
      rsize = (int(np.rint(float(img.size[0]) / shrink)),
               int(np.rint(float(img.size[1]) / shrink)))
      img = img.resize(rsize, Image.ANTIALIAS)
      quad /= shrink
      qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
      img = img.crop(crop)
      quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if self.enable_padding and max(pad) > border - 4:
      pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
      img = np.pad(np.float32(img),
                   ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                   'reflect')
      h, w, _ = img.shape
      y, x, _ = np.ogrid[:h, :w, :1]
      mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                         np.float32(w - 1 - x) / pad[2]),
                        1.0 - np.minimum(np.float32(y) / pad[1],
                                         np.float32(h - 1 - y) / pad[3]))
      blur = qsize * 0.02
      blurred_image = scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
      img += blurred_image * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
      img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
      img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
      quad += pad[:2]

    # Transform.
    img = img.transform((self.align_size * 4, self.align_size * 4), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    img = img.resize((self.align_size, self.align_size), Image.ANTIALIAS)

    return np.array(img)