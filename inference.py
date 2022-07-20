from commons import get_image
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import io
import base64


def get_predicted_image(model, category, image):
      np_image = get_image(image)

      input_tensor = tf.convert_to_tensor(np_image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

      # input_tensor = np.expand_dims(image_np, 0)
      detections = model(input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
      detections['num_detections'] = num_detections

      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

      image_np_with_detections = np_image.copy()

      viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
      
      img = Image.fromarray(image_np_with_detections)
      # name = "static/images/predicted_image.jpeg"
      data = io.BytesIO()
      img.save(data, 'JPEG')
      encoded_img = base64.b64encode(data.getvalue())
      decoded_img = encoded_img.decode('utf-8')
      img_data = f"data:image/jpeg;base64,{decoded_img}"
      
      return img_data