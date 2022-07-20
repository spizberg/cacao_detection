import io
import tensorflow as tf
from PIL import Image 
from object_detection.utils import label_map_util
import numpy as np


PATH_TO_MODEL_DIR = 'my_model'
PATH_TO_LABELS = 'my_model/label_map.pbtxt'

def get_model():

	PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"


	# Load saved model and build the detection function
	detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

	return detect_fn, category_index

def get_image(image_bytes):
	image = Image.open(io.BytesIO(image_bytes))
	basewidth = 640
	wpercent = (basewidth/float(image.size[0]))
	hsize = int((float(image.size[1])*float(wpercent)))
	image = image.resize((basewidth,hsize), Image.ANTIALIAS)
	return np.array(image)