import os
from flask import Flask, render_template, request
from commons import get_model
import tensorflow as tf
app = Flask(__name__)

from inference import get_predicted_image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model, category = get_model()

@app.route('/',methods=['GET','POST'])
def hello_world():
	if request.method=='POST':
		try:
			file=request.files['file']
			image=file.read()
			data_img=get_predicted_image(model, category, image)
			# data_img = None
			return render_template('result.html',predicted_image=data_img)

		except:

			return render_template('index.html')
	return render_template('index.html')
		

    

if __name__ == '__main__':
	app.run(debug=True, port=os.getenv('PORT',5000))
	




				
		    
		    
		    