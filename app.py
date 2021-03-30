from __future__ import division, print_function
import os
from tensorflow.keras.models import Sequential
import numpy as np
from keras.preprocessing import image 
#from keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import model_from_json

global graph
graph=tf.compat.v1.get_default_graph()
# Define a flask app
app = Flask(__name__)


# Load your trained model
json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("final_model.h5")

print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('digital.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = loaded_model.predict_classes(x)
        found = ["The great Indian bustard is a bustard found on the Indian subcontinent. A large bird with a horizontal body and long bare legs, giving it an ostrich like appearance, this bird is among the heaviest of the flying birds. It belongs to Otididae family and is listed among critically endangered species.",
                 "The spoon-billed sandpiper is a small wader which breeds in northeastern Russia and winters in Southeast Asia. It belongs to Scolopacidae family and is listed among critically endangered species.",
                 "Amorphophallus Titanum is endemic to sumantra. Due to its odor, like that of a rotting corpse, the titan arum is characterized as a Carrion Flower or Corpse Flower. It belongs to Araceae family.",
                 "Lady's slipper, (subfamily Cypripedioideae), also called lady slipper or slipper orchid, subfamily of five genera of orchids (family Orchidaceae), in which the lip of the flower is slipper-shaped.",
                 "Pangolins, sometimes known as scaly anteaters, are of the order Pholidota. Often thought of as a reptile, but pangolins are actually mammals. They are the most trafficked mammals.",
                 "The white deer found at Seneca Army Depot are a natural variation of the white-tailed deer (Odocoileus virginianus), which usually have brown coloring. The Seneca White Deer are leucistic, meaning they lack all pigmentation in the hair, but have the normal brown-colored eyes."]
        text = found[preds[0]]
        return text

if __name__ == '__main__':
    app.run(threaded = False)

