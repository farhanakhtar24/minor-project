from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from urllib.request import Request, urlopen
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
from derain_image import run_eval, load_model
import base64

app = Flask(__name__)

CORS(app)
#API Routes

@app.route("/")
def hello():
    return "Image de-raining using GAN"

@app.route('/derain', methods=['POST'])
def derain():
    try:
        data = request.get_json()
        image_url = str(data.get('url'))
        print(image_url)
        req = Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req)
        rain_image = response.read()
        with open('../../../minor_proj/rain_image.jpg', 'wb') as f:
            f.write(rain_image)
        model = load_model('unet_512')
        run_eval(net_G=model, save_output=True, output_dir='../../../minor_proj/output/', path_to_rainy_image='../../../minor_proj/rain_image.jpg', in_size=512)
        result = {'url': f'{image_url}'}

        fname = 'rain_image'

        img1 = mpimg.imread(f'../../../minor_proj/output/{fname}_input.png')
        img2 = mpimg.imread(f'../../../minor_proj/output/{fname}_output.png')
        combined_img = np.hstack((img1, img2))
        plt.imsave(f'../../../minor_proj/output/{fname}_combined.png', combined_img)
        
        # Open the image file in binary mode
        with open(f'../../../minor_proj/output/{fname}_output.png', 'rb') as f:
            # Read the file and encode it as base64
            image_data = base64.b64encode(f.read())
            # Decode the base64 bytes to ASCII
            base64_string = image_data.decode('ascii')
        
        result = {'image': base64_string}
        
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify(e), 400

if __name__ == "__main__":
    app.run(debug=True)