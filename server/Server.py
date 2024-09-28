import Util
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the Celebrity Face Recognition API"


@app.route("/Hello")
def hello():
    return "Hello You Are Currently Visiting Celebrity Face Recognition API"


@app.route("/Classify_Image", methods=["GET", "POST"])
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(Util.Classify_Image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction....")
    Util.Load_Saved_Artifacts()
    app.run()
