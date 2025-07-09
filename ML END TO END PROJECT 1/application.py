from flask import Flask, request, render_template
import pickle
import numpy as np



application = Flask(__name__)
app = application
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load your model and scaler (ensure the paths are correct)
ridge_model = pickle.load(open('MODELS/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('MODELS/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        temperature = float(request.form.get("temperature"))
        rh = float(request.form.get("rh"))
        ws = float(request.form.get("ws"))
        rain = float(request.form.get("rain"))
        ffmc = float(request.form.get("ffmc"))
        dmc = float(request.form.get("dmc"))
        isi = float(request.form.get("isi"))
        classes = float(request.form.get("classes"))
        region = float(request.form.get("region"))

        data = [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]
        new_data_scaled = standard_scaler.transform(data)
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", results = result[0])

    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
