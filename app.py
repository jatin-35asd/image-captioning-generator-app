from flask import Flask, render_template, request
import os
from utils import preprocess_image, generate_caption, efficientnet

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img_array = preprocess_image(image_path)
            feature = efficientnet.predict(img_array, verbose=0)

            caption = generate_caption(feature)
            caption = caption.replace('startseq', '').replace('endseq', '').strip()

    return render_template("index.html", caption=caption, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
