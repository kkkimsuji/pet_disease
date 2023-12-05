import base64
import io
import torch
from PIL import Image
from flask import Flask, render_template, request, abort
import requests, json

app = Flask(__name__, static_url_path='/static')

# 학습시킨 모델
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/petom_weights.pt', force_reload=True)


@app.route('/')
def index():
    return render_template("about.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/skin_detect', methods=['GET', 'POST'])
def skin_detect():
    if request.method == 'POST':

        im_file = request.files['file']
        if im_file != '':
            im_bytes = im_file.read()
            img = Image.open(io.BytesIO(im_bytes))

            results = model(img, size=640)  # inference

            results.ims  # array of original images (as np array) passed to model for inference
            results.render()  # updates results.imgs with boxes and labels
            for img in results.ims:  # 'JpegImageFile' -> bytes-like object
                buffered = io.BytesIO()
                img_base64 = Image.fromarray(img)
                img_base64.save(buffered, format="JPEG")
                encoded_img_data = base64.b64encode(buffered.getvalue()).decode(
                    'utf-8')  # base64 encoded image with results
                return render_template('result.html', img_data=encoded_img_data)
        else:
            abort(404)

    else:
        return render_template("skin_detect.html")


@app.route('/eye_detect', methods=['GET', 'POST'])
def eye_detect():
    if request.method == 'POST':

        im_file = request.files['file']
        if im_file != '':
            im_bytes = im_file.read()
            img = Image.open(io.BytesIO(im_bytes))

            results = model(img, size=640)  # inference

            results.ims  # array of original images (as np array) passed to model for inference
            results.render()  # updates results.imgs with boxes and labels
            for img in results.ims:  # 'JpegImageFile' -> bytes-like object
                buffered = io.BytesIO()
                img_base64 = Image.fromarray(img)
                img_base64.save(buffered, format="JPEG")
                encoded_img_data = base64.b64encode(buffered.getvalue()).decode(
                    'utf-8')  # base64 encoded image with results
                return render_template('result.html', img_data=encoded_img_data)
        else:
            abort(404)

    else:
        return render_template("eye_detect.html")


@app.route('/bcs')
def bcs():
    return render_template("bcs.html")


@app.route('/hospital')
def hospital():
    return render_template("hospital.html",api_keys='e4780035288430c0c50d0571a16174f1', markers=markers)


@app.route('/dog_bcs')
def dog_bcs():
    return render_template("dog_bcs.html")


@app.route('/cat_bcs')
def cat_bcs():
    return render_template("cat_bcs.html")


@app.route('/bcs_1')
def bcs_1():
    return render_template("bcs_1.html")


@app.route('/bcs_2')
def bcs_2():
    return render_template("bcs_2.html")


@app.route('/bcs_3')
def bcs_3():
    return render_template("bcs_3.html")


@app.route('/bcs_4')
def bcs_4():
    return render_template("bcs_4.html")


@app.route('/bcs_5')
def bcs_5():
    return render_template("bcs_5.html")


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
