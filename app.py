import base64
import io
import torch
from PIL import Image
from flask import Flask, render_template, request, abort
import requests, json
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
import pandas as pd

app = Flask(__name__, static_url_path='/static')

sqlite_db_path = 'sqlite:///DB.db'
engine = create_engine(sqlite_db_path)

def get_matching_data(model_result):

    df = pd.read_sql_table('disaese',engine)
    filtered_df = df[df['result'] == model_result]
    matching_data = filtered_df.to_dict(orient='records')
    return matching_data


model = YOLO("best.pt")


# 학습시킨 모델
# model = torch.hub.load('ultralytics/ultralytics', 'custom', path='models_train/best.pt', force_reload=True)
# model = YOLO("models_train/best.pt")  # load a pretrained model (recommended for training)


#
# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

@app.route('/')
def index():
    return render_template("about.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/skin_detect', methods=['GET', 'POST'])
def skin_detect():
    if request.method == 'POST':

        img_file = request.files['file']

        results = model.predict(source='static/images/result_sample_eu.png', save=True)
        model_result =  result.split(' ')[1]
        matching_data = get_matching_data(model_result)
        
        image_directory = 'runs/classify/predict/'
    
        # 이미지 파일명을 동적으로 얻어오기
        image_files = os.listdir(image_directory)
        if not image_files:
            return "No image files found."
    
        # 첫 번째 이미지 파일 사용
        dynamic_image_filename = image_files[0]
        image_path = os.path.join(image_directory, dynamic_image_filename)

        return render_template('result.html', matching_data=matching_data,img_path=img_path)

        # if im_file != '':
        #     im_bytes = im_file.read()
        #     img = Image.open(io.BytesIO(im_bytes))

        #     results = model.predict(source='static/images/result_sample_eu.png', save=True)



            # results = model.predict(img)  # inference
            #
            # results.ims  # array of original images (as np array) passed to model for inference
            # results.render()  # updates results.imgs with boxes and labels
            # for img in results.ims:  # 'JpegImageFile' -> bytes-like object
            #     buffered = io.BytesIO()
            #     img_base64 = Image.fromarray(img)
            #     img_base64.save(buffered, format="JPEG")
            #     encoded_img_data = base64.b64encode(buffered.getvalue()).decode(
            #         'utf-8')  # base64 encoded image with results
            #     return render_template('result.html', img_data=encoded_img_data)
        # else:
        #     abort(404)

    else:
        return render_template("skin_detect.html")


@app.route('/eye_detect', methods=['GET', 'POST'])
def eye_detect():
    if request.method == 'POST':

        im_file = request.files['file']
        if im_file != '':
            im_bytes = im_file.read()
            img = Image.open(io.BytesIO(im_bytes))

            results = model(img)  # inference

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
    return render_template("hospital.html")


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
# import base64
# import io
# import torch
# from PIL import Image
# from flask import Flask, render_template, request, abort
# #import requests, json
# from sqlalchemy import create_engine
# import pandas as pd
#
# app = Flask(__name__, static_url_path='/static')
#
# # # 학습시킨 모델
# # model = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/petom_weights.pt', force_reload=True)
# from torchvision import models
#
# # SQLite 데이터베이스 경로
# sqlite_db_path = 'sqlite:///your_database.db'
#
# # SQLite 데이터베이스 연결
# engine = create_engine(sqlite_db_path)
#
# # 모델 결과에 해당하는 데이터를 가져오는 함수
# def get_matching_data(model_result):
#     # SQLite 데이터베이스에서 데이터프레임으로 읽기
#     df = pd.read_sql_table('your_table_name', engine)
#
#     # 모델 결과와 일치하는 행 필터링
#     filtered_df = df[df['model_result_column_name'] == model_result]
#
#     # 필터링된 데이터를 딕셔너리로 변환하여 반환
#     matching_data = filtered_df.to_dict(orient='records')
#     return matching_data
#
# @app.route('/')
# def index():
#     return render_template("about.html")
#
# device = torch.device('cpu')
# model = models.resnet50(pretrained=True)
# model = model.to(device)
# model.eval()
#
# @app.route('/about')
# def about():
#     return render_template("about.html")
#
#
# @app.route('/skin_detect', methods=['GET', 'POST'])
# def skin_detect():
#     if request.method == 'POST':
#
#         im_file = request.files['file']
#         if im_file != '':
#             im_bytes = im_file.read()
#             img = Image.open(io.BytesIO(im_bytes))
#
#             results = model(img, size=640)  # inference
#
#             results.ims  # array of original images (as np array) passed to model for inference
#             results.render()  # updates results.imgs with boxes and labels
#             for img in results.ims:  # 'JpegImageFile' -> bytes-like object
#                 buffered = io.BytesIO()
#                 img_base64 = Image.fromarray(img)
#                 img_base64.save(buffered, format="JPEG")
#                 encoded_img_data = base64.b64encode(buffered.getvalue()).decode(
#                     'utf-8')  # base64 encoded image with results
#                 return render_template('result.html', img_data=encoded_img_data)
#         else:
#             abort(404)
#
#     else:
#         return render_template("skin_detect.html")
#
# @app.route('/eye_detect', methods=['GET', 'POST'])
# def eye_detect():
#     if request.method == 'POST':
#
#         im_file = request.files['file']
#         # if im_file != '':
#         #     im_bytes = im_file.read()
#         #     img = Image.open(io.BytesIO(im_bytes))
#
#         #     results = model(img, size=640)  # inference
#
#         #     results.ims  # array of original images (as np array) passed to model for inference
#         #     results.render()  # updates results.imgs with boxes and labels
#         #     for img in results.ims:  # 'JpegImageFile' -> bytes-like object
#         #         buffered = io.BytesIO()
#         #         img_base64 = Image.fromarray(img)
#         #         img_base64.save(buffered, format="JPEG")
#         #         encoded_img_data = base64.b64encode(buffered.getvalue()).decode(
#         #             'utf-8')  # base64 encoded image with results
#         #         return render_template('result.html', img_data=encoded_img_data)
#
#             # 이미지를 모델 입력 형식으로 변환 및 모델 예측 코드 생략
#
#             # 모델 결과를 얻어온다고 가정
#             model_result = 42  # 실제 모델 결과 값으로 대체
#
#             # 모델 결과에 해당하는 데이터 가져오기
#             matching_data = get_matching_data(model_result)
#
#             # 결과를 HTML 템플릿으로 전달
#             return render_template('result.html', matching_data=matching_data)
#         else:
#             abort(404)
#
#     else:
#         return render_template("eye_detect.html")
#
# @app.route('/bcs')
# def bcs():
#     return render_template("bcs.html")
#
# @app.route('/hospital')
# def hospital():
#     return render_template("hospital.html")
#
# @app.route('/dog_bcs')
# def dog_bcs():
#     return render_template("dog_bcs.html")
#
# @app.route('/cat_bcs')
# def cat_bcs():
#     return render_template("cat_bcs.html")
#
# @app.route('/bcs_1')
# def bcs_1():
#     return render_template("bcs_1.html")
# @app.route('/bcs_2')
# def bcs_2():
#     return render_template("bcs_2.html")
# @app.route('/bcs_3')
# def bcs_3():
#     return render_template("bcs_3.html")
# @app.route('/bcs_4')
# def bcs_4():
#     return render_template("bcs_4.html")
# @app.route('/bcs_5')
# def bcs_5():
#     return render_template("bcs_5.html")
#
# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('404.html'), 404
#
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)
