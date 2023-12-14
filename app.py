import base64
import io
import torch
from PIL import Image
from flask import Flask, render_template, request, abort, send_from_directory, send_file
import requests, json
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
import pandas as pd
import os
import shutil

app = Flask(__name__, static_url_path='/static')

sqlite_db_path = 'sqlite:///DB.db'
engine = create_engine(sqlite_db_path)


def get_matching_data(model_result):
    df = pd.read_sql_table('disease', engine)
    filtered_df = df[df['id'] == model_result]
    matching_data = filtered_df.to_dict(orient='records')
    return matching_data


def delete_directory(directory_path):
    try:
        # 디렉토리 및 하위 항목 삭제
        shutil.rmtree(directory_path)
    except OSError as e:
        print(f"디렉토리 삭제 중 오류 발생: {e}")


def get_image(filename):
    # 정적 파일을 저장한 디렉토리의 경로를 지정
    directory = os.path.join(app.root_path, 'runs', 'classify', 'predict')

    # 해당 파일을 반환
    return send_from_directory(directory, filename)


def convert_image_format(input_path, output_path, new_format):
    try:
        # 이미지 열기
        with Image.open(input_path) as img:
            # 이미지를 지정된 형식으로 저장
            img.save(output_path, format=new_format)
            print(f"이미지 형식이 {new_format}으로 변경되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


# def get_image():
#     # 이미지 파일 경로를 설정합니다.
#     image_path = 'static/sample_image.jpg'
#
#     # MIME 타입 설정
#     mime_type = 'image/jpeg'
#
#     return send_file(image_path, mimetype=mime_type)
#

model = YOLO("best.pt")


# 학습시킨 모델
# model = torch.hub.load('ultralytics/ultralytics', 'custom', path='models_train/best.pt', force_reload=True)
# model = YOLO("models_train/best.pt")  # load a pretrained model (recommended for training)

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
        img = Image.open(img_file)

        # RGB 형식으로 변환
        img = img.convert("RGB")

        # 변환된 이미지를 사용하여 모델에 전달
        results = model.predict(source=img, save=True)

        model_result = results.probs.top5

        matching_data = get_matching_data(model_result[0])

        image_directory = 'runs/classify/predict/'

        # 이미지 파일명을 동적으로 얻어오기
        image_files = os.listdir(image_directory)
        if not image_files:
            return "No image files found."

        # 첫 번째 이미지 파일 사용
        dynamic_image_filename = image_files[0]
        img_path = os.path.join(image_directory, dynamic_image_filename)

        return render_template('result.html', matching_data=matching_data, img_path=img_path)

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
        directory_to_delete = "runs/classify/predict/"
        delete_directory(directory_to_delete)

        img_file = request.files['file']
        img = Image.open(img_file)

        # RGB 형식으로 변환
        img = img.convert("RGB")

        # 변환된 이미지를 사용하여 모델에 전달
        results = model.predict(source=img, save=True)
        for result in results:
            top5_list = result.probs.top5

            # 가져온 데이터 사용 예시
            if top5_list is not None:
                # print("Top 5 Classes:", top5_list)
                model_result = top5_list[0]

        matching_data = get_matching_data(model_result)

        # img_path = get_image('image0.jpg')
        # image_directory = 'runs/classify/predict/'
        #
        # # 이미지 파일명을 동적으로 얻어오기
        # image_files = os.listdir(image_directory)
        # if not image_files:
        #     return "No image files found."
        #
        # # 첫 번째 이미지 파일 사용
        # dynamic_image_filename = image_files[0]
        # img_path = os.path.join(image_directory, dynamic_image_filename)
        # mime_type = 'image/jpeg'
        source_path = 'runs/classify/predict/image0.jpg'

        # 대상 디렉토리
        target_directory = 'static/images/'

        # 대상 파일 경로
        target_path = os.path.join(target_directory, "result_img.jpg")

        try:
            # 이미지 파일을 대상 디렉토리로 이동
            shutil.move(source_path, target_path)
        except Exception as e:
            print(f"이동 중 오류 발생: {e}")

        return render_template('result.html', matching_data=matching_data)


    else:
        return render_template("eye_detect.html")


# @app.route('/eye_detect', methods=['GET', 'POST'])
# def eye_detect():
#     if request.method == 'POST':
#
#         im_file = request.files['file']
#         if im_file != '':
#             im_bytes = im_file.read()
#             img = Image.open(io.BytesIO(im_bytes))
#
#             results = model(img)  # inference
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
#         return render_template("eye_detect.html")


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
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
