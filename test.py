# import base64
# import io
# import torch
# from PIL import Image
# from flask import Flask, render_template, request, abort
# import requests, json
# from ultralytics import YOLO
# import numpy as np
# import re
#
# app = Flask(__name__, static_url_path='/static')
#
# model = YOLO("best.pt")
# #
# # results = model.predict("static/images/nodule_tumor_sample1.jpg")
#
# img_file = 'static/images/nodule_tumor_sample1.jpg'
# img = Image.open(img_file)
#
# # RGB 형식으로 변환
# img = img.convert("RGB")
# print('-----------0------------')
# # 변환된 이미지를 사용하여 모델에 전달
# results = model.predict(source=img, save=True)
# print('-----------1------------')
# for result in results:
#     top5_list = result.probs.top5
#
#     # 가져온 데이터 사용 예시
#     if top5_list is not None:
#         print("Top 5 Classes:", top5_list)
#         print(top5_list[0])
# print('-----------2------------')
# for result in results:
#     probs_tensor = result.probs
#     keypoints_list = result.keypoints
#
#     # 가져온 데이터 사용 예시
#     if probs_tensor is not None:
#         print("Probabilities:", probs_tensor)
#     print('-----------3------------')
#
# # start_index = results[0].find("0: ")
# # if start_index != -1:
# #     extracted_string = results[0][start_index + len("0: "):]
# #
# #     print(extracted_string)
#
# # print('-----------1------------')
# # print(results[0])
# # print('-------------2-------------')
# # model_result = results[0].split(' ')[1]
# # print(model_result)
# # print('-------------3-------------')
#
#
# # print(results)
# # print('--------------')
# #
# # if isinstance(results, list):
# #     for item in results:
# #         print(type(item))