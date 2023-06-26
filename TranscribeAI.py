# pip install pytube,transformers,diffusers, torch
from pytube import YouTube
import os
from transformers import pipeline
import math
import json
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import torch
import base64
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers import StableDiffusionUpscalePipeline
import requests
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import YolosImageProcessor, YolosForObjectDetection
import cv2
from transformers import pipeline, set_seed
from ImageBot import ImageBot
from TextBot import TextBot


class TranscendAI:

    def __init__(self):

        self.image_bot = ImageBot()
        self.text_bot = TextBot()

    def run_pipeline(self, url, t_only):
        return self.text_bot.run_pipeline(url, t_only)

    def generate_image(self, text):
        return self.image_bot.generate_image(text)

    def img_2_img(self, text, url):
        return self.image_bot.img_2_img(text, url)

    def upscale_image(self, url):
        return self.image_bot.upscale_image( url)

    def generate_video(self, text):
        return self.image_bot.generate_video(text)

    def classify(self, url):
        return self.image_bot.classify(url)

    def yolo(self, url):
        return self.image_bot.yolo(url)

    def ask_qa_bot(self, question, context):
        return self.text_bot.ask_qa_bot(question, context)

    def text_generation(self, text, multiple):
        return self.text_bot.text_generation(text, multiple)

