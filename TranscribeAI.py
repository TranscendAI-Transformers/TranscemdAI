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


class TranscendAI:

    def __init__(self):
        self.negative_prompt = None
        self.url = None
        self.title = None
        self.res = {}
        self.yt = None
        self.audio_format = '.wav'
        self.output_format = '.json'
        self.temp_location = './temp/'
        self.transcribe_model = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", device=0, framework="pt")
        self.summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0,
                                         framework="pt")
        self.nlp = pipeline('question-answering', model="deepset/roberta-large-squad2",
                            tokenizer="deepset/roberta-large-squad2", device=0, framework="pt")
        self.video_diffuser = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",
                                                                torch_dtype=torch.float16,
                                                                variant="fp16")
        self.video_diffuser.scheduler = DPMSolverMultistepScheduler.from_config(self.video_diffuser.scheduler.config)
        self.video_diffuser.enable_model_cpu_offload()
        self.img_img_diffusion = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                                revision="fp16",
                                                                                torch_dtype=torch.float16,
                                                                                safety_checker=None)
        self.img_img_diffusion = self.img_img_diffusion.to("cuda")
        self.upscaler = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",
                                                                       revision="fp16", torch_dtype=torch.float16)
        self.upscaler = self.upscaler.to("cuda")
        self.upscaler.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.upscaler.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.upscaler.enable_model_cpu_offload()
        self.upscaler.enable_attention_slicing("max")
        self.diffusion2 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base",
                                                            torch_dtype=torch.float16, revision="fp16")
        self.diffusion2.scheduler = DPMSolverMultistepScheduler.from_config(self.diffusion2.scheduler.config)
        self.diffusion2 = self.diffusion2.to("cuda")
        self.give_n_prompts()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.yolo_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.yolo_image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.text_generator = pipeline('text-generation', model='gpt2-large', device=0)

    def give_n_prompts(self):
        self.negative_prompt = "split image, out of frame, amputee, mutated, mutation, deformed, severed, " \
                               "dismembered," \
                               " corpse, photograph, poorly drawn, bad anatomy, blur, blurry, lowres, bad hands, " \
                               "error, missing fingers, extra digit, fewer digits, cropped, worst quality, " \
                               "low quality," \
                               " normal quality, jpeg artifacts, signature, watermark, " \
                               "username, artist name, ugly, symbol, " \
                               "hieroglyph,, extra fingers,  six fingers per hand, " \
                               "four fingers per hand, disfigured hand, " \
                               "monochrome, missing limb, disembodied limb, linked limb, connected limb, " \
                               "interconnected limb, broken finger, broken hand, broken wrist, broken leg, " \
                               "split limbs, no thumb, missing hand, missing arms, missing legs, fused finger, " \
                               "fused digit, missing digit, bad digit, extra knee, extra elbow, storyboard, " \
                               "split arms, split hands, split fingers, twisted fingers, disfigured butt, " \
                               "deformed hands,  watermark, text, deformed fingers, blurred faces, irregular face," \
                               " irrregular body shape, ugly eyes, deformed face, squint, tiling, poorly drawn hands," \
                               " poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, " \
                               "disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark," \
                               " grainy, signature, cut off, draft, ugly eyes, squint, tiling, poorly drawn hands, " \
                               "poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs," \
                               " disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, " \
                               "grainy, signature, cut off, draft, disfigured, kitsch, ugly, oversaturated, grain, " \
                               "low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation," \
                               " mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs," \
                               " disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly," \
                               " disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, " \
                               "2 heads, 2 faces, no repeat, elongated waist, long waist, long legs, elongated body"

    # downloading youtube video
    def download_video(self):
        title = self.title + self.audio_format
        if not os.path.isfile(self.temp_location + title):
            video = self.yt.streams.filter(only_audio=True).first()
            out_file = video.download(output_path=self.temp_location)
            base, ext = os.path.splitext(out_file)
            title = self.temp_location + title
            try:
                os.rename(out_file, title)
            except FileExistsError:
                os.remove(title)
                os.rename(out_file, title)
            return title
        print('downloaded video->', title)
        print('location->', self.temp_location + title)
        return './temp/' + title

    # for transcribing the audio
    def transcribe_audio(self):
        new_file = self.download_video()
        text = self.transcribe_model(new_file, chunk_length_s=60)['text']
        self.res['text'] = text.lower()
        print('transcribe done')
        torch.cuda.empty_cache()

    # used for generating summary
    def generate_summary(self):
        prev = 0
        text = self.res['text']
        self.res['summary'] = []
        if len(text) > 3000:
            batches = math.ceil(len(text) / 3000)
            for i in range(1, batches + 1):
                summary = self.summarizer_model(text[prev:i * 3000], max_length=150, min_length=30, do_sample=False)
                self.res['summary'].append(summary[0]['summary_text'].lower())
                prev += 3000
        else:
            summary = self.summarizer_model(text, max_length=150, min_length=30, do_sample=False)
            self.res['summary'].append(summary[0]['summary_text'].lower())
        print('summarization done')
        torch.cuda.empty_cache()

    def generate_image(self, text):
        prompt = text
        image = self.diffusion2(prompt, negative_prompt=self.negative_prompt, num_inference_steps=100,
                                height=768, width=768).images[0]
        image_name = self.temp_location + text + ".png"
        image.save(image_name)
        with open(image_name, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + my_string.decode('utf-8')
        os.remove(image_name)
        torch.cuda.empty_cache()
        return resp

    def img_2_img(self, text, url):
        import requests
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((768, 512))
        prompt = text
        images = self.img_img_diffusion(prompt=prompt, image=init_image, num_inference_steps=100,
                                        negative_prompt="bad, deformed, ugly, bad anatomy").images
        images[0].save("fantasy_landscape.png")
        with open("fantasy_landscape.png", "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + my_string.decode('utf-8')
        os.remove("fantasy_landscape.png")
        torch.cuda.empty_cache()
        return resp

    def upscale_image(self, prompt, url):
        # url = "https://cdn.britannica.com/30/94430-050-D0FC51CD/Niagara-Falls.jpg"
        response = requests.get(url)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((512, 512))
        prompt = "a bird sitting on a branch"
        upscaled_image = self.upscaler(prompt=prompt, image=low_res_img).images[0]
        temp_name = "upsampled_cat.png"
        upscaled_image.save(temp_name)
        with open(temp_name, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + my_string.decode('utf-8')
        os.remove(temp_name)
        torch.cuda.empty_cache()
        return resp

    def generate_video(self, text):
        prompt = text
        video_frames = self.video_diffuser(prompt, num_inference_steps=50, num_frames=80,
                                           negative_prompt=self.negative_prompt, height=256, width=256).frames
        video_path = export_to_video(video_frames)
        print(video_path)
        torch.cuda.empty_cache()
        return video_path

    def ask_qa_bot(self, question, context):
        qa_input = {
            'question': question,
            'context': context
        }
        ans = self.nlp(qa_input)
        torch.cuda.empty_cache()
        return ans

    # to save the file as json
    def store_output(self):
        with open(self.temp_location + self.title + self.output_format, "w") as outfile:
            json.dump(self.res, outfile)

    # to process the url
    def process_url(self, url):
        self.url = url
        self.yt = YouTube(self.url)
        title = self.yt.title
        title = title.replace('.', '')
        title = title.replace('(', '')
        title = title.replace(')', '')
        title = title.replace('?', '')
        title = title.replace('|', '')
        title = title.replace(':', '')
        print(title)
        title = title.replace(' ', '_')
        self.title = title

    def check_cache(self):
        prev_file = self.temp_location + self.title + self.output_format
        if os.path.isfile(prev_file):
            with open(prev_file) as json_file:
                self.res = json.load(json_file)
                return self.res
        else:
            return None

    def classify(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((512, 512))
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.resnet_model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return self.resnet_model.config.id2label[predicted_label]

    def yolo(self, url):
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.yolo_image_processor(images=image, return_tensors="pt")
        outputs = self.yolo_model(**inputs)

        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        target_sizes = torch.tensor([image.size[::-1]])
        results = \
        self.yolo_image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        output = []
        image.save('original.png')
        img = cv2.imread("original.png")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            out = f"Detected {self.yolo_model.config.id2label[label.item()]} with confidence " \
                  f"{round(score.item(), 3)} at location {box}"
            output.append(out)
            print(box)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (0, 0, 255), 3)
        cv2.imwrite('modified_image.jpg', img)
        with open('modified_image.jpg', "rb") as img_file:
            img_str = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + img_str.decode('utf-8')
        os.remove("original.png")
        os.remove("modified_image.jpg")
        return {'text': output, 'image': resp}

    def text_generation(self, text):
        return self.text_generator(text, max_length=150, num_return_sequences=1, top_k=0,
                                   temperature=0.8, do_sample=True, )[0]

    # this is the pipeline sequence
    def run_pipeline(self, url):
        self.process_url(url)
        cache_hit = self.check_cache()
        if cache_hit is not None:
            return cache_hit
        self.transcribe_audio()
        self.generate_summary()
        self.store_output()
        return self.res
