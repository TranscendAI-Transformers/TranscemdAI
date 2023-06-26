import base64
import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, \
    StableDiffusionUpscalePipeline
from PIL import Image
from io import BytesIO
import requests
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import YolosImageProcessor, YolosForObjectDetection
import cv2
from diffusers.utils import export_to_video
import random


class ImageBot:

    def __init__(self):
        self.diffusion2 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base",
                                                            torch_dtype=torch.float16, revision="fp16")
        self.diffusion2.scheduler = DPMSolverMultistepScheduler.from_config(self.diffusion2.scheduler.config)
        self.diffusion2 = self.diffusion2.to("cuda")
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
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.yolo_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.yolo_image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.video_diffuser = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",
                                                                torch_dtype=torch.float16,
                                                                variant="fp16")
        self.video_diffuser.scheduler = DPMSolverMultistepScheduler.from_config(self.video_diffuser.scheduler.config)
        self.video_diffuser.enable_model_cpu_offload()
        self.temp_location = './temp/'
        self.negative_prompt = None
        self.give_n_prompts()

    def generate_image(self, text):
        prompt = text
        image = self.diffusion2(prompt, negative_prompt=self.negative_prompt, num_inference_steps=100,
                                height=768, width=768).images[0]
        text = "dummy"
        image_name = self.temp_location + text + ".png"
        image.save(image_name)
        with open(image_name, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + my_string.decode('utf-8')
        os.remove(image_name)
        torch.cuda.empty_cache()
        return resp

    def img_2_img(self, text, url):
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

    def upscale_image(self, url):
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
        results = self.yolo_image_processor.post_process_object_detection(outputs, threshold=0.9,
                                                                          target_sizes=target_sizes)[0]
        output = []
        colors=[]
        image.save('original.png')
        img = cv2.imread("original.png")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            out = f"Detected {self.yolo_model.config.id2label[label.item()]} with confidence " \
                  f"{round(score.item(), 3)} at location {box}"
            output.append(out)
            print(box)
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color, 3)
            colors.append('rgb'+color[::-1].__repr__())
        cv2.imwrite('modified_image.jpg', img)
        with open('modified_image.jpg', "rb") as img_file:
            img_str = base64.b64encode(img_file.read())
            resp = 'data:image/png;base64,' + img_str.decode('utf-8')
        os.remove("original.png")
        os.remove("modified_image.jpg")
        return {'text': output, 'image': resp,'colors':colors}

    def generate_video(self, text):
        prompt = text
        video_frames = self.video_diffuser(prompt, num_inference_steps=50, num_frames=32,
                                           negative_prompt=self.negative_prompt).frames
        video_path = export_to_video(video_frames)
        print(video_path)
        torch.cuda.empty_cache()
        return video_path

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
