import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import base64
# video_diffuser = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
# video_diffuser.scheduler = DPMSolverMultistepScheduler.from_config(video_diffuser.scheduler.config)
# video_diffuser.enable_model_cpu_offload()

# prompt = "A panda eating bamboo on a rock"
# video_frames = video_diffuser(prompt, num_inference_steps=25).frames
# video_path = export_to_video(video_frames)
# print(video_path)
# with open(video_path, "rb") as vid_file:
#         vid_string=base64.b64encode(vid_file.read())
#         print(vid_string)
# from huggingface_hub import snapshot_download
#
# from modelscope.pipelines import pipeline
# from modelscope.outputs import OutputKeys
# import pathlib
#
# model_dir = pathlib.Path('weights')
# snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
#                    repo_type='model', local_dir=model_dir)
#
# pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
# test_text = {
#         'text': 'A panda eating bamboo on a rock.',
#     }
# output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]
# print('output_video_path:', output_video_path)
# import requests
# import torch
# from PIL import Image
# from io import BytesIO
#
# from diffusers import StableDiffusionImg2ImgPipeline
#
# device = "cuda"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(device)
#
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
#
# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))
#
# prompt = "A fantasy landscape, trending on artstation"
#
# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# images[0].save("fantasy_landscape.png")


# import PIL
# import requests
# import torch
# from io import BytesIO
#
# from diffusers import StableDiffusionInpaintPipeline
#
#
# def download_image(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")
#
#
# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture" \
#           "-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture" \
#            "-creations-5sI6fQgYIuo_mask.png"
#
# init_image = download_image(img_url).resize((512, 512))
# mask_image = download_image(mask_url).resize((512, 512))
#
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")
#
# prompt = "small robot , high resolution, sitting on a park bench"
# image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
# image.save('demo.png')
#
# torch.cuda.empty_cache()


import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(
#     model_id, revision="fp16", torch_dtype=torch.float16
# )
# pipeline = pipeline.to("cuda")

# let's download an  image
# url = "https://cdn.britannica.com/30/94430-050-D0FC51CD/Niagara-Falls.jpg"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((250, 250))
# prompt = "a white dog"


#
# import torch
# from diffusers import DiffusionPipeline
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
#
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipe = StableDiffusionUpscalePipeline.from_pretrained(
#     model_id, revision="fp16", torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# # Workaround for not accepting attention shape using VAE for Flash Attention
# pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
# pipe.enable_model_cpu_offload()
# upscaled_image = pipe(prompt=prompt, image=low_res_img, output_type="np").images[0]
# upscaled_image.save("upsampled_cat.png")
#
# torch.cuda.empty_cache()

# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# import torch
#
# repo_id = "stabilityai/stable-diffusion-2-base"
# pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
#
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
# negative_prompt="split image, out of frame, amputee, mutated, mutation, deformed, severed, dismembered, corpse, photograph, poorly drawn, bad anatomy, blur, blurry, lowres, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, ugly, symbol, hieroglyph,, extra fingers,  six fingers per hand, four fingers per hand, disfigured hand, monochrome, missing limb, disembodied limb, linked limb, connected limb, interconnected limb,  broken finger, broken hand, broken wrist, broken leg, split limbs, no thumb, missing hand, missing arms, missing legs, fused finger, fused digit, missing digit, bad digit, extra knee, extra elbow, storyboard, split arms, split hands, split fingers, twisted fingers, disfigured butt, deformed hands,  watermark, text, deformed fingers, blurred faces, irregular face, irrregular body shape, ugly eyes, deformed face, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, ugly eyes, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, 2 heads, 2 faces"
# prompt = "medium shot of medieval warrior from behind, river, mountains, clouds, castle, artstation"
# image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=100, height=768, width= 768).images[0]
# image.save("astronaut.png")

import torch
from transformers import pipeline

# path to the audio file to be transcribed
# audio = "./temp/test.wav"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
# transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2", chunk_length_s=30, device=device)
# transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
#
# print('Transcription: ', transcribe(audio)["text"])
# from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch
# from datasets import load_dataset
#
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
#
# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
#
# inputs = processor(image, return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])


# from huggingface_hub import snapshot_download
#
# from modelscope.pipelines import pipeline
# from modelscope.outputs import OutputKeys
# import pathlib
#
# model_dir = pathlib.Path('weights')
# snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
#                    repo_type='model', local_dir=model_dir)
#
# pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
# test_text = {
#         'text': 'spiderman running',
#     }
# output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]
#
# with open('test.mp4', "wb") as out_file:  # open for [w]riting as [b]inary
#     out_file.write(output_video_path)
# print('output_video_path:', )
# from transformers import YolosImageProcessor, YolosForObjectDetection
# from PIL import Image
# import torch
# import requests
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
# image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
#
# inputs = image_processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
#
# # model predicts bounding boxes and corresponding COCO classes
# logits = outputs.logits
# bboxes = outputs.pred_boxes
#
#
# # print results
# target_sizes = torch.tensor([image.size[::-1]])
# results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2-large')
# set_seed(42)
# print(generator("what is java?", max_length=150, num_return_sequences=1))
# from ImageBot import ImageBot
# bot = ImageBot()
# # bot.generate_image('kid playing in park')
# # bot.img_2_img('kid running', 'https://cdn.outsideonline.com/wp-content/uploads/2017/12/13/kids-running-raising-rippers_h.jpg')
# # bot.upscale_image(None,'https://cdn.outsideonline.com/wp-content/uploads/2017/12/13/kids-running-raising-rippers_h.jpg')
# bot.yolo('https://cdn.outsideonline.com/wp-content/uploads/2017/12/13/kids-running-raising-rippers_h.jpg')
# from TextBot import  TextBot
# bot= TextBot()
# bot.run_pipeline('https://www.youtube.com/watch?v=HSh6EgMwMOY&ab_channel=CNBC')

from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

print(image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png"))

