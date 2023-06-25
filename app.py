from flask import Flask, request
from TranscribeAI import TranscribeAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
bot = TranscribeAI()


@app.route('/', methods=['POST'])
def transcribe_summary():
    url = request.json['url']
    return bot.run_pipeline(url)


@app.route('/imagegen', methods=['POST'])
def generate_image():
    text = request.json['text']
    return bot.generate_image(text)


@app.route('/img2img', methods=['POST'])
def img_2_img():
    text = request.json['text']
    url = request.json['url']
    return bot.img_2_img(text, url)


@app.route('/aiupscaler', methods=['POST'])
def upscale_image():
    # text = request.json['text']
    url = request.json['url']
    print(url)
    return bot.upscale_image(None, url)


@app.route('/videogen', methods=['POST'])
def generate_video():
    text = request.json['text']
    return bot.generate_video(text)


@app.route('/ask_bot', methods=['POST'])
def hello():
    question = request.json['question']
    context = request.json['context']
    return bot.ask_qa_bot(question,context)
