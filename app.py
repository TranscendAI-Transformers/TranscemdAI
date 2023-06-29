from flask import Flask, request, make_response
from TranscribeAI import TranscendAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
bot = TranscendAI()


@app.route('/', methods=['POST'])
def transcribe_summary():
    url = request.json['url']
    t_only = request.json['tOnly']
    try:
        return bot.run_pipeline(url, t_only)
    except Exception as ex:
        print(ex)
        return make_response('Pipeline Error', 400)


@app.route('/imagegen', methods=['POST'])
def generate_image():
    text = request.json['text']
    return bot.generate_image(text)


@app.route('/img2img', methods=['POST'])
def img_2_img():
    try:
        text = request.json['text']
        url = request.json['url']
        return bot.img_2_img(text, url)
    except Exception as ex:
        print(ex)
        return make_response('Image Url Error', 400)


@app.route('/aiupscaler', methods=['POST'])
def upscale_image():
    try:
        url = request.json['url']
        print(url)
        return bot.upscale_image(url)
    except Exception as ex:
        print(ex)
        return make_response('Image Url Error', 400)


@app.route('/videogen', methods=['POST'])
def generate_video():
    text = request.json['text']
    return bot.generate_video(text)


@app.route('/ask_bot', methods=['POST'])
def hello():
    question = request.json['question']
    context = request.json['context']
    return bot.ask_qa_bot(question, context)


@app.route("/classify", methods=['POST'])
def classify():
    try:
        url = request.json['url']
        print(url)
        return bot.classify(url)
    except Exception as ex:
        print(ex)
        return make_response('Image Url Error', 400)


@app.route("/yolo", methods=['POST'])
def yolo():
    try:
        url = request.json['url']
        print(url)
        return bot.yolo(url)
    except Exception as ex:
        print(ex)
        return make_response('Image Url Error', 400)


@app.route('/text_generation', methods=['POST'])
def text_generation():
    try:
        text = request.json['text']
        multiple = request.json['multiple']
        return bot.text_generation(text, multiple)
    except Exception as ex:
        print(ex)
        return make_response('Text Generation Error', 400)


@app.route('/summary', methods=['POST'])
def summary():
    try:
        return bot.summary(request.json['text'])
    except Exception as e:
        return make_response('Pipeline Error', 400)


@app.route('/image_caption', methods=['POST'])
def image_caption():
    try:
        url = request.json['url']
        return bot.image_caption(url)
    except Exception as e:
        return make_response('Image Url Error', 400)


@app.route('/image_qa', methods=['POST'])
def image_qa():
    try:
        url = request.json['url']
        text = request.json['text']
        return bot.image_qa(url, text)
    except Exception as e:
        make_response('Invalid Url Error', 400)
