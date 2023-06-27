from transformers import pipeline, set_seed
import os
from pytube import YouTube
import json
import torch
import math


class TextBot:

    def __init__(self):
        self.transcribe_model = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", device=0, framework="pt")
        self.summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0,
                                         framework="pt")
        self.nlp = pipeline('question-answering', model="deepset/roberta-large-squad2",
                            tokenizer="deepset/roberta-large-squad2", device=0, framework="pt")
        self.text_generator = pipeline('text-generation', model='gpt2-large', device=0)
        self.url = None
        self.title = None
        self.res = {}
        self.yt = None
        self.audio_format = '.wav'
        self.output_format = '.json'
        self.temp_location = './temp/'
        torch.cuda.empty_cache()

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

    def transcribe_audio(self):
        new_file = self.download_video()
        text = self.transcribe_model(new_file, chunk_length_s=60)['text']
        self.res['text'] = text.lower()
        print('transcribe done')
        torch.cuda.empty_cache()

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

    def summary(self,text):
        prev = 0

        summary = []
        if len(text) > 3000:
            batches = math.ceil(len(text) / 3000)
            for i in range(1, batches + 1):
                t_sum = self.summarizer_model(text[prev:i * 3000], max_length=150, min_length=30, do_sample=False)
                summary.append(t_sum[0]['summary_text'].lower())
                prev += 3000
        else:
            t_sum = self.summarizer_model(text, max_length=150, min_length=30, do_sample=False)
            summary.append(t_sum[0]['summary_text'].lower())
        print('summarization done')
        torch.cuda.empty_cache()
        return summary

    def store_output(self):
        with open(self.temp_location + self.title + self.output_format, "w") as outfile:
            json.dump(self.res, outfile)

    def ask_qa_bot(self, question, context):
        qa_input = {
            'question': question,
            'context': context
        }
        ans = self.nlp(qa_input)
        torch.cuda.empty_cache()
        return ans

    def text_generation(self, text, multiple):
        num = 1
        if multiple:
            num = 3
        resp = self.text_generator(text, max_length=150, num_return_sequences=num, top_k=0,
                                   temperature=0.8, do_sample=True, )
        torch.cuda.empty_cache()
        return resp

    def run_pipeline(self, url, t_only):
        self.process_url(url)
        cache_hit = self.check_cache()
        if cache_hit is not None:
            return cache_hit
        self.transcribe_audio()
        if not t_only:
            self.generate_summary()
        self.store_output()
        torch.cuda.empty_cache()
        return self.res
