import discord
import os 
from whispercpp import Whisper
from discord.ext import commands
from discord.ext import voice_recv
from discord.ext import tasks
import asyncio
from discord.ext.voice_recv import FFmpegSink
from discord.utils import MISSING, SequenceProxy
import Levenshtein
from discord.ext import voice_recv
from transformers import pipeline
import os 
import json
import datetime
from typing import Callable, Optional, Any, IO, Sequence, Tuple, Generator, Union, Dict, List
import numpy as np
import samplerate
import pyaudio
import io
import time
import pydub
import scipy
from dotenv import load_dotenv

load_dotenv()
APPLICATION_ID = os.getenv('APPLICATION_ID')
PUBLIC_KEY = os.getenv('PUBLIC_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

SCORE_FILE = './score_data'
whisper_instance = Whisper('base')
pipe = pipeline("text-generation", model="google/gemma-2-2b-it", token = HUGGINGFACE_TOKEN, max_new_tokens = 100)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    channel = discord.utils.get(client.get_all_channels(), name='General')
    voice_channel = client.get_channel(channel.id)
    vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient, timeout = 60, reconnect = True)
    sink = voice_recv.BasicSink(callback_listen_voice)
    vc.listen(sink)

history_audio = b''
def callback_listen_voice(user, data):
    global history_audio
    history_audio += data.pcm
    print ('hisotry_audio', len(history_audio))
    if len(history_audio) > 300000:
        tmp_audio = history_audio
        history_audio = b''
        audio_data = np.frombuffer(tmp_audio, dtype=np.int32, offset=0).astype(np.int16)
        np_data = audio_data.astype(np.float32) * (1 / 32768.0) # 2^15
        np_data = samplerate.resample(np_data, 1.0 / 3, 'sinc_best')
        original_text_list = audio2text(np_data)

        original_text = "".join(original_text_list)
        print (f"original_text: {original_text}\noriginal_text_list: {original_text_list}")
        correct_text_list = [textCorrect(text) for text in original_text_list]
        print (f'correct_text_list: {correct_text_list}')
        correct_text = "".join(correct_text_list)
        print (f"correct_text: {correct_text}")

        output_list = [edit_display(original_text_list[i], correct_text_list[i]) for i in range(len(correct_text_list))]
        
        # print (f"output_list: {output_list}")
        output_text = "".join(output_list)
        print (f'output_text: {output_text}')
        
        channel = discord.utils.get(client.get_all_channels(), name='speaking')
        channel = client.get_channel(channel.id)
        channel.send(f"orginal: {original_text}\n\ncorrect: {correct_text}\n\nupdate: {output_text}")
        # await update_score(original_text_list, correct_text_list, message)

        # now_second = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        # scipy.io.wavfile.write(f'./voice_data/recoding_2_{now_second}.wav', 48000, audio_data)


async def audio2text(file_name):
    result = whisper_instance.transcribe(file_name)
    text = whisper_instance.extract_text(result)
    text = [x.strip(' ') for x in text]
    return text

def audio2text(file_name):
    result = whisper_instance.transcribe(file_name)
    text = whisper_instance.extract_text(result)
    text = [x.strip(' ') for x in text]
    return text

def textCorrect(text):
    messages = [
        {"role": "user", "content": f"Assuming you are a senior IELTS speaking examiner, please help me to correct the grammatical mistakes, tense error or other simple mistaks in the following sentence, just output correct sentence, not any other extra words. \n\n{text}"},
    ]
    print (f'messages: {messages}')
    output = pipe(messages)
    content = 'default'
    for every_dick in output[0]['generated_text']:
        if every_dick['role'] == 'assistant':
            content = every_dick['content'].strip('\n').strip(' ')
    return str(content)


async def textCorrect(text):
    messages = [
        {"role": "user", "content": f"Assuming you are a senior IELTS speaking examiner, please help me to correct the grammatical mistakes, tense error or other simple mistaks in the following sentence, just output correct sentence, not any other extra words. \n\n{text}"},
    ]
    print (f'messages: {messages}')
    output = pipe(messages)
    content = 'default'
    for every_dick in output[0]['generated_text']:
        if every_dick['role'] == 'assistant':
            content = every_dick['content'].strip('\n').strip(' ')
    return content

async def edit_display(raw, res):
    edit_operation = Levenshtein.editops(raw, res)
    if raw == res:
        output = raw 
        return output
    output = raw[: edit_operation[0][1]]
    history_pre = ''
    for i in range(len(edit_operation)):
        raw_index = edit_operation[i][1]
        if edit_operation[i][0] in ('delete', 'replace'): # delete/replace
            symbol = raw[edit_operation[i][1]]
            if symbol.isdigit() or symbol.isalpha():
                output += "~~{}~~".format(symbol)
            else:
                output += symbol
            raw_index += 1
        if edit_operation[i][0] in ('replace', 'insert'):
            symbol = res[edit_operation[i][2]]
            if symbol.isdigit() or symbol.isalpha():
                history_pre += "**__{}__**".format(symbol)
            else:
                history_pre += symbol
        if i < len(edit_operation) - 1 and raw_index < edit_operation[i + 1][1]:
            output += history_pre + raw[raw_index: edit_operation[i + 1][1]]
            history_pre = ''
    output += history_pre + raw[edit_operation[len(edit_operation) - 1][1] + 1: ]
    output = output.replace('__****__', '')
    output = output.replace('~~~~', '')
    return output


def edit_display(raw, res):
    edit_operation = Levenshtein.editops(raw, res)
    if raw == res:
        output = raw 
        return output
    output = raw[: edit_operation[0][1]]
    history_pre = ''
    for i in range(len(edit_operation)):
        raw_index = edit_operation[i][1]
        if edit_operation[i][0] in ('delete', 'replace'): # delete/replace
            symbol = raw[edit_operation[i][1]]
            if symbol.isdigit() or symbol.isalpha():
                output += "~~{}~~".format(symbol)
            else:
                output += symbol
            raw_index += 1
        if edit_operation[i][0] in ('replace', 'insert'):
            symbol = res[edit_operation[i][2]]
            if symbol.isdigit() or symbol.isalpha():
                history_pre += "**__{}__**".format(symbol)
            else:
                history_pre += symbol
        if i < len(edit_operation) - 1 and raw_index < edit_operation[i + 1][1]:
            output += history_pre + raw[raw_index: edit_operation[i + 1][1]]
            history_pre = ''
    output += history_pre + raw[edit_operation[len(edit_operation) - 1][1] + 1: ]
    output = output.replace('__****__', '')
    output = output.replace('~~~~', '')
    return output


async def update_score(original_text_list, correct_text_list, message):
    user = message.author.name
    dura = message.attachments[0].duration
    day = str(message.created_at.strftime('%Y%m%d'))
    print (f'day: {day}')
    edit_list = [Levenshtein.distance(original_text_list[i], correct_text_list[i]) for i in range(len(correct_text_list))]
    len_list = [len(text) for text in original_text_list]
    error_rate = sum(edit_list) / sum(len_list)
    with open(SCORE_FILE, 'r') as load_f:
        score_dict = json.load(load_f)
    if user not in score_dict:
        score_dict[user] = {}
    if day not in score_dict[user]:
        score_dict[user][day] = (0, 0)
    score_dict[user][day] = (score_dict[user][day][0] + dura, 
        (score_dict[user][day][0] * score_dict[user][day][1] + dura * error_rate) / (score_dict[user][day][0] + dura))
    with open(SCORE_FILE, "w") as f:
        json.dump(score_dict, f, ensure_ascii=False)

async def get_score(message):
    user = message.author.name
    with open(SCORE_FILE, 'r') as load_f:
        score_dict = json.load(load_f)
    if user not in score_dict:
        await message.reply("There is not your recording")
        return 
    output_list = []
    for day, value in score_dict[user].items():
        minute = '{:.2f}min'.format(value[0] / 60)
        error_rate = '{:.2f}%'.format(value[1] * 100)
        output_list.append(f'learn: {minute}, error_rate: {error_rate} on {day}')
    await message.reply('\n'.join(output_list))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content == 'close bot':
        print (f'close bot')
        await client.close()
        return
    if message.content in ('hi bot', 'Hi bot'):
        await message.reply("I'm here")
    print ('message.content')
    print (message.content)
    print (message.content in ('score', 'Score'))
    if message.content in ('score', 'Score') and message.channel.name == 'speaking':
        await get_score(message)

    print (f'some one sent {message.content}')
    message_len = len(message.attachments)

    # audio 
    if message_len >= 1 and message.attachments[0].content_type == 'audio/ogg' \
            and message.channel.name == 'speaking':
        file_name = "/Users/wa007/code/voiceCorrection/ogg.ogg"
        await message.attachments[0].save(file_name)
        original_text_list = await audio2text(file_name)
        original_text = "".join(original_text_list) 
        print (f"original_text: {original_text}")
        if len(original_text) <= 7 and ('score' in original_text or 'Score' in original_text):
            await get_score(message)

        correct_text_list = [await textCorrect(text) for text in original_text_list]
        # print (f'correct_text_list: {correct_text_list}')
        correct_text = "".join(correct_text_list)
        print (f"correct_text: {correct_text}")

        output_list = [await edit_display(original_text_list[i], correct_text_list[i]) for i in range(len(correct_text_list))]
        
        # print (f"output_list: {output_list}")
        output_text = "".join(output_list)
        print (f'output_text: {output_text}')
        
        await message.reply(f"orginal: {original_text}\n\ncorrect: {correct_text}\n\nupdate: {output_text}")
        await update_score(original_text_list, correct_text_list, message)
        print ('\n\n\n')


client.run(DISCORD_TOKEN)
