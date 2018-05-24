#!/usr/bin/env python
import audioop
import aiohttp
import base64
import json
import math
import multiprocessing
import os
import requests
import subprocess
import tempfile
import ujson
import wave

from autosub.constants import (
    LANGUAGE_CODES, GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL,
)
from autosub.formatters import FORMATTERS

DEFAULT_SUBTITLE_FORMAT = 'vtt'
DEFAULT_CONCURRENCY = int(os.environ.get('SKPATION_CORES', 4))
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'en'
EXECUTABLE = os.environ.get('FFMPEG_PATH', 'ffmpeg')


def percentile(arr, percent):
    arr = sorted(arr)
    k = (len(arr) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return arr[int(k)]
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    return d0 + d1


class FLACConverter(object):
    def __init__(self, source_path, include_before=0.25, include_after=0.25):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after

    def __call__(self, region):
        try:
            start, end = region
            start = max(0, start - self.include_before)
            end += self.include_after
            temp = tempfile.NamedTemporaryFile(suffix='.flac')
            command = [EXECUTABLE, "-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", temp.name]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            audio = temp.read()
            return base64.b64encode(audio)

        except KeyboardInterrupt:
            return


class SpeechRecognizer(object):
    def __init__(self, language="en", rate=44100, retries=3, api_key=GOOGLE_SPEECH_API_KEY):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries
        self.url = GOOGLE_SPEECH_API_URL + '?key={}'.format(self.api_key)


    def extract(self, response):
        for key, value in response.items():
            try:
                alt = value[0]['alternatives'][0]
                line = alt['transcript']
                confidence = alt['confidence']
                return line[:1].upper() + line[1:], confidence
            except Exception as e:
                # no result
                continue

    async def fetch(self, audio):
        body = {
            "audio": {
                "content": audio
            },
            "config": {
                "enableAutomaticPunctuation": True,
                "encoding": "FLAC",
                "languageCode": "en-US"
            }
        }
        async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
            async with session.post(self.url, json=body) as response:
                return await response.json()

def which(program):
    def is_exe(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def extract_audio(filename, channels=1, rate=16000, s3=True):
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    if not s3:
        if not os.path.isfile(filename):
            print("The given file does not exist: {0}".format(filename))
            raise Exception("Invalid filepath: {0}".format(filename))
        if not which(EXECUTABLE):
            print("ffmpeg: Executable not found on machine.")
            raise Exception("Dependency not found: ffmpeg")
    command = [EXECUTABLE, "-y", "-i", filename, "-ac", str(channels), "-ar", str(rate), "-loglevel", "error",
               temp.name]
    use_shell = True if os.name == "nt" else False
    subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
    return temp.name, rate


def find_speech_regions(filename, frame_width=4096, min_region_size=0.5, max_region_size=6):
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
    rate = reader.getframerate()
    n_channels = reader.getnchannels()
    chunk_duration = float(frame_width) / rate

    n_chunks = int(math.ceil(reader.getnframes() * 1.0 / frame_width))
    energies = []

    for i in range(n_chunks):
        chunk = reader.readframes(frame_width)
        energies.append(audioop.rms(chunk, sample_width * n_channels))

    threshold = percentile(energies, 0.2)

    elapsed_time = 0

    regions = []
    region_start = None

    for energy in energies:
        is_silence = energy <= threshold
        max_exceeded = region_start and elapsed_time - region_start >= max_region_size

        if (max_exceeded or is_silence) and region_start:
            if elapsed_time - region_start >= min_region_size:
                regions.append((region_start, elapsed_time))
                region_start = None

        elif (not region_start) and (not is_silence):
            region_start = elapsed_time
        elapsed_time += chunk_duration
    return regions


async def generate_subtitles(
        audio_filename,
        concurrency=DEFAULT_CONCURRENCY,
        src_language=DEFAULT_SRC_LANGUAGE,
        verbose=False,
        api_key=GOOGLE_SPEECH_API_KEY,
):
    regions = find_speech_regions(audio_filename)
    is_parallel = concurrency > 0
    if is_parallel:
        pool = multiprocessing.Pool(concurrency)
    converter = FLACConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer(language=src_language,
                                  api_key=api_key)

    transcripts = []
    confidences = []
    if regions:
        try:
            extracted_regions = []
            if is_parallel:
                for i, extracted_region in enumerate(pool.imap(converter, regions)):
                    extracted_regions.append(extracted_region)

                for i, audio in enumerate(extracted_regions):
                    response = await recognizer.fetch(audio)
                    if response:
                        transcript, confidence = recognizer.extract(response)
                        transcripts.append(transcript)
                        confidences.append(confidence)
                    else:
                        transcripts.append('')
                        confidences.append(0)
            else:
                for region in regions:
                    caption, confidence = recognizer(converter(region))
                    if verbose:
                        print(caption)
                    transcripts.append(caption)
                    confidences.append(confidence)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print("Cancelling transcription")
            raise

    timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
    timed_confidences = [(r, c) for r, c in zip(regions, confidences) if c]
    pool.close()
    return timed_subtitles, timed_confidences

def to_subs(timed_subtitles, subtitle_file_format=DEFAULT_SUBTITLE_FORMAT):
    formatter = FORMATTERS.get(subtitle_file_format)
    return formatter(timed_subtitles)

def persist_subtitles(timed_subtitles, output):
    formatted_subtitles = to_subs(timed_subtitles)

    with open(output, 'wb') as f:
        f.write(formatted_subtitles.encode("utf-8"))

    return output
