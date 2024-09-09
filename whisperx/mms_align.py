#from uroman import uroman # pip install uroman-python



try:
    import uroman as ur
    uroman = ur.Uroman()
    uroman_type = 0
except:
    from uroman import uroman # Python 3.10.14
    uroman_type = 1
    # uroman == 1.3.1.1
    # uroman-python = 1.2.8.1

import re
from itertools import chain
import jieba
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle # It was trained with 23,000 hours of audio from 1100+ languages
from typing import List
#from lab.formatter import save_audacity_label, save_praat_textgrid
#import pysrt # pip install pysrt
import os, math
import pdb
import pandas as pd
from tqdm import tqdm
from .audio import SAMPLE_RATE, load_audio
from .alignment import Segment
from typing import Iterable, Union, List
from .types import AlignedTranscriptionResult, SingleSegment, SingleAlignedSegment, SingleWordSegment
import numpy as np
from .utils import interpolate_nans, ISO_639_1_TO_3

from ctc_forced_aligner import (
    #load_audio,
    load_alignment_model,
    generate_emissions,
    #preprocess_text,
    get_alignments,
    get_spans,
    #postprocess_results,
    text_normalize,
    get_uroman_tokens
)

def tokenize_for_mer(text):
    # https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/utils.py#L22
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def normalize_uroman(text, verbose=False, enable_uroman=True):
    # modified from https://pytorch.org/audio/2.1.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html#chinese
    # tokenize the plain text
    token_list = tokenize_for_mer(text)
    token_text = " ".join(token_list)

    # romanize the token text
    if enable_uroman:
        if uroman_type == 0:
            roman_text = uroman.romanize_string(token_text)
        else:
            roman_text = uroman(token_text)

    # 大寫轉小寫
    if enable_uroman: roman_text = roman_text.lower()
    token_text = token_text.lower()

    # 替換特殊引號
    if enable_uroman: roman_text = roman_text.replace("’", "'")
    token_text = token_text.replace("’", "'")

    # 僅保留小寫字母、空格和單引號，替換大寫字母和其他所有符號或標點
    # case 1. follow pytorch official document
    #if enable_uroman: roman_text = re.sub("([^a-z' ])", " ", roman_text)

    # case 2. insert star token according to the original paper
    if enable_uroman:
        roman_text = re.sub("([^a-z' ])", "*", roman_text)
        roman_text = re.sub('\*+', '*', roman_text)# 合併多個star token為一個star token

    # 合併多個空格為一個空格
    if enable_uroman: roman_text = re.sub(' +', ' ', roman_text)
    token_text = re.sub(' +', ' ', token_text)

    # 去除首尾空格
    if enable_uroman: roman_text = roman_text.strip()
    token_text = token_text.strip()

    # check results
    if enable_uroman:
        roman_cnt = len(roman_text.split())
        token_cnt = len(token_text.split())
        assert roman_cnt == token_cnt, "inconsistent length: #roman(%d) v.s. #token(%d)"%(roman_cnt, token_cnt)
        if verbose:
            print(roman_text)
            print(token_text)
        return roman_text, token_text
    else:
        return None, token_text

def load_norm_audio(audio_path):
    # read audio
    waveform, sample_rate = torchaudio.load(audio_path)
    # check sample_rate
    if sample_rate != 16000:
        # Resample audio to the expected sampling rate
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = 16000
    assert sample_rate == bundle.sample_rate
    # convert multi-ch to single-ch
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # get audio duration
    duration_in_seconds = waveform.shape[1] / sample_rate
    return waveform, sample_rate, duration_in_seconds

def check_model_name(model_name):
    model_name = model_name.lower()
    assert model_name in ["mms_fa", "ctc-forced-aligner"], f"model_name {model_name} is unsupported"

def load_mms_fa(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name.lower()
    check_model_name(model_name)
    if model_name == "mms_fa":
        print("Ours")
        model = bundle.get_model().to(device)
        dictionary = bundle.get_dict()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
    elif model_name == "ctc-forced-aligner":
        print("CTC-ForcedAligner")
        dtype=torch.float16 if device == 'cuda' else torch.float32
        model, tokenizer, dictionary = load_alignment_model(device, dtype=dtype)
        aligner = None
    else:
        raise NotImplementedError
    return {
        'model_name': model_name,
        'device': device,
        'model': model,
        'dictionary': dictionary,
        'tokenizer': tokenizer,
        'aligner': aligner
    }


def mms_align(
    transcript: Iterable[SingleSegment],
    model_args: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    language: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    assert language in ISO_639_1_TO_3, f"language {language} is unsupported"
    language = ISO_639_1_TO_3[language]
    
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_name = model_args['model_name']
    check_model_name(model_name)
    model_dictionary = model_args['dictionary']
    model = model_args['model']
    aligner = model_args['aligner']
    tokenizer = model_args['tokenizer']

    # 1. Preprocess
    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        if model_name == "mms_fa":
            # split into words and romans
            norm_roman, norm_token = normalize_uroman(segment["text"])
            per_word = norm_roman.split(" ")
            text = norm_token.split(" ")
        else:
            text_split = tokenize_for_mer(segment["text"])
            norm_text = [text_normalize(line.strip(), language) for line in text_split]
            tokens = get_uroman_tokens(norm_text, language)
            assert len(norm_text) == len(tokens), "length mismatch"

            tokens_starred = []
            [tokens_starred.extend(["<star>", token]) for token in tokens]
            text_starred = []
            [text_starred.extend(["<star>", chunk]) for chunk in text_split]

            segment["tokens_starred"] = tokens_starred
            segment["text_starred"] = text_starred

            per_word = tokens
            text = norm_text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            clean_char.append(char_)
            clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd]):
                clean_wdx.append(wdx)

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["norm_text"] = text
        segment["norm_roman"] = per_word

        segment["sentence_spans"] = [(0, len(per_word))]
    
    
    



    

    aligned_segments: List[SingleAlignedSegment] = []
    
    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        tokens = segment["norm_roman"]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]



        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None


        char_segments = []
        if model_name == "ctc-forced-aligner":
            emission, stride = generate_emissions(model, waveform_segment[0].to(model.dtype).to(device), batch_size=1)
            tokens_starred = segment["tokens_starred"]
            text_starred =  segment["text_starred"]
            # get_alignments 有機會撞到 segmentation fault... 可能C++ implementation 有問題
            segments, scores, blank_id = get_alignments(emission, tokens_starred, model_dictionary)
            spans = get_spans(tokens_starred, segments, tokenizer.decode(blank_id))
            #word_timestamps = postprocess_results(text_starred, spans, stride, scores)

            for i, chars in enumerate(text_starred):
                if chars == "<star>": continue
                seg_start_idx = spans[i][0].start
                seg_end_idx = spans[i][-1].end
                score = np.exp(scores[seg_start_idx: seg_end_idx].sum().item())
                char_segments.append(Segment(
                    chars,
                    seg_start_idx,
                    seg_end_idx,
                    score
                ))
            num_frames = emission.size(0)
        else:
            with torch.inference_mode():
                tmp = tokenizer(segment["norm_roman"])
                # obtain emission (the frame-wise probability over tokens)
                emission, _ = model(waveform_segment.to(device))
                # emission.shape = torch.Size([1, 1368, 29])
                token_spans = aligner(emission[0], tmp)
            for t_spans, chars in zip(token_spans, segment['norm_text']): 
                char_segments.append(Segment(
                    chars,
                    t_spans[0].start,
                    t_spans[-1].end,
                    sum(s.score * len(s) for s in t_spans) / sum(len(s) for s in t_spans),
                ))
            num_frames = emission.size(1)

        duration = t2 -t1
        ratio = duration * waveform_segment.size(0) / (num_frames)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(segment['norm_text']):
            start, end, score = None, None, None
            if cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )
            word_idx += 1

        #import pdb; pdb.set_trace()
            
        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx
        
            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable 
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)
            
            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        # if model_lang in LANGUAGES_WITHOUT_SPACES:
        #     agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}


if __name__ == "__main__":
    import json
    with open('before_align_results.json') as f:
        results = json.load(f)
    model_name = "mms_fa"
    model_name = "ctc-forced-aligner"
    check_model_name(model_name)
    align_args = load_mms_fa(model_name)
    device = 'cuda'
    interpolate_method = 'nearest'
    return_char_alignments = False
    print_progress = False

    language = "zh"
    tmp_results = results
    results = []
    for result, audio_path in tmp_results:
        input_audio = audio_path
        result = mms_align(result["segments"], align_args, input_audio, device, language, interpolate_method=interpolate_method, return_char_alignments=return_char_alignments, print_progress=print_progress)
        import pdb; pdb.set_trace()