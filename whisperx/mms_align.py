from uroman import uroman # pip install uroman-python
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
from .audio import SAMPLE_RATE
from typing import Iterable, Union, List
from .types import AlignedTranscriptionResult, SingleSegment, SingleAlignedSegment, SingleWordSegment
import numpy as np

def tokenize_for_mer(text):
    # https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/utils.py#L22
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def normalize_uroman(text, verbose=True, enable_uroman=True):
    # modified from https://pytorch.org/audio/2.1.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html#chinese
    # tokenize the plain text
    token_list = tokenize_for_mer(text)
    token_text = " ".join(token_list)

    # romanize the token text
    if enable_uroman: roman_text = uroman(token_text)

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


def load_mms_fa(model_type=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == 0:
        print("Ours")
        model = bundle.get_model().to(device)
        dictionary = bundle.get_dict()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
    elif model_type == 1:
        print("CTC-ForcedAligner")
        dtype=torch.float16 if device == 'cuda' else torch.float32
        model, tokenizer, dictionary = load_alignment_model(device, dtype=dtype)
        aligner = None
    else:
        raise NotImplementedError
    return {
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
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = model_args['dictionary']

    # 1. Preprocess
    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        # split into words and romans
        norm_roman, norm_token = normalize_uroman(segment["text"])
        per_word = norm_roman.split(" ")
        text = norm_token.split(" ")

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
        segment["norm_text"] = norm_token
        segment["norm_roman"] = norm_roman
    
    
    



    

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

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]

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
            
        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 -t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
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

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1
            
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
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
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








class Aligner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The model instantiated by MMS_FA’s get_model() method by default includes the feature dimension for <star> token. You can disable this by passing with_star=False.
        self.model = bundle.get_model()
        self.model.to(self.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.split_chunk_size_in_seconds = 60 * 10
        assert 16000 == bundle.sample_rate, "please check the bundle.sample_rate"
    
    def compute_alignments(self, waveform: torch.Tensor, transcript: List[str]):
        with torch.inference_mode():
            # e.g.1 transcript = ['i', 'had', 'that', 'curiosity', 'beside', 'me', 'at', 'this', 'moment']
            # e.g.2 ['li', 'si', 'da', 'biao', 'shi', 'zi', 'ji', 'gen', 'zhou', 'yun', 'lu', 'bing', 'mei', 'you', 'shen', 'chou', 'da', 'hen']
            tmp = self.tokenizer(transcript) # Tokens(tmp) are numerical expression of transcripts.
            # e.g.1 tmp = [[2], [15, 1, 13], [7, 15, 1, 7], [20, 6, 9, 2, 5, 8, 2, 7, 16], [17, 3, 8, 2, 13, 3], [10, 3], [1, 7], [7, 15, 2, 8], [10, 5, 10, 3, 4, 7]]
            # e.g.2 tmp = [[12, 2], [8, 2], [13, 1], [17, 2, 1, 5], [8, 15, 2], [23, 2], [22, 2], [14, 3, 4], [23, 15, 5, 6], [16, 6, 4], [12, 6], [17, 2, 4, 14], [10, 3, 2], [16, 5, 6], [8, 15, 3, 4], [20, 15, 5, 6], [13, 1], [15, 3, 4]]
            #pdb.set_trace()

            emission, _ = self.model(waveform.to(self.device)) # obtain emission
            # Emission reperesents the frame-wise probability distribution over tokens, and it can be obtained by passing waveform to an acoustic model.

            token_spans = self.aligner(emission[0], tmp)
            # aligment process takes emission and token sequences and outputs timestaps of the tokens and their scores.
            # the alignment result (token_spans) is expressed in the frame cordinate of the emission, which is different from the original waveform.
            # note that token_spans return the word-level alignments. you can extract token-level & frame-level alignemtns also.
        return emission, token_spans

    def _score(self, spans):
        # Compute average score weighted by the span length
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)
    
    def forward(self, audio_path, transcript, human_utt_list, debug=False):
        # load normalized audio (16khz mono)
        waveform, sample_rate, duration_in_seconds = load_norm_audio(audio_path)
        assert sample_rate == 16000, "sample rate can be 16khz only"

        # tokenize the transcript
        norm_roman, norm_token = normalize_uroman(transcript, verbose=False)
        norm_roman_list = norm_roman.split()
        norm_token_list = norm_token.split()
        assert len(norm_roman_list) == len(norm_token_list), "ERROR: inconsistent length between roman and token"

        # split audio
        split_seconds_list = [0.0]
        audio_max_length = duration_in_seconds
        norm_roman_list_list = []
        norm_token_list_list = []
        if duration_in_seconds > self.split_chunk_size_in_seconds:
            print("split audio cuz audio is too long.")
            #squeeze_waveform = waveform.squeeze(0)
            #speech_timestamps = self.vad.get_speech_timestamps(squeeze_waveform, self.vad.model, sampling_rate=16000)
            #seg_list = [ (i['start']/16000, i['end']/16000) for i in speech_timestamps]

            # add audio_max_length to save VRAM
            audio_max_length = human_utt_list[-1][1] + 10.0 if human_utt_list[-1][1] + 10.0 < duration_in_seconds else duration_in_seconds
            
            tmp = len(split_seconds_list) * self.split_chunk_size_in_seconds
            for curr_utt, next_utt in zip(human_utt_list[:-1], human_utt_list[1:]):
                curr_utt_start = curr_utt[0]
                curr_utt_end = curr_utt[1]
                next_utt_start = next_utt[0]
                if curr_utt_start < tmp < next_utt_start:
                    split_seconds_list.append( (curr_utt_end + next_utt_start) / 2.0 )
                    tmp = len(split_seconds_list) * self.split_chunk_size_in_seconds
            split_seconds_list.append(audio_max_length)

            #pdb.set_trace()
            
            num = 1
            tmp_norm_roman_list = []
            tmp_norm_token_list = []
            counter = 0
            for utt in human_utt_list:
                start, end, text = utt
                _, utt_norm_token = normalize_uroman(text, verbose=False, enable_uroman=False)
                length = len(utt_norm_token.split())
                if start < split_seconds_list[num]:
                    tmp_norm_roman_list.extend(norm_roman_list[counter: counter+length])
                    tmp_norm_token_list.extend(norm_token_list[counter: counter+length])
                else:
                    norm_roman_list_list.append(tmp_norm_roman_list)
                    norm_token_list_list.append(tmp_norm_token_list)
                    tmp_norm_roman_list = norm_roman_list[counter: counter+length]
                    tmp_norm_token_list = norm_token_list[counter: counter+length]
                    num += 1
                counter += length
            if len(tmp_norm_roman_list):
                norm_roman_list_list.append(tmp_norm_roman_list)
                norm_token_list_list.append(tmp_norm_token_list)
            #pdb.set_trace()
            #print("=====================")
        else:
            norm_roman_list_list = [norm_roman_list]
            norm_token_list_list = [norm_token_list]
            split_seconds_list.append(audio_max_length)
        
        

        tmp_seg_list = []
        tmp_align_seg_list = []
        for i, (sub_norm_roman_list, sub_norm_token_list) in enumerate(zip(norm_roman_list_list, norm_token_list_list)):
            if len(norm_roman_list_list) == 1:
                sub_waveform = waveform
            else:
                start = int(16000 * split_seconds_list[i])
                end = int(16000 * split_seconds_list[i+1])
                sub_waveform = waveform[:, start:end]
            # force alignment
            emission, token_spans = self.compute_alignments(sub_waveform, sub_norm_roman_list)
            # format the output
            for t_spans, chars in zip(token_spans, sub_norm_token_list):
                num_frames = emission.size(1)
                num_points = sub_waveform.size(1)
                ratio = num_points / num_frames / 16000.0
                t0, t1 = t_spans[0].start * ratio, t_spans[-1].end * ratio
                t0, t1 = t0 + split_seconds_list[i], t1 + split_seconds_list[i]
                t0, t1, chars = float(t0), float(t1), str(chars)
                tmp_seg_list.append((t0, t1, chars))
                tmp_align_seg_list.append((t0, t1, self._score(t_spans), chars))
        seg_list = tmp_seg_list
        align_seg_list = tmp_align_seg_list

        # output
        if debug:
            debug_dir = 'debug'
            if not os.path.isdir(debug_dir): os.mkdir(debug_dir)
            prefix = os.path.splitext(os.path.basename(audio_path))[0]
            save_audacity_label(os.path.join(debug_dir, '%s_audacity.txt'%(prefix)), seg_list)
            save_praat_textgrid(os.path.join(debug_dir, '%s_praat.TextGrid'%(prefix)), seg_list, duration_in_seconds)
        return align_seg_list
    
    def group(self, align_seg_list, human_utt_list):
        align_utt_list = []
        i0, i1 = 0, 0

        for human_utt in human_utt_list:
            _, norm_token = normalize_uroman(human_utt[-1], verbose=True, enable_uroman=False)
            norm_token_list = norm_token.split()
            i1 = i0 + len(norm_token_list) # set the end point
            # obtain utterance timing
            start_second = align_seg_list[i0][0]
            end_second = align_seg_list[i1-1][1]
            # score utterance according to CTC-Segmentation
            # align_seg_list[1671] = (459.7517410496066, 460.1517512649889, 0.12133800704032183, 'appq')
            utt_score = min([seg[2] for seg in align_seg_list[i0:i1]])
            text_token_list = [seg[3] for seg in align_seg_list[i0:i1]]
            # 修正assertion (2024.05.26 01:55)
            assert norm_token_list == text_token_list, "inconsistent token length between the original one and restored one"
            # if norm_token_list != text_token_list:
            #     pdb.set_trace()
            #     print("something wrong")
            text = ''.join([seg[3] for seg in align_seg_list[i0:i1]])
            #align_utt_list.append((start_second, end_second, utt_score, text))
            align_utt_list.append({
                'start' : start_second,
                'end' : end_second,
                'score' : utt_score,
                'text' : text
            })
            i0 = i1 # set the next start point
        assert len(align_utt_list) == len(human_utt_list), "inconsistent utterance length: #align(%d) v.s. #human(%d)"%(len(align_utt_list), len(human_utt_list))
        return align_utt_list