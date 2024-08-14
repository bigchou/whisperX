dependencies = ['torch', 'torchaudio']
import torch
import os
import json
from utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       drop_chunks,
                       Validator,
                       OnnxWrapper)


def versiontuple(v):
    return tuple(map(int, (v.split('+')[0].split("."))))


def silero_vad(onnx=False, force_onnx_cpu=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    if not onnx:
        installed_version = torch.__version__
        supported_version = '1.12.0'
        if versiontuple(installed_version) < versiontuple(supported_version):
            raise Exception(f'Please install torch {supported_version} or greater ({installed_version} installed)')

    model_dir = os.path.join(os.path.dirname(__file__), 'files')

    if onnx:
        model = OnnxWrapper(os.path.join(model_dir, 'silero_vad.onnx'), force_onnx_cpu)
    else:
        model = init_jit_model(os.path.join(model_dir, 'silero_vad.jit'))
    utils = (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks)
    #return model, utils
    model.read_audio = read_audio
    return model
