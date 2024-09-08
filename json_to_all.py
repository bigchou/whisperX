from whisperx.utils import get_writer
import json, os
import argparse


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

def optional_int(string):
    return None if string == "None" else int(string)

parser = argparse.ArgumentParser()
parser.add_argument("jsonfile", type=str, help="input json file")
parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
parser.add_argument("--highlight_words", type=str2bool, default=True, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
args = parser.parse_args()


output_format = args.output_format
output_dir = args.output_dir
writer = get_writer(output_format, output_dir)
word_options = {
    "highlight_words": args.highlight_words,
    "max_line_count": args.max_line_count,
    "max_line_width": args.max_line_width
}
writer_args = {arg: word_options[arg] for arg in word_options}
with open(args.jsonfile) as f:
    result = json.load(f)

audio_path = os.path.splitext(args.jsonfile)[0]+'.mp4'
writer(result, audio_path, writer_args)
