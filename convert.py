import sys
import json

jsonpath = '0524.json'
jsonpath = sys.argv[1]
with open(jsonpath) as f: data = json.load(f)



def seconds_to_hmsms(seconds):
    # 计算总小时、分钟、秒数和毫秒数
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    # 格式化输出为 00:00:00:00 格式
    return f"{hours:02}:{minutes:02}:{secs:02}:{milliseconds:03}"


results = []
tmp = data['segments'][0]['words'][0]

i = -1
for item in data['segments']:
    for word in item['words']:
        i+=1
        if i == 0:
            continue
        
        if 'speaker' not in word or word['speaker'] == tmp['speaker']:
            tmp['word'] += word['word']
            if 'start' in word:
                tmp['start'] = min(tmp['start'], word['start'])
            if 'end' in word:
                tmp['end'] = max(tmp['end'], word['end'])
        else:
            results.append(tmp)
            tmp = word
print("DONE")
#import pdb; pdb.set_trace()

out = []
for result in results:
    speaker = result['speaker']
    start = seconds_to_hmsms(result['start'])
    end = seconds_to_hmsms(result['end'])
    text = result['word']
    out.append("%s [%s - %s]: %s"%(speaker, start, end, text))

out = "\n".join(out)
with open('%s_merge.txt'%(jsonpath), 'w') as f:
    f.write(out)
        
