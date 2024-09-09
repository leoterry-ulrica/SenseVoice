#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
from utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"
input_file = (
    "C:\\Users\\75423\\Desktop\\882d7492-e1b8-4a1b-bab7-07e3a0e00b3d.mp3"
    #"https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
)

res = model.generate(
    input=input_file,
    cache={},
    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    ## 标点符号恢复
    use_itn=True,
)
print(f'origin text: {res[0]["text"]}')
text = rich_transcription_postprocess(res[0]["text"])

print(text)
