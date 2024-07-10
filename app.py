from typing import Union
import uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from funasr import AutoModel
from utils.postprocess_utils import rich_transcription_postprocess
import os
from fastapi.middleware.cors import CORSMiddleware
#from whisper import tokenizer

#LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有源
    allow_credentials=True,
    allow_methods=["*"], # 允许所有HTTP方法
    allow_headers=["*"], # 允许所有HTTP头部
)
# 从环境变量中获取模型目录
model_dir = os.environ.get("MODEL_DIR", "/app/models/SenseVoiceSmall")
print(f"model dir: {model_dir}")
# Load the model
model_instance = AutoModel(
    model=model_dir,
    trust_remote_code=True,
)

# 创建临时文件夹
temp_dir = "./tmp/uploaded_files"
os.makedirs(temp_dir, exist_ok=True)

@app.post("/v1/audio/transcriptions", tags=["Audio"], summary="Create transcription")
async def transcribe( 
        file: UploadFile = File(...),
        model: Union[str, None] = Form(default="whisper-1"),
        language: Union[str, None] = Form(default="auto", enum=["zn", "en", "yue", "ja", "ko", "nospeech"]),
        prompt: Union[str, None] = Form(default=None, description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language."),
        temperature: float = Form(default=0, description="Defaults to 0.The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."),
        response_format: Union[str, None] = Form(default="json", enum=["text", "vtt", "srt", "tsv", "json", "verbose_json"])):

    # 生成唯一的文件名
    unique_id = uuid.uuid4()
    file_extension = os.path.splitext(file.filename)[1]
    temp_file_name = f"{unique_id}{file_extension}"
    temp_file_path = os.path.join(temp_dir, temp_file_name)
    # 保存文件到临时目录
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    # Read the file
    #contents = await file.read()

    # Generate transcription
    res = model_instance.generate(
        input=temp_file_path,
        cache={},
        language=language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
    )

    # Postprocess the transcription
    text = rich_transcription_postprocess(res[0]["text"])
    print(f"text: {text}")
    return JSONResponse(content={"text": text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)