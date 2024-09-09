# 使用Python 3.10的基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有内容到容器的/app目录下
COPY . /app

# 更新包列表并安装ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# 设置为清华同衡镜像源
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 暴露端口
EXPOSE 8000

# 运行脚本
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
