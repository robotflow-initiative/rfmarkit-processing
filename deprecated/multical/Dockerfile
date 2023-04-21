FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Shanghai"

RUN apt-get update -y && \
    apt-get install libgl1-mesa-glx libxext-dev libxrender1 libglib2.0-dev python3 python3-pip -y

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install opencv-python-headless && \
    pip3 install multical

WORKDIR /input
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME /input

ENTRYPOINT ["/app/entrypoint.sh"]

