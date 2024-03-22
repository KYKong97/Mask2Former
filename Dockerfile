FROM tverous/pytorch-notebook:base

WORKDIR /app
COPY . .


RUN pip install --progress-bar off 'git+https://github.com/facebookresearch/detectron2.git' \
    'git+https://github.com/cocodataset/panopticapi.git' \
    'git+https://github.com/mcordts/cityscapesScripts.git' \
    opencv-python-headless timm
RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
