from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

MODEL_NAME = 'face_detection_yunet_2023mar.onnx'
MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
URLS = [
    f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/{MODEL_NAME}',
    f'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/{MODEL_NAME}',
]


def download_model() -> int:
    output = Path(__file__).resolve().parent / 'models' / MODEL_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == MODEL_SHA256:
        print('YuNet 模型已安装，校验通过。')
        return 0
    temporary = output.with_suffix('.onnx.part')
    for url in URLS:
        try:
            print('正在下载 OpenCV 官方 YuNet 模型……')
            request = urllib.request.Request(url, headers={'User-Agent': 'idphoto-batch-tool'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(5 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
                raise ValueError('模型校验失败，下载内容可能不完整或为 Git LFS 指针。')
            temporary.write_bytes(data)
            temporary.replace(output)
            print(f'YuNet 下载并校验完成：{output}')
            return 0
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            print(f'下载源失败：{exc}')
    print('下载失败，原有模型已保留。无可用模型时仍可使用 Haar 备用检测器。')
    return 1


if __name__ == '__main__':
    raise SystemExit(download_model())
