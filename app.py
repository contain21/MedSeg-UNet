from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import psutil
from datetime import datetime
from werkzeug.utils import secure_filename
from utils import load_model, process_image_full_pipeline, analyze_segmentation
import cv2
import json
import time
import hashlib
import hmac
import base64
import websocket  # 星火使用WebSocket协议
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 配置路径
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 模型配置
MODELS = {
    "UNet": "models/best_model_zengqiang.pth",
    "MultiResUNet": "models/best_multiresunet8015.pth",
    "DoubleU-Net": "models/best_doubleunet.pth",
    "Improved DoubleU-Net": "models/best_fuse.pth",
}


@app.route('/')
def home():
    return render_template('index.html', models=list(MODELS.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    # 记录开始时间和初始内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_mem = float(process.memory_info().rss / 1024 / 1024)  # MB

    # 1. 检查文件
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. 获取模型选择
    model_name = request.form.get('model', 'UNet')
    if model_name not in MODELS:
        return jsonify({"error": "Invalid model selected"}), 400

    # 3. 保存原始文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = secure_filename(f"upload_{timestamp}_{file.filename}")
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(upload_path)
    img = cv2.imread(upload_path)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(upload_path, img)  # 保存调整后的图片

    try:
        # 4. 加载模型并记录内存变化
        model_load_start = time.time()
        model = load_model(MODELS[model_name],model_name)
        model_load_time = round(time.time() - model_load_start, 2)
        model_load_mem = process.memory_info().rss / 1024 / 1024 - initial_mem

        # 5. 完整处理流程
        pipeline_start = time.time()
        result_files = process_image_full_pipeline(
            model=model,
            input_path=upload_path,
            output_dir=app.config['RESULT_FOLDER'],
            timestamp=timestamp
        )
        pipeline_time = round(time.time() - pipeline_start, 2)

        # 6. 分析分割结果
        seg_metrics = analyze_segmentation(result_files['seg_path'])

        # 计算总时间和内存使用
        total_time = float(round(time.time() - start_time, 2))
        final_mem = float(process.memory_info().rss / 1024 / 1024)
        mem_usage = float(round(final_mem - initial_mem, 2))

        # 7. 返回结果URL和指标
        return jsonify({
            "original_url": url_for('static', filename=f'uploads/{original_filename}'),
            "segmentation_url": url_for('static', filename=f'results/seg_{timestamp}.png'),
            "overlay_url": url_for('static', filename=f'results/overlay_{timestamp}.png'),
            "performance_metrics": {
                "total_processing_time": f"{total_time}秒",
                "model_loading": {
                    "time": f"{float(model_load_time)}秒",
                    "memory_increase": f"{float(model_load_mem)}MB"
                },
                "inference_time": f"{float(pipeline_time)}秒",
                "memory_usage": f"{mem_usage}MB",
                "image_resolution": "256×256像素"
            },
            "segmentation_metrics": seg_metrics,
            "model_info": {
                "name": model_name,
                "input_size": "256×256",
                "benchmark_metrics": {
                    "iou": 0.7939,
                    "dice": 0.86,
                    "accuracy": 0.92
                }
            }
        })

    except Exception as e:
        # 清理可能已创建的文件
        for f in [upload_path,
                  os.path.join(app.config['RESULT_FOLDER'], f"seg_{timestamp}.png"),
                  os.path.join(app.config['RESULT_FOLDER'], f"overlay_{timestamp}.png")]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        app.logger.error(f"Error in /predict: {str(e)}")
        return jsonify({
            "error": f"Processing failed: {str(e)}"
        }), 500


# 星火大模型的API接口地址
SPARK_API_URL = "https://spark-api-open.xf-yun.com/v1/chat/completions"  # 替换为实际的API URL
# API Key 或认证信息
API_KEY = "HFBzVlMeaorrZYzquLDI:BgWtKqEgTuitujYCspXD"  # 替换为你的API密钥

@app.route('/chat', methods=['POST'])
def chat():
    """处理文本聊天请求"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request"}), 400
    # 获取用户消息
    user_message = data['message']
    # 调用星火大模型的API
    response = call_spark_model(user_message)
    if response:
        return jsonify({"reply": response['choices'][0]['message']['content']})
    else:
        return jsonify({"error": "Failed to get response from model"}), 500


def call_spark_model(message):
    """调用星火大模型API并返回回复"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 构造请求体
    payload = {
        "model": "generalv3",  # 如果有多个模型，可以根据需求修改
        "messages": [
        {
            "role": "user",  # 用户角色
            "content": message  # 用户输入的文本
        }
    ]
    }

    # 发送请求
    try:
        response = requests.post(SPARK_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败，会抛出异常
        response_data = response.json()
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"Error calling Spark model: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)