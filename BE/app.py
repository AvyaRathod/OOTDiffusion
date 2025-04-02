from flask import Flask, request, jsonify
import requests
from io import BytesIO
import base64
from PIL import Image
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_dc import OOTDiffusionDC
from ootd.inference_ootd_hd import OOTDiffusionHD  # 👈 Added HD model
from run.utils_ootd import get_mask_location

app = Flask(__name__)

CATEGORY_DICT = ['upperbody', 'lowerbody', 'dress']
CATEGORY_UTILS = ['upper_body', 'lower_body', 'dresses']

def load_image_from_base64_or_url(base64_key, url_key, data, resize_dim=(768, 1024)):
    try:
        if base64_key in data:
            img_data = base64.b64decode(data[base64_key])
            img = Image.open(BytesIO(img_data)).convert("RGB").resize(resize_dim)
            return img
        elif url_key in data:
            resp = requests.get(data[url_key])
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB").resize(resize_dim)
        return None
    except Exception as e:
        print(f"Failed to load image from {base64_key} or {url_key}: {e}")
        return None

def run_ootd_inference(model_img, cloth_img, category, image_scale, n_samples, n_steps, seed, gpu_id, model_type='dc'):
    model_img_proc = model_img.resize((384, 512))
    openpose = OpenPose(gpu_id)
    parsing = Parsing(gpu_id)

    keypoints = openpose(model_img_proc)
    model_parse, _ = parsing(model_img_proc)
    mask, mask_gray = get_mask_location(model_type, CATEGORY_UTILS[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    masked_vton_img = Image.composite(mask_gray, model_img, mask)

    model = OOTDiffusionHD(gpu_id) if model_type == 'hd' else OOTDiffusionDC(gpu_id)

    result_images = model(
        model_type=model_type,
        category=CATEGORY_DICT[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    return result_images[0]  # Return only the first image for now

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "API is up and running!"}), 200

@app.route("/inference", methods=["POST"])
def inference():
    data = request.get_json()

    try:
        cloth_type = data.get("cloth_type", "top").lower()

        category_map = {
            "top": 0,
            "bottoms": 1,
            "full body dress": 2
        }

        category = category_map.get(cloth_type)
        if category is None:
            return jsonify({"error": "Invalid cloth_type provided."}), 400

        # Load images
        model_img = load_image_from_base64_or_url("model_image_base64", "model_image_url", data)
        cloth_img = load_image_from_base64_or_url("cloth_image_base64", "cloth_image_url", data)

        if model_img is None or cloth_img is None:
            return jsonify({"error": "Failed to load model or cloth image (check base64 or URL)."}), 400

        model_type = 'dc'
        # 'hd' if cloth_type == 'top' else 'dc'

        output_img = run_ootd_inference(
            model_img=model_img,
            cloth_img=cloth_img,
            category=category,
            image_scale=data.get("scale", 2.0),
            n_samples=data.get("sample", 4),
            n_steps=data.get("step", 20),
            seed=data.get("seed", -1),
            gpu_id=data.get("gpu_id", 0),
            model_type=model_type
        )

        buffered = BytesIO()
        output_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": img_base64}), 200

    except Exception as e:
        return jsonify({"error": "Unexpected server error.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
