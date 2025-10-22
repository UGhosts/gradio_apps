from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ExifTags
import io
import os
import uuid
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="../model/ocr/configs/OCR.yaml")

app = Flask(__name__)

def get_json(filepath):
    output = pipeline.predict(
        input=filepath,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    for res in output:
        # res.print()
        res.save_to_img(save_path="./output/ocr_rest")
        res.save_to_json(save_path="./output/ocr_rest")

def resize_image_preserve_orientation(img, max_dimension=4000):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass

    width, height = img.size
    original_width, original_height = width, height

    if width <= max_dimension and height <= max_dimension:
        return img, False

    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"图片尺寸过大，已等比调整至: {new_width}x{new_height} (原始尺寸: {original_width}x{original_height})")
    return resized_img, True

def compress_and_process_image(img, target_size=300, max_dimension=4000, invert=True):
    target_size_bytes = target_size * 1024
    min_acceptable = target_size_bytes * 0.95
    max_acceptable = target_size_bytes * 1.05

    if img.mode in ('RGBA', 'LA'):
        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        background.paste(img, img.split()[-1])
        img = background
    elif img.mode == 'P':
        img = img.convert('RGB')

    img, resized = resize_image_preserve_orientation(img, max_dimension)

    quality = 95
    step = 5
    best_quality = 95
    best_size = float('inf')
    max_attempts = 20
    attempts = 0
    compressed_img = None

    while quality > 0 and attempts < max_attempts:
        attempts += 1
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True, progressive=True)
        current_size = buffer.tell()

        size_diff = abs(current_size - target_size_bytes)
        if size_diff < abs(best_size - target_size_bytes):
            best_size = current_size
            best_quality = quality
            buffer.seek(0)
            compressed_img = Image.open(buffer)

        if min_acceptable <= current_size <= max_acceptable:
            break
        elif current_size > max_acceptable:
            step = 10 if current_size > target_size_bytes * 1.5 else (2 if quality < 50 else 5)
            quality = max(quality - step, 0)
        else:
            quality += step
            if quality > 100:
                quality = 100
                break

    if compressed_img is None:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=best_quality, optimize=True, progressive=True)
        buffer.seek(0)
        compressed_img = Image.open(buffer)

    if invert:
        inverted_img = ImageOps.invert(compressed_img.convert('L')).convert('RGB')
    else:
        inverted_img = compressed_img

    output_buffer = io.BytesIO()
    inverted_img.save(output_buffer, format='JPEG', quality=best_quality, optimize=True, progressive=True)
    output_buffer.seek(0)
    return output_buffer, best_quality, inverted_img.size

@app.route('/ocr_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No image part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    img = Image.open(file.stream)


    output_buffer, quality, size = compress_and_process_image(
        img,
        target_size=600,
        max_dimension=800,
        invert=False
    )

    output_filename = f"./output/ocr_rest/"+file.filename
    with open(output_filename, 'wb') as f:
        f.write(output_buffer.getvalue())

    get_json(output_filename)
    path = './output/ocr_rest'
    json_name = file.filename.rsplit('.', 1)
    json_name = path + '/' + json_name[0] + '_res.json'
    import json
    with open(json_name, 'r', encoding='utf-8') as file:
        # 解析JSON数据
        json_data = json.load(file)

    return jsonify(json_data), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5005)