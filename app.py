from flask import Flask, render_template, request, redirect, url_for, abort
import base64
import tempfile
from PIL import Image
import io
import numpy as np
import os
from mlmodel.test import cycle_gan

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        folderpath = tempfile.mkdtemp()
        filepath = os.path.join(folderpath,"a.png")
        #filepath = "{}/".format(tempfile.gettempdir()) + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        
        f.save(filepath)
        image = Image.open(filepath)
        
        # 画像処理部分
        image = np.asarray(image)
        width = len(image[0])
        height = len(image)
               
        filtered = Image.fromarray(cycle_gan(folderpath))
        filtered = filtered.resize(size=(width,height), resample=Image.LANCZOS)
        
        # base64でエンコード
        buffer = io.BytesIO()
        filtered.save(buffer, format="PNG")
        img_string = base64.b64encode(buffer.getvalue()).decode().replace("'", "")
        
        result = "image size {}×{}".format(width, height)
        return render_template("index.html", filepath=filepath, result=result, img_data=img_string)
    
