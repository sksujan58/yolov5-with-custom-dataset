from flask import Flask,request
import torch
import pandas
import io
from PIL import Image
import io, base64

app=Flask(__name__)


model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')  # local repo
model.conf = 0.5

@app.route("/predict",methods=["POST"])
def prediction():
    base64_str = request.json[0]["text"]
    img=base64.b64decode(base64_str)
    with open("p2.jpg", "wb") as f:
        f.write(img)
        f.close()
    results = model("p2.jpg")
    results.render()
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save("output.jpg", format="jpeg")
        with open("output.jpg", "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
    return my_string




if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)