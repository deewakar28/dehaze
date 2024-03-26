# backend.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional
import cv2
import numpy as np
import shutil
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,x
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def dark_channel(img, size=15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

def get_atmo(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

def guided_filter(p, i, r, e):
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * i + mean_b
    return q

def dehaze_image(file_path):
    try:
        im = cv2.imread(file_path)
        img = im.astype('float64') / 255
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
        trans_guided = cv2.max(trans_guided, 0.25)
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/dehaze/")
async def dehaze(upload_file: UploadFile = File(...)):
    try:
        with open(f"uploads/{upload_file.filename}", "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        dehazed_image = dehaze_image(f"uploads/{upload_file.filename}")
        if dehazed_image is not None:
            output_file_path = f"uploads/dehazed_{upload_file.filename}"
            cv2.imwrite(output_file_path, dehazed_image * 255)
            return FileResponse(output_file_path)
        else:
            return {"error": "Failed to process image"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        return FileResponse(f"uploads/{filename}")
    except Exception as e:
        return {"error": str(e)}
    
uvicorn.run(app)
