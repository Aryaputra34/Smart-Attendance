from fastapi import FastAPI, Body, HTTPException
import torch
import cv2
import numpy as np  # for NumPy operations
import pathlib
import time
import requests
url = "https://eyecatching-image-ghhipha43a-uc.a.run.app/api/users/attendance-logs"

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLO model weights and configuration
# model = torch.hub.load('yolov5', 'custom', 'yolov5/runs/train/yolov5l_results3/weights/best.pt', source='local')
model = torch.hub.load('yolov5', 'custom', 'yolov5/runs/train/yolov5l_results2/weights/best.pt', source='local')

app = FastAPI()

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

@app.post("/api/face_detection")
async def receive_face_image(image_data: bytes = Body(...)):
  """
  API endpoint to receive image data as bytes and run YOLO model for face detection.
  """
  try:
    start = time.time()
    # Decode image bytes using NumPy
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray for blurry image filtering

    # Convert image to RGB format (YOLOv8 expects RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output.jpg', image)

    # blurry image filtering
    fm = variance_of_laplacian(gray)
    blur_threshold = 24
    # 200 - 250 di Gedung PUT
    # 24 default nya
    if fm > blur_threshold:
      print("Image - Not Blurry: "+str(fm))
    if fm < blur_threshold:
      print("Image - Blurry: "+str(fm))
      return {"message": "image is detected"}

    # Run YOLO model prediction on the imageip -
    model.conf = 0.85
    results = model(rgb_img, size=640, augment=True)

    end_inference = time.time()
    print(f"Inferencing time : {end_inference - start}")

    print(f"Threshold : {model.conf}")
    for *xyxy, conf, cls in results.xyxy[0]:
        # print(f'Coordinates: {xyxy} | Confidence: {conf} | Class: {cls}')
        print(f'Confidence: {conf}')

    # results.show()
    results = str(results)
    print(results)
    results = results.splitlines(False)[0]
    results = results.split("_")
    results[0] = results[0].split(" ")[4]
    
    if len(results) > 2:
        print(results[0])
        print(results[1])
        print(len(results))
        return {"message": "there are multiple faces detected"}

    results = '_'.join(results)

    if results == "detections)":
          print("face not detected")
          return ""
    
    print(results) #output nya string

    nim, nama_user = results.split("_")
    floor = "20"
    status="Present"
    nim = int(nim)

    # Data to be sent
    data = {
        'user_id': nim,
        'name': nama_user,
        'floor': floor,
        'status': status
    }

    # Files to be sent (if any)
    files = {
        'image_file': open('output.jpg', 'rb')  # Replace 'file.txt' with the path to your file
    }

    # Send the request
    response = requests.post(url, data=data, files=files)
    print(response.json())
    end = time.time()
    print(f"Time lapsed : {end - start}")
    return {"message": "face detected, saved to database","nim": nim, "name": nama_user}

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
