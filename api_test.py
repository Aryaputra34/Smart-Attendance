from fastapi import FastAPI, Body, HTTPException
import torch
import cv2
import numpy as np  # for NumPy operations
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLO model weights and configuration
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
    # Decode image bytes using NumPy and then convert color to rgb and gray
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #rgb for inferencing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray for blurry image filtering

    # save an output image
    cv2.imwrite('output.jpg', image)

    # blurry image filtering
    fm = variance_of_laplacian(gray)
    blur_threshold = 24
    if fm > blur_threshold:
      print("Image - Not Blurry: "+str(fm))
    if fm < blur_threshold:
      print("Image - Blurry: "+str(fm))
      return {"message": "image is blurry"}


    # setting up confidence threshold and inferencing model
    model.conf = 0.85
    results = model(rgb_img, size=640, augment=True)

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
          return {"message": "face not registered"}
    
    print(results) #output nya string
    nim, nama = results.split("_")

    return {"message": "face detected","nim": nim, "name": nama}

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
