import asyncio
import cv2
import threading
import aiohttp

image_path='wili-kacamata.jpeg'
url="http://localhost:8000/api/face_detection"
# url="https://cc3a-182-0-101-96.ngrok-free.app/api/face_detection"

async def send_frame_image(image_bytes, url):
  async with aiohttp.ClientSession() as session:
    async with session.post(url, data=image_bytes) as response:
      if response.status == 200:
        print(await response.text())  # Access response message
      else:
        print("Error sending frame data to API:", await response.text())

gambar = cv2.imread(image_path)

_, encoded_image_buffer = cv2.imencode('.jpg', gambar, [cv2.IMWRITE_JPEG_QUALITY, 100])
image_bytes = encoded_image_buffer.tobytes()

# Send full frame data asynchronously within the thread
asyncio.run(send_frame_image(image_bytes, url))
