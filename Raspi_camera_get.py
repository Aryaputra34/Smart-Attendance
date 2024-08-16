import asyncio
import cv2
import threading
import aiohttp  # Use aiohttp for asynchronous requests

# Load the Haar cascade classifier for face detection (optional)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Get camera frame dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the center coordinates of the frame
center_x = width // 2
center_y = height // 2

# Define rectangle dimensions relative to center
rectangle_width = int(width * 0.5)  # Adjust width as needed (0 to 1 for frame proportion)
rectangle_height = int(height * 0.7)  # Adjust height as needed (0 to 1 for frame proportion)

# Define rectangle top-left corner relative to center
rectangle_x = center_x - rectangle_width // 2
rectangle_y = center_y - rectangle_height // 2


async def send_frame_image(image_bytes, url):
  async with aiohttp.ClientSession() as session:
    async with session.post(url, data=image_bytes) as response:
      if response.status == 200:
        print(await response.text())  # Access response message
      else:
        print("Error sending frame data to API:", await response.text())


class FrameProcessingThread(threading.Thread):
  def __init__(self, frame, url):
    super().__init__()
    self.frame = frame
    self.url = url

  def run(self):
    # Convert frame to bytes (assuming JPEG format)
    _, encoded_image_buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    image_bytes = encoded_image_buffer.tobytes()

    # Send full frame data asynchronously within the thread
    asyncio.run(send_frame_image(image_bytes, self.url))


async def main():
  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    raw_img = frame.copy()
    # Convert frame to grayscale (improves efficiency for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame (optional)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle for reference (optional)
    cv2.rectangle(frame, (rectangle_x, rectangle_y), (rectangle_x + rectangle_width, rectangle_y + rectangle_height), (0, 255, 0), 2)

    # Check for faces and send full frame if inside rectangle
    for (x, y, w, h) in faces:
      # Calculate face center
      face_center_x = x + w // 2
      face_center_y = y + h // 2

      # Check if face center is inside the rectangle (alternative check)
      if rectangle_x < face_center_x < rectangle_x + rectangle_width and rectangle_y < face_center_y < rectangle_y + rectangle_height:
        # Face is completely inside the rectangle, send full frame
        frame_processing_thread = FrameProcessingThread(raw_img, url="http://localhost:8000/api/face_detection")
        frame_processing_thread.start()  # Start the thread to send full frame

    # Display the resulting frame (regardless of sending)
    cv2.imshow('frame', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
      break

  # Release the capture and destroy all windows
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  asyncio.run(main())
