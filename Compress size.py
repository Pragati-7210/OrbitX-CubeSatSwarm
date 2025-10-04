import os
import cv2
import numpy as np
import json
import zlib
import base64

# Full absolute path to your file
img_path = r"C:\Users\bishn\Downloads\real_asteroid.jpg"  # or .jpeg

# Load original image
raw_img = cv2.imread(img_path)
if raw_img is None:
    raise FileNotFoundError(f"⚠️ Could not load image at {img_path}")

# Compression demo
_, encoded_webp = cv2.imencode(".webp", raw_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
webp_bytes = encoded_webp.tobytes()

compressed_bytes = zlib.compress(webp_bytes, level=9)
b64_data = base64.b64encode(compressed_bytes).decode('utf-8')

# Save packet
relay_packet = {
    "asteroid_id": "2025-AB",
    "image_format": "webp",
    "compressed_data": b64_data
}
with open("relay_packet_real.json", "w") as f:
    json.dump(relay_packet, f)

# Reconstruct
received = json.load(open("relay_packet_real.json"))
received_bytes = base64.b64decode(received["compressed_data"])
decompressed = zlib.decompress(received_bytes)
img_array = cv2.imdecode(np.frombuffer(decompressed, np.uint8), cv2.IMREAD_COLOR)

cv2.imwrite("asteroid_received_real.png", img_array)

# Size check
orig_size = os.path.getsize(img_path) / 1024          # KB
compressed_size = len(relay_packet["compressed_data"]) / 1024  # KB
reduction_percent = ((orig_size - compressed_size) / orig_size) * 100

print(f"Original size: {orig_size:.2f} KB")
print(f"Compressed size (after zlib+WebP+Base64): {compressed_size:.2f} KB")
print(f"Size reduced by: {reduction_percent:.2f}%")

# --- Side-by-side visualization ---
height = 450
orig_resized = cv2.resize(raw_img, (int(raw_img.shape[1] * height / raw_img.shape[0]), height))
recv_resized = cv2.resize(img_array, (int(img_array.shape[1] * height / img_array.shape[0]), height))

comparison = cv2.hconcat([orig_resized, recv_resized])
cv2.imshow("Original vs Received", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
