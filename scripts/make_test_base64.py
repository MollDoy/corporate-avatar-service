import base64
from io import BytesIO

from PIL import Image, ImageDraw


image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(image)

draw.ellipse((170, 80, 342, 252), fill=(230, 190, 160))
draw.rectangle((190, 260, 322, 460), fill=(40, 80, 140))

buffer = BytesIO()
image.save(buffer, format="PNG")

encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

print(encoded)