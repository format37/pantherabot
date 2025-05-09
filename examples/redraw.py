from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
from PIL import Image
import io

# Set your API key

# Path to your input image (ensure it exists in the specified path)
input_image_path = "input_image.jpg"

# (Optional) Path to your mask image (if you want to limit edits to a specific area)
# For full-image editing, you can omit the mask parameter
mask_image_path = None  # Set to None for no mask, or provide path like "mask.png"

# Define the text prompt for editing
text_prompt = "Redraw the image in a surreal, dreamlike style with brighter colors."

# Open and get dimensions of the input image
input_image = Image.open(input_image_path)
width, height = input_image.size

# Prepare the mask
if mask_image_path:
    # Use the provided mask image
    with open(mask_image_path, "rb") as mask_file:
        mask_data = mask_file.read()
else:
    # Create an empty mask (fully transparent)
    empty_mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask_buffer = io.BytesIO()
    empty_mask.save(mask_buffer, format="PNG")
    mask_data = mask_buffer.getvalue()

# Call the DALL·E image editing endpoint
response = client.images.generate(image=open(input_image_path, "rb"),
mask=io.BytesIO(mask_data),
prompt=text_prompt,
n=1,
size="1024x1024")

# Retrieve the URL of the edited image from the response
edited_image_url = response.data[0].url
print("Edited image URL:", edited_image_url)
