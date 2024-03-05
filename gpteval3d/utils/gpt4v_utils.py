import base64
import io
import os.path as osp
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_and_encode_image(image_path, resize_ratio):
    # Open the original image
    with Image.open(image_path) as img:
        # Calculate the new size
        new_width = int(img.width * resize_ratio)
        new_height = int(img.height * resize_ratio)

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Save the resized image to a buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")  # You can change the format if needed
        buffer.seek(0)

        # Encode the buffer to Base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def call_gpt_4v(client, user_prompt, user_img_path, 
                max_tokens=4096, n_choices=3):
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    base64_image = encode_image(user_img_path)
    # base64_image = resize_and_encode_image(user_img_path, 0.5)
    conversation_history = [{
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": user_prompt
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ],
    }]
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=conversation_history,
            max_tokens=max_tokens,
            n=n_choices
        )
        err = None
    except Exception as e:
        print(e)
        err = e
        response = None
    return response, err


def show_content(response):
    print(response.choices[0].message.content)
    
    
def make_comparison_prompt(text_prompt, instruction_file):
    assert osp.isfile(instruction_file)
    with open(instruction_file, 'r') as f:
        instruction = f.read()
    postfix_temp = 'Following is the text prompt from which these two 3D objects are generated:\n"{}"\nPlease compare these two 3D objects as instructed.'
    postfix = postfix_temp.format(text_prompt)
    return instruction + postfix
