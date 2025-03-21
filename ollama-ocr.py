import ollama
from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='llama3.2-vision:11b') 

# Step 1: OCR the image
result = ocr.process_image(
    image_path="images/img1.png",
    format_type="markdown"  # Options: markdown, text, json, structured, key_value
)
print(result)

# Step 2: Initialize Chat (Llama3 model example)
messages = [
    {"role": "system", "content": "You are an AI assistant. Here's some extracted text from an image in markdown format:"},
    {"role": "user", "content": f"{result}\n\nYou can now ask questions about this content."}
]

while True:
    user_query = input("Ask something about the OCR result: ")
    messages.append({"role": "user", "content": user_query})

    response = ollama.chat(
        model='llama3',  # Any chat-capable model
        messages=messages
    )
    print(f"AI: {response['message']['content']}")
    messages.append({"role": "assistant", "content": response['message']['content']})
