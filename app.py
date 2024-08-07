# Import necessary libraries
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from ai71 import AI71
import os
import json

import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Load the Florence-2 model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Function to run Florence-2 inference
def run_inference(image, prompt="<OCR>"):
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))

        return parsed_answer
    except Exception as e:
        return f"Error in run_inference: {str(e)}"

# Function to extract book details from text using AI71
def extract_details(text):
    try:
       

        AI71_api_key = os.getenv("AI71_API_KEY")
        
        client = AI71(AI71_api_key)

        # Simple invocation
        result = client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=[
                {"role": "user", "content": f"Extract book title, description, author names, publisher, and suggest genre from the following text: {text} and generate JSON output."},
            ],
        )

        content = result.choices[0].message.content
        return content
    except Exception as e:
        return f"Error in extracting details: {str(e)}"

# Function to handle the combined pipeline
def combined_pipeline(image):
    try:
        # Run Florence-2 inference on the uploaded image
        extracted_text = run_inference(image)

        # Extract details from the text using Falcon
        book_details = extract_details(extracted_text)

        return book_details
    except Exception as e:
        return f"Error in combined_pipeline: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=combined_pipeline,
    inputs=gr.Image(type="pil", mirror_webcam=False),
    outputs=gr.Textbox(),
    title="Automate Book Inventory using AI",
    description="Upload an image of a BOOK cover to run Florence-2 inference to extract OCR and understand and prepare structured book information using Falcon."
)

# Launch the Gradio app
interface.launch(share=True)
