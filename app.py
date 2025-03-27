import os
import json
import pytesseract
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator

# Set Tesseract OCR path (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
FEEDBACK_FILE = 'feedback.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load BLIP image captioning model
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
translator = Translator()
print("BLIP model loaded successfully!")

# Load feedback data if exists
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'r') as f:
        feedback_data = json.load(f)
else:
    feedback_data = {}

def generate_caption(image_path):
    """Generate a description for the uploaded image using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def extract_text(image_path):
    """Extract text from the image using Tesseract OCR."""
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'language' not in request.form:
            return render_template('index.html', error="File or language not provided.")

        file = request.files['file']
        language = request.form['language']

        if file.filename == '':
            return render_template('index.html', error="No selected file.")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate caption
            if filename in feedback_data:
                caption = feedback_data[filename]
            else:
                caption = generate_caption(file_path)

            # Translate caption
            translated_caption = translator.translate(caption, dest=language).text

            # Extract text using OCR
            extracted_text = extract_text(file_path)
            translated_text = translator.translate(extracted_text, dest=language).text if extracted_text else "No text found."

            return render_template('index.html', filename=filename, caption=translated_caption, extracted_text=translated_text, language=language)

    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handles user feedback on the generated caption."""
    filename = request.form['filename']
    feedback = request.form['feedback']
    
    if feedback == "no":
        user_caption = request.form['user_caption']
        feedback_data[filename] = user_caption
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_data, f)
        return redirect(url_for('upload_file', message="Your suggestion is noted!"))
    
    return redirect(url_for('upload_file', message="Thank you for your suggestion!"))

if __name__ == '__main__':
    app.run(debug=True)