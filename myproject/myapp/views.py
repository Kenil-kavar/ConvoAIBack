from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import torch
import os
import base64, time
import subprocess
import numpy as np
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from scipy.io.wavfile import write
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset 
from django.conf import settings
from scipy.io.wavfile import write
from django.shortcuts import render
from .forms import YourForm, CareerForm, EnquireForm
from .mongo_utils import get_mongo_client
from django.http import HttpResponse
import datetime
from django.contrib.auth.hashers import make_password

# Load environment variables
load_dotenv()

@csrf_exempt
def Enquire_Form(request):
    if request.method == 'POST':
        form = EnquireForm(request.POST)
        if form.is_valid():
            try:
                db = get_mongo_client()
                collection = db["EnquireForm"]  # Change collection name if needed

                # Convert form data to a dictionary and save it in MongoDB
                form_data = form.cleaned_data
                form_data["area_of_interest"] = list(form.cleaned_data["area_of_interest"])  # Convert QueryDict to list
                form_data["looking_for"] = list(form.cleaned_data["looking_for"])  # Convert QueryDict to list

                collection.insert_one(form_data)  # Insert into MongoDB

                return JsonResponse({"success": True, "message": "Form submitted successfully!"})
            except Exception as e:
                return JsonResponse({"success": False, "error": str(e)}, status=500)
        else:
            return JsonResponse({"success": False, "errors": form.errors}, status=400)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt  # Remove this if using CSRF protection in frontend
def submit_form(request):
    if request.method == 'POST':
        form = YourForm(request.POST)
        if form.is_valid():
            try:
                db = get_mongo_client()

                # Get form data
                form_data = form.cleaned_data

                # Hash the password before saving
                if "password" in form_data:
                    form_data["password"] = make_password(form_data["password"])

                # Save to MongoDB
                db["signup"].insert_one(form_data)

                return JsonResponse({"success": True, "message": "Form data saved successfully"})
            except Exception as e:
                return JsonResponse({"success": False, "error": str(e)}, status=500)
        else:
            return JsonResponse({"success": False, "errors": form.errors}, status=400)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)

# Ensure the upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@csrf_exempt  # Remove if using CSRF protection in frontend
def Career_form(request):
    if request.method == 'POST':
        form = CareerForm(request.POST, request.FILES)  # Include request.FILES for file handling

        if form.is_valid():
            try:
                db = get_mongo_client()

                # Get form data
                form_data = form.cleaned_data

                # Hash password if it exists in form
                if "password" in form_data:
                    form_data["password"] = make_password(form_data["password"])

                # Handle file upload
                if "resume" in request.FILES:
                    resume_file = request.FILES["resume"]
                    file_path = os.path.join(UPLOAD_DIR, resume_file.name)  # Store in 'uploads' directory
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        for chunk in resume_file.chunks():
                            f.write(chunk)

                    form_data["resume"] = file_path  # Store file path in MongoDB

                # Save to MongoDB
                db["careerform"].insert_one(form_data)

                return JsonResponse({"success": True, "message": "CareerForm data saved successfully"})
            except Exception as e:
                return JsonResponse({"success": False, "error": str(e)}, status=500)
        else:
            return JsonResponse({"success": False, "errors": form.errors}, status=400)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)



# Global variables
model = None
tokenizer = None
messages = []

def initialize_chat_model():
    """Initialize the chat model and tokenizer."""
    global model, tokenizer, messages

    try:
        # Load the saved model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/content/ConvoAIBack/myproject/myapp/whisper_streaming/lora_model",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

        # Initialize system prompt
        system_prompt = {
            "role": "system",
            "content": (
                "You are a friendly and professional customer care assistant. "
                "Assist users with clear and accurate guidance for their requests within 15 words."
            )
        }

        # Initialize messages
        messages = [
            system_prompt,
            {"role": "assistant", "content": "Hi, how can I help you?"}
        ]

        print("Chat model successfully initialized.")

    except Exception as e:
        print(f"Error initializing model: {e}")

def chat_with_model(user_input):
    """Chat with the language model and return the AI's response."""
    global messages, model, tokenizer

    if model is None or tokenizer is None:
        print("Error: Model or tokenizer is not initialized.")
        return "Sorry, I encountered an error."

    # Append user input to messages
    messages.append({"role": "user", "content": user_input})

    try:
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Move input to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # Generate the output
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            temperature=1.0,
            min_p=0.1
        )

        # Decode the output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = decoded_output.split("assistant")[-1].strip()

        # Append AI response to messages
        messages.append({"role": "assistant", "content": ai_response})

        return ai_response

    except Exception as e:
        print(f"Error during chat generation: {e}")
        return "Sorry, I encountered an error during response generation."

def process_text_file(file_path):
    """Reads and processes the text file output from Whisper."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return ""

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Combine lines into a single string and replace newlines with spaces
        single_line = ' '.join(line.strip() for line in lines)

        print("Processed text:")
        return single_line

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

@csrf_exempt
def handle_audio(request):
    if request.method == 'POST' and 'audio_file' in request.FILES:
        try:
            start_time = time.time()
            
            audio_file = request.FILES['audio_file']
            
            os.makedirs("/content/ConvoAIBack/myproject/myapp/tmp", exist_ok=True)  # Create tmp directory if it doesn't exist
            temp_audio_path = f"/content/ConvoAIBack/myproject/myapp/tmp/{audio_file.name}"           
            with open(temp_audio_path, 'wb') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            # Verify file exists and has data
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                print(f"Audio file saved successfully: {temp_audio_path}")
            else:
                raise Exception("Audio file was not saved correctly")

            # Run Whisper speech-to-text model
            whisper_command = (
                f"python3 /content/ConvoAIBack/myproject/myapp/whisper_streaming/whisper_online.py "
                f"{temp_audio_path} --language en --min-chunk-size 1 > /content/ConvoAIBack/myproject/myapp/tmp/text.txt"
            )

            subprocess.run(whisper_command, shell=True, check=True)
            print("Whisper processing complete.")

            # Process the output text file
            file_path = "/content/ConvoAIBack/myproject/myapp/tmp/text.txt"
            user_input = process_text_file(file_path)

            # Initialize the chat model
            initialize_chat_model()

            # Get AI response
            ai_response = chat_with_model(user_input) if user_input else "No valid input found."
            print("AI Response:", ai_response)

            # Load SpeechT5 model for text-to-speech
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            inputs = processor(text=ai_response, return_tensors="pt")

            # Load speaker embeddings
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[5000]["xvector"]).unsqueeze(0)

            # Generate speech
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

            output_speech_path = "/content/ConvoAIBack/myproject/myapp/tmp/speech.wav"
            sf.write(output_speech_path, speech.numpy(), samplerate=20000)

            end_time = time.time()
            print(f"Finished processing in {end_time - start_time:.4f} sec")

            # Return the generated audio file
            return FileResponse(open(output_speech_path, 'rb'), content_type='audio/wav', as_attachment=True, filename="speech.wav")

        except subprocess.CalledProcessError as e:
            return JsonResponse({'error': f'Whisper model error: {str(e)}'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request. Please upload an audio file.'}, status=400)
