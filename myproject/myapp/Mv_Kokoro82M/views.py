from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import torch , os, base64, subprocess
import numpy as np
import os, subprocess
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from .models import build_model  
from django.conf import settings
from myapp.Kokoro82M.kokoro import generate
from scipy.io.wavfile import write


load_dotenv()




model = None
tokenizer = None
messages = []

def initialize_chat_model():
    """Initialize the chat model and tokenizer."""
    global model, tokenizer, messages

    # load the saved model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
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
            "Assist users with clear and accurate guidance for their requests."
        )
    }

    # Initialize messages
    messages = [
        system_prompt,
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]



def chat_with_model(user_input):
    """Chat with the language model and return the AI's response."""
    global messages

    # append the input to messages
    messages.append({"role": "user", "content": user_input})

    # tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

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

    # append ai response to messages
    messages.append({"role": "assistant", "content": ai_response})

    return ai_response

def generate_and_play_audio(text, model, voicepack, voice_name):
    """Generate audio from text and play it."""
    # split the text into chunks
    chunk_size = 490  # Set the chunk size
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Generate audio for each chunk and concatenate the results
    audio_chunks = []
    for chunk in text_chunks:
        audio, _ = generate(model, chunk, voicepack, lang=voice_name[0])
        audio_chunks.append(audio)

    # Combine audio chunks
    combined_audio = np.concatenate(audio_chunks)

    # Save the combined audio
    write("output.wav", 25000, combined_audio)
    
def process_text_file(file_path):
    try:
        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Combine lines into a single string and replace newlines with spaces
        single_line = ' '.join(line.strip() for line in lines)
        
        # Print or return the result
        print("Processed text:")
               
        return single_line
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

@csrf_exempt
def handle_audio(request):
      try:
          # Get the uploaded audio file
          audio_file = request.FILES['audio_file']
          audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
          # Save the uploaded file
          with open(audio_file_path, 'wb+') as destination:
              for chunk in audio_file.chunks():
                  destination.write(chunk)
          # Run Whisper speech-to-text model
          command = f"python3 ../whisper_streaming/whisper_online.py {audio_file_path} --language en --min-chunk-size 1 > text.txt"
          subprocess.run(command, shell=True, check=True)
          # Process the output text file
          file_path = "../whisper_streaming/text.txt"
          user_input = process_text_file(file_path)
          if not user_input:
              return JsonResponse({'error': 'Transcription failed'}, status=500)
              
          # Get AI response
          ai_response = chat_with_model(user_input)
          if not ai_response:
              return JsonResponse({'error': 'No response from the chat model'}, status=500)
              
          # Generate AI Voice Response
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          kokoro_model = build_model('kokoro-v0_19.pth', device)
          voice_name = 'af'  # Select a voice
          voicepack = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)
          output_audio_path = os.path.join(settings.MEDIA_ROOT, "output.wav")
          generate_and_play_audio(ai_response, kokoro_model, voicepack, voice_name, output_audio_path)
          
          # Encode the generated audio file in Base64 for response
          with open(output_audio_path, "rb") as audio_file:
              encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
          return JsonResponse({'response': ai_response, 'audio_file': encoded_audio})
      except subprocess.CalledProcessError as e:
          return JsonResponse({'error': f'Whisper model error: {str(e)}'}, status=500)
      except Exception as e:
          return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

      return JsonResponse({'error': 'Invalid request'}, status=400)
