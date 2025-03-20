# Backend Repository

This repository contains the backend components for the project, which include:

- **Whisper**: Speech-to-Text (STT)
- **LLM**: For generating responses
- **microsoft/speecht5_tts Model**: Text-to-Speech (TTS)

## How to Run the Backend

Follow the steps below if you are using Google Colab to set up and run the backend:

### Step 1: Clone the Repo
```sh
!git clone https://github.com/Kenil-kavar/ConvoAIBack.git
```

### Step 2: Install Additional Dependencies
Run the following command to install extra dependencies:
```sh
!pip install faster-whisper torch torchaudio python-dotenv unsloth munch TTS Django requests
```

### Step 3: Run the Server
Start the Django development server by running:
```sh
!python /content/ConvoAIBack/myproject/manage.py runserver 0.0.0.0:8000 &> /dev/null &
```

The backend should now be up and running in the Background!

### Step 4: Make it Live with NGROK
```sh
!pip install pyngrok  ## If it shows error then restart session on colab
!ngrok config add-authtoken 2k6bQMxNdBdGf1ufHl7UwXc9g1h_81AUdX77yAhinTYoVDiGG

from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)
```

# Output files are saved in tmp folder
