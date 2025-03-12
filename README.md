# Backend Repository

This repository contains the backend components for the project, which include:

- **Whisper**: Speech-to-Text (STT)
- **LLM**: For generating responses
- **Kokoro Model**: Text-to-Speech (TTS)

## How to Run the Backend

Follow the steps below to set up and run the backend:

### Step 1: Install Required Dependencies
Run the following command to install the necessary dependencies for the Kokoro model:
```sh
pip install -r myproject/myapp/Mv_Kokoro82M/requirements.txt
```

### Step 2: Install Additional Dependencies
Run the following command to install extra dependencies:
```sh
pip install -q phonemizer torch transformers scipy munch
```
Additionally, install `espeak-ng`:
```sh
apt-get -qq -y install espeak-ng > /dev/null 2>&1
```

### Step 3: Install Django and Other Required Modules (if needed)
If you encounter errors related to missing Django or other dependencies, install them using:
```sh
pip install Django python-dotenv requests munch numpy torch unsloth
```

### Step 4: Run the Server
Start the Django development server by running:
```sh
python manage.py runserver 8000
```

The backend should now be up and running!

