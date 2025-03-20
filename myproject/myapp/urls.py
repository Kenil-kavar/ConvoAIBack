from django.urls import path
from .views import handle_audio  # Import the view


urlpatterns = [
    # Route for handling audio files
    path('handle-audio/', handle_audio, name='handle_audio'),
]


