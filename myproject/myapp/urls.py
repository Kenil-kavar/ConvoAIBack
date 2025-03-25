from django.urls import path
from django.contrib import admin
from .views import handle_audio, submit_form, Career_form 


urlpatterns = [
    # Route for handling audio files
    path('admin/', admin.site.urls),
    path('handle-audio/', handle_audio, name='handle_audio'),
    path('submit/', submit_form, name='submit_form'),
    path('career-submit/', Career_form, name='Career_form'),

]


