# your_app/forms.py
from django import forms

class YourForm(forms.Form):
    fullname = forms.CharField(max_length=100)
    mobilenumber = forms.CharField(max_length=15)  # Assuming mobile numbers are strings
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
