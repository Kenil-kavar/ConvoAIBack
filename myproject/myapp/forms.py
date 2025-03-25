from django import forms
from django.core.validators import URLValidator


class YourForm(forms.Form):
    fullname = forms.CharField(max_length=100)
    mobilenumber = forms.CharField(max_length=15)  # Assuming mobile numbers are strings
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)


class CareerForm(forms.Form):
    fullname = forms.CharField(
        max_length=100, 
        widget=forms.TextInput(attrs={'placeholder': 'Full Name'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'placeholder': 'Email'})
    )
    linkedin = forms.CharField(
        max_length=255,
        validators=[URLValidator()],
        widget=forms.URLInput(attrs={'placeholder': 'LinkedIn Profile Link'})
    )
    github = forms.CharField(
        max_length=255,
        validators=[URLValidator()],
        widget=forms.URLInput(attrs={'placeholder': 'GitHub Profile Link'})
    )
    resume = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'accept': '.pdf,.doc,.docx'})
    )

