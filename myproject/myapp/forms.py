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


class EnquireForm(forms.Form):
    fullname = forms.CharField(max_length=100)
    email = forms.EmailField()
    company_name = forms.CharField(max_length=100, required=False)
    phone_number = forms.CharField(max_length=15, required=False)
    
    BUDGET_CHOICES = [
        ("25000-50000", "25,000-50,000"),
        ("50000-100000", "50,000-1,00,000"),
        ("100000+", "> 1,00,000"),
    ]
    estimated_budget = forms.ChoiceField(choices=BUDGET_CHOICES, widget=forms.RadioSelect())

    AREA_OF_INTEREST_CHOICES = [
        ("AI Customer Service", "AI Customer Service"),
        ("Data Analytics", "Data Analytics"),
        ("Custom AI Solutions", "Custom AI Solutions"),
        ("Workflow Automation", "Workflow Automation"),
        ("AI Chatbots", "AI Chatbots"),
    ]
    area_of_interest = forms.MultipleChoiceField(
        choices=AREA_OF_INTEREST_CHOICES,
        widget=forms.CheckboxSelectMultiple()
    )

    LOOKING_FOR_CHOICES = [
        ("Proof of concept", "Proof of concept"),
        ("Full solution implementation", "Full solution implementation"),
        ("Integration with existing systems", "Integration with existing systems"),
        ("Consultation services", "Consultation services"),
        ("Training for team", "Training for team"),
        ("Ongoing support", "Ongoing support"),
    ]
    looking_for = forms.MultipleChoiceField(
        choices=LOOKING_FOR_CHOICES,
        widget=forms.CheckboxSelectMultiple(),
        required=False
    )

    project_description = forms.CharField(widget=forms.Textarea(), required=False)

