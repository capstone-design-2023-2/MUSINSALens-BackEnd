import os
from django import forms
from .models import UploadedImage
from django.conf import settings

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['file']

    file = forms.ImageField(label='file', required=True)