import os
from django import forms
from .models import UploadedImage
from django.conf import settings

class ImageUploadForm(forms.ModelForm):
    file = forms.ImageField(label='file', required=True)
    sort_criterion = forms.CharField(label='Sort Criterion', required=False)

    class Meta:
        model = UploadedImage
        fields = ['file', 'sort_criterion']  