import os
from django import forms
from .models import UploadedImage
from django.conf import settings

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['category', 'file', 'filename']

    category = forms.CharField(max_length=255, label='category', required=True)
    filename = forms.CharField(max_length=255, label='filename', required=True)
    file = forms.ImageField(label='file', required=True)

    def save(self, commit=True):
        instance = super(ImageUploadForm, self).save(commit=False)
        category = self.cleaned_data.get('category', '')
        filename = self.cleaned_data.get('filename', '')

        # 사용자 정의 경로 및 파일 이름 생성
        file_name = f"{filename}"
        file_path = os.path.join(category, file_name)

        # MEDIA_ROOT에 디렉터리 생성
        media_root = getattr(settings, 'MEDIA_ROOT', None)
        if media_root:
            full_path = os.path.join(media_root, category)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

        # 파일 저장
        if self.cleaned_data.get('file'):
            instance.image_path = self.cleaned_data['file']
            instance.image_path.name = file_path + '.png'

        if commit:
            instance.save()

        return instance