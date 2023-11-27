import os
import hashlib
from django.shortcuts import render, redirect
from django.conf import settings 
from .forms import ImageUploadForm

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            instance = form.save(commit=False)

            # 사용자가 업로드한 이미지 파일의 파일명을 file_name에 저장
            instance.file_name = request.FILES['file'].name

            # 사용자가 입력한 파일명을 해싱하여 image_path에 저장
            hashed_name = hashlib.sha256(instance.file_name.encode('utf-8')).hexdigest()
            instance.image_path = f'uploaded_images/{hashed_name}.png'  # 확장자는 이미지 종류에 따라 변경

            # 이미지를 서버 로컬에 저장
            save_path = os.path.join(settings.BASE_DIR, instance.image_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in request.FILES['file'].chunks():
                    f.write(chunk)

            # sort_criterion을 변수에 저장
            sort_criterion = form.cleaned_data['sort_criterion']

            instance.save()
            
            return render(request, 'index.html', {'form': form})
        
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
