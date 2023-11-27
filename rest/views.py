import os
import hashlib
from django.shortcuts import render, redirect
from django.conf import settings 
from .forms import ImageUploadForm
import subprocess

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

            # detectron2결과 변수에 저장
            result = requestML(save_path) # {"category": "long_sleeve_top", "score": 0.8854095935821533, "path": "D:\\dev\\vscode\\musinsa\\MUSINSALens-BackEnd\\uploaded_images/27869c4cd00e53d5590e783ad80102db912bf8bc7d7563bdc4a048cde9124279.png"} 

            print(result)

            instance.save()
            
            return render(request, 'index.html', {'form': form})
        
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})

def requestML(image_path):
    current_working_directory = os.getcwd()
    print("Current Working Directory:", current_working_directory)
    print("====================================")
    print(image_path)
    result = subprocess.run(['python', '../Detectron2/detectron2/run.py', image_path], capture_output=True, text=True)
    return result.stdout
