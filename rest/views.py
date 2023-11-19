from django.shortcuts import render, redirect
from .forms import ImageUploadForm

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        print(form)
        if form.is_valid():
            form.save()  # 이미지 저장
            return render(request, 'index.html', {'form': form})
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
