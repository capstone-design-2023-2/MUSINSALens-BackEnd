# from django.shortcuts import render
# from django.http import HttpResponse

# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

# myapp/views.py

from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return HttpResponse("This is the index view.")

def accounts_view(request):
    return render(request, 'login.html')

def kakao_redirect(request):
    return render(request, 'kakao-redirect.html')