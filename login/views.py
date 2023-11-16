from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import requests
from django.conf import settings


# kakao api key 값 및 필요 url
KAKAO_API_KEY = settings.KAKAO_API_KEY
KAKAO_REDIRECT_URL = settings.KAKAO_REDIRECT_URL
KAKAO_SECRET_KEY = settings.KAKAO_SECRET_KEY
KAKAO_PROFILE_URI = settings.KAKAO_PROFILE_URI

def index(request):
    context = {'check': False}
    if request.session.get('access_token'):
        context['check'] = True
    return render(request, 'index.html', context)

def get_kakao_auth_url():
    return f'https://kauth.kakao.com/oauth/authorize?client_id={KAKAO_API_KEY}&redirect_uri={KAKAO_REDIRECT_URL}&response_type=code'

def kakao_login(request):
    return redirect(get_kakao_auth_url())

def get_kakao_access_token(code):
    url = f'https://kauth.kakao.com/oauth/token?grant_type=authorization_code&client_id={KAKAO_API_KEY}&redirect_uri={KAKAO_REDIRECT_URL}&code={code}&client_secret={KAKAO_SECRET_KEY}'
    response = requests.post(url)
    result = response.json()
    print("---------------------------")
    print(result) # {'access_token': '2dedBYEOZ5QGVNK8e-S3AHV8uh5YQTmoUCcKKcjZAAABi8QVTo1b9Pmr5eg_ZA', 'token_type': 'bearer', 'refresh_token': 'MYdFIOoQqI67UK17Ea_5GcpOlE4VXp1XwgwKKcjZAAABi8QVTohb9Pmr5eg_ZA', 'expires_in': 21599, 'refresh_token_expires_in': 5183999}


    # 카카오톡 정보 요청
    info_access_token = f"Bearer {result.get('access_token')}"
    auth_headers = {
        "Authorization": info_access_token,
        "Content-type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    user_info_res = requests.get(KAKAO_PROFILE_URI, headers=auth_headers)
    user_info_json = user_info_res.json()

    print(user_info_json) # {'id': 3160011217, 'connected_at': '2023-11-12T14:37:14Z'}

    ## TODO : 회원가입 처리 구현(insert db)
    



    return result.get('access_token')

def kakao_login_redirect(request):
    code = request.GET.get('code')
    if not code:
        return HttpResponse("bad request", status=400)

    access_token = get_kakao_access_token(code)
    request.session['access_token'] = access_token
    request.session.modified = True
    return render(request, 'loginSuccess.html')

def kakao_logout(request):
    access_token = request.session.get('access_token')
    if not access_token:
        return render(request, 'logoutError.html')

    url = 'https://kapi.kakao.com/v1/user/logout'
    headers = {'Authorization': f'bearer {access_token}'}
    response = requests.post(url, headers=headers)
    result = response.json()

    if result.get('id'):
        del request.session['access_token']
        return render(request, 'loginoutSuccess.html')
    else:
        return render(request, 'logoutError.html')
