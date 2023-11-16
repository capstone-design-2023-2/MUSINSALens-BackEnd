from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='kakao'),
    path('oauth/kakao/login/', views.kakao_login),
    path('oauth/kakao/callback/', views.kakao_login_redirect),
    path('oauth/kakao/logout/', views.kakao_logout),
]