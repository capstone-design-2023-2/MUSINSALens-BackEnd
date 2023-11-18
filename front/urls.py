from django.contrib import admin
from django.urls import path, include
from .views import ItemListCreateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/items', ItemListCreateView.as_view(), name = 'item-list-create'),
]