from django.urls import path
from classifier import views

urlpatterns = [
    path('start', views.start, name='start'),
    path('stop/<int:pk>', views.stop, name='stop'),
    path('stop_all', views.stop_all, name='stop_all')
]
