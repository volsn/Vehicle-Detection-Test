from django.urls import path
from classifier import views

urlpatterns = [
    path('visualize/<int:pk>', views.stream_video_view, name='visualize'),
    path('classify/<int:sec>', views.classify_view, name='classify'),
]
