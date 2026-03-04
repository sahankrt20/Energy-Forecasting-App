from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health),
    path('forecast/', views.forecast),
    path('model-info/', views.model_info),
    path('compare/', views.compare_models),
]
