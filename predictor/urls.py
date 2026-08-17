from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('heart/', views.heart_predict, name='heart'),
    path('churn/', views.churn_predict, name='churn'),
    path('attrition/', views.attrition_predict, name='attrition'),
    path('census/', views.census_predict, name='census'),
    path('ames/', views.ames_predict, name='ames'),
    path('student/', views.student_predict, name='student'),
    path('privacy/', views.privacy, name='privacy'),
    path('terms/', views.terms, name='terms'),
    path('ecommerce/', views.segmentation_predict, name='ecommerce'),
    path('mbti/', views.mbti_predict, name='mbti'),
    path('plant-disease/', views.plant_disease_predict, name='plant_disease'),
]