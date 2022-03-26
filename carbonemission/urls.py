"""carbonemission URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from carbonemission import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns 


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('result/', views.result, name='result'),
    path('visualize/', views.visualize, name='visualize'),
    path('A_visualize/', views.A_visualize, name='A_visualize'),
    path('N_visualize/', views.N_visualize, name='N_visualize'),
    path('input_visualize/', views.input_visualize, name='input_visualize'),
    path('input_predict/', views.input_predict, name='input_predict'),
    path('input_Compare/', views.input_Compare, name='input_Compare'),
    path('com_res/', views.com_res, name='com_res'),
    path('solution/', views.solution, name='solution'),
    path('A_InputVisualize/', views.A_InputVisualize, name='A_InputVisualize'),
    path('N_InputVisualize/', views.N_InputVisualize, name='N_InputVisualize'),
]
urlpatterns +=staticfiles_urlpatterns()
