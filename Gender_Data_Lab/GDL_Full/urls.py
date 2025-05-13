from django.urls import path
from . import views

urlpatterns = [
    # Main & Gender App Views
    path('', views.index, name='index'),
    path('ai_analytics/', views.ai_analytics, name='ai_analytics'),
    path('instruction/', views.instruction, name='instruction'),

    # Auth & Dashboard
    path('admin_login/', views.admin_login, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout, name='logout'),
    path('warehouse/', views.dashboard, name='warehouse'),

    # File Uploads
    path('fileupload/', views.fileupload, name='fileupload'),
    path('upload/', views.upload_file, name='upload_file'),

    # Indicator Search Integration
    path('indicator_search/', views.indicator_search, name='indicator_search'),
    path('upload_indicator_file/', views.upload_indicator_file, name='upload_indicator_file'),
    path('analyze_indicator/<int:survey_id>/', views.analyze_indicator, name='analyze_indicator'),
    path('get-indicators/<int:survey_id>/', views.get_indicators, name='get_indicators'),
    path('download/', views.download_report, name='download_report'),
    path("chat/", views.chat, name="chat")
]
