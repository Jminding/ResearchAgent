"""
URL configuration for agents app.
"""

from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Authentication
    path('', views.dashboard_view, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='agents/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Dashboard and Profile
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('settings/', views.profile_settings_view, name='profile_settings'),

    # Research Operations
    path('research/new/', views.new_research_view, name='new_research'),
    path('research/<str:session_id>/', views.session_detail_view, name='session_detail'),
    path('research/<str:session_id>/progress/', views.session_progress_api, name='session_progress_api'),
    path('research/<str:session_id>/delete/', views.delete_session_view, name='delete_session'),

    # File Downloads
    path('research/<str:session_id>/paper/', views.download_paper_view, name='download_paper'),
    path('research/<str:session_id>/paper/view/', views.view_paper_view, name='view_paper'),
    path('files/<int:file_id>/download/', views.download_file_view, name='download_file'),

    # Peer Review Feedback
    path('research/<str:session_id>/feedback/', views.submit_feedback_view, name='submit_feedback'),
]
