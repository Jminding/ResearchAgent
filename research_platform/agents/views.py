"""
Views for the research agent platform.
"""

import threading
from datetime import datetime
from pathlib import Path
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, FileResponse, Http404, JsonResponse
from django.core.paginator import Paginator

from .models import ResearchSession, ResearchPaper, GeneratedFile, PeerReviewFeedback
from .forms import SignUpForm, UserProfileForm, ResearchSubmissionForm, PeerReviewFeedbackForm
from .services import run_research_sync, process_feedback_sync
from .progress_tracker import SessionProgressTracker


def signup_view(request):
    """User registration view."""
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully! Welcome to the Research Platform.')
            return redirect('dashboard')
    else:
        form = SignUpForm()

    return render(request, 'agents/signup.html', {'form': form})


@login_required
def dashboard_view(request):
    """
    Dashboard view showing user's research sessions.
    """
    # Get user's research sessions
    sessions = ResearchSession.objects.filter(owner=request.user).order_by('-created_at')

    # Pagination
    paginator = Paginator(sessions, 10)  # 10 sessions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Check if user has API key set
    has_api_key = request.user.profile.has_api_key

    context = {
        'page_obj': page_obj,
        'has_api_key': has_api_key,
        'total_sessions': sessions.count(),
        'completed_sessions': sessions.filter(status='completed').count(),
        'running_sessions': sessions.filter(status='running').count(),
    }

    return render(request, 'agents/dashboard.html', context)


@login_required
def profile_settings_view(request):
    """User profile settings view."""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile settings updated successfully.')
            return redirect('profile_settings')
    else:
        form = UserProfileForm(instance=request.user.profile)

    return render(request, 'agents/profile_settings.html', {'form': form})


@login_required
def new_research_view(request):
    """View for submitting a new research request."""
    # Check if user has API key
    if not request.user.profile.has_api_key:
        messages.error(
            request,
            'Please set your Anthropic API key in profile settings before starting research.'
        )
        return redirect('profile_settings')

    if request.method == 'POST':
        form = ResearchSubmissionForm(request.POST)
        if form.is_valid():
            # Create session
            session = form.save(commit=False)
            session.owner = request.user

            # Generate session ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session.session_id = f"session_{timestamp}"

            session.save()

            # Start research in background thread
            api_key = request.user.profile.get_anthropic_api_key()

            def run_in_background():
                """Run research in background thread."""
                run_research_sync(session, api_key)

            thread = threading.Thread(target=run_in_background, daemon=True)
            thread.start()

            messages.success(
                request,
                f'Research started! Session ID: {session.session_id}. '
                'Check the dashboard for progress.'
            )
            return redirect('session_detail', session_id=session.session_id)
    else:
        form = ResearchSubmissionForm()
        # Set default mode from user profile
        form.initial['mode'] = request.user.profile.default_research_mode

    return render(request, 'agents/new_research.html', {'form': form})


@login_required
def session_detail_view(request, session_id):
    """View for session details."""
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    # Get generated files
    generated_files = session.generated_files.all()

    # Get feedback submissions
    feedbacks = session.feedback_submissions.all().order_by('-submitted_at')

    # Check if paper exists
    has_paper = hasattr(session, 'paper')

    # Get progress if session is running
    current_step = None
    if session.status == 'running' and session.session_directory:
        try:
            tracker = SessionProgressTracker(Path(session.session_directory))
            current_step = tracker.get_current_step()
        except Exception:
            pass

    context = {
        'session': session,
        'generated_files': generated_files,
        'feedbacks': feedbacks,
        'has_paper': has_paper,
        'current_step': current_step,
    }

    return render(request, 'agents/session_detail.html', context)


@login_required
def download_paper_view(request, session_id):
    """Download research paper PDF."""
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    if not hasattr(session, 'paper'):
        raise Http404("Paper not found")

    paper = session.paper
    response = FileResponse(paper.pdf_file.open('rb'), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{paper.pdf_file.name}"'

    return response


@login_required
def view_paper_view(request, session_id):
    """View research paper PDF inline."""
    import os
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    if not hasattr(session, 'paper'):
        raise Http404("Paper not found")

    paper = session.paper
    filename = os.path.basename(paper.pdf_file.name)

    # Open the file and create response
    pdf_file = paper.pdf_file.open('rb')
    response = FileResponse(pdf_file, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    response['X-Frame-Options'] = 'SAMEORIGIN'  # Allow iframe from same origin

    return response


@login_required
def download_file_view(request, file_id):
    """Download a generated file."""
    gen_file = get_object_or_404(GeneratedFile, id=file_id)

    # Check ownership
    if gen_file.session.owner != request.user:
        raise Http404("File not found")

    # Determine content type
    content_types = {
        'csv': 'text/csv',
        'json': 'application/json',
        'txt': 'text/plain',
        'log': 'text/plain',
        'png': 'image/png',
    }
    content_type = content_types.get(gen_file.file_type, 'application/octet-stream')

    response = FileResponse(gen_file.file.open('rb'), content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="{gen_file.filename}"'

    return response


@login_required
def submit_feedback_view(request, session_id):
    """Submit peer review feedback for a session."""
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    # Check if session is completed
    if session.status not in ['completed', 'revision']:
        messages.error(
            request,
            'Can only submit feedback for completed research sessions.'
        )
        return redirect('session_detail', session_id=session_id)

    if request.method == 'POST':
        form = PeerReviewFeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.session = session
            feedback.save()

            # Process feedback in background thread
            api_key = request.user.profile.get_anthropic_api_key()

            def process_in_background():
                """Process feedback in background thread."""
                process_feedback_sync(feedback, api_key)

            thread = threading.Thread(target=process_in_background, daemon=True)
            thread.start()

            messages.success(
                request,
                'Feedback submitted! The research agent is processing your feedback.'
            )
            return redirect('session_detail', session_id=session_id)
    else:
        form = PeerReviewFeedbackForm()

    context = {
        'session': session,
        'form': form,
    }

    return render(request, 'agents/submit_feedback.html', context)


@login_required
def session_progress_api(request, session_id):
    """
    API endpoint for real-time session progress.

    Returns JSON with current progress information.
    """
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    if not session.session_directory:
        return JsonResponse({
            'status': session.status,
            'current_step': 'Initializing...',
            'progress': {}
        })

    try:
        tracker = SessionProgressTracker(Path(session.session_directory))
        progress = tracker.get_progress()
        current_step = tracker.get_current_step()

        return JsonResponse({
            'status': session.status,
            'current_step': current_step,
            'progress': progress,
            'session_id': session.session_id
        })
    except Exception as e:
        return JsonResponse({
            'status': session.status,
            'current_step': 'Loading...',
            'error': str(e)
        }, status=500)


@login_required
def delete_session_view(request, session_id):
    """Delete a research session (soft delete)."""
    session = get_object_or_404(
        ResearchSession,
        session_id=session_id,
        owner=request.user
    )

    if request.method == 'POST':
        session_topic = session.title or session.topic[:50]
        session.delete()

        messages.success(
            request,
            f'Research session "{session_topic}..." deleted successfully.'
        )
        return redirect('dashboard')

    return render(request, 'agents/delete_session.html', {'session': session})
