"""
Django forms for the research agent platform.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, ResearchSession, PeerReviewFeedback


class SignUpForm(UserCreationForm):
    """
    Extended user registration form with email and API key.
    """
    email = forms.EmailField(
        max_length=254,
        required=True,
        help_text='Required. Enter a valid email address.',
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email'
        })
    )

    anthropic_api_key = forms.CharField(
        max_length=200,
        required=False,
        help_text='Optional. You can add this later in settings.',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Anthropic API Key (optional)'
        })
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Username'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm Password'
        })

    def save(self, commit=True):
        """Save user and set API key in profile if provided."""
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']

        if commit:
            user.save()
            # Set API key in profile if provided
            api_key = self.cleaned_data.get('anthropic_api_key')
            if api_key:
                user.profile.set_anthropic_api_key(api_key)
                user.profile.save()

        return user


class UserProfileForm(forms.ModelForm):
    """
    Form for updating user profile settings.
    """
    anthropic_api_key = forms.CharField(
        max_length=200,
        required=False,
        help_text='Your Anthropic API key (stored encrypted)',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter new API key to update'
        })
    )

    class Meta:
        model = UserProfile
        fields = ['default_research_mode']
        widgets = {
            'default_research_mode': forms.Select(attrs={
                'class': 'form-control'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Show if API key is set (but not the actual key)
        if self.instance.has_api_key:
            self.fields['anthropic_api_key'].help_text = '✓ API key is set. Enter new key to update.'
        else:
            self.fields['anthropic_api_key'].help_text = '⚠ No API key set. You need this to run research.'

    def save(self, commit=True):
        """Save profile and update API key if provided."""
        profile = super().save(commit=False)

        # Update API key if provided
        api_key = self.cleaned_data.get('anthropic_api_key')
        if api_key:
            profile.set_anthropic_api_key(api_key)

        if commit:
            profile.save()

        return profile


class ResearchSubmissionForm(forms.ModelForm):
    """
    Form for submitting a new research request.
    """
    class Meta:
        model = ResearchSession
        fields = ['topic', 'mode']
        widgets = {
            'topic': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter your research topic or question...\n\nExample: "Investigate the use of graph neural networks for anomaly detection in financial transaction networks"'
            }),
            'mode': forms.Select(attrs={
                'class': 'form-control'
            })
        }
        help_texts = {
            'topic': 'Describe what you want to research. Be specific and detailed.',
            'mode': 'Discovery mode runs follow-up experiments; Demo mode only lists them.'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make topic label more descriptive
        self.fields['topic'].label = 'Research Topic'
        self.fields['mode'].label = 'Research Mode'


class PeerReviewFeedbackForm(forms.ModelForm):
    """
    Form for submitting peer review feedback for revision.
    """
    class Meta:
        model = PeerReviewFeedback
        fields = ['feedback_text']
        widgets = {
            'feedback_text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 10,
                'placeholder': 'Paste peer review feedback here...\n\nExample:\n- The methodology section lacks detail on parameter selection\n- Figure 2 needs better labeling\n- Add comparison with baseline method XYZ\n- Clarify the assumptions in Section 3.2'
            })
        }
        help_texts = {
            'feedback_text': 'Paste any peer review feedback or revision requests. The research agent will address them.'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['feedback_text'].label = 'Peer Review Feedback'
