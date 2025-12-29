"""
Database models for the research agent platform.
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from .encryption import encrypt_api_key, decrypt_api_key


class UserProfile(models.Model):
    """
    Extended user profile with encrypted API key storage.

    Linked to Django's built-in User model via OneToOneField.
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
    )

    # Encrypted Anthropic API key
    encrypted_anthropic_api_key = models.BinaryField(
        blank=True,
        null=True,
        help_text="Encrypted Anthropic API key"
    )

    # Preferences
    default_research_mode = models.CharField(
        max_length=20,
        choices=[
            ('discovery', 'Discovery Mode'),
            ('demo', 'Demo Mode'),
        ],
        default='demo',
        help_text="Default mode for new research sessions"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'

    def __str__(self):
        return f"Profile for {self.user.username}"

    def set_anthropic_api_key(self, api_key: str):
        """
        Encrypt and store the Anthropic API key.

        Args:
            api_key: Plain text API key
        """
        if api_key:
            self.encrypted_anthropic_api_key = encrypt_api_key(api_key)
        else:
            self.encrypted_anthropic_api_key = None

    def get_anthropic_api_key(self) -> str:
        """
        Decrypt and return the Anthropic API key.

        Returns:
            Decrypted API key string, or empty string if not set
        """
        if self.encrypted_anthropic_api_key:
            return decrypt_api_key(self.encrypted_anthropic_api_key)
        return ''

    @property
    def has_api_key(self) -> bool:
        """Check if user has an API key set."""
        return bool(self.encrypted_anthropic_api_key)


class ResearchSession(models.Model):
    """
    Represents a single research session/run.

    Stores metadata about the research topic, status, and timestamps.
    """

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('revision', 'Under Revision'),
    ]

    # Ownership
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='research_sessions'
    )

    # Research details
    topic = models.CharField(
        max_length=500,
        help_text="Research topic or query"
    )

    title = models.CharField(
        max_length=200,
        blank=True,
        help_text="Short display title (auto-generated from topic)"
    )

    session_id = models.CharField(
        max_length=100,
        unique=True,
        help_text="Unique session identifier (e.g., session_20251223_120000)"
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )

    mode = models.CharField(
        max_length=20,
        choices=[
            ('discovery', 'Discovery'),
            ('demo', 'Demo'),
        ],
        default='demo'
    )

    # Paths
    session_directory = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to session directory"
    )

    transcript_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to session transcript"
    )

    # Error tracking
    error_message = models.TextField(
        blank=True,
        help_text="Error message if session failed"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Research Session'
        verbose_name_plural = 'Research Sessions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', '-created_at']),
            models.Index(fields=['session_id']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.session_id} - {self.title or self.topic[:50]}"

    def save(self, *args, **kwargs):
        """Override save to auto-generate title if not set."""
        if not self.title and self.topic:
            # Generate title from first 100 chars of topic
            self.title = self.topic[:100].strip()
            # Clean up title - remove newlines, extra spaces
            self.title = ' '.join(self.title.split())
            # Add ellipsis if truncated
            if len(self.topic) > 100:
                self.title = self.title.rstrip('.,!?') + '...'
        super().save(*args, **kwargs)

    def mark_running(self):
        """Mark session as running."""
        self.status = 'running'
        self.started_at = timezone.now()
        # Ensure title is set before saving
        if not self.title and self.topic:
            self.title = self.topic[:100].strip()
            self.title = ' '.join(self.title.split())
            if len(self.topic) > 100:
                self.title = self.title.rstrip('.,!?') + '...'
            self.save(update_fields=['status', 'started_at', 'title', 'updated_at'])
        else:
            self.save(update_fields=['status', 'started_at', 'updated_at'])

    def mark_completed(self):
        """Mark session as completed."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'completed_at', 'updated_at'])

    def mark_failed(self, error_message: str = ''):
        """Mark session as failed with error message."""
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'error_message', 'completed_at', 'updated_at'])

    def mark_revision(self):
        """Mark session as under revision."""
        self.status = 'revision'
        self.save(update_fields=['status', 'updated_at'])

    @property
    def duration(self):
        """Calculate session duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return timezone.now() - self.started_at
        return None

    @property
    def duration_display(self):
        """Return duration formatted in a human-readable way (rounded to minutes)."""
        duration = self.duration
        if not duration:
            return None

        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


class ResearchPaper(models.Model):
    """
    Stores the final research paper (PDF) for a session.
    """

    session = models.OneToOneField(
        ResearchSession,
        on_delete=models.CASCADE,
        related_name='paper'
    )

    # PDF file
    pdf_file = models.FileField(
        upload_to='research_papers/%Y/%m/%d/',
        help_text="Final research paper PDF"
    )

    # Metadata
    title = models.CharField(
        max_length=500,
        blank=True,
        help_text="Paper title extracted from PDF"
    )

    file_size = models.IntegerField(
        default=0,
        help_text="File size in bytes"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Research Paper'
        verbose_name_plural = 'Research Papers'

    def __str__(self):
        return f"Paper for {self.session.session_id}"

    def save(self, *args, **kwargs):
        """Override save to update file size."""
        if self.pdf_file:
            self.file_size = self.pdf_file.size
        super().save(*args, **kwargs)


class GeneratedFile(models.Model):
    """
    Stores auxiliary files generated during research (CSVs, logs, etc.).
    """

    FILE_TYPE_CHOICES = [
        ('csv', 'CSV Data'),
        ('json', 'JSON Data'),
        ('log', 'Log File'),
        ('txt', 'Text File'),
        ('png', 'Image'),
        ('other', 'Other'),
    ]

    session = models.ForeignKey(
        ResearchSession,
        on_delete=models.CASCADE,
        related_name='generated_files'
    )

    # File details
    file = models.FileField(
        upload_to='research_files/%Y/%m/%d/',
        help_text="Generated file"
    )

    file_type = models.CharField(
        max_length=20,
        choices=FILE_TYPE_CHOICES,
        help_text="Type of generated file"
    )

    filename = models.CharField(
        max_length=255,
        help_text="Original filename"
    )

    description = models.TextField(
        blank=True,
        help_text="Description of file contents"
    )

    file_size = models.IntegerField(
        default=0,
        help_text="File size in bytes"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Generated File'
        verbose_name_plural = 'Generated Files'
        ordering = ['created_at']

    def __str__(self):
        return f"{self.filename} ({self.file_type})"

    def save(self, *args, **kwargs):
        """Override save to update file size and extract filename."""
        if self.file:
            self.file_size = self.file.size
            if not self.filename:
                self.filename = self.file.name
        super().save(*args, **kwargs)


class PeerReviewFeedback(models.Model):
    """
    Stores peer review feedback submitted by users for revision.
    """

    session = models.ForeignKey(
        ResearchSession,
        on_delete=models.CASCADE,
        related_name='feedback_submissions'
    )

    # Feedback details
    feedback_text = models.TextField(
        help_text="Peer review feedback or revision requests"
    )

    # Status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )

    # Response
    revision_notes = models.TextField(
        blank=True,
        help_text="Notes from the revision process"
    )

    error_message = models.TextField(
        blank=True,
        help_text="Error message if processing failed"
    )

    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Peer Review Feedback'
        verbose_name_plural = 'Peer Review Feedbacks'
        ordering = ['-submitted_at']

    def __str__(self):
        return f"Feedback for {self.session.session_id} at {self.submitted_at}"

    def mark_processing(self):
        """Mark feedback as being processed."""
        self.status = 'processing'
        self.save(update_fields=['status'])

    def mark_completed(self, revision_notes: str = ''):
        """Mark feedback processing as completed."""
        self.status = 'completed'
        self.revision_notes = revision_notes
        self.processed_at = timezone.now()
        self.save(update_fields=['status', 'revision_notes', 'processed_at'])

    def mark_failed(self, error_message: str):
        """Mark feedback processing as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.processed_at = timezone.now()
        self.save(update_fields=['status', 'error_message', 'processed_at'])
