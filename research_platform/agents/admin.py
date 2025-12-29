"""
Django admin configuration for agents app.
"""

from django.contrib import admin
from .models import UserProfile, ResearchSession, ResearchPaper, GeneratedFile, PeerReviewFeedback


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """Admin interface for UserProfile."""
    list_display = ['user', 'has_api_key', 'default_research_mode', 'created_at']
    list_filter = ['default_research_mode', 'created_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('API Configuration', {
            'fields': ('encrypted_anthropic_api_key',),
            'description': 'API key is stored encrypted'
        }),
        ('Preferences', {
            'fields': ('default_research_mode',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ResearchSession)
class ResearchSessionAdmin(admin.ModelAdmin):
    """Admin interface for ResearchSession."""
    list_display = ['session_id', 'owner', 'topic_short', 'status', 'mode', 'created_at']
    list_filter = ['status', 'mode', 'created_at']
    search_fields = ['session_id', 'topic', 'owner__username']
    readonly_fields = ['session_id', 'created_at', 'started_at', 'completed_at', 'updated_at']
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Session Info', {
            'fields': ('session_id', 'owner', 'status', 'mode')
        }),
        ('Research Details', {
            'fields': ('topic',)
        }),
        ('Paths', {
            'fields': ('session_directory', 'transcript_path'),
            'classes': ('collapse',)
        }),
        ('Error Tracking', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'completed_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def topic_short(self, obj):
        """Return shortened topic for display."""
        return obj.topic[:50] + '...' if len(obj.topic) > 50 else obj.topic

    topic_short.short_description = 'Topic'


@admin.register(ResearchPaper)
class ResearchPaperAdmin(admin.ModelAdmin):
    """Admin interface for ResearchPaper."""
    list_display = ['session', 'title', 'file_size_mb', 'created_at']
    search_fields = ['title', 'session__session_id']
    readonly_fields = ['file_size', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'

    def file_size_mb(self, obj):
        """Display file size in MB."""
        return f"{obj.file_size / (1024 * 1024):.2f} MB"

    file_size_mb.short_description = 'File Size'


@admin.register(GeneratedFile)
class GeneratedFileAdmin(admin.ModelAdmin):
    """Admin interface for GeneratedFile."""
    list_display = ['filename', 'session', 'file_type', 'file_size_kb', 'created_at']
    list_filter = ['file_type', 'created_at']
    search_fields = ['filename', 'description', 'session__session_id']
    readonly_fields = ['file_size', 'created_at']
    date_hierarchy = 'created_at'

    def file_size_kb(self, obj):
        """Display file size in KB."""
        return f"{obj.file_size / 1024:.2f} KB"

    file_size_kb.short_description = 'File Size'


@admin.register(PeerReviewFeedback)
class PeerReviewFeedbackAdmin(admin.ModelAdmin):
    """Admin interface for PeerReviewFeedback."""
    list_display = ['session', 'status', 'submitted_at', 'processed_at']
    list_filter = ['status', 'submitted_at']
    search_fields = ['feedback_text', 'session__session_id']
    readonly_fields = ['submitted_at', 'processed_at']
    date_hierarchy = 'submitted_at'

    fieldsets = (
        ('Session', {
            'fields': ('session', 'status')
        }),
        ('Feedback', {
            'fields': ('feedback_text',)
        }),
        ('Response', {
            'fields': ('revision_notes', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('submitted_at', 'processed_at'),
            'classes': ('collapse',)
        }),
    )
