from django.apps import AppConfig


class AgentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'agents'
    verbose_name = 'Research Agents'

    def ready(self):
        """Import signals when app is ready."""
        import agents.signals
