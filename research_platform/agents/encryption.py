"""
Encryption utilities for securely storing API keys.

Uses Fernet symmetric encryption from the cryptography library.
The master encryption key should be stored in environment variables.
"""

from cryptography.fernet import Fernet
from django.conf import settings


class EncryptionService:
    """Service for encrypting and decrypting sensitive data."""

    def __init__(self):
        """Initialize with encryption key from settings."""
        if not settings.ENCRYPTION_KEY:
            raise ValueError(
                "ENCRYPTION_KEY not set in settings. "
                "Set the ENCRYPTION_KEY environment variable."
            )

        # Ensure key is bytes
        if isinstance(settings.ENCRYPTION_KEY, str):
            self._key = settings.ENCRYPTION_KEY.encode()
        else:
            self._key = settings.ENCRYPTION_KEY

        self._fernet = Fernet(self._key)

    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt plaintext string.

        Args:
            plaintext: The string to encrypt

        Returns:
            Encrypted bytes
        """
        if not plaintext:
            return b''

        plaintext_bytes = plaintext.encode('utf-8')
        encrypted = self._fernet.encrypt(plaintext_bytes)
        return encrypted

    def decrypt(self, encrypted_bytes: bytes) -> str:
        """
        Decrypt encrypted bytes back to string.

        Args:
            encrypted_bytes: The encrypted data

        Returns:
            Decrypted string

        Raises:
            cryptography.fernet.InvalidToken: If decryption fails
        """
        if not encrypted_bytes:
            return ''

        decrypted = self._fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')


# Global encryption service instance
_encryption_service = None


def get_encryption_service() -> EncryptionService:
    """Get or create the global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def encrypt_api_key(api_key: str) -> bytes:
    """Convenience function to encrypt an API key."""
    return get_encryption_service().encrypt(api_key)


def decrypt_api_key(encrypted_key: bytes) -> str:
    """Convenience function to decrypt an API key."""
    return get_encryption_service().decrypt(encrypted_key)
