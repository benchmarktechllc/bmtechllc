"""Ed25519 verification of Discord interaction requests."""

from __future__ import annotations

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey


def verify_signature(
    public_key_hex: str, signature_hex: str | None, timestamp: str | None, body: bytes
) -> bool:
    """True only if `signature_hex` signs `timestamp + body` under Discord's public key.

    Any missing header or malformed hex counts as invalid; this never raises.
    """
    if not signature_hex or not timestamp:
        return False
    try:
        key = VerifyKey(bytes.fromhex(public_key_hex))
        key.verify(timestamp.encode() + body, bytes.fromhex(signature_hex))
    except (ValueError, TypeError, BadSignatureError):
        return False
    return True
