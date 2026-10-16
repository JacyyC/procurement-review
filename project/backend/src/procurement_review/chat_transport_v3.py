"""Opt-in safe-egress.v2 transport; the legacy V2 configuration stays unchanged."""

import hashlib
from pathlib import Path
from typing import Literal
from pydantic import Field

from .business_v3_security import ProtectedOutputV3
from .chat_transport_v2 import (
    ChatTransportV2,
    EndpointConfigV2,
    decode_response,
    AttemptResult,
)
from .gold_v3_contracts import payload_sha256


class EndpointConfigV3(EndpointConfigV2):
    egress_contract: Literal["safe-egress.v2"] = "safe-egress.v2"
    protection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def fingerprint(self):
        return payload_sha256(
            {
                "parent": super().fingerprint,
                "egress": self.egress_contract,
                "implementation": {
                    name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()
                    ).hexdigest()
                    for name in ("business_v3_security.py", "chat_transport_v3.py")
                },
            }
        )


class ChatTransportV3(ChatTransportV2):
    def __init__(
        self,
        config: EndpointConfigV3,
        credential,
        *,
        protection: ProtectedOutputV3,
        transport=None,
    ):
        if not isinstance(config, EndpointConfigV3):
            raise ValueError("explicit safe-egress.v2 config required")
        if config.protection_sha256 != protection.fingerprint:
            raise ValueError("protection registry mismatch")
        super().__init__(config, credential, transport=transport)
        self.protection = protection

    @property
    def configuration_fingerprint(self):
        return payload_sha256(
            {
                "endpoint": self.config.fingerprint,
                "protection": self.protection.fingerprint,
            }
        )

    async def send(self, payload, request, seconds):
        if self.config.protection_sha256 != self.protection.fingerprint:
            return AttemptResult(error_code="security_boundary")
        return await super().send(payload, request, seconds)

    def decode(self, response, request, credential):
        return decode_response(
            response,
            request,
            self.config,
            credential,
            output_guard=self.protection.assert_safe,
        )
