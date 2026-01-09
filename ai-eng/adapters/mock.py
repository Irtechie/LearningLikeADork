# Purpose: Mock adapter for testing.
# Created: 2026-01-05
# Author: MWR

class MockAdapter:
    def generate(self, user_input: str) -> str:
        return (
            "Quirk Quirk has the abilities: UIPower 36.8% Attack (0SP) Noxious Strike (1SP) Regenerate (2SP) Poison Spit (3SP)."
        )
