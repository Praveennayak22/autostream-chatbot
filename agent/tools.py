"""
Tool: mock_lead_capture
Called ONLY after name, email, and platform have all been collected.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API that simulates submitting a qualified lead to a CRM."""
    separator = "=" * 55
    print(f"\n{separator}")
    print("  🎯  LEAD CAPTURED SUCCESSFULLY!")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{separator}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"
