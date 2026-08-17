from __future__ import annotations

from typing import Callable

MIN_STRIPE_ROWS_PER_UNIT = 4


def install() -> None:
    """Restore a physically useful minimum width for Cu/Au stripe models.

    Ratio labels remain Cu:Au = 1:1, 2:1, 3:1, 1:2, 1:3, but one ratio
    unit corresponds to four atomic rows rather than two.  This gives:
      1:1 -> 4 Cu + 4 Au rows
      2:1 -> 8 Cu + 4 Au rows
      3:1 -> 12 Cu + 4 Au rows
      1:2 -> 4 Cu + 8 Au rows
      1:3 -> 4 Cu + 12 Au rows
    """
    from tools import chat_workflow_patch as chat
    from tools import nersc_portability_patch as portability
    from tools import stripe_interface_patch as stripe

    stripe.STRIPE_ROW_BASE_UNIT = MIN_STRIPE_ROWS_PER_UNIT

    if getattr(portability, "_stripe_width_compat_installed", False):
        return

    original_source_patch: Callable[[str], str] = portability.patched_main_source

    def patched_source(source: str) -> str:
        patched = original_source_patch(source)
        patched = patched.replace("2-row base unit", "4-row base unit")
        patched = patched.replace('"stripe_row_base_unit": 2', '"stripe_row_base_unit": 4')
        return patched

    portability.patched_main_source = patched_source
    portability._stripe_width_compat_installed = True

    # Keep all references used by the chat layer on the same live stripe module.
    chat.stripe.STRIPE_ROW_BASE_UNIT = MIN_STRIPE_ROWS_PER_UNIT
