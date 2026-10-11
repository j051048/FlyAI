"""Verify fresh signed suffixes and the separate, bounded prefix-restore contract.

A restore declaration is an audit statement, not a proof of honest arithmetic.
The expected binding comes from the coordinator's authenticated commit barrier.
"""
from .receipt import ReceiptError, validate_conversation_restore

RESTORED_SCOPE = "fresh_suffix_with_signed_prefix_restore"
COMPLETE_SCOPE = "complete_final_attempt"
RESTORE_STATEMENT = "restored prefix state; receipt chunks count only new forwards"


def verify_prefix_restore(receipts, binding, assignments):
    if not isinstance(binding, dict) or set(binding) != {
            "transaction_id", "prefix", "identity_sha256", "stages"}:
        raise ReceiptError("complete authenticated prefix-restore binding required")
    rows = binding["stages"]
    if not isinstance(rows, list) or len(rows) != len(assignments) or len(rows) > 256:
        raise ReceiptError("prefix-restore binding omits assigned stages")
    expected = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"stage", "pubkey", "entry_id", "state_digest"}:
            raise ReceiptError("invalid prefix-restore stage binding")
        pub = row["pubkey"]
        if type(row["stage"]) is not int or row["stage"] != index or pub not in assignments or pub in expected:
            raise ReceiptError("prefix-restore signer/stage is not in the assignment")
        note = {"schema": "v4-conversation-restore/1", "transaction_id": binding["transaction_id"],
                "prefix": binding["prefix"], "identity_sha256": binding["identity_sha256"],
                "entry_id": row["entry_id"], "state_digest": row["state_digest"], "scope": RESTORE_STATEMENT}
        try:
            expected[pub] = validate_conversation_restore(note)
        except (ValueError, TypeError) as error:
            raise ReceiptError("invalid authenticated restore metadata") from error
    if set(expected) != set(assignments):
        raise ReceiptError("prefix-restore assignment coverage differs")
    seen = set()
    for receipt in receipts:
        pub = receipt.get("pubkey")
        if pub in seen or pub not in expected or receipt.get("conversation_restore") != expected[pub]:
            raise ReceiptError("fresh receipt does not bind its authenticated prefix restore")
        seen.add(pub)
    if seen != set(expected):
        raise ReceiptError("fresh suffix omits prefix-restore participants")
