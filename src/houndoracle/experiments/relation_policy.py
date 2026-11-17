"""Experimental evidence-only inference contract, without dataset-specific examples."""


def audit_instructions():
    return (
        "Before selecting a verdict, check each required relation in the proposition against "
        "the supplied passages. Preserve subject, predicate, object, date, amount, comparison "
        "and logical scope. Resolve pronouns only when the text or page title grounds them. "
        "A page title identifies context; it does not establish unmentioned properties. "
        "YES requires every necessary relation to follow from the passages. "
        "NO requires an explicitly incompatible fact about the same entity and scope; "
        "missing information is not contradiction. If neither is established, choose MAYBE. "
        "Do not fill missing facts using memory, plausibility, external aliases or typical "
        "real-world behavior. Paraphrases may preserve a relation, but entity identity and "
        "quantities cannot be guessed. For compound claims, one grounded contradiction can "
        "refute a conjunction; a missing conjunct alone cannot. Mentally audit both a proposed "
        "verdict and its strongest evidence-based alternative, then emit only the required JSON."
    )
