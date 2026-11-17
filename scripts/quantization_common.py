"""Explicit campaign identities for fixed quantization recipes."""


def campaign_index(args):
    return 12 if args.mixed else 11 if args.per_channel else 10


def variant_names(args):
    if args.mixed:
        return ["fp32", "int8-mixed1", "int8-mixed2", "int8-mixed3"]
    return (
        ["fp32", "int8-dense-pc", "int8-full-pc"]
        if args.per_channel
        else ["fp32", "int8-dense", "int8-full"]
    )
