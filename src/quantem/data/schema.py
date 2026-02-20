"""Metadata schema definition and validation for quantem.data datasets."""

SCHEMA_VERSION = "1.0"

VALID_TECHNIQUES = [
    "4dstem",
    "hrtem",
    "eels",
    "tomo",
    "diffraction",
    "complex",
    "image",
]

# Fields that must be present in every metadata JSON.
REQUIRED_FIELDS = {
    "schema_version",
    "name",
    "technique",
    "description",
    "data",
    "attribution",
}

REQUIRED_DATA_FIELDS = {"shape", "dtype"}
REQUIRED_ATTRIBUTION_FIELDS = {"contributor", "license"}


def validate(meta: dict) -> list[str]:
    """Validate a metadata dict against the schema.

    Returns a list of error strings (empty if valid).
    """
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in meta:
            errors.append(f"Missing required field: {field!r}")

    if "technique" in meta and meta["technique"] not in VALID_TECHNIQUES:
        errors.append(
            f"Invalid technique {meta['technique']!r}. "
            f"Must be one of: {VALID_TECHNIQUES}"
        )

    if "data" in meta:
        data = meta["data"]
        if not isinstance(data, dict):
            errors.append("'data' must be a dict")
        else:
            for field in REQUIRED_DATA_FIELDS:
                if field not in data:
                    errors.append(f"Missing required field: data.{field!r}")

    if "attribution" in meta:
        attr = meta["attribution"]
        if not isinstance(attr, dict):
            errors.append("'attribution' must be a dict")
        else:
            for field in REQUIRED_ATTRIBUTION_FIELDS:
                if field not in attr:
                    errors.append(
                        f"Missing required field: attribution.{field!r}"
                    )

    return errors


def make_template(
    name: str,
    technique: str,
    shape: list[int] | tuple[int, ...],
    dtype: str = "float32",
    description: str = "",
    contributor: str = "",
    license: str = "CC-BY-4.0",
) -> dict:
    """Create a metadata dict with required fields pre-filled."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "technique": technique,
        "description": description,
        "data": {
            "shape": list(shape),
            "dtype": dtype,
        },
        "instrument": {},
        "calibration": {},
        "processing": {},
        "attribution": {
            "contributor": contributor,
            "license": license,
        },
    }
