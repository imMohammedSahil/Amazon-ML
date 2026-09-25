import re
import unicodedata
import polars as pl

# Legal business suffixes dictionary for normalization
LEGAL_SUFFIXES = [
    r"\bincorporated\b", r"\binc\b",
    r"\blimited liability company\b", r"\bllc\b", r"\bllp\b",
    r"\blimited\b", r"\bltd\b",
    r"\bcorporation\b", r"\bcorp\b",
    r"\bgmbh\b", r"\bag\b",
    r"\bsarl\b", r"\bsas\b", r"\bsa\b",
    r"\bpvt\b", r"\bprivate\b",
    r"\bco\b", r"\bcompany\b"
]
LEGAL_SUFFIX_REGEX = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)

# Address abbreviations mapping
ADDRESS_ABBR = {
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\broad\b": "rd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bfloor\b": "fl",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w"
}


def strip_accents(text: str) -> str:
    """Removes unicode accents (e.g., 'café' -> 'cafe', 'Société' -> 'Societe')."""
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(text))
        if unicodedata.category(c) != 'Mn'
    )


def clean_text_field(text: str) -> str:
    """Standardizes string: lowercase, strips accents, removes punctuation."""
    if text is None or text == "":
        return ""
    text = strip_accents(str(text).lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name_field(text: str) -> str:
    """Cleans business name and removes legal corporate suffixes."""
    cleaned = clean_text_field(text)
    if not cleaned:
        return ""
    cleaned = LEGAL_SUFFIX_REGEX.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_phone_field(phone: str) -> str:
    """Extracts only raw digits from phone string, taking the last 10 digits if longer."""
    if phone is None or phone == "":
        return ""
    digits = re.sub(r"\D", "", str(phone))
    return digits[-10:] if len(digits) >= 10 else digits


def clean_address_field(address: str) -> str:
    """Cleans address and normalizes road/unit abbreviations."""
    cleaned = clean_text_field(address)
    if not cleaned:
        return ""
    for pattern, repl in ADDRESS_ABBR.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class DataPreprocessor:
    """
    High-performance preprocessor for business entity tables.
    Works natively on Polars DataFrames using vectorized map operations.
    """
    def __init__(self):
        pass

    def clean_table(self, df: pl.DataFrame, source_name: str = "") -> pl.DataFrame:
        """
        Cleans and normalizes all fields in an entity table.
        Adds clean_* columns for downstream blocking and feature extraction.
        """
        cols = df.columns
        
        # Build vectorized transformations
        cleaned_df = df.with_columns([
            # Source indicator column
            pl.lit(source_name).alias("source") if "source" not in cols else pl.col("source"),
            
            # ID column as string
            pl.col("id").cast(pl.Utf8).alias("id"),
            
            # Clean Name
            pl.col("name").fill_null("").map_elements(
                clean_name_field, return_dtype=pl.Utf8
            ).alias("clean_name"),
            
            # Raw cleaned Name (with legal suffix retained for feature comparison)
            pl.col("name").fill_null("").map_elements(
                clean_text_field, return_dtype=pl.Utf8
            ).alias("clean_name_raw"),
            
            # Clean Address
            pl.col("address").fill_null("").map_elements(
                clean_address_field, return_dtype=pl.Utf8
            ).alias("clean_address"),
            
            # Clean City & State
            pl.col("city").fill_null("").map_elements(
                clean_text_field, return_dtype=pl.Utf8
            ).alias("clean_city"),
            pl.col("state").fill_null("").map_elements(
                clean_text_field, return_dtype=pl.Utf8
            ).alias("clean_state"),
            
            # Clean Zip & Phone
            pl.col("zip").fill_null("").cast(pl.Utf8).str.replace_all(r"\D", "").alias("clean_zip"),
            pl.col("phone").fill_null("").map_elements(
                clean_phone_field, return_dtype=pl.Utf8
            ).alias("clean_phone"),
            
            # Country normalization
            pl.col("country").fill_null("").map_elements(
                clean_text_field, return_dtype=pl.Utf8
            ).alias("clean_country")
        ])
        
        # Combined full address string for cross-field blocking
        cleaned_df = cleaned_df.with_columns(
            (pl.col("clean_address") + " " + pl.col("clean_city") + " " + pl.col("clean_state") + " " + pl.col("clean_zip"))
            .str.strip_chars()
            .alias("clean_full_address")
        )
        
        return cleaned_df
