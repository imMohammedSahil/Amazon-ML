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
        Automatically maps entity_id -> id, business_name -> name, business_address -> address.
        Adds clean_* columns for downstream blocking and feature extraction.
        """
        cols = df.columns
        
        # Column standardization mapping
        id_col = "entity_id" if "entity_id" in cols else ("id" if "id" in cols else cols[0])
        name_col = "business_name" if "business_name" in cols else ("name" if "name" in cols else None)
        addr_col = "business_address" if "business_address" in cols else ("address" if "address" in cols else None)
        country_col = "country" if "country" in cols else None
        phone_col = "phone" if "phone" in cols else None
        city_col = "city" if "city" in cols else None
        state_col = "state" if "state" in cols else None
        zip_col = "zip" if "zip" in cols else None

        expressions = [
            # ID column
            pl.col(id_col).cast(pl.Utf8).alias("id"),
            # Source indicator column
            pl.lit(source_name).alias("source") if "source" not in cols else pl.col("source"),
            
            # Clean Name
            (pl.col(name_col).fill_null("").map_elements(clean_name_field, return_dtype=pl.Utf8) if name_col 
             else pl.lit("")).alias("clean_name"),
            
            # Raw Clean Name
            (pl.col(name_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if name_col 
             else pl.lit("")).alias("clean_name_raw"),
            
            # Clean Address
            (pl.col(addr_col).fill_null("").map_elements(clean_address_field, return_dtype=pl.Utf8) if addr_col 
             else pl.lit("")).alias("clean_address"),
            
            # Clean Country
            (pl.col(country_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if country_col 
             else pl.lit("")).alias("clean_country"),
            
            # Clean City, State, Zip, Phone (if present, else empty strings)
            (pl.col(city_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if city_col 
             else pl.lit("")).alias("clean_city"),
            (pl.col(state_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if state_col 
             else pl.lit("")).alias("clean_state"),
            (pl.col(zip_col).fill_null("").cast(pl.Utf8).str.replace_all(r"\D", "") if zip_col 
             else pl.lit("")).alias("clean_zip"),
            (pl.col(phone_col).fill_null("").map_elements(clean_phone_field, return_dtype=pl.Utf8) if phone_col 
             else pl.lit("")).alias("clean_phone")
        ]
        
        cleaned_df = df.with_columns(expressions)
        
        # Combined full address string for cross-field blocking
        cleaned_df = cleaned_df.with_columns(
            (pl.col("clean_address") + " " + pl.col("clean_city") + " " + pl.col("clean_state") + " " + pl.col("clean_zip"))
            .str.strip_chars()
            .alias("clean_full_address")
        )
        
        return cleaned_df
