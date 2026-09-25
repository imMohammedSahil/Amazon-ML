import re
import unicodedata
import polars as pl
import jellyfish

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
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(text))
        if unicodedata.category(c) != 'Mn'
    )


def clean_text_field(text: str) -> str:
    if text is None or text == "":
        return ""
    text = strip_accents(str(text).lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_name_field(text: str) -> str:
    cleaned = clean_text_field(text)
    if not cleaned:
        return ""
    cleaned = LEGAL_SUFFIX_REGEX.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_phone_field(phone: str) -> str:
    if phone is None or phone == "":
        return ""
    digits = re.sub(r"\D", "", str(phone))
    return digits[-10:] if len(digits) >= 10 else digits


def clean_address_field(address: str) -> str:
    cleaned = clean_text_field(address)
    if not cleaned:
        return ""
    for pattern, repl in ADDRESS_ABBR.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def precompute_phonetic_metaphone(name: str) -> str:
    if not name:
        return ""
    try:
        return jellyfish.metaphone(name[:25])
    except Exception:
        return ""


def precompute_phonetic_soundex(name: str) -> str:
    if not name:
        return ""
    try:
        return jellyfish.soundex(name[:25])
    except Exception:
        return ""


def extract_street_digits_str(addr: str) -> str:
    if not addr:
        return ""
    digits = re.findall(r"\b\d+\b", str(addr))
    return " ".join(digits) if digits else ""


def extract_first_two_tokens(name: str) -> str:
    if not name:
        return ""
    tokens = [t for t in name.split() if len(t) > 1]
    return " ".join(tokens[:2]) if tokens else ""


class DataPreprocessor:
    """
    High-performance, precomputing preprocessor.
    Precomputes phonetics, first tokens, and street digits ONCE per entity to eliminate
    redundant computations during pairwise feature extraction.
    """
    def __init__(self):
        pass

    def clean_table(self, df: pl.DataFrame, source_name: str = "") -> pl.DataFrame:
        cols = df.columns
        
        id_col = "entity_id" if "entity_id" in cols else ("id" if "id" in cols else cols[0])
        name_col = "business_name" if "business_name" in cols else ("name" if "name" in cols else None)
        addr_col = "business_address" if "business_address" in cols else ("address" if "address" in cols else None)
        country_col = "country" if "country" in cols else None
        phone_col = "phone" if "phone" in cols else None
        city_col = "city" if "city" in cols else None
        state_col = "state" if "state" in cols else None
        zip_col = "zip" if "zip" in cols else None

        expressions = [
            pl.col(id_col).cast(pl.Utf8).alias("id"),
            pl.lit(source_name).alias("source") if "source" not in cols else pl.col("source"),
            
            (pl.col(name_col).fill_null("").map_elements(clean_name_field, return_dtype=pl.Utf8) if name_col 
             else pl.lit("")).alias("clean_name"),
            
            (pl.col(name_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if name_col 
             else pl.lit("")).alias("clean_name_raw"),
            
            (pl.col(addr_col).fill_null("").map_elements(clean_address_field, return_dtype=pl.Utf8) if addr_col 
             else pl.lit("")).alias("clean_address"),
            
            (pl.col(country_col).fill_null("").map_elements(clean_text_field, return_dtype=pl.Utf8) if country_col 
             else pl.lit("")).alias("clean_country"),
            
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
        
        # Combined full address
        cleaned_df = cleaned_df.with_columns(
            (pl.col("clean_address") + " " + pl.col("clean_city") + " " + pl.col("clean_state") + " " + pl.col("clean_zip"))
            .str.strip_chars()
            .alias("clean_full_address")
        )
        
        # Precompute phonetic keys, first tokens, and street digits ONCE
        cleaned_df = cleaned_df.with_columns([
            pl.col("clean_name").map_elements(precompute_phonetic_metaphone, return_dtype=pl.Utf8).alias("meta_key"),
            pl.col("clean_name").map_elements(precompute_phonetic_soundex, return_dtype=pl.Utf8).alias("soundex_key"),
            pl.col("clean_name").map_elements(extract_first_two_tokens, return_dtype=pl.Utf8).alias("name_first_tokens"),
            pl.col("clean_full_address").map_elements(extract_street_digits_str, return_dtype=pl.Utf8).alias("street_digits_str")
        ])
        
        return cleaned_df
