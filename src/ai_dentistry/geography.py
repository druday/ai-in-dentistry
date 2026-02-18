from __future__ import annotations

import difflib
import re
import unicodedata


def _normalize_token(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().strip()
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return folded


WHO_REGION_MAP = {
    # AFRO
    "algeria": "AFRO",
    "angola": "AFRO",
    "benin": "AFRO",
    "botswana": "AFRO",
    "burkina faso": "AFRO",
    "burundi": "AFRO",
    "cabo verde": "AFRO",
    "cameroon": "AFRO",
    "central african republic": "AFRO",
    "chad": "AFRO",
    "comoros": "AFRO",
    "congo": "AFRO",
    "cote d ivoire": "AFRO",
    "democratic republic of the congo": "AFRO",
    "equatorial guinea": "AFRO",
    "eritrea": "AFRO",
    "eswatini": "AFRO",
    "ethiopia": "AFRO",
    "gabon": "AFRO",
    "gambia": "AFRO",
    "ghana": "AFRO",
    "guinea": "AFRO",
    "guinea bissau": "AFRO",
    "kenya": "AFRO",
    "lesotho": "AFRO",
    "liberia": "AFRO",
    "madagascar": "AFRO",
    "malawi": "AFRO",
    "mali": "AFRO",
    "mauritania": "AFRO",
    "mauritius": "AFRO",
    "mozambique": "AFRO",
    "namibia": "AFRO",
    "niger": "AFRO",
    "nigeria": "AFRO",
    "rwanda": "AFRO",
    "sao tome and principe": "AFRO",
    "senegal": "AFRO",
    "seychelles": "AFRO",
    "sierra leone": "AFRO",
    "south africa": "AFRO",
    "south sudan": "AFRO",
    "togo": "AFRO",
    "uganda": "AFRO",
    "united republic of tanzania": "AFRO",
    "tanzania": "AFRO",
    "zambia": "AFRO",
    "zimbabwe": "AFRO",
    # PAHO
    "antigua and barbuda": "PAHO",
    "argentina": "PAHO",
    "bahamas": "PAHO",
    "barbados": "PAHO",
    "belize": "PAHO",
    "bolivia": "PAHO",
    "brazil": "PAHO",
    "canada": "PAHO",
    "chile": "PAHO",
    "colombia": "PAHO",
    "costa rica": "PAHO",
    "cuba": "PAHO",
    "dominica": "PAHO",
    "dominican republic": "PAHO",
    "ecuador": "PAHO",
    "el salvador": "PAHO",
    "grenada": "PAHO",
    "guatemala": "PAHO",
    "guyana": "PAHO",
    "haiti": "PAHO",
    "honduras": "PAHO",
    "jamaica": "PAHO",
    "mexico": "PAHO",
    "nicaragua": "PAHO",
    "panama": "PAHO",
    "paraguay": "PAHO",
    "peru": "PAHO",
    "saint kitts and nevis": "PAHO",
    "saint lucia": "PAHO",
    "saint vincent and the grenadines": "PAHO",
    "suriname": "PAHO",
    "trinidad and tobago": "PAHO",
    "united states": "PAHO",
    "usa": "PAHO",
    "uruguay": "PAHO",
    "venezuela": "PAHO",
    # SEARO
    "bangladesh": "SEARO",
    "bhutan": "SEARO",
    "dpr korea": "SEARO",
    "india": "SEARO",
    "indonesia": "SEARO",
    "maldives": "SEARO",
    "myanmar": "SEARO",
    "nepal": "SEARO",
    "sri lanka": "SEARO",
    "thailand": "SEARO",
    "timor leste": "SEARO",
    # EMRO
    "afghanistan": "EMRO",
    "bahrain": "EMRO",
    "djibouti": "EMRO",
    "egypt": "EMRO",
    "iran": "EMRO",
    "iraq": "EMRO",
    "jordan": "EMRO",
    "kuwait": "EMRO",
    "lebanon": "EMRO",
    "libya": "EMRO",
    "morocco": "EMRO",
    "oman": "EMRO",
    "pakistan": "EMRO",
    "palestine": "EMRO",
    "qatar": "EMRO",
    "saudi arabia": "EMRO",
    "somalia": "EMRO",
    "sudan": "EMRO",
    "syrian arab republic": "EMRO",
    "syria": "EMRO",
    "tunisia": "EMRO",
    "united arab emirates": "EMRO",
    "yemen": "EMRO",
    # EURO
    "albania": "EURO",
    "andorra": "EURO",
    "armenia": "EURO",
    "austria": "EURO",
    "azerbaijan": "EURO",
    "belarus": "EURO",
    "belgium": "EURO",
    "bosnia and herzegovina": "EURO",
    "bulgaria": "EURO",
    "croatia": "EURO",
    "cyprus": "EURO",
    "czech republic": "EURO",
    "denmark": "EURO",
    "estonia": "EURO",
    "finland": "EURO",
    "france": "EURO",
    "georgia": "EURO",
    "germany": "EURO",
    "greece": "EURO",
    "hungary": "EURO",
    "iceland": "EURO",
    "ireland": "EURO",
    "israel": "EURO",
    "italy": "EURO",
    "kazakhstan": "EURO",
    "kyrgyzstan": "EURO",
    "latvia": "EURO",
    "lithuania": "EURO",
    "luxembourg": "EURO",
    "malta": "EURO",
    "monaco": "EURO",
    "montenegro": "EURO",
    "netherlands": "EURO",
    "north macedonia": "EURO",
    "norway": "EURO",
    "poland": "EURO",
    "portugal": "EURO",
    "republic of moldova": "EURO",
    "romania": "EURO",
    "russian federation": "EURO",
    "san marino": "EURO",
    "serbia": "EURO",
    "slovakia": "EURO",
    "slovenia": "EURO",
    "spain": "EURO",
    "sweden": "EURO",
    "switzerland": "EURO",
    "tajikistan": "EURO",
    "turkey": "EURO",
    "turkmenistan": "EURO",
    "ukraine": "EURO",
    "united kingdom": "EURO",
    "england": "EURO",
    "scotland": "EURO",
    "uzbekistan": "EURO",
    # WPRO
    "australia": "WPRO",
    "brunei darussalam": "WPRO",
    "cambodia": "WPRO",
    "china": "WPRO",
    "fiji": "WPRO",
    "japan": "WPRO",
    "lao people s democratic republic": "WPRO",
    "laos": "WPRO",
    "malaysia": "WPRO",
    "marshall islands": "WPRO",
    "micronesia": "WPRO",
    "mongolia": "WPRO",
    "nauru": "WPRO",
    "new zealand": "WPRO",
    "palau": "WPRO",
    "papua new guinea": "WPRO",
    "philippines": "WPRO",
    "republic of korea": "WPRO",
    "south korea": "WPRO",
    "korea south": "WPRO",
    "samoa": "WPRO",
    "singapore": "WPRO",
    "solomon islands": "WPRO",
    "tonga": "WPRO",
    "tuvalu": "WPRO",
    "vanuatu": "WPRO",
    "viet nam": "WPRO",
    "vietnam": "WPRO",
}


def extract_country_from_affiliation(affiliation: str, cutoff: float = 0.85) -> str | None:
    tokens = [t.strip() for t in re.split(r"[,;]", affiliation) if t.strip()]
    if not tokens:
        return None

    normalized_tokens = [_normalize_token(token) for token in tokens]

    # Favor right-most tokens, where country names typically appear.
    for token in reversed(normalized_tokens):
        if token in WHO_REGION_MAP:
            return token

    all_countries = list(WHO_REGION_MAP.keys())
    for token in reversed(normalized_tokens):
        match = difflib.get_close_matches(token, all_countries, n=1, cutoff=cutoff)
        if match:
            return match[0]

    return None


def map_country_to_who_region(country: str | None) -> str:
    if not country:
        return "Unknown"
    normalized = _normalize_token(country)
    return WHO_REGION_MAP.get(normalized, "Unknown")
