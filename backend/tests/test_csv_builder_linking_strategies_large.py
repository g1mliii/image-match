"""
Large CSV Builder linking strategy tests.

These tests mirror the linking behavior implemented in:
    backend/static/csv-builder.js

Goal:
- Verify every implemented strategy can link correctly.
- Verify the high-volume logical flow (perform + finalize) stays correct.
- Keep tests deterministic and fast enough for regular CI runs.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


MAX_LINK_INDEX_TOKEN_CANDIDATES = 400
MAX_SEARCH_FALLBACK_SCAN = 10000

LINK_FILENAME_FIELDS = ["filename", "product", "image", "image_name", "file", "photo", "picture"]
LINK_PRIORITY_FIELDS = [
    "sku",
    "item",
    "product",
    "product_id",
    "item_number",
    "image",
    "filename",
    "file",
    "photo",
    "name",
    "product_name",
]


def normalize_lookup_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def strip_extension(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\.[^.]+$", "", value)


def normalize_name_lookup(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_lookup_value(value).replace("_", " ").replace("-", " ")).strip()


def tokenize_lookup_value(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip() for token in re.split(r"[^a-z0-9]+", value) if len(token.strip()) >= 3]


def normalize_for_fuzzy_match(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    text = value.lower()
    text = re.sub(r"[_\-.]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_fuzzy_score(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    if str1 in str2 or str2 in str1:
        shorter, longer = (str1, str2) if len(str1) < len(str2) else (str2, str1)
        return len(shorter) / len(longer)

    words1 = [w for w in str1.split(" ") if len(w) > 1]
    words2 = [w for w in str2.split(" ") if len(w) > 1]
    if not words1 or not words2:
        return 0.0

    matching_words = 0
    for w1 in words1:
        for w2 in words2:
            if w1 == w2 or w1 in w2 or w2 in w1:
                matching_words += 1
                break
    return matching_words / max(len(words1), len(words2))


def _set_first_map_value(target: Dict[str, Dict[str, Any]], key: str, data: Dict[str, Any]) -> None:
    if key and key not in target:
        target[key] = data


def _add_token_candidate(token_map: Dict[str, List[int]], token: str, data_index: int) -> None:
    if not token:
        return
    candidates = token_map.setdefault(token, [])
    if len(candidates) >= MAX_LINK_INDEX_TOKEN_CANDIDATES:
        return
    if candidates and candidates[-1] == data_index:
        return
    if data_index in candidates:
        return
    candidates.append(data_index)


@dataclass
class CsvBuilderLinkHarness:
    products: List[Dict[str, Any]]
    imported_data: List[Dict[str, Any]]
    sku_pattern: str = r"[A-Z]+-\d+"
    sku_link_field: str = "__auto__"
    link_indexes: Optional[Dict[str, Any]] = None
    fuzzy_index: Dict[str, List[int]] = field(default_factory=dict)
    linked_products: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_images: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_data: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.build_fuzzy_index()

    def get_active_sku_fields(self) -> List[str]:
        fields: List[str] = []
        if self.sku_link_field and self.sku_link_field != "__auto__":
            fields.append(self.sku_link_field)
        if "sku" not in fields:
            fields.append("sku")
        return fields

    def build_link_indexes(self) -> None:
        indexes: Dict[str, Any] = {
            "sku_map": {},
            "sku_no_ext_map": {},
            "name_map": {},
            "filename_field_map": {},
            "search_exact_map": {},
            "search_token_map": {},
            "source_size": len(self.imported_data),
        }

        for data_index, data in enumerate(self.imported_data):
            if not isinstance(data, dict):
                continue

            seen_sku_values: set[str] = set()
            for sku_field in self.get_active_sku_fields():
                sku = normalize_lookup_value(data.get(sku_field))
                if not sku or sku in seen_sku_values:
                    continue
                seen_sku_values.add(sku)
                sku_no_ext = strip_extension(sku)
                _set_first_map_value(indexes["sku_map"], sku, data)
                _set_first_map_value(indexes["sku_map"], sku_no_ext, data)
                _set_first_map_value(indexes["sku_no_ext_map"], sku_no_ext, data)
                _set_first_map_value(indexes["search_exact_map"], sku, data)
                _set_first_map_value(indexes["search_exact_map"], sku_no_ext, data)

            normalized_name = normalize_name_lookup(data.get("name"))
            if normalized_name:
                name_no_ext = strip_extension(normalized_name)
                _set_first_map_value(indexes["name_map"], normalized_name, data)
                _set_first_map_value(indexes["name_map"], name_no_ext, data)
                _set_first_map_value(indexes["search_exact_map"], normalized_name, data)
                _set_first_map_value(indexes["search_exact_map"], name_no_ext, data)

            for field in LINK_FILENAME_FIELDS:
                field_value = normalize_lookup_value(data.get(field))
                if not field_value:
                    continue
                _set_first_map_value(indexes["filename_field_map"], field_value, data)
                _set_first_map_value(indexes["filename_field_map"], strip_extension(field_value), data)

            for field_key, raw_value in data.items():
                if raw_value is None or raw_value == "":
                    continue
                normalized_value = normalize_lookup_value(raw_value)
                if not normalized_value:
                    continue

                normalized_no_ext = strip_extension(normalized_value)
                _set_first_map_value(indexes["search_exact_map"], normalized_value, data)
                _set_first_map_value(indexes["search_exact_map"], normalized_no_ext, data)

                should_tokenize = field_key in LINK_PRIORITY_FIELDS or len(normalized_value) <= 80
                if should_tokenize:
                    for token in tokenize_lookup_value(normalized_no_ext):
                        _add_token_candidate(indexes["search_token_map"], token, data_index)

        self.link_indexes = indexes

    def ensure_link_indexes(self) -> Dict[str, Any]:
        if self.link_indexes is None or self.link_indexes.get("source_size") != len(self.imported_data):
            self.build_link_indexes()
        assert self.link_indexes is not None
        return self.link_indexes

    def build_fuzzy_index(self) -> None:
        self.fuzzy_index = {}
        for idx, data in enumerate(self.imported_data):
            if not isinstance(data, dict) or not data.get("name"):
                continue
            clean_name = normalize_for_fuzzy_match(data["name"])
            if not clean_name:
                continue
            for word in [w for w in clean_name.split(" ") if len(w) > 1]:
                items = self.fuzzy_index.setdefault(word, [])
                if idx not in items:
                    items.append(idx)

    def get_fuzzy_index_candidates(self, clean_filename: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.fuzzy_index:
            return self.imported_data

        file_words = [w for w in clean_filename.split(" ") if len(w) > 1]
        if not file_words:
            return self.imported_data

        candidate_scores: Dict[int, int] = {}
        for word in file_words:
            if word in self.fuzzy_index:
                for idx in self.fuzzy_index[word]:
                    candidate_scores[idx] = candidate_scores.get(idx, 0) + 1

        if not candidate_scores:
            return self.imported_data[:limit]

        top_indices = [
            idx
            for idx, _ in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : max(limit, int((len(self.imported_data) + 9) / 10))
            ]
        ]
        return [self.imported_data[i] for i in top_indices]

    def link_by_filename_equals_sku(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None
        indexes = self.ensure_link_indexes()
        filename_with_ext = normalize_lookup_value(product["filename"])
        filename_no_ext = strip_extension(filename_with_ext)
        if not filename_no_ext:
            return None
        return (
            indexes["sku_map"].get(filename_no_ext)
            or indexes["sku_map"].get(filename_with_ext)
            or indexes["sku_no_ext_map"].get(filename_no_ext)
            or indexes["sku_no_ext_map"].get(filename_with_ext)
        )

    def link_by_filename_contains_sku(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None
        try:
            indexes = self.ensure_link_indexes()
            match = re.search(self.sku_pattern, product["filename"], flags=re.IGNORECASE)
            if not match:
                return None
            extracted = normalize_lookup_value(match.group(0))
            extracted_no_ext = strip_extension(extracted)
            return (
                indexes["sku_map"].get(extracted)
                or indexes["sku_map"].get(extracted_no_ext)
                or indexes["sku_no_ext_map"].get(extracted)
                or indexes["sku_no_ext_map"].get(extracted_no_ext)
            )
        except re.error:
            return None

    def link_by_folder_equals_sku(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("category"):
            return None
        indexes = self.ensure_link_indexes()
        folder_name = normalize_lookup_value(product["category"])
        folder_no_ext = strip_extension(folder_name)
        if not folder_name:
            return None
        return (
            indexes["sku_map"].get(folder_name)
            or indexes["sku_map"].get(folder_no_ext)
            or indexes["sku_no_ext_map"].get(folder_name)
            or indexes["sku_no_ext_map"].get(folder_no_ext)
        )

    def link_by_fuzzy_name(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None
        clean_filename = normalize_for_fuzzy_match(strip_extension(product["filename"]))
        if not clean_filename or len(clean_filename) < 2:
            return None

        candidates = self.get_fuzzy_index_candidates(clean_filename)
        best_match = None
        best_score = 0.0
        for data in candidates:
            if not data or not data.get("name"):
                continue
            clean_name = normalize_for_fuzzy_match(data["name"])
            if not clean_name:
                continue
            score = calculate_fuzzy_score(clean_filename, clean_name)
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = data
        return best_match

    def link_by_sku_equals_filename(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.link_by_filename_equals_sku(product)

    def link_by_metadata_filename(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None
        indexes = self.ensure_link_indexes()
        image_filename = normalize_lookup_value(product["filename"])
        image_filename_no_ext = strip_extension(image_filename)
        return indexes["filename_field_map"].get(image_filename) or indexes["filename_field_map"].get(
            image_filename_no_ext
        )

    def link_by_name_equals_filename(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None
        indexes = self.ensure_link_indexes()
        filename_with_ext = normalize_name_lookup(product["filename"])
        filename_no_ext = normalize_name_lookup(strip_extension(product["filename"]))
        if not filename_no_ext:
            return None
        return indexes["name_map"].get(filename_no_ext) or indexes["name_map"].get(filename_with_ext)

    def link_by_search_all_fields(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not product or not product.get("filename"):
            return None

        indexes = self.ensure_link_indexes()
        clean_filename_with_ext = normalize_lookup_value(product["filename"])
        clean_filename = strip_extension(clean_filename_with_ext)
        if not clean_filename:
            return None

        exact_match = indexes["search_exact_map"].get(clean_filename_with_ext) or indexes["search_exact_map"].get(
            clean_filename
        )
        if exact_match:
            return exact_match

        def values_match(raw_value: Any) -> bool:
            if not raw_value:
                return False
            field_value = normalize_lookup_value(raw_value)
            if not field_value:
                return False
            if field_value == clean_filename or field_value == clean_filename_with_ext:
                return True
            field_without_ext = strip_extension(field_value)
            if field_without_ext == clean_filename:
                return True
            if field_value in clean_filename or clean_filename in field_value:
                return min(len(field_value), len(clean_filename)) >= 3
            return False

        token_candidates: set[int] = set()
        for token in tokenize_lookup_value(clean_filename):
            for idx in indexes["search_token_map"].get(token, []):
                token_candidates.add(idx)

        if token_candidates:
            candidate_pool = [self.imported_data[i] for i in token_candidates if 0 <= i < len(self.imported_data)]
        elif len(self.imported_data) <= MAX_SEARCH_FALLBACK_SCAN:
            candidate_pool = self.imported_data
        else:
            return None

        for data in candidate_pool:
            if not isinstance(data, dict):
                continue
            for key in LINK_PRIORITY_FIELDS:
                if key in data and values_match(data.get(key)):
                    return data
            for key, value in data.items():
                if key in LINK_PRIORITY_FIELDS:
                    continue
                if values_match(value):
                    return data
        return None

    def perform_linking(self, strategy: str) -> Dict[str, Any]:
        strategy_map = {
            "filename_equals_sku": self.link_by_filename_equals_sku,
            "filename_contains_sku": self.link_by_filename_contains_sku,
            "folder_equals_sku": self.link_by_folder_equals_sku,
            "fuzzy_name": self.link_by_fuzzy_name,
            "sku_equals_filename": self.link_by_sku_equals_filename,
            "metadata_filename": self.link_by_metadata_filename,
            "name_equals_filename": self.link_by_name_equals_filename,
            "search_all_fields": self.link_by_search_all_fields,
        }

        fn = strategy_map.get(strategy)
        linked = 0
        results = []
        for product in self.products:
            matched_data = fn(product) if fn else None
            if matched_data is not None:
                linked += 1
                results.append({"image": product["filename"], "data": matched_data, "matched": True})
            else:
                results.append({"image": product["filename"], "data": {}, "matched": False})
        return {"linked": linked, "unlinked": len(self.products) - linked, "results": results}

    def finalize_linking(self, matches: Dict[str, Any]) -> None:
        for idx, result in enumerate(matches["results"]):
            if not result.get("matched") or idx >= len(self.products):
                continue
            data = result.get("data") or {}
            product = self.products[idx]
            if data.get("sku"):
                product["sku"] = data["sku"]
            if data.get("name"):
                product["name"] = data["name"]
            if data.get("price"):
                product["price"] = data["price"]
            if data.get("category"):
                product["category"] = data["category"]

        self.linked_products = [dict(p) for p in self.products]
        self.unmatched_images = [
            {**self.products[i], "index": i}
            for i, result in enumerate(matches["results"])
            if i < len(self.products) and not result.get("matched")
        ]

        matched_data_ids = {id(r["data"]) for r in matches["results"] if r.get("matched") and r.get("data")}
        self.unmatched_data = [d for d in self.imported_data if id(d) not in matched_data_ids]

    def apply_linking(self, strategy: str) -> Dict[str, Any]:
        matches = self.perform_linking(strategy)
        self.finalize_linking(matches)
        return matches


def _repo_file(*parts: str) -> str:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(tests_dir))
    return os.path.join(repo_root, *parts)


def test_strategy_inventory_matches_current_js_and_ui() -> None:
    js_path = _repo_file("backend", "static", "csv-builder.js")
    html_path = _repo_file("backend", "static", "csv-builder.html")

    with open(js_path, "r", encoding="utf-8") as f:
        js_text = f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    js_cases = set(re.findall(r"case '([^']+)':", js_text))
    expected_js = {
        "filename_equals_sku",
        "filename_contains_sku",
        "folder_equals_sku",
        "fuzzy_name",
        "sku_equals_filename",
        "metadata_filename",
        "name_equals_filename",
        "search_all_fields",
    }
    assert expected_js.issubset(js_cases)

    ui_strategies = set(re.findall(r'name="linkStrategy" value="([^"]+)"', html_text))
    assert ui_strategies == {
        "search_all_fields",
        "filename_equals_sku",
        "filename_contains_sku",
        "metadata_filename",
    }

    # Strategies implemented in JS but currently not exposed in the UI.
    hidden = expected_js - ui_strategies
    assert hidden == {"folder_equals_sku", "fuzzy_name", "sku_equals_filename", "name_equals_filename"}


@pytest.mark.parametrize(
    ("strategy", "product", "expected_sku"),
    [
        ("filename_equals_sku", {"filename": "PM-001.jpg", "category": ""}, "PM-001"),
        ("filename_contains_sku", {"filename": "hero_DW-002_front.png", "category": ""}, "DW-002"),
        ("folder_equals_sku", {"filename": "anything.jpg", "category": "BOX-777"}, "BOX-777"),
        ("fuzzy_name", {"filename": "blue_ceramic_placemat.jpg", "category": ""}, "PM-001"),
        ("sku_equals_filename", {"filename": "DW-002.jpeg", "category": ""}, "DW-002"),
        ("metadata_filename", {"filename": "kitchen-shot.png", "category": ""}, "DW-002"),
        ("name_equals_filename", {"filename": "Special Name Match.jpg", "category": ""}, "ALIAS-42"),
        ("search_all_fields", {"filename": "mirror-frame-deluxe.png", "category": ""}, "MR-500"),
    ],
)
def test_each_strategy_links_expected_record(
    strategy: str, product: Dict[str, Any], expected_sku: str
) -> None:
    imported_data = [
        {
            "sku": "PM-001",
            "name": "Blue Ceramic Placemat",
            "price": "29.99",
            "filename": "pm001-main.jpg",
        },
        {
            "sku": "DW-002",
            "name": "White Dinner Plate Set",
            "price": "45.00",
            "image_name": "kitchen-shot.png",
        },
        {"sku": "BOX-777", "name": "Mega Storage Box", "price": "99.00"},
        {"sku": "ALIAS-42", "name": "Special Name Match", "price": "11.00"},
        {
            "sku": "MR-500",
            "name": "Mirror Product",
            "price": "20.00",
            "vendor_identifier": "mirror-frame-deluxe",
        },
    ]

    harness = CsvBuilderLinkHarness(products=[product], imported_data=imported_data)
    matches = harness.perform_linking(strategy)

    assert matches["linked"] == 1
    assert matches["results"][0]["data"]["sku"] == expected_sku


def test_invalid_filename_contains_regex_is_safe_and_returns_unmatched() -> None:
    harness = CsvBuilderLinkHarness(
        products=[{"filename": "hero_PM-001.jpg", "category": ""}],
        imported_data=[{"sku": "PM-001", "name": "Blue Placemat"}],
        sku_pattern="(",
    )
    matches = harness.perform_linking("filename_contains_sku")
    assert matches["linked"] == 0
    assert matches["unlinked"] == 1


def test_custom_selected_header_is_used_for_sku_strategies() -> None:
    imported_data = [
        {"product_code": "VC-1001", "name": "Vendor Product One", "price": "19.99"},
        {"product_code": "VC-1002", "name": "Vendor Product Two", "price": "29.99"},
    ]
    harness = CsvBuilderLinkHarness(
        products=[{"filename": "VC-1002.jpg", "category": ""}],
        imported_data=imported_data,
        sku_link_field="product_code",
    )
    matches = harness.perform_linking("filename_equals_sku")
    assert matches["linked"] == 1
    assert matches["results"][0]["data"]["name"] == "Vendor Product Two"


def test_auto_mode_keeps_default_sku_behavior() -> None:
    imported_data = [
        {"sku": "PM-001", "product_code": "VC-1001", "name": "Blue Placemat"},
        {"sku": "PM-002", "product_code": "VC-1002", "name": "White Plate"},
    ]
    harness = CsvBuilderLinkHarness(
        products=[{"filename": "PM-002.jpg", "category": ""}],
        imported_data=imported_data,
        sku_link_field="__auto__",
    )
    matches = harness.perform_linking("filename_equals_sku")
    assert matches["linked"] == 1
    assert matches["results"][0]["data"]["sku"] == "PM-002"


def test_large_search_all_fields_logical_flow() -> None:
    random.seed(7)
    total = 6000
    imported_data = []
    for i in range(total):
        imported_data.append(
            {
                "sku": f"SKU-{i:05d}",
                "name": f"Product Name {i}",
                "filename": f"img_{i:05d}.jpg",
                "custom_ref": f"REF-{i:05d}",
                "price": f"{(i % 250) + 0.99:.2f}",
            }
        )

    products: List[Dict[str, Any]] = []
    for i in range(2000):
        products.append({"filename": f"SKU-{i:05d}.jpg", "category": ""})
    for i in range(2000, 4000):
        products.append({"filename": f"REF-{i:05d}.png", "category": ""})
    for i in range(2000):
        products.append({"filename": f"NO_MATCH-{i:05d}.jpg", "category": ""})

    harness = CsvBuilderLinkHarness(products=products, imported_data=imported_data)
    matches = harness.apply_linking("search_all_fields")

    assert matches["linked"] == 4000
    assert matches["unlinked"] == 2000
    assert len(harness.unmatched_images) == 2000
    assert len(harness.linked_products) == len(products)

    # Spot check merged metadata.
    assert harness.products[10]["sku"] == "SKU-00010"
    assert harness.products[10]["name"] == "Product Name 10"
    assert harness.products[2500]["sku"] == "SKU-02500"
    assert harness.products[2500]["name"] == "Product Name 2500"
    assert harness.products[-1].get("sku", "") in ("", None)


def test_large_multi_strategy_batches() -> None:
    imported_data: List[Dict[str, Any]] = []
    for i in range(5000):
        imported_data.append(
            {
                "sku": f"ABC-{i:05d}",
                "name": f"Catalog Product {i}",
                "filename": f"media_{i:05d}.jpg",
                "image": f"hero_{i:05d}.png",
                "item_number": f"ITM-{i:05d}",
            }
        )

    # Large batches per strategy.
    products_filename_equals = [{"filename": f"ABC-{i:05d}.jpg", "category": ""} for i in range(1000)]
    products_filename_contains = [{"filename": f"shot_ABC-{i:05d}_v2.png", "category": ""} for i in range(1000, 2000)]
    products_metadata_filename = [{"filename": f"hero_{i:05d}.png", "category": ""} for i in range(2000, 3000)]
    products_name_equals = [{"filename": f"Catalog Product {i}.jpg", "category": ""} for i in range(3000, 3500)]
    products_folder_equals = [{"filename": f"x_{i}.jpg", "category": f"ABC-{i:05d}"} for i in range(3500, 4000)]

    batches = [
        ("filename_equals_sku", products_filename_equals, 1000),
        ("filename_contains_sku", products_filename_contains, 1000),
        ("metadata_filename", products_metadata_filename, 1000),
        ("name_equals_filename", products_name_equals, 500),
        ("folder_equals_sku", products_folder_equals, 500),
    ]

    for strategy, batch, expected_links in batches:
        harness = CsvBuilderLinkHarness(products=batch, imported_data=imported_data)
        matches = harness.perform_linking(strategy)
        assert matches["linked"] == expected_links
        assert matches["unlinked"] == 0
