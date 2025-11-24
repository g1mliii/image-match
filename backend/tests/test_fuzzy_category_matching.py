"""
Test fuzzy category matching for misspellings and capitalization

This test verifies that the system can handle:
- Misspellings (placemat → placemats, dinerware → dinnerware)
- Capitalization (PlaceMats → placemats)
- Pluralization (placemat → placemats)
- Extra spaces (place mats → placemats)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from product_matching import (
    normalize_category,
    fuzzy_match_category,
    levenshtein_distance
)


def test_levenshtein_distance():
    """Test Levenshtein distance calculation"""
    print("\n[TEST] Levenshtein distance...")
    
    # Test 1: Identical strings
    assert levenshtein_distance("cat", "cat") == 0, "Identical strings should have distance 0"
    print("  [✓] Identical strings: distance 0")
    
    # Test 2: One character difference
    assert levenshtein_distance("cat", "cats") == 1, "One insertion"
    assert levenshtein_distance("cat", "bat") == 1, "One substitution"
    assert levenshtein_distance("cats", "cat") == 1, "One deletion"
    print("  [✓] One character difference: distance 1")
    
    # Test 3: Multiple differences
    assert levenshtein_distance("placemat", "placemats") == 1, "Pluralization"
    assert levenshtein_distance("dinerware", "dinnerware") == 1, "Misspelling"
    assert levenshtein_distance("cat", "dog") == 3, "Completely different"
    print("  [✓] Multiple differences calculated correctly")
    
    # Test 4: Real-world examples
    assert levenshtein_distance("placemats", "placemats") == 0, "Exact match"
    assert levenshtein_distance("placemat", "placemats") == 1, "Singular vs plural"
    assert levenshtein_distance("dinnerware", "dinerware") == 1, "Missing letter"
    assert levenshtein_distance("dinnerware", "dinnerwar") == 1, "Missing letter at end"
    print("  [✓] Real-world examples work correctly")
    
    print("[PASS] Levenshtein distance works correctly")


def test_normalize_category():
    """Test category normalization"""
    print("\n[TEST] Category normalization...")
    
    # Test 1: Case normalization
    assert normalize_category("Placemats") == "placemats", "Should lowercase"
    assert normalize_category("DINNERWARE") == "dinnerware", "Should lowercase"
    assert normalize_category("PlaceMats") == "placemats", "Should lowercase"
    print("  [✓] Case normalization works")
    
    # Test 2: Whitespace trimming
    assert normalize_category("  placemats  ") == "placemats", "Should trim"
    assert normalize_category("\tdinnerware\n") == "dinnerware", "Should trim tabs/newlines"
    print("  [✓] Whitespace trimming works")
    
    # Test 3: Empty/NULL handling
    assert normalize_category(None) is None, "None should stay None"
    assert normalize_category("") is None, "Empty should become None"
    assert normalize_category("   ") is None, "Whitespace should become None"
    print("  [✓] Empty/NULL handling works")
    
    # Test 4: Unknown variations
    assert normalize_category("unknown") is None, "Unknown should become None"
    assert normalize_category("uncategorized") is None, "Uncategorized should become None"
    assert normalize_category("n/a") is None, "N/A should become None"
    print("  [✓] Unknown variations handled")
    
    print("[PASS] Category normalization works correctly")


def test_fuzzy_match_category():
    """Test fuzzy category matching"""
    print("\n[TEST] Fuzzy category matching...")
    
    available_categories = ["placemats", "dinnerware", "glassware", "flatware"]
    
    # Test 1: Exact match (case-insensitive)
    assert fuzzy_match_category("placemats", available_categories) == "placemats", "Exact match"
    assert fuzzy_match_category("Placemats", available_categories) == "placemats", "Case-insensitive exact match"
    assert fuzzy_match_category("DINNERWARE", available_categories) == "dinnerware", "Uppercase exact match"
    print("  [✓] Exact matches work")
    
    # Test 2: Misspellings (1-2 character difference)
    assert fuzzy_match_category("placemat", available_categories) == "placemats", "Missing 's'"
    assert fuzzy_match_category("dinerware", available_categories) == "dinnerware", "Missing 'n'"
    assert fuzzy_match_category("glasware", available_categories) == "glassware", "Missing 's'"
    assert fuzzy_match_category("flatwear", available_categories) == "flatware", "Wrong 'ea' vs 'a'"
    print("  [✓] Misspellings matched correctly")
    
    # Test 3: Extra spaces/hyphens
    assert fuzzy_match_category("place mats", available_categories) == "placemats", "Space removed"
    assert fuzzy_match_category("dinner-ware", available_categories) == "dinnerware", "Hyphen removed"
    print("  [✓] Spaces and hyphens handled")
    
    # Test 4: No match (too different)
    assert fuzzy_match_category("xyz", available_categories, threshold=2) is None, "Too different"
    assert fuzzy_match_category("abc123", available_categories, threshold=2) is None, "Completely different"
    print("  [✓] No match for very different strings")
    
    # Test 5: Threshold testing
    assert fuzzy_match_category("placemat", available_categories, threshold=1) == "placemats", "Within threshold"
    assert fuzzy_match_category("placemat", available_categories, threshold=0) is None, "Outside threshold"
    print("  [✓] Threshold works correctly")
    
    # Test 6: Empty inputs
    assert fuzzy_match_category("", available_categories) is None, "Empty input"
    assert fuzzy_match_category("placemats", []) is None, "Empty available categories"
    print("  [✓] Empty inputs handled")
    
    print("[PASS] Fuzzy category matching works correctly")


def test_real_world_scenarios():
    """Test real-world misspelling scenarios"""
    print("\n[TEST] Real-world scenarios...")
    
    available_categories = [
        "placemats",
        "dinnerware", 
        "glassware",
        "flatware",
        "table linens",
        "serving pieces"
    ]
    
    # Common misspellings
    test_cases = [
        ("placemat", "placemats", "Singular form"),
        ("Placemats", "placemats", "Capitalized"),
        ("PLACEMATS", "placemats", "All caps"),
        ("PlaceMats", "placemats", "Mixed case"),
        ("place mats", "placemats", "Two words"),
        ("place-mats", "placemats", "Hyphenated"),
        ("dinerware", "dinnerware", "Missing 'n'"),
        ("dinnerwar", "dinnerware", "Missing 'e'"),
        ("glasware", "glassware", "Missing 's'"),
        ("glasswar", "glassware", "Missing 'e'"),
        ("flatwear", "flatware", "Wrong spelling"),
        ("tablelinens", "table linens", "No space"),
        ("servingpieces", "serving pieces", "No space"),
    ]
    
    passed = 0
    failed = 0
    
    for input_cat, expected, description in test_cases:
        result = fuzzy_match_category(input_cat, available_categories, threshold=2)
        if result == expected:
            print(f"  [✓] '{input_cat}' → '{result}' ({description})")
            passed += 1
        else:
            print(f"  [✗] '{input_cat}' → '{result}' (expected '{expected}') ({description})")
            failed += 1
    
    if failed == 0:
        print(f"\n[PASS] All {passed} real-world scenarios passed")
    else:
        print(f"\n[FAIL] {failed}/{len(test_cases)} scenarios failed")
        assert False, f"{failed} test cases failed"


def main():
    """Run all fuzzy matching tests"""
    print("=" * 70)
    print("FUZZY CATEGORY MATCHING TESTS")
    print("=" * 70)
    
    tests = [
        ("Levenshtein Distance", test_levenshtein_distance),
        ("Category Normalization", test_normalize_category),
        ("Fuzzy Category Matching", test_fuzzy_match_category),
        ("Real-World Scenarios", test_real_world_scenarios)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "[+]" if result else "[X]"
        print(f"{symbol} {status:8} {test_name}")
    
    print("=" * 70)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All fuzzy matching tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
