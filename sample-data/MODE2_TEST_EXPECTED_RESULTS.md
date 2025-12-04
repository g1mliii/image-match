# Mode 2 Metadata Matching - Expected Test Results

## Test Overview
This document outlines expected matching results for Mode 2 (metadata-only) testing.
Use this to validate the matching algorithm's behavior against known scenarios.

---

## Test Cases & Expected Matches

### ✅ TEST 1: Perfect Match (Exact Duplicate)
**New Product:** NEW-001 - USB-C Cable 6ft Black  
**Expected Match:** HIST-001 - USB-C Cable 6ft Black  
**Expected Confidence:** ~95-100%  
**Reasoning:**
- Identical name ✓
- Same category ✓
- Same price ✓
- Similar performance trend ✓

---

### ✅ TEST 2: Name Variation (Spelling/Format Differences)
**New Product:** NEW-002 - USB C Cable 6 foot Black  
**Expected Match:** HIST-001 - USB-C Cable 6ft Black  
**Expected Confidence:** ~85-95%  
**Reasoning:**
- Very similar name (USB-C vs USB C, 6ft vs 6 foot) ✓
- Same category ✓
- Similar price ($13.49 vs $12.99 = 3.8% difference) ✓
- Similar performance ✓
**Tests:** Name fuzzy matching, format tolerance

---

### ✅ TEST 3: Price Sensitivity (Same Product, Different Price)
**New Product:** NEW-003 - USB-C Cable 6ft Black  
**Expected Match:** HIST-001 - USB-C Cable 6ft Black  
**Expected Confidence:** ~75-85% (lower due to price)  
**Reasoning:**
- Identical name ✓
- Same category ✓
- Large price difference ($25.99 vs $12.99 = 100% increase) ✗
- Similar performance ✓
**Tests:** Price weighting impact on confidence

---

### ❌ TEST 4: Wrong Category (Negative Test)
**New Product:** NEW-004 - USB-C Cable 6ft Black  
**Expected Match:** Should NOT match HIST-001 or match with LOW confidence (<60%)  
**Expected Confidence:** <60%  
**Reasoning:**
- Identical name ✓
- WRONG category (Clothing vs Electronics) ✗✗
- Same price ✓
- Similar performance ✓
**Tests:** Category mismatch penalty, false positive prevention

---

### ✅ TEST 5: Name Variation with Abbreviation
**New Product:** NEW-005 - HDMI Cable 10 feet  
**Expected Match:** HIST-003 - HDMI Cable 10ft  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Very similar name (feet vs ft) ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Abbreviation handling

---

### ✅ TEST 6: Similar Product, Different Variant
**New Product:** NEW-006 - Wireless Gaming Mouse RGB  
**Expected Match:** HIST-005 - Wireless Mouse Gaming RGB  
**Expected Confidence:** ~85-95%  
**Reasoning:**
- Very similar name (word order difference) ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Word order tolerance

---

### ✅ TEST 7: Partial Name Match
**New Product:** NEW-007 - Wireless Mouse  
**Expected Match:** HIST-004 - Wireless Mouse Ergonomic  
**Expected Confidence:** ~75-85%  
**Reasoning:**
- Partial name match (missing "Ergonomic") ✓
- Same category ✓
- Same price ✓
- Similar performance trend ✓
**Tests:** Partial name matching

---

### ✅ TEST 8: Generic Name Match
**New Product:** NEW-008 - Mechanical Keyboard  
**Expected Match:** HIST-006 - Mechanical Keyboard RGB  
**Expected Confidence:** ~80-90%  
**Reasoning:**
- Partial name match (missing "RGB") ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Generic vs specific product names

---

### ✅ TEST 9: Abbreviation in Name
**New Product:** NEW-009 - Cotton T-Shirt Red L  
**Expected Match:** HIST-008 - Cotton T-Shirt Red Large  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Very similar name (L vs Large) ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Size abbreviation handling

---

### ✅ TEST 10: Spelling Variation
**New Product:** NEW-010 - Cotton Tshirt Red Large  
**Expected Match:** HIST-008 - Cotton T-Shirt Red Large  
**Expected Confidence:** ~85-95%  
**Reasoning:**
- Very similar name (Tshirt vs T-Shirt) ✓
- Same category ✓
- Similar price ($21.99 vs $19.99 = 10% difference) ✓
- Similar performance ✓
**Tests:** Hyphenation tolerance

---

### ✅ TEST 11: Abbreviation Variation
**New Product:** NEW-011 - Running Shoes Sz 10  
**Expected Match:** HIST-010 - Running Shoes Size 10  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Very similar name (Sz vs Size) ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Common abbreviations

---

### ✅ TEST 12: Partial Name (Missing Details)
**New Product:** NEW-012 - Laptop Stand  
**Expected Match:** HIST-011 - Laptop Stand Aluminum  
**Expected Confidence:** ~80-90%  
**Reasoning:**
- Partial name match ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Material descriptor handling

---

### ✅ TEST 13: Word Order Variation
**New Product:** NEW-013 - iPhone 14 Phone Case  
**Expected Match:** HIST-012 - Phone Case iPhone 14  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Same words, different order ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Word order independence

---

### ✅ TEST 14: Descriptor Order Change
**New Product:** NEW-014 - Water Bottle 32oz Stainless Steel  
**Expected Match:** HIST-013 - Water Bottle Stainless 32oz  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Same descriptors, different order ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Descriptor position tolerance

---

### ✅ TEST 15: Generic Name Match
**New Product:** NEW-015 - Ceramic Coffee Mug  
**Expected Match:** HIST-014 - Coffee Mug Ceramic White  
**Expected Confidence:** ~80-90%  
**Reasoning:**
- Similar name (missing "White") ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Color descriptor handling

---

### ✅ TEST 16: Simplified Name
**New Product:** NEW-016 - Spiral Notebook  
**Expected Match:** HIST-015 - Notebook Spiral 100 Pages  
**Expected Confidence:** ~80-90%  
**Reasoning:**
- Core name match (missing page count) ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Specification detail handling

---

### ✅ TEST 17: Exact Match Different Performance
**New Product:** NEW-017 - USB Flash Drive 64GB  
**Expected Match:** HIST-017 - USB Flash Drive 64GB  
**Expected Confidence:** ~95-100%  
**Reasoning:**
- Identical name ✓
- Same category ✓
- Same price ✓
- Similar performance trend ✓
**Tests:** Performance history impact

---

### ✅ TEST 18: Simplified Name Match
**New Product:** NEW-018 - Desk Lamp LED  
**Expected Match:** HIST-019 - Desk Lamp LED Adjustable  
**Expected Confidence:** ~85-90%  
**Reasoning:**
- Partial name match ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Feature descriptor handling

---

### ✅ TEST 19: Descriptor Position Change
**New Product:** NEW-019 - 15 inch Laptop Backpack  
**Expected Match:** HIST-020 - Backpack Laptop 15inch  
**Expected Confidence:** ~85-95%  
**Reasoning:**
- Same words, different order ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Size format tolerance (15 inch vs 15inch)

---

### ✅ TEST 20: Word Order Variation
**New Product:** NEW-020 - Wireless Keyboard Office  
**Expected Match:** HIST-007 - Office Keyboard Wireless  
**Expected Confidence:** ~90-95%  
**Reasoning:**
- Same words, different order ✓
- Same category ✓
- Same price ✓
- Similar performance ✓
**Tests:** Adjective position tolerance

---

## Edge Case Tests

### 🔧 EDGE CASE 1: Missing Required Fields
**New Product:** NEW-021 - USB-C Cable  
**Expected Behavior:** Should still attempt matching but with lower confidence  
**Expected Match:** Possibly HIST-001 or HIST-002  
**Expected Confidence:** ~40-60% (reduced due to missing data)  
**Missing Fields:**
- Category (empty)
- Price (empty)
- Performance History (empty)
**Tests:** Graceful handling of missing data

---

### 🔧 EDGE CASE 2: Malformed Data
**New Product:** NEW-022 - Mystery Product  
**Expected Behavior:** Should handle malformed price and performance data  
**Expected Match:** Possibly HIST-003 (if name similarity is high enough)  
**Expected Confidence:** ~50-70% (reduced due to data quality issues)  
**Issues:**
- Price has $ symbol and quotes: "$19.99"
- Performance has invalid value: "100, 105, invalid, 110"
**Tests:** Data validation and error handling

---

### 🔧 EDGE CASE 3: Extra Whitespace
**New Product:** NEW-023 - "  Wireless Mouse  " (with leading/trailing spaces)  
**Expected Match:** HIST-004 - Wireless Mouse Ergonomic  
**Expected Confidence:** ~75-85%  
**Issues:**
- Leading and trailing whitespace in name
**Tests:** String normalization and trimming

---

### ❌ EDGE CASE 4: No Match Expected (Negative Test)
**New Product:** NEW-024 - BLUETOOTH SPEAKER  
**Expected Match:** No good match (all confidence <70%)  
**Expected Confidence:** <70% for all matches  
**Reasoning:**
- No similar products in historical data
- Different category (Audio) not present in historical
**Tests:** False positive prevention, threshold validation

---

### ❌ EDGE CASE 5: Wrong Category (Negative Test)
**New Product:** NEW-025 - Blue Jeans Size 32  
**Expected Match:** Should NOT match any clothing items strongly  
**Expected Confidence:** <70% for all matches  
**Reasoning:**
- Different product type (Jeans vs T-Shirts/Shoes)
- Different subcategory (Pants vs Shirts/Footwear)
**Tests:** Category specificity, false positive prevention

---

## Scoring Breakdown (Approximate Weights)

Based on the metadata matching algorithm, confidence scores are calculated from:

1. **Name Similarity (40-50%):** Fuzzy string matching
2. **Category Match (20-30%):** Exact or hierarchical match
3. **Price Similarity (15-20%):** Percentage difference
4. **Performance Similarity (10-15%):** Trend correlation

**Note:** Exact weights may vary based on implementation. Use this as a guide.

---

## Testing Checklist

### Before Testing:
- [ ] Clear existing database or use fresh instance
- [ ] Upload `mode2_historical_products_test.csv` as Historical Products
- [ ] Verify 20 historical products loaded successfully
- [ ] Upload `mode2_new_products_test.csv` as New Products
- [ ] Verify 25 new products loaded successfully

### During Testing:
- [ ] Run matching in Mode 2 (metadata only)
- [ ] Check confidence scores for each match
- [ ] Verify top matches align with expected results
- [ ] Note any unexpected matches or confidence scores
- [ ] Test edge cases (NEW-021 through NEW-025)
- [ ] Verify negative tests (NEW-004, NEW-024, NEW-025) don't produce false positives

### After Testing:
- [ ] Document any discrepancies from expected results
- [ ] Note confidence score ranges for different scenarios
- [ ] Identify any algorithm improvements needed
- [ ] Verify threshold (70%) is appropriate
- [ ] Check if any edge cases need special handling

---

## Success Criteria

✅ **Pass:** 
- 90%+ of expected matches are correct (top match)
- Confidence scores within ±10% of expected ranges
- Edge cases handled gracefully (no crashes)
- Negative tests produce confidence <70%

⚠️ **Review Needed:**
- 70-90% of expected matches correct
- Confidence scores vary by >15% from expected
- Some edge cases cause errors

❌ **Fail:**
- <70% of expected matches correct
- Multiple crashes on edge cases
- False positives on negative tests

---

## Notes for Manual Testing

1. **Confidence Score Interpretation:**
   - 90-100%: Excellent match (likely same product)
   - 80-90%: Very good match (high confidence)
   - 70-80%: Good match (acceptable)
   - 60-70%: Marginal match (review needed)
   - <60%: Poor match (likely different products)

2. **Common Issues to Watch For:**
   - Case sensitivity problems (USB-C vs usb-c)
   - Whitespace handling
   - Special character handling ($, -, etc.)
   - Number format variations (6ft vs 6 ft vs 6-ft)
   - Missing data causing crashes vs graceful degradation

3. **Performance Notes:**
   - Mode 2 should be fast (no image processing)
   - 25 products should match in <5 seconds
   - Check for any timeout or performance issues

---

## Expected Output Format

For each new product, the UI should show:
```
NEW-001: USB-C Cable 6ft Black
├─ Match: HIST-001 (95% confidence)
│  └─ USB-C Cable 6ft Black
│     Category: Electronics > Cables
│     Price: $12.99
│
└─ Reasoning:
   • Name: Exact match
   • Category: Exact match
   • Price: Exact match
   • Performance: Similar trend
```

---

## Troubleshooting

**If matches are way off:**
- Check if name similarity is working (fuzzy matching)
- Verify category matching logic
- Check price similarity calculation
- Review performance history comparison

**If confidence scores are too high/low:**
- Review weighting of different factors
- Check if threshold needs adjustment
- Verify normalization of scores

**If edge cases crash:**
- Check null/empty field handling
- Verify data type validation
- Review error handling in matching logic

---

## Additional Test Scenarios to Consider

If initial testing goes well, consider adding:
- Products with very long names (>100 characters)
- Products with special characters (!@#$%^&*)
- Products with numbers in different formats (1st, first, 1)
- Products with multiple categories (hierarchical matching)
- Products with extreme prices ($0.01, $999,999.99)
- Products with negative or zero performance values
- Products with very short names (1-2 words)
- Products with identical names but different SKUs

---

**Generated:** 2025-12-03  
**Version:** 1.0  
**Test Mode:** Mode 2 (Metadata Only)
