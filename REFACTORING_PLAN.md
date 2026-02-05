# Non-Breaking Refactoring Plan

## Overview
Refactor app.py (6,418 lines) and app.js (8,614 lines) into smaller, more maintainable modules without breaking any existing functionality.

## Key Constraint
**This is a non-breaking refactor.** Every step must maintain full functionality. We'll use a "strangler fig" pattern - extract code incrementally while keeping the original working.

---

## Part 1: Dead Code Removal (app.js)

### Confirmed Dead Code to Remove (~150 lines)

| Function | Lines | Reason |
|----------|-------|--------|
| `generateSparkline()` | 4870-4902 | Never called anywhere |
| `getChartColor()` | 4857-4859 | Only used by generateSparkline |
| `setChartColor()` | 4861-4868 | Only used by generateSparkline |
| `showColorPicker()` | 6365-6408 | Only used by generateSparkline |
| Chart localStorage logic | ~5 lines | Related to above |

**NOTE:** `parsePriceHistory()` is USED in both app.js (line 4224) and csv-builder.js - DO NOT REMOVE.

---

## Part 2: Duplicate Code Consolidation (app.js)

### High-Value Consolidation Targets

#### 2.1 File Upload Handlers (~150 lines saved)
`handleHistoricalFiles()` and `handleNewFiles()` are nearly identical.

**Current:**
```javascript
function handleHistoricalFiles(files) { ... } // ~65 lines
function handleNewFiles(files) { ... }        // ~65 lines
```

**Refactored:**
```javascript
function handleFiles(files, type) {
    const config = type === 'historical'
        ? { storage: historicalFiles, infoId: 'historicalInfo', btnId: 'processHistoricalBtn', csv: historicalCsv, advancedMode: historicalAdvancedMode }
        : { storage: newFiles, infoId: 'newInfo', btnId: 'processNewBtn', csv: newCsv, advancedMode: newAdvancedMode };
    // Unified logic...
}
```

#### 2.2 Process Catalog Functions (~200 lines saved)
`processHistoricalCatalog()` and `processNewProducts()` share 80% of their code.

**Refactored:**
```javascript
async function processCatalog(type) {
    const config = getCatalogConfig(type); // Returns type-specific settings
    // Unified processing logic...
}
```

#### 2.3 Catalog Options (~100 lines saved)
`initCatalogOptions()` / `initNewCatalogOptions()` and related handlers are duplicated.

---

## Part 3: Progress Estimation Simplification (app.js)

### Current State
- `startProgressEstimation()` (~80 lines) with complex time calculations
- User reports it's inaccurate anyway

### Simplified Version (~30 lines)
```javascript
function showProgress(containerId, totalItems) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="progress-estimation">
            <h4>Processing ${totalItems} items...</h4>
            <div class="progress-bar-modern">
                <div class="progress-fill-modern"></div>
            </div>
            <div class="progress-time">
                <span class="items-processed">0 / ${totalItems}</span>
            </div>
        </div>
    `;
    return {
        update: (processed) => { /* Update count and bar width */ },
        complete: (msg) => { /* Show completion */ }
    };
}
```

---

## Part 4: app.js Module Extraction

### Phase 1: Extract Utilities (Low Risk)
Create `backend/static/js/utils.js`:
- `escapeHtml()`, `formatSeconds()`, `normalizeDateString()`
- Memory utilities: `cleanupMemory()`, `revokeAllBlobUrls()`
- Event tracking: `addTrackedListener()`, `removeTrackedListeners()`
- ~200 lines

### Phase 2: Extract UI Components (Medium Risk)
Create `backend/static/js/ui-components.js`:
- `showToast()`, `showLoadingSpinner()`
- Modal functions: `showCsvHelp()`, `closeCsvHelp()`, etc.
- Tooltip functions: `initTooltips()`, `positionTooltip()`
- ~300 lines

### Phase 3: Extract Mobile Integration (Medium Risk)
Create `backend/static/js/mobile-integration.js`:
- `startMatchResultsPolling()`, `stopMatchResultsPolling()`
- `openMobileModal()`, `closeMobileModal()`
- `generateQRCode()`, `generateNewPassword()`
- ~200 lines

### Phase 4: Extract Catalog Management (Medium Risk)
Create `backend/static/js/catalog-manager-main.js`:
- `initCatalogOptions()`, `handleCatalogOptionChange()`
- `refreshCatalogInfo()`, `startStateChecking()`
- ~400 lines

### Implementation Strategy
1. Each module exposes functions via `window.ModuleName = { ... }`
2. Main app.js imports modules via script tags
3. Replace function calls incrementally
4. Test after each extraction

---

## Part 5: app.py Route Extraction

### Largest Routes (Current)
| Route | Lines | Location |
|-------|-------|----------|
| `batch_upload_products` | 465 | line 2251 |
| `match_products` | 333 | line 2716 |
| `upload_product` | 295 | line 1956 |
| `mobile_upload_and_match` | 284 | line 1081 |
| `create_metadata_products_batch` | 268 | line 1558 |

### Strategy: Extract Business Logic to Services

Rather than using Blueprints (which would change URLs), extract the business logic into service functions.

#### Create `backend/services/upload_service.py`:
```python
def process_single_upload(file, metadata, is_historical):
    """Business logic extracted from upload_product()"""
    pass

def process_batch_upload(files, metadata_list, is_historical):
    """Business logic extracted from batch_upload_products()"""
    pass
```

#### Create `backend/services/matching_service.py`:
```python
def match_single_product(product_id, threshold, limit):
    """Business logic extracted from match_products()"""
    pass

def match_batch_products(product_ids, threshold, limit):
    """Business logic extracted from batch_match_products()"""
    pass
```

#### Create `backend/services/mobile_service.py`:
```python
def process_mobile_upload_and_match(image, threshold, limit):
    """Business logic extracted from mobile_upload_and_match()"""
    pass
```

### Result
- Route handlers become thin wrappers (~20-30 lines each)
- Business logic is testable independently
- No URL changes, no Blueprint complexity

---

## Part 6: Implementation Order

### Week 1: Safe Cleanup
1. [ ] Remove dead code (generateSparkline, chart functions)
2. [ ] Simplify progress estimation
3. [ ] Test thoroughly

### Week 2: JS Consolidation
4. [ ] Create unified `handleFiles(files, type)` function
5. [ ] Create unified `processCatalog(type)` function
6. [ ] Create unified catalog options handlers
7. [ ] Test thoroughly

### Week 3: JS Module Extraction
8. [ ] Extract utils.js
9. [ ] Extract ui-components.js
10. [ ] Extract mobile-integration.js
11. [ ] Test thoroughly

### Week 4: Python Service Extraction
12. [ ] Create upload_service.py
13. [ ] Create matching_service.py
14. [ ] Create mobile_service.py
15. [ ] Final testing

---

## Verification Checklist

After each change:
- [ ] App loads without console errors
- [ ] Historical catalog upload works
- [ ] New products upload works
- [ ] Matching works (all 3 modes)
- [ ] Mobile upload works
- [ ] Results display correctly
- [ ] Filters work
- [ ] Export works
- [ ] Catalog manager works
- [ ] Snapshot save/load works

---

## Expected Results

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| app.js | 8,614 lines | ~6,000 lines | ~30% |
| app.py | 6,418 lines | ~4,500 lines | ~30% |

New files created:
- `backend/static/js/utils.js` (~200 lines)
- `backend/static/js/ui-components.js` (~300 lines)
- `backend/static/js/mobile-integration.js` (~200 lines)
- `backend/static/js/catalog-manager-main.js` (~400 lines)
- `backend/services/upload_service.py` (~400 lines)
- `backend/services/matching_service.py` (~300 lines)
- `backend/services/mobile_service.py` (~200 lines)

---

## Risk Mitigation

1. **Git branches**: Work on `refactor/cleanup` branch
2. **Small commits**: One logical change per commit
3. **Test after each change**: Use the verification checklist
4. **Rollback ready**: If something breaks, revert immediately
5. **No URL changes**: All API endpoints remain identical
