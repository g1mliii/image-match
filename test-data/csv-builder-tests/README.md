# CSV Builder Linking Strategy Test Files

Test files to verify all CSV Builder linking strategies work correctly.

## Source Images
**Path:** `C:\Users\subai\Downloads\drive-download-20251230T180235Z-1-001\new screen`

### Folder Structure (46 total files):
| Folder | Files | Sample Filenames |
|--------|-------|------------------|
| clothing/ | 6 | `Screenshot 2025-11-24 234109.png`, `...234109 - Copy.png` |
| coding/ | 4 | `Screenshot 2025-09-22 200846.png` |
| extension/ | 9 | `Screenshot 2025-09-16 123009.png` |
| imagematch/ | 4 | `Screenshot 2025-11-20 191259.png` |
| potplayer/ | 6 | `Screenshot 2025-11-28 125548.png` |
| website/ | 10 | `Gemini_Generated_Image_xxx.png`, `Untitled design.png` |
| (root) | 7 | `IMG_1139.jpg`, `Screenshot 2025-09-30 225158.png` |

---

## How to Test

1. Open main app → click **CSV BUILDER** in header
2. **Step 1:** Upload the `new screen` folder
   - **Expected:** 46 images loaded
   - **Expected:** Categories detected: clothing, coding, extension, imagematch, potplayer, website
3. **Step 2:** Click **IMPORT FROM FILE** → select test CSV
4. **Select the matching linking strategy**
5. **Verify:** Linked count = 46, Unlinked = 0
6. Click **APPLY LINKING** and check data table

---

## Test Files

### 1. `test_filename_equals_sku.csv`
**Strategy:** Filename = SKU

| Field | Expected |
|-------|----------|
| CSV Rows | 46 |
| **Linked** | **46** |
| Unlinked | 0 |

**How it works:** The `sku` column contains exact filename (without .png/.jpg extension)

**Sample rows:**
```
sku,name,category,price
Screenshot 2025-11-24 234109,Winter Jacket,clothing,89.99
IMG_1139,Phone Photo,personal,0
Untitled design,Design Asset,website,15.00
```

---

### 2. `test_metadata_filename.csv`
**Strategy:** Metadata Filename Column

| Field | Expected |
|-------|----------|
| CSV Rows | 46 |
| **Linked** | **46** |
| Unlinked | 0 |

**How it works:** CSV has explicit `filename` column with full filename INCLUDING extension

**Sample rows:**
```
filename,sku,name,category,price
Screenshot 2025-11-24 234109.png,CLO-001,Winter Jacket,clothing,89.99
IMG_1139.jpg,ROOT-001,Phone Capture,personal,0
Untitled design.png,WEB-010,Design Asset,website,15.00
```

---

### 3. `test_name_equals_filename.csv`
**Strategy:** Name = Image Filename

| Field | Expected |
|-------|----------|
| CSV Rows | 46 |
| **Linked** | **46** |
| Unlinked | 0 |

**How it works:** The `name` column contains exact filename (without extension)

**Sample rows:**
```
sku,name,category,price
CLO-001,Screenshot 2025-11-24 234109,clothing,89.99
ROOT-001,IMG_1139,personal,0
WEB-010,Untitled design,website,15.00
```

---

### 4. `test_search_all_fields.csv`
**Strategy:** Search All Fields

| Field | Expected |
|-------|----------|
| CSV Rows | 46 |
| **Linked** | **46** |
| Unlinked | 0 |

**How it works:** Filename appears somewhere in ANY column (description, notes, etc.)

**Sample rows:**
```
sku,name,description,notes
CLO-001,Winter Jacket,Screenshot 2025-11-24 234109 shows our jacket,winter
ROOT-001,Phone Photo,IMG_1139 from phone camera,photo
WEB-010,Design Asset,Untitled design file,asset
```

---

## Verification Checklist

For EACH test file, verify:

- [ ] **Linked count = 46** (all images matched)
- [ ] **Unlinked count = 0** (no failures)
- [ ] Root level files link (IMG_1139, misc screenshots)
- [ ] Categorized files link (clothing/, coding/, etc.)
- [ ] Special filenames work:
  - [ ] `Screenshot 2025-11-24 234109 - Copy.png` (has " - Copy")
  - [ ] `Gemini_Generated_Image_cklaevcklaevckla (1).png` (has parentheses)
  - [ ] `Untitled design.png` (has space)
  - [ ] `IMG_1139.jpg` (jpg not png)

---

## If Tests Fail

If linked count < 46, check:

1. **Case sensitivity** - filenames might need exact case match
2. **Extension handling** - some strategies need .png, others don't
3. **Special characters** - spaces, parentheses, hyphens
4. **Path vs filename** - strategy might be matching full path not just filename
