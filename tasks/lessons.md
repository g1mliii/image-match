# Lessons Learned

Record failures, detection signals, and prevention rules here.

---

## Template
```
### [Date] - Brief Title
**Failure mode:**
**Detection signal:**
**Prevention rule:**
```

---

## Lessons

### 2025-02-04 - Lucide Icons Re-initialization
**Failure mode:** Icons not appearing after dynamic content render
**Detection signal:** `data-lucide` attributes present but no SVG icons visible
**Prevention rule:** Always call `IconManager.reinit(container)` after:
- Dynamic content rendering (results, modals)
- Status updates that change icon elements
- Any DOM manipulation that adds Lucide icon placeholders

See `.claude/projects/.../memory/ICON_SYSTEM_AUDIT.md` for full details.

### 2025-02-04 - CSV/Snapshot Relationship
**Failure mode:** User confusion about CSV files still appearing after snapshot deletion
**Detection signal:** User reports CSV still visible in UI after deleting snapshot
**Prevention rule:** Understand that:
- CSV content is stored INSIDE snapshot database files, not separately
- The "Download CSV" buttons in upload section pull from MAIN database, not snapshots
- Deleting a snapshot doesn't affect the main database
- To clear CSV data, user must clear the working directory (deletes main DB products)
