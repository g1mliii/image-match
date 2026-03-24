/*
 * Catalog options/state/snapshot UI module extracted from app.core.js.
 */

const runWhenDomReadyCatalog = window.__catalogMatchOnDomReady || function(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback, { once: true });
        return;
    }
    setTimeout(callback, 0);
};

// ============ Catalog Options ============

let existingCatalogStats = null;

function initCatalogOptions() {
    // Check if there's an existing catalog
    checkExistingCatalog();

    // Add event listeners for catalog options
    const radioButtons = document.querySelectorAll('input[name="catalogLoadOption"]');
    radioButtons.forEach(radio => {
        radio.addEventListener('change', handleCatalogOptionChange);
    });
}

async function checkExistingCatalog() {
    try {
        const response = await fetch('/api/catalogs/main-db-stats');
        // Use the same endpoint as the active catalog info bar
        if (!response.ok) throw new Error('Failed to fetch catalog stats');

        const data = await response.json();
        existingCatalogStats = data;

        const catalogOptions = document.getElementById('catalogOptions');
        const statsEl = document.getElementById('existingCatalogStats');

        if (data.exists) {
            // Show catalog options - ALWAYS visible when there's an existing catalog
            catalogOptions.style.display = 'block';

            let statsText = `<strong>${data.total_products.toLocaleString()}</strong> products`;
            if (data.historical_products > 0) {
                statsText = `<strong>${data.historical_products.toLocaleString()}</strong> historical products`;
                if (data.new_products > 0) {
                    statsText += ` | <strong>${data.new_products.toLocaleString()}</strong> new products`;
                }
            }
            if (data.loaded_snapshot && data.loaded_snapshot.loaded) {
                statsText += ` | <strong>${data.loaded_snapshot.name}</strong>`;
            }

            statsEl.innerHTML = statsText;

            // Check for large database warning
            if (data.database_size_mb && data.database_size_mb > 500) {
                showToast('Database is large (' + data.database_size_mb.toFixed(0) + ' MB). Consider cleaning up old products.', 'warning', 8000);
            }
        } else {
            // No existing catalog - still show options but with different message
            catalogOptions.style.display = 'block';
            statsEl.innerHTML = `<em>No existing catalog</em>`;
            // Disable "use existing" option when there's no catalog
            const useExistingRadio = document.querySelector('input[name="catalogLoadOption"][value="use_existing"]');
            if (useExistingRadio) {
                useExistingRadio.disabled = true;
                // Select "add_to_existing" by default when no catalog exists
                const addToExistingRadio = document.querySelector('input[name="catalogLoadOption"][value="add_to_existing"]');
                if (addToExistingRadio) {
                    addToExistingRadio.checked = true;
                }
            }
        }

        // Initialize UI state based on selected option (fixes initial load bug)
        handleCatalogOptionChange();
    } catch (error) {
        console.error('Error checking existing catalog:', error);
        // Still show options even on error
        const catalogOptions = document.getElementById('catalogOptions');
        const statsEl = document.getElementById('existingCatalogStats');
        if (catalogOptions && statsEl) {
            catalogOptions.style.display = 'block';
            statsEl.innerHTML = `<em>Unable to load catalog info</em>`;
        }
    }
}

async function handleCatalogOptionChange(e) {
    // Support both event-based calls and direct calls
    const option = e && e.target ? e.target.value : getCatalogLoadOption();
    const dropZone = document.getElementById('historicalDropZone');
    const processBtn = document.getElementById('processHistoricalBtn');
    const downloadDiv = document.getElementById('downloadExistingHistoricalDiv');

    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process, hide download
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING CATALOG';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // AUTO-LOAD CSV WHEN "USE EXISTING" IS SELECTED
        // This populates the CSV file label immediately so user can see it's loaded
        autoLoadCatalogCSV();
    } else if (option === 'replace') {
        // Replace catalog - show warning, hide download
        if (e && e.target && existingCatalogStats && existingCatalogStats.historical_products > 0) {
            // Only show confirmation dialog when user manually changes to replace (not on initial load)
            const confirmed = await window.showAppConfirmDialog({
                title: 'Replace Historical Catalog',
                message: `Delete all ${existingCatalogStats.historical_products.toLocaleString()} existing historical products and replace them with a new catalog?`,
                details: 'A backup snapshot will be created automatically.',
                confirmLabel: 'REPLACE',
                danger: true
            });
            if (!confirmed) {
                // Revert to use_existing
                document.querySelector('input[name="catalogLoadOption"][value="use_existing"]').checked = true;
                handleCatalogOptionChange();
                return;
            }
        }
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'REPLACE & PROCESS';
        if (downloadDiv) downloadDiv.style.display = 'none';
    } else {
        // Add to existing - show download button
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'ADD & PROCESS';
        if (downloadDiv) downloadDiv.style.display = 'block';
    }
}

function getCatalogLoadOption() {
    const selected = document.querySelector('input[name="catalogLoadOption"]:checked');
    return selected ? selected.value : 'add_to_existing';
}

// Modify processHistoricalCatalog to handle catalog options
const originalProcessHistoricalCatalog = typeof processHistoricalCatalog === 'function' ? processHistoricalCatalog : null;

// Override processHistoricalCatalog to handle catalog options
async function processHistoricalCatalogWithOptions() {
    const option = getCatalogLoadOption();

    if (option === 'use_existing') {
        // Skip upload, use existing catalog
        showToast('Using existing catalog', 'success');

        // Load existing products from database
        try {
            const response = await fetch('/api/catalog/products?type=historical&limit=10000');
            if (!response.ok) throw new Error('Failed to load existing products');

            const data = await response.json();
            historicalProducts = data.products.map(p => ({
                id: p.id,
                filename: p.filename,
                category: p.category,
                sku: p.sku,
                name: p.product_name,
                is_historical: true,
                hasFeatures: p.has_features  // Use actual feature status from DB
            }));

            // Update UI
            document.getElementById('historicalStatus').innerHTML =
                `<p class="success">Loaded ${historicalProducts.length} products from existing catalog</p>`;

            // Show next section
            showNewSectionAfterHistoricalStep();

            // Load metadata schema to populate sliders if research was already done
            await loadMetadataSchema();

            // Auto-load CSV if available
            await autoLoadCatalogCSV();

        } catch (error) {
            console.error('Error loading existing catalog:', error);
            showToast('Failed to load existing catalog', 'error');
        }
        return;
    }

    if (option === 'replace') {
        // Create automatic backup snapshot before replacing (debounced to avoid duplicates in batch operations)
        const now = Date.now();
        if (now - lastAutoBackupTime > AUTO_BACKUP_DEBOUNCE_MS) {
            try {
                console.log('[REPLACE] Creating automatic backup snapshot...');
                showToast('Creating backup snapshot...', 'info');

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                const snapshotName = `auto-backup-before-replace-${timestamp}`;

                const snapshotResponse = await fetch('/api/catalogs/save-current', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: snapshotName,
                        description: 'Automatic backup created before batch replace operations',
                        tags: ['auto-backup', 'replace'],
                        skip_if_empty: true
                    })
                });
                const snapshotResult = await snapshotResponse.json().catch(() => ({}));

                if (snapshotResponse.ok) {
                    if (snapshotResult.skipped) {
                        console.log('[REPLACE] Skipping backup snapshot (catalog is empty)');
                        showToast('No existing catalog to back up', 'info');
                    } else {
                        console.log('[REPLACE] Backup snapshot created:', snapshotName);
                        showToast('Backup snapshot created', 'success');
                        // Wait a moment to ensure snapshot is complete
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                    lastAutoBackupTime = now;
                } else {
                    console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                    showToast('Warning: Could not create backup snapshot', 'warning');
                }
            } catch (error) {
                console.warn('[REPLACE] Error creating backup snapshot:', error);
                showToast('Warning: Could not create backup snapshot', 'warning');
                // Continue with replace even if snapshot fails
            }
        } else {
            console.log('[REPLACE] Skipping backup (within debounce window) - batch operation detected');
            showToast('Batch operation detected - using previous backup', 'info');
        }

        // Clear existing catalog
        try {
            console.log('[REPLACE] Starting catalog cleanup...');
            showToast('Clearing existing catalog...', 'info');
            const response = await fetch('/api/catalog/cleanup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'historical' })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('[REPLACE] Cleanup failed:', errorData);
                throw new Error('Failed to clear catalog');
            }

            const result = await response.json();
            console.log('[REPLACE] Cleanup successful:', result);
            showToast(`Existing catalog cleared (${result.products_deleted} products deleted)`, 'success');

            // Wait a moment to ensure cleanup is complete
            await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
            console.error('[REPLACE] Error clearing catalog:', error);
            showToast('Failed to clear existing catalog', 'error');
            return;
        }
    }

    // Continue with normal processing (add_to_existing or replace after clearing)
    await processHistoricalCatalog();
}


function initNewCatalogOptions() {
    // Check if there's an existing new products catalog
    checkExistingNewCatalog();

    // Add event listeners for new catalog options
    const radioButtons = document.querySelectorAll('input[name="newCatalogLoadOption"]');
    radioButtons.forEach(radio => {
        radio.addEventListener('change', handleNewCatalogOptionChange);
    });
}

async function checkExistingNewCatalog() {
    try {
        const response = await fetch('/api/catalog/stats');
        if (!response.ok) throw new Error('Failed to fetch catalog stats');

        const data = await response.json();

        const statsEl = document.getElementById('existingNewCatalogStats');

        if (data.new_products > 0) {
            statsEl.innerHTML = `<strong>${data.new_products.toLocaleString()}</strong> new products in database`;

            // Enable "use existing" option
            const useExistingRadio = document.querySelector('input[name="newCatalogLoadOption"][value="use_existing"]');
            if (useExistingRadio) {
                useExistingRadio.disabled = false;
            }
        } else {
            statsEl.innerHTML = `<em>No existing new products</em>`;

            // Disable "use existing" option when there's no new products
            const useExistingRadio = document.querySelector('input[name="newCatalogLoadOption"][value="use_existing"]');
            if (useExistingRadio) {
                useExistingRadio.disabled = true;
                // Select "add_to_existing" by default
                const addToExistingRadio = document.querySelector('input[name="newCatalogLoadOption"][value="add_to_existing"]');
                if (addToExistingRadio) {
                    addToExistingRadio.checked = true;
                }
            }
        }

        // Initialize UI state based on selected option (fixes initial load bug)
        handleNewCatalogOptionChange();
    } catch (error) {
        console.error('Error checking existing new catalog:', error);
    }
}

async function handleNewCatalogOptionChange() {
    const option = getNewCatalogLoadOption();
    const dropZone = document.getElementById('newDropZone');
    const processBtn = document.getElementById('processNewBtn');
    const downloadDiv = document.getElementById('downloadExistingNewDiv');

    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process, hide download
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // AUTO-LOAD CSV WHEN "USE EXISTING" IS SELECTED
        // This populates the CSV file label immediately so user can see it's loaded
        autoLoadCatalogCSV();
    } else if (option === 'replace') {
        // Replacing - enable upload, show warning, hide download
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // Show warning
        if (existingCatalogStats && existingCatalogStats.new_products > 0) {
            const confirmed = await window.showAppConfirmDialog({
                title: 'Replace New Products',
                message: `Delete all ${existingCatalogStats.new_products} existing new products and replace them with a new catalog?`,
                details: 'A backup snapshot will be created automatically.',
                confirmLabel: 'REPLACE',
                danger: true
            });
            if (!confirmed) {
                // Revert to add_to_existing
                document.querySelector('input[name="newCatalogLoadOption"][value="add_to_existing"]').checked = true;
                handleNewCatalogOptionChange();
                return;
            }
        }
    } else {
        // add_to_existing - enable upload, show download button
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'block';
    }
}

function getNewCatalogLoadOption() {
    const selected = document.querySelector('input[name="newCatalogLoadOption"]:checked');
    return selected ? selected.value : 'add_to_existing';
}

// Override processNewProducts to handle catalog options
async function processNewCatalogWithOptions() {
    const option = getNewCatalogLoadOption();

    if (option === 'use_existing') {
        // Skip upload, use existing catalog
        showToast('Using existing new products', 'success');

        // Load existing new products from database
        try {
            const response = await fetch('/api/catalog/products?type=new&limit=10000');
            if (!response.ok) throw new Error('Failed to load existing new products');

            const data = await response.json();
            newProducts = data.products.map(p => ({
                id: p.id,
                filename: p.filename,
                category: p.category,
                sku: p.sku,
                name: p.product_name,
                is_historical: false,
                hasFeatures: p.has_features  // Use actual feature status from DB
            }));

            // Update UI
            document.getElementById('newStatus').innerHTML =
                `<p class="success">Loaded ${newProducts.length} new products from existing catalog</p>`;

            // Show next section
            document.getElementById('matchSection').style.display = 'block';
            document.getElementById('matchSection').scrollIntoView({ behavior: 'smooth' });

            // Load metadata schema to populate sliders
            await loadMetadataSchema();

            // Auto-load CSV if available
            await autoLoadCatalogCSV();

        } catch (error) {
            console.error('Error loading existing new products:', error);
            showToast('Failed to load existing new products', 'error');
        }
        return;
    }

    if (option === 'replace') {
        // Create automatic backup snapshot before replacing (debounced to avoid duplicates in batch operations)
        const now = Date.now();
        if (now - lastAutoBackupTime > AUTO_BACKUP_DEBOUNCE_MS) {
            try {
                console.log('[REPLACE] Creating automatic backup snapshot...');
                showToast('Creating backup snapshot...', 'info');

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                const snapshotName = `auto-backup-before-replace-${timestamp}`;

                const snapshotResponse = await fetch('/api/catalogs/save-current', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: snapshotName,
                        description: 'Automatic backup created before batch replace operations',
                        tags: ['auto-backup', 'replace'],
                        skip_if_empty: true
                    })
                });
                const snapshotResult = await snapshotResponse.json().catch(() => ({}));

                if (snapshotResponse.ok) {
                    if (snapshotResult.skipped) {
                        console.log('[REPLACE] Skipping backup snapshot (catalog is empty)');
                        showToast('No existing catalog to back up', 'info');
                    } else {
                        console.log('[REPLACE] Backup snapshot created:', snapshotName);
                        showToast('Backup snapshot created', 'success');
                        // Wait a moment to ensure snapshot is complete
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                    lastAutoBackupTime = now;
                } else {
                    console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                    showToast('Warning: Could not create backup snapshot', 'warning');
                }
            } catch (error) {
                console.warn('[REPLACE] Error creating backup snapshot:', error);
                showToast('Warning: Could not create backup snapshot', 'warning');
                // Continue with replace even if snapshot fails
            }
        } else {
            console.log('[REPLACE] Skipping backup (within debounce window) - batch operation detected');
            showToast('Batch operation detected - using previous backup', 'info');
        }

        // Clear existing new products
        try {
            console.log('[REPLACE] Starting new products cleanup...');
            showToast('Clearing existing new products...', 'info');
            const response = await fetch('/api/catalog/cleanup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'new' })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('[REPLACE] Cleanup failed:', errorData);
                throw new Error('Failed to clear new products');
            }

            const result = await response.json();
            console.log('[REPLACE] Cleanup successful:', result);
            showToast(`Existing new products cleared (${result.products_deleted} products deleted)`, 'success');

            // Wait a moment to ensure cleanup is complete
            await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
            console.error('[REPLACE] Error clearing new products:', error);
            showToast('Failed to clear existing new products', 'error');
            return;
        }
    }

    // Continue with normal processing (add_to_existing or replace after clearing)
    await processNewProducts();
}

// Hook into the process button
runWhenDomReadyCatalog(() => {
    setTimeout(() => {
        const processBtn = document.getElementById('processHistoricalBtn');
        if (processBtn) {
            // Store original handler
            const originalHandler = processBtn.onclick;

            processBtn.onclick = async (e) => {
                // Always use the processHistoricalCatalogWithOptions which handles all cases
                await processHistoricalCatalogWithOptions();
            };
        }

        // Initialize new catalog options
        initNewCatalogOptions();

        const processNewBtn = document.getElementById('processNewBtn');
        if (processNewBtn) {
            const originalNewHandler = processNewBtn.onclick;

            processNewBtn.onclick = async (e) => {
                await processNewCatalogWithOptions();
            };
        }
    }, 100);
});


let lastKnownCatalogState = {
    totalProducts: 0,
    historicalProducts: 0,
    newProducts: 0,
    lastChecked: null
};

// Check if catalog has changed since last check
async function checkCatalogStateChanged() {
    try {
        const response = await fetch('/api/catalog/stats');
        if (!response.ok) return false;

        const stats = await response.json();

        const hasChanged = (
            lastKnownCatalogState.totalProducts !== stats.total_products ||
            lastKnownCatalogState.historicalProducts !== stats.historical_products ||
            lastKnownCatalogState.newProducts !== stats.new_products
        );

        // Update last known state
        lastKnownCatalogState = {
            totalProducts: stats.total_products,
            historicalProducts: stats.historical_products,
            newProducts: stats.new_products,
            lastChecked: Date.now()
        };

        return hasChanged;
    } catch (error) {
        console.error('Error checking catalog state:', error);
        return false;
    }
}

// Reset app state when catalog changes are detected
async function resetAppState(reason = 'Catalog data has changed') {
    console.log('Resetting app state:', reason);

    // Clear in-memory product data
    historicalProducts = [];
    newProducts = [];
    matchResults = [];
    historicalFiles = [];
    newFiles = [];
    historicalCsv = null;
    newCsv = null;

    // Clear saved state (webview only)
    await clearSavedState();

    // TRIGGER BACKEND CLEANUP: Wipe match results from DB to save space
    try {
        fetch('/api/cleanup-matches', { method: 'POST' })
            .then(res => res.json())
            .then(d => console.log('[CLEANUP] Transient matches wiped:', d))
            .catch(e => console.warn('[CLEANUP] Failed to wipe matches:', e));
    } catch (e) {
        console.warn('Backend cleanup error:', e);
    }

    // Reset UI to initial state
    const historicalSection = document.getElementById('historicalSection');
    const newSection = document.getElementById('newSection');
    const matchSection = document.getElementById('matchSection');
    const resultsSection = document.getElementById('resultsSection');

    if (newSection) newSection.style.display = 'none';
    if (matchSection) matchSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    setSectionCollapsed('historicalSection', false);
    setSectionCollapsed('newSection', false);

    // Clear status messages
    const historicalStatus = document.getElementById('historicalStatus');
    const newStatus = document.getElementById('newStatus');
    const historicalInfo = document.getElementById('historicalInfo');
    const newInfo = document.getElementById('newInfo');

    if (historicalStatus) historicalStatus.innerHTML = '';
    if (newStatus) newStatus.innerHTML = '';
    if (historicalInfo) historicalInfo.innerHTML = '';
    if (newInfo) newInfo.innerHTML = '';

    // Reset buttons
    const processHistoricalBtn = document.getElementById('processHistoricalBtn');
    const processNewBtn = document.getElementById('processNewBtn');

    if (processHistoricalBtn) {
        processHistoricalBtn.disabled = true;
        processHistoricalBtn.textContent = 'PROCESS';
    }
    if (processNewBtn) {
        processNewBtn.disabled = true;
    }

    // Reset drop zones
    const historicalDropZone = document.getElementById('historicalDropZone');
    const newDropZone = document.getElementById('newDropZone');

    if (historicalDropZone) {
        historicalDropZone.style.opacity = '1';
        historicalDropZone.style.pointerEvents = 'auto';
    }
    if (newDropZone) {
        newDropZone.style.opacity = '1';
        newDropZone.style.pointerEvents = 'auto';
    }

    // Refresh catalog options
    checkExistingCatalog();

    // Scroll to top
    if (historicalSection) {
        historicalSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Validate that products in memory still exist in database
async function validateProductsExist(productIds) {
    if (!productIds || productIds.length === 0) return { valid: true, missing: [] };

    try {
        const response = await fetch('/api/catalog/products?limit=10000');
        if (!response.ok) return { valid: false, missing: productIds };

        const data = await response.json();
        const existingIds = new Set(data.products.map(p => p.id));

        const missing = productIds.filter(id => !existingIds.has(id));

        return {
            valid: missing.length === 0,
            missing: missing
        };
    } catch (error) {
        console.error('Error validating products:', error);
        return { valid: false, missing: [] };
    }
}

// Check state before critical operations
async function ensureStateValid() {
    const hasChanged = await checkCatalogStateChanged();

    if (hasChanged) {
        // Validate that our in-memory products still exist
        const historicalIds = historicalProducts.map(p => p.id).filter(id => id);
        const newIds = newProducts.map(p => p.id).filter(id => id);

        const historicalValidation = await validateProductsExist(historicalIds);
        const newValidation = await validateProductsExist(newIds);

        if (!historicalValidation.valid || !newValidation.valid) {
            showToast('Database has changed. Resetting to sync with current data.', 'warning', 5000);
            resetAppState('Products were deleted from Catalog Manager');
            return false;
        }
    }

    return true;
}


document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === 'visible') {
        // User returned to this tab - check if catalog changed
        const hasChanged = await checkCatalogStateChanged();

        if (hasChanged) {
            // Check if we have any in-progress work
            const hasHistoricalData = historicalProducts.length > 0;
            const hasNewData = newProducts.length > 0;
            const hasResults = matchResults.length > 0;

            if (hasHistoricalData || hasNewData || hasResults) {
                // Validate our data is still valid
                const historicalIds = historicalProducts.map(p => p.id).filter(id => id);
                const newIds = newProducts.map(p => p.id).filter(id => id);

                const historicalValidation = await validateProductsExist(historicalIds);
                const newValidation = await validateProductsExist(newIds);

                if (!historicalValidation.valid || !newValidation.valid) {
                    showToast('Catalog was modified. Resetting app state.', 'warning', 5000);
                    resetAppState('Catalog modified while away');
                } else {
                    // Just refresh the catalog options display
                    checkExistingCatalog();
                }
            } else {
                // No in-progress work, just refresh catalog options
                checkExistingCatalog();
            }
        }
    }
});

// Periodic state check (every 30 seconds if tab is visible)
function startStateChecking() {
    if (stateCheckInterval) return;

    stateCheckInterval = setInterval(async () => {
        if (document.visibilityState === 'visible') {
            // Only check if we have in-progress work
            if (historicalProducts.length > 0 || newProducts.length > 0) {
                await ensureStateValid();
            }
        }
    }, 30000); // Check every 30 seconds
}

function stopStateChecking() {
    if (stateCheckInterval) {
        clearInterval(stateCheckInterval);
        stateCheckInterval = null;
    }
}

// Start state checking when page loads
runWhenDomReadyCatalog(() => {
    // Initialize last known state
    checkCatalogStateChanged();

    // Start periodic checking
    startStateChecking();
});

// Expose reset function globally for Catalog Manager to call
window.resetMainAppState = resetAppState;
window.checkCatalogStateChanged = checkCatalogStateChanged;

// ============ Catalog Info Integration ============

// Initialize catalog info bar on page load
setTimeout(() => {
    initCatalogInfo();
}, 500);

function initCatalogInfo() {
    loadCatalogInfo();
    initCatalogChangeListener();
}

// Load and display catalog info
async function loadCatalogInfo() {
    try {
        const response = await fetch('/api/catalogs/main-db-stats');
        if (!response.ok) {
            throw new Error('Failed to load catalog stats');
        }

        const data = await response.json();

        const infoBar = document.getElementById('catalogInfoBar');
        const summary = document.getElementById('activeCatalogSummary');

        if (!infoBar || !summary) return;

        if (data.exists) {
            let text = `${data.total_products} products (${data.historical_products} historical, ${data.new_products} new)`;

            if (data.loaded_snapshot && data.loaded_snapshot.loaded) {
                text += ` | Loaded from: "${data.loaded_snapshot.name}"`;
            }

            summary.textContent = text;
            infoBar.style.display = 'block';

            // Also load metadata schema if a catalog exists
            loadMetadataSchema();
        } else {
            summary.textContent = 'No catalog loaded';
            infoBar.style.display = 'block';
        }

    } catch (error) {
        console.error('Error loading catalog info:', error);
        const summary = document.getElementById('activeCatalogSummary');
        if (summary) {
            summary.textContent = 'Unable to load catalog info';
        }
    }
}

// Refresh catalog info
function refreshCatalogInfo() {
    loadCatalogInfo();
    showToast('Catalog info refreshed', 'success');
}

// Open Catalog Manager (delegates to index.html's guarded version if available)
function openCatalogManager() {
    // If the guarded version from index.html is available, it handles dedup
    // This function exists as fallback for standalone catalog page usage
    if (window.pywebview && window.pywebview.api) {
        console.log('[NAV] Opening Catalog Manager in child window (webviewer)...');
        try {
            window.pywebview.api.open_catalog_manager();
        } catch (e) {
            console.error('[NAV] Error opening catalog manager:', e);
            window.open('/catalog-manager', '_blank');
        }
    } else {
        console.log('[NAV] Pywebview not available, opening in browser tab...');
        window.open('/catalog-manager', '_blank');
    }
}

// Listen for catalog changes from Catalog Manager (browser mode only)
// PyWebview mode uses window.handleChildWindowEvent in index.html
function initCatalogChangeListener() {
    // Listen via BroadcastChannel (browser mode only)
    try {
        catalogChannel = new BroadcastChannel('catalog_changes');
        catalogChannel.onmessage = (event) => {
            // In browser mode, call the same handler that pywebview uses
            if (typeof handleCatalogChanged === 'function') {
                handleCatalogChanged({ action: event.data.action, details: event.data.details });
            }
        };
    } catch (e) {
        // BroadcastChannel not supported, use polling
        if (catalogPollingInterval) clearInterval(catalogPollingInterval);
        catalogPollingInterval = setInterval(checkCatalogChangesInMainApp, 2000);
    }

    // Also check on visibility change (when user switches back to this tab)
    const catalogVisibilityHandler = () => {
        if (!document.hidden) {
            checkCatalogChangesInMainApp();

            // Restart polling if it was stopped and BroadcastChannel not available
            if (!catalogChannel && !catalogPollingInterval) {
                if (catalogPollingInterval) clearInterval(catalogPollingInterval);
                catalogPollingInterval = setInterval(checkCatalogChangesInMainApp, 2000);
            }
        } else {
            // Page is hidden, pause polling to save resources
            if (catalogPollingInterval && !catalogChannel) {
                clearInterval(catalogPollingInterval);
                catalogPollingInterval = null;
            }
        }
    };

    addTrackedListener(document, 'visibilitychange', catalogVisibilityHandler, 'general');
}

// Check for catalog changes via sessionStorage (browser mode fallback)
function checkCatalogChangesInMainApp() {
    const changeData = sessionStorage.getItem('catalogManagerChange');
    if (changeData) {
        try {
            const change = JSON.parse(changeData);
            // Only process recent changes (within last 30 seconds)
            if (Date.now() - change.timestamp < 30000) {
                // Call the same handler that pywebview uses
                if (typeof handleCatalogChanged === 'function') {
                    handleCatalogChanged({ action: change.action, details: change.details });
                }
            }
        } catch (e) {
            console.error('Error parsing catalog change:', e);
        }
    }
}


let pendingSaveOperation = null;

/**
 * Show the save dialog to user for snapshot creation
 * @param {string} operation - The operation that triggered the save (e.g., 'comparison_complete', 'manual_save')
 * @param {string} defaultName - Default snapshot name to prefill
 */
function showSaveDialog(operation = 'snapshot', defaultName = null) {
    const dialog = document.getElementById('saveDialog');
    const nameInput = document.getElementById('snapshotNameInput');

    // Generate default name if not provided
    if (!defaultName) {
        const now = new Date();
        const date = now.toISOString().split('T')[0];
        const time = now.toTimeString().split(' ')[0].replace(/:/g, '-');
        defaultName = `${operation}-${date}_${time}`;
    }

    nameInput.value = defaultName;
    nameInput.focus();
    nameInput.select();

    // Store the operation for later use
    pendingSaveOperation = operation;

    // Show dialog
    dialog.classList.add('show');
}

/**
 * Close the save dialog without saving
 */
function closeSaveDialog() {
    const dialog = document.getElementById('saveDialog');
    dialog.classList.remove('show');
    pendingSaveOperation = null;
}

/**
 * Submit the save dialog
 */
let _saveDialogSubmitting = false;
async function submitSaveDialog() {
    if (_saveDialogSubmitting) return;

    const nameInput = document.getElementById('snapshotNameInput');
    const saveType = document.querySelector('input[name="saveType"]:checked').value;
    const snapshotName = nameInput.value.trim();

    if (!snapshotName) {
        showToast('Please enter a snapshot name', 'error');
        return;
    }

    if (saveType === 'skip') {
        closeSaveDialog();
        return;
    }

    _saveDialogSubmitting = true;
    // Disable confirm button to give visual feedback
    const confirmBtn = document.querySelector('#saveDialog .btn-primary');
    if (confirmBtn) confirmBtn.disabled = true;

    try {
        const operation = pendingSaveOperation || 'snapshot';
        console.log(`[SAVE] Saving snapshot "${snapshotName}" as ${saveType}`);

        const response = await fetch('/api/catalogs/save-current', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: snapshotName,
                description: `Saved from ${operation}`,
                tags: [saveType, operation]
            })
        });

        const result = await response.json();

        if (response.ok && result.status === 'success') {
            const typeLabel = saveType === 'persistent' ? 'Persistent snapshot' : 'Session snapshot';
            console.log(`[SAVE] ✓ Snapshot saved successfully: ${snapshotName}`);
            showToast(`${typeLabel} saved: "${snapshotName}"`, 'success');
            closeSaveDialog();
        } else {
            console.error(`[SAVE] Error saving snapshot:`, result);
            showToast(`Error saving snapshot: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('[SAVE] Exception:', error);
        showToast('Error saving snapshot', 'error');
    } finally {
        _saveDialogSubmitting = false;
        const confirmBtn = document.querySelector('#saveDialog .btn-primary');
        if (confirmBtn) confirmBtn.disabled = false;
    }
}


let crashRecoveryData = null;

/**
 * Check for crash recovery on app startup
 */
async function checkForCrashRecovery() {
    try {
        const response = await fetch('/api/catalogs/check-crash-recovery');
        const result = await response.json();

        if (result.crash_detected && result.recovery_snapshot) {
            crashRecoveryData = result.recovery_snapshot;
            showCrashRecoveryDialog(result.recovery_snapshot);
        }
    } catch (error) {
        console.error('Error checking for crash recovery:', error);
    }
}

/**
 * Show crash recovery dialog to user
 */
function showCrashRecoveryDialog(snapshotInfo) {
    const dialog = document.getElementById('crashRecoveryDialog');
    const detailsDiv = document.getElementById('recoveryDetails');

    // Format snapshot info for display
    let html = `
        <div><strong>Name:</strong> ${escapeHtml(snapshotInfo.name || 'Unknown')}</div>
        <div><strong>Created:</strong> ${new Date(snapshotInfo.created_at).toLocaleString()}</div>
        <div><strong>Products:</strong> ${snapshotInfo.product_count || 'Unknown'}</div>
    `;

    if (snapshotInfo.created_by_operation) {
        html += `<div><strong>From:</strong> ${escapeHtml(snapshotInfo.created_by_operation)}</div>`;
    }

    // Show expiry warning since this is a session snapshot
    html += `<div style="margin-top: 10px; color: #c00;"><strong>WARNING:</strong> This recovery snapshot will expire in 1 hour if not restored</div>`;

    detailsDiv.innerHTML = html;
    dialog.classList.add('show');
}

/**
 * Close crash recovery dialog
 */
function closeCrashRecoveryDialog() {
    const dialog = document.getElementById('crashRecoveryDialog');
    dialog.classList.remove('show');
    crashRecoveryData = null;
}

/**
 * Discard the crash recovery snapshot
 */
async function discardCrashRecovery() {
    if (!crashRecoveryData) {
        closeCrashRecoveryDialog();
        return;
    }

    try {
        const snapshotName = encodeURIComponent(crashRecoveryData.id);
        const response = await fetch(`/api/catalogs/${snapshotName}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Recovery snapshot discarded', 'info');
        } else {
            const data = await response.json();
            showToast(`Error discarding snapshot: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error discarding recovery snapshot:', error);
        showToast('Error discarding recovery snapshot', 'error');
    }

    closeCrashRecoveryDialog();
}

/**
 * Restore from crash recovery snapshot
 */
async function restoreCrashRecovery() {
    if (!crashRecoveryData) {
        closeCrashRecoveryDialog();
        return;
    }

    try {
        // Load the snapshot
        const snapshotName = encodeURIComponent(crashRecoveryData.id);
        const response = await fetch(`/api/catalogs/load/${snapshotName}`, {
            method: 'POST'
        });

        const result = await response.json();

        if (response.ok) {
            showToast(`Recovered snapshot "${crashRecoveryData.name}" loaded`, 'success');

            // Refresh the UI
            await loadCatalogInfo();

            // Reload products if needed
            if (result.total_products && result.total_products > 0) {
                showToast(`${result.total_products} products restored`, 'info');
            }
        } else {
            showToast(`Error loading recovery snapshot: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error restoring recovery snapshot:', error);
        showToast('Error restoring recovery snapshot', 'error');
    }

    closeCrashRecoveryDialog();
}

async function autoLoadCatalogCSV() {
    try {
        // Load historical CSV
        const historicalResponse = await fetch('/api/catalogs/csv-content?section=historical');
        const historicalData = await historicalResponse.json();

        if (historicalData.has_csv) {
            const blob = new Blob([historicalData.csv_content], { type: 'text/csv' });
            const file = new File([blob], historicalData.filename || 'historical.csv');
            historicalCsv = file;

            // Update UI EXACTLY like manual upload does - just show the filename
            const historicalFileLabel = document.getElementById('historicalFileLabel');
            if (historicalFileLabel) {
                historicalFileLabel.textContent = historicalData.filename;
            }

            // Also update status section if it exists
            const historicalStatus = document.getElementById('historicalStatus');
            if (historicalStatus) {
                historicalStatus.innerHTML += `<p class="success">[✓] Historical CSV loaded: ${historicalData.filename} (${historicalData.row_count} rows)</p>`;
            }
        }

        // Load new CSV
        const newResponse = await fetch('/api/catalogs/csv-content?section=new');
        const newData = await newResponse.json();

        if (newData.has_csv) {
            const blob = new Blob([newData.csv_content], { type: 'text/csv' });
            const file = new File([blob], newData.filename || 'new.csv');
            newCsv = file;

            // Update UI EXACTLY like manual upload does - just show the filename
            const newFileLabel = document.getElementById('newFileLabel');
            if (newFileLabel) {
                newFileLabel.textContent = newData.filename;
            }

            // Also update status section if it exists
            const newStatus = document.getElementById('newStatus');
            if (newStatus) {
                newStatus.innerHTML += `<p class="success">[✓] New CSV loaded: ${newData.filename} (${newData.row_count} rows)</p>`;
            }
        }

        // Update CSV warnings after auto-load
        updateCsvWarning('historical');
        updateCsvWarning('new');
    } catch (error) {
        // Silently fail if no CSV available - this is normal if Mode 1 only
    }
}


// Mobile modal/connectivity logic moved to /static/app.mobile.js
