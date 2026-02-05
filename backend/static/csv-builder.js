// CSV Builder Application State
const state = {
    currentStep: 1,
    products: [],
    importedData: [],
    linkedProducts: [],
    unmatchedImages: [],
    unmatchedData: [],
    selectedProductIndex: null,
    undoStack: [],
    redoStack: [],
    autoSaveTimer: null,
    linkingStrategy: 'filename_equals_sku',
    skuPattern: '[A-Z]+-\\d+'
};

// MEMORY OPTIMIZATION: Limits to prevent unbounded state growth (200-500MB possible)
const MAX_PRODUCTS = 50000;  // Maximum products in state
const MAX_UNDO_STACK = 10;   // Maximum undo history items
const MAX_REDO_STACK = 10;   // Maximum redo history items

// Expose state to window for loadToApp function (defined in html)
window.state = state;

// Track event listeners for cleanup
const eventListeners = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeStep1();
    loadFromLocalStorage();
    checkForMainAppFiles();
});

// Check if files were passed from the main app (via server staging or sessionStorage fallback)
async function checkForMainAppFiles() {
    try {
        // First try to get window_id from query params (webview mode)
        const params = new URLSearchParams(window.location.search);
        const windowId = params.get('window_id');
        const section = params.get('section');

        if (windowId) {
            console.log('[CSV-BUILDER] Fetching staged data for window:', windowId);

            // Fetch staged data from server
            const response = await fetch(`/api/csv-builder/get-staged/${windowId}`);

            if (response.ok) {
                const data = await response.json();
                const fileData = data.file_data;
                const source = data.section || section;

                if (fileData && fileData.length > 0) {
                    console.log('[CSV-BUILDER] Found pre-populated files from main app:', fileData.length);

                    // Create product entries from file data
                    state.products = fileData.map(f => ({
                        filename: f.filename,
                        category: f.category || '',
                        sku: '',
                        name: '',
                        price: '',
                        priceHistory: [],
                        performanceHistory: []
                    }));

                    // Store source for later
                    state.mainAppSource = source;

                    // Display file info in Step 1
                    displayPrePopulatedFiles(fileData, source);

                    // Enable next button
                    document.getElementById('nextToLink').disabled = false;

                    showToast(`${fileData.length} images loaded from main app`, 'success');
                    return;
                }
            } else {
                console.log('[CSV-BUILDER] No staged data found for window:', windowId);
            }
        }

        // Fallback: Try sessionStorage (browser mode)
        const filesStr = sessionStorage.getItem('csvBuilderFiles');
        const source = sessionStorage.getItem('csvBuilderSource');

        if (filesStr) {
            const fileData = JSON.parse(filesStr);

            if (fileData && fileData.length > 0) {
                console.log('[CSV-BUILDER] Found pre-populated files from sessionStorage:', fileData.length);

                // Create product entries from file data
                state.products = fileData.map(f => ({
                    filename: f.filename,
                    category: f.category || '',
                    sku: '',
                    name: '',
                    price: '',
                    priceHistory: [],
                    performanceHistory: []
                }));

                // Store source for later
                state.mainAppSource = source;

                // Display file info in Step 1
                displayPrePopulatedFiles(fileData, source);

                // Clear sessionStorage after reading
                sessionStorage.removeItem('csvBuilderFiles');
                sessionStorage.removeItem('csvBuilderSource');

                // Enable next button
                document.getElementById('nextToLink').disabled = false;

                showToast(`${fileData.length} images loaded from main app`, 'success');
            }
        }
    } catch (error) {
        console.error('[CSV-BUILDER] Error loading files from main app:', error);
    }
}

// Display pre-populated files in Step 1 UI
function displayPrePopulatedFiles(fileData, source) {
    const info = document.getElementById('imageInfo');
    if (!info) return;

    // Count categories
    const categoryCount = {};
    fileData.forEach(f => {
        if (f.category) {
            categoryCount[f.category] = (categoryCount[f.category] || 0) + 1;
        }
    });

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No categories detected</div>';

    const displayLimit = 50;
    const hasMore = fileData.length > displayLimit;

    const sourceLabel = source === 'historical' ? 'Historical Products' : 'New Products';

    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearCsvBuilderUpload()" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>Loaded from Main App: ${fileData.length} images</h4>
        <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px;">
            <strong>Destination:</strong> <span id="destinationLabel">${sourceLabel}</span>
        </div>
        ${categorySummary}
        <div class="file-list" id="csvBuilderFileList">
            ${fileData.slice(0, displayLimit).map(f =>
                `<div>${escapeHtml(f.filename)}${f.category ? ` <span style="color: #667eea;">[${f.category}]</span>` : ''}</div>`
            ).join('')}
        </div>
        ${hasMore ? `
            <div style="text-align: center; margin-top: 10px;">
                <button class="btn" onclick="showAllCsvBuilderFiles(${fileData.length})" style="font-size: 12px; padding: 5px 15px;">
                    SHOW ALL ${fileData.length} FILES
                </button>
            </div>
        ` : ''}
        <div style="margin-top: 15px; padding: 15px; background: #e7f5ff; border: 1px solid #74c0fc; border-radius: 4px;">
            <p style="margin: 0 0 10px 0;"><strong>Next Step:</strong> Download CSV template below, fill in product details, then load it in Step 2 to link metadata with images.</p>
            <button class="btn" style="background-color: #667eea; color: white;" onclick="downloadCsvTemplateFromBuilder(${fileData.length})" data-tooltip="Download CSV template with pre-filled filenames and categories">
                DOWNLOAD CSV TEMPLATE
            </button>
        </div>
    `;
    info.classList.add('show');
}

// Cleanup on window unload
window.addEventListener('beforeunload', () => {
    cleanupResources();
});

// Centralized cleanup function
function cleanupResources() {
    // Clear auto-save timer
    if (state.autoSaveTimer) {
        clearTimeout(state.autoSaveTimer);
        state.autoSaveTimer = null;
    }

    // Remove event listeners
    eventListeners.forEach(({ element, event, handler }) => {
        if (element && element.removeEventListener) {
            element.removeEventListener(event, handler);
        }
    });
    eventListeners.length = 0;

    // Clean up global window properties
    if (window.savedWorkData) {
        window.savedWorkData = null;
        delete window.savedWorkData;
    }

    // Remove all lingering modals to prevent DOM accumulation
    ['loadSavedWorkModal', 'destinationModal', 'confirmSendModal'].forEach(modalId => {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.remove();
        }
    });

    // Clear unmatched arrays to free memory
    state.unmatchedImages = [];
    state.unmatchedData = [];

    // Clear fuzzy index
    state.fuzzyIndex = {};
}

// Native folder selection helper for pywebview (same as app.js)
async function selectFolderNative(handleFilesCallback) {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.select_folder) {
        try {
            const filesInfo = await window.pywebview.api.select_folder();
            if (filesInfo && filesInfo.length > 0) {
                // MEMORY OPTIMIZATION: Don't load images into memory
                // Create file-like objects with paths for backend processing
                const files = filesInfo.map((info) => {
                    const file = {
                        name: info.name,
                        type: 'image/' + info.name.split('.').pop().toLowerCase(),
                        path: info.path,  // Absolute file path
                        size: info.size
                    };
                    // Add webkitRelativePath for category detection
                    file.webkitRelativePath = info.relativePath;
                    return file;
                });
                handleFilesCallback(files);
            }
        } catch (e) {
            console.error('Native folder selection error:', e);
            showToast('Error selecting folder: ' + e.message, 'error');
        }
        return true; // Handled natively
    }
    return false; // Fall back to HTML input
}

// ===== STEP 1: Upload Images =====
function initializeStep1() {
    // MEMORY OPTIMIZATION: Clear old event listeners to prevent accumulation on re-initialization
    eventListeners.forEach(({ element, event, handler }) => {
        if (element && element.removeEventListener) {
            element.removeEventListener(event, handler);
        }
    });
    eventListeners.length = 0;

    // Helper to track event listeners
    const addTrackedListener = (element, event, handler) => {
        if (element) {
            element.addEventListener(event, handler);
            eventListeners.push({ element, event, handler });
        }
    };

    // ===== Image Upload =====
    const imageDropZone = document.getElementById('imageDropZone');
    const imageInput = document.getElementById('imageInput');
    const imageBrowseBtn = document.getElementById('imageBrowseBtn');
    const nextBtn = document.getElementById('nextToLink');

    const imageBrowseBtnHandler = async (e) => {
        e.stopPropagation();
        const handled = await selectFolderNative(handleImageFiles);
        if (!handled) {
            imageInput.click();
        }
    };
    addTrackedListener(imageBrowseBtn, 'click', imageBrowseBtnHandler);

    const imageDropZoneClickHandler = async () => {
        const handled = await selectFolderNative(handleImageFiles);
        if (!handled) {
            imageInput.click();
        }
    };
    addTrackedListener(imageDropZone, 'click', imageDropZoneClickHandler);

    const imageDragoverHandler = (e) => {
        e.preventDefault();
        imageDropZone.classList.add('drag-over');
    };
    addTrackedListener(imageDropZone, 'dragover', imageDragoverHandler);

    const imageDragleaveHandler = () => {
        imageDropZone.classList.remove('drag-over');
    };
    addTrackedListener(imageDropZone, 'dragleave', imageDragleaveHandler);

    const imageDropHandler = (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('drag-over');
        handleImageFiles(Array.from(e.dataTransfer.files));
    };
    addTrackedListener(imageDropZone, 'drop', imageDropHandler);

    const imageInputChangeHandler = (e) => {
        handleImageFiles(Array.from(e.target.files));
    };
    addTrackedListener(imageInput, 'change', imageInputChangeHandler);

    // ===== Direct CSV Upload (Step 1) =====
    const csvDropZone = document.getElementById('csvDropZone');
    const csvInput = document.getElementById('csvInput');
    const csvBrowseBtn = document.getElementById('csvBrowseBtn');

    if (csvBrowseBtn) {
        const csvBrowseBtnHandler = (e) => {
            e.stopPropagation();
            csvInput.click();
        };
        addTrackedListener(csvBrowseBtn, 'click', csvBrowseBtnHandler);
    }

    if (csvDropZone) {
        const csvDropZoneClickHandler = () => {
            csvInput.click();
        };
        addTrackedListener(csvDropZone, 'click', csvDropZoneClickHandler);

        const csvDragoverHandler = (e) => {
            e.preventDefault();
            csvDropZone.classList.add('drag-over');
        };
        addTrackedListener(csvDropZone, 'dragover', csvDragoverHandler);

        const csvDragleaveHandler = () => {
            csvDropZone.classList.remove('drag-over');
        };
        addTrackedListener(csvDropZone, 'dragleave', csvDragleaveHandler);

        const csvDropHandler = (e) => {
            e.preventDefault();
            csvDropZone.classList.remove('drag-over');
            const files = Array.from(e.dataTransfer.files);
            const csvFile = files.find(f => f.name.endsWith('.csv'));
            if (csvFile) {
                handleDirectCsvUpload(csvFile);
            } else {
                showToast('Please drop a CSV file', 'error');
            }
        };
        addTrackedListener(csvDropZone, 'drop', csvDropHandler);
    }

    if (csvInput) {
        const csvInputChangeHandler = (e) => {
            if (e.target.files.length > 0) {
                handleDirectCsvUpload(e.target.files[0]);
            }
        };
        addTrackedListener(csvInput, 'change', csvInputChangeHandler);
    }

    // ===== Next Button =====
    const nextBtnHandler = () => {
        goToStep(2);
    };
    addTrackedListener(nextBtn, 'click', nextBtnHandler);

    // ===== Import Inputs for Step 2 =====
    const importInput = document.getElementById('importFileInput');
    if (importInput) {
        addTrackedListener(importInput, 'change', handleImportFile);
    }
}

/**
 * Handle direct CSV upload from Step 1
 */
function handleDirectCsvUpload(file) {
    // Show immediate feedback that upload started
    console.log('[CSV-BUILDER] Processing CSV file:', file.name);
    showToast(`Processing ${file.name}...`, 'info');

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const content = e.target.result;
            parseImportedData(content, 'direct upload');

            // Copy imported data to products for export
            if (state.importedData.length > 0) {
                state.products = state.importedData.map(item => ({
                    filename: item.filename || '',
                    category: item.category || '',
                    sku: item.sku || '',
                    name: item.name || '',
                    price: item.price || '',
                    ...item // Include all other columns
                }));

                // Update UI
                const csvInfo = document.getElementById('csvInfo');
                if (csvInfo) {
                    csvInfo.innerHTML = `<p style="color: #4CAF50; font-weight: bold;">[✓] ${state.products.length} rows loaded from ${file.name}</p>`;
                }

                // Enable next button
                const nextBtn = document.getElementById('nextToLink');
                if (nextBtn) nextBtn.disabled = false;

                // Show prominent success message
                console.log('[CSV-BUILDER] Successfully loaded', state.products.length, 'rows');
                showToast(`Successfully loaded ${state.products.length} rows from CSV!`, 'success', 4000);
            } else {
                showToast('No valid data found in CSV', 'error');
            }
        } finally {
            // Clean up reader reference
            reader.onload = null;
            reader.onerror = null;
        }
    };
    reader.onerror = () => {
        showToast('Error reading CSV file', 'error');
        reader.onload = null;
        reader.onerror = null;
    };
    reader.readAsText(file);
}


function handleImageFiles(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));

    if (imageFiles.length === 0) {
        showToast('No image files found in folder', 'error');
        return;
    }

    // Extract categories and create product entries
    state.products = imageFiles.map(file => {
        const category = extractCategoryFromPath(file.webkitRelativePath || file.name);
        return {
            filename: file.name,
            category: category || '',
            sku: '',
            name: '',
            price: '',
            priceHistory: [],
            performanceHistory: []
        };
    });

    // Display file info
    const categoryCount = {};
    state.products.forEach(p => {
        if (p.category) {
            categoryCount[p.category] = (categoryCount[p.category] || 0) + 1;
        }
    });

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';

    const info = document.getElementById('imageInfo');
    const displayLimit = 50;
    const hasMore = imageFiles.length > displayLimit;
    
    // Show destination selector if not from main app
    const destinationSection = !state.mainAppSource ? `
        <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px;">
            <strong>Destination:</strong> <span id="destinationLabel">${state.mainAppSource || 'Not set'}</span>
            <button class="btn-small" onclick="showDestinationSelector()" style="margin-left: 10px;">SELECT</button>
        </div>
    ` : '';
    
    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearCsvBuilderUpload()" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>${imageFiles.length} images loaded</h4>
        ${destinationSection}
        ${categorySummary}
        <div class="file-list" id="csvBuilderFileList">
            ${state.products.slice(0, displayLimit).map(p =>
                `<div>${escapeHtml(p.filename)}${p.category ? ` <span style="color: #667eea;">[${p.category}]</span>` : ''}</div>`
            ).join('')}
        </div>
        ${hasMore ? `
            <div style="text-align: center; margin-top: 10px;">
                <button class="btn" onclick="showAllCsvBuilderFiles(${imageFiles.length})" style="font-size: 12px; padding: 5px 15px;">
                    SHOW ALL ${imageFiles.length} FILES
                </button>
            </div>
        ` : ''}
        <div style="margin-top: 15px; padding: 15px; background: #e7f5ff; border: 1px solid #74c0fc; border-radius: 4px;">
            <p style="margin: 0 0 10px 0;"><strong>Next Step:</strong> Download CSV template below, fill in product details, then load it in Step 2 to link metadata with images.</p>
            <button class="btn" style="background-color: #667eea; color: white;" onclick="downloadCsvTemplateFromBuilder(${imageFiles.length})" data-tooltip="Download CSV template with pre-filled filenames and categories">
                DOWNLOAD CSV TEMPLATE
            </button>
        </div>
    `;
    info.classList.add('show');

    document.getElementById('nextToLink').disabled = false;
    showToast(`${imageFiles.length} images loaded from ${Object.keys(categoryCount).length || 0} categories`, 'success');

    saveState();
}

function extractCategoryFromPath(path) {
    if (!path) return null;
    
    const parts = path.split('/');
    if (parts.length === 1) return null;
    
    const category = parts[parts.length - 2];
    const ignoredFolders = ['historical_products', 'new_products', 'products', 'images', 'uploads'];
    
    if (ignoredFolders.includes(category.toLowerCase())) {
        if (parts.length > 2) {
            return parts[parts.length - 3];
        }
        return null;
    }
    
    return category;
}


// ===== STEP 2: Link Data =====

function importFromFile() {
    document.getElementById('importFileInput').click();
}

function handleImportFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        parseImportedData(content, file.name);
    };
    reader.readAsText(file);
}

function importFromClipboard() {
    navigator.clipboard.readText().then(text => {
        parseImportedData(text, 'clipboard');
    }).catch(() => {
        showToast('Failed to read clipboard. Please grant permission.', 'error');
    });
}

function parseImportedData(content, source) {
    // Handle null/undefined content
    if (!content || typeof content !== 'string') {
        showToast('No data found or invalid format', 'error');
        return;
    }
    
    // Clean content - handle different line endings and BOM
    content = content.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    
    const lines = content.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
        showToast('No data found in file', 'error');
        return;
    }
    
    // Parse CSV with error tracking
    const parseErrors = [];
    const rows = lines.map((line, index) => {
        try {
            return parseCSVLine(line);
        } catch (e) {
            parseErrors.push(`Line ${index + 1}: ${e.message}`);
            return [];
        }
    }).filter(row => row.length > 0);
    
    if (rows.length === 0) {
        showToast('Could not parse any valid rows', 'error');
        return;
    }
    
    // Detect headers - check first row for common header names
    const headers = rows[0].map(h => (h || '').toLowerCase().trim());
    const commonHeaders = ['filename', 'sku', 'name', 'price', 'category', 'product_name', 'product_id'];
    const hasHeaders = headers.some(h => commonHeaders.some(ch => h.includes(ch)));

    // Track which standard fields were found in headers
    const foundFields = {
        filename: false,
        sku: false,
        name: false,
        price: false,
        category: false,
        price_history: false,
        performance_history: false
    };

    if (hasHeaders) {
        headers.forEach(header => {
            const normalized = normalizeHeaderName(header);
            if (normalized in foundFields) {
                foundFields[normalized] = true;
            }
        });
    }

    const dataRows = hasHeaders ? rows.slice(1) : rows;
    
    // Track data quality issues
    const dataQuality = {
        emptyRows: 0,
        missingFields: 0,
        invalidPrices: 0,
        duplicateSKUs: new Set()
    };
    
    // Map to objects with validation
    const skuSet = new Set();
    state.importedData = dataRows.map((row, index) => {
        // Skip completely empty rows
        if (row.every(cell => !cell || !cell.trim())) {
            dataQuality.emptyRows++;
            return null;
        }
        
        let obj;
        if (hasHeaders) {
            obj = {};
            headers.forEach((header, i) => {
                const value = sanitizeField(row[i]);
                // Normalize header names
                const normalizedHeader = normalizeHeaderName(header);
                obj[normalizedHeader] = value;
            });

            // Fallback for missing critical fields: attempt positional mapping
            if (!obj.filename && !obj.sku && row.length >= 3) {
                // If both filename and SKU are missing, try to infer from available data
                if (!obj.filename && row[0]) obj.filename = sanitizeField(row[0]);
                if (!obj.sku && row[2]) obj.sku = sanitizeField(row[2]);
            }
        } else {
            // No headers detected - use positional mapping
            // Assume order: filename, category, sku, name, price
            // Handle rows with fewer than 5 columns gracefully
            obj = {
                filename: sanitizeField(row[0] || ''),
                category: sanitizeField(row[1] || ''),
                sku: sanitizeField(row[2] || ''),
                name: sanitizeField(row[3] || ''),
                price: sanitizeField(row[4] || '')
            };

            // Track incomplete rows when using positional mapping
            if (row.length < 3) {
                dataQuality.missingFields++;
            }
        }
        
        // Validate and clean price
        if (obj.price) {
            const cleanPrice = parsePrice(obj.price);
            if (cleanPrice === null) {
                dataQuality.invalidPrices++;
                obj.price = '';
            } else {
                obj.price = cleanPrice;
            }
        }
        
        // Track duplicate SKUs
        if (obj.sku) {
            if (skuSet.has(obj.sku.toLowerCase())) {
                dataQuality.duplicateSKUs.add(obj.sku);
            }
            skuSet.add(obj.sku.toLowerCase());
        }
        
        // Track missing required fields
        if (!obj.sku && !obj.name && !obj.filename) {
            dataQuality.missingFields++;
        }
        
        return obj;
    }).filter(obj => obj !== null);

    // Build fuzzy index for faster fuzzy matching (after CSV import)
    if (state.importedData.length > 0) {
        buildFuzzyIndex();
    }

    // Build status message
    let statusMessage = `Imported ${state.importedData.length} products from ${source}`;
    const warnings = [];
    const infoMessages = [];

    // Header-specific warnings
    if (hasHeaders) {
        // Only warn if MOST basic fields are missing (supports dynamic metadata CSVs)
        const basicFields = ['filename', 'sku', 'name', 'price', 'category'];
        const missingBasicFields = basicFields.filter(f => !foundFields[f]);

        // Only warn if more than 3 out of 5 basic fields are missing
        if (missingBasicFields.length > 3) {
            warnings.push(`Missing most basic headers: ${missingBasicFields.join(', ')} (will use positional mapping as fallback)`);
        } else if (missingBasicFields.length > 0) {
            infoMessages.push(`ℹ️ Optional headers not found: ${missingBasicFields.join(', ')}`);
        }

        // Show info about history fields if found
        if (foundFields.price_history) {
            infoMessages.push(`price_history column detected`);
        }
        if (foundFields.performance_history) {
            infoMessages.push(`performance_history column detected`);
        }

        // Check for dynamic metadata columns (columns beyond the basic ones)
        const knownFields = Object.keys(foundFields);
        const dynamicColumns = headers.filter(h => {
            const normalized = normalizeHeaderName(h);
            return !knownFields.includes(normalized) && h.trim().length > 0;
        });
        if (dynamicColumns.length > 0) {
            infoMessages.push(`Dynamic metadata columns found: ${dynamicColumns.slice(0, 3).join(', ')}${dynamicColumns.length > 3 ? ` (+${dynamicColumns.length - 3} more)` : ''}`);
        }
    } else {
        warnings.push(`ℹ️ No headers detected - using positional mapping (Col 1=filename, Col 2=category, Col 3=sku, Col 4=name, Col 5=price). Ensure your CSV matches this order!`);
    }

    if (dataQuality.emptyRows > 0) {
        warnings.push(`${dataQuality.emptyRows} empty row(s) skipped`);
    }
    if (dataQuality.invalidPrices > 0) {
        warnings.push(`${dataQuality.invalidPrices} invalid price(s) cleared`);
    }
    if (dataQuality.duplicateSKUs.size > 0) {
        warnings.push(`${dataQuality.duplicateSKUs.size} duplicate SKU(s) found`);
    }
    if (parseErrors.length > 0) {
        warnings.push(`${parseErrors.length} parse error(s)`);
    }

    // Display info messages first (green)
    if (infoMessages.length > 0) {
        statusMessage += `<br><span style="color: #48bb78;">${infoMessages.join('<br>')}</span>`;
    }

    // Display warnings (orange)
    if (warnings.length > 0) {
        statusMessage += `<br><span style="color: #ed8936;">${warnings.join('<br>')}</span>`;
    }

    // Show import status
    document.getElementById('importStatus').style.display = 'block';
    document.getElementById('importStatusText').innerHTML = statusMessage;

    // Show linking panel
    document.getElementById('linkingPanel').style.display = 'block';
    document.getElementById('skipLinkingActions').style.display = 'none';

    // Auto-preview with default strategy
    previewLinking();

    // Save detected schema to backend for dynamic weight sliders
    if (hasHeaders && headers.length > 0) {
        saveDetectedSchemaToBackend(headers, dataRows);
    }

    // Update column preview and data preview
    updateColumnPreview();
    updateDataPreviewTable();

    showToast(`Imported ${state.importedData.length} products`, 'success');
}

/**
 * Detect column types and save schema to backend
 * @param {Array} headers - Array of header names
 * @param {Array} dataRows - Array of data row arrays
 */
async function saveDetectedSchemaToBackend(headers, dataRows) {
    try {
        // Detect column types based on data
        const columns = headers.map(header => {
            const colIndex = headers.indexOf(header);
            const values = dataRows.map(row => row[colIndex]).filter(v => v && v.trim());
            const dataType = detectColumnType(values);

            return {
                column_name: normalizeHeaderName(header) || header.toLowerCase().replace(/\s+/g, '_'),
                data_type: dataType,
                display_name: header.toUpperCase()
            };
        }).filter(col => col.column_name); // Remove empty columns

        console.log('[CSV-BUILDER] Detected schema:', columns);

        // Save to backend
        const response = await fetch('/api/metadata-schema', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ columns: columns, clear_existing: true })
        });

        if (response.ok) {
            console.log('[CSV-BUILDER] Schema saved to backend');
        } else {
            console.warn('[CSV-BUILDER] Failed to save schema:', response.statusText);
        }
    } catch (error) {
        console.error('[CSV-BUILDER] Error saving schema:', error);
    }
}

/**
 * Detect if a column is numeric or string based on its values
 * @param {Array} values - Sample values from the column
 * @returns {string} 'numeric' or 'string'
 */
function detectColumnType(values) {
    if (!values || values.length === 0) return 'string';

    // Sample up to 50 values
    const sampleSize = Math.min(50, values.length);
    const sample = values.slice(0, sampleSize);

    // Count how many values parse as numbers
    let numericCount = 0;
    sample.forEach(val => {
        const cleaned = val.toString().replace(/[$,]/g, '').trim();
        if (cleaned && !isNaN(parseFloat(cleaned))) {
            numericCount++;
        }
    });

    // If >80% of values are numeric, consider it a numeric column
    return (numericCount / sample.length) >= 0.8 ? 'numeric' : 'string';
}

function normalizeHeaderName(header) {
    // Map common variations to standard names
    const headerMap = {
        'product_name': 'name',
        'productname': 'name',
        'product name': 'name',
        'item_name': 'name',
        'itemname': 'name',
        'product_sku': 'sku',
        'productsku': 'sku',
        'item_sku': 'sku',
        'itemsku': 'sku',
        'product_id': 'sku',
        'productid': 'sku',
        'item_id': 'sku',
        'itemid': 'sku',
        'file_name': 'filename',
        'file': 'filename',
        'image': 'filename',
        'image_name': 'filename',
        'imagename': 'filename',
        'cat': 'category',
        'product_category': 'category',
        'productcategory': 'category',
        'unit_price': 'price',
        'unitprice': 'price',
        'cost': 'price',
        'amount': 'price',
        'pricehistory': 'price_history',
        'price history': 'price_history',
        'performancehistory': 'performance_history',
        'performance history': 'performance_history'
    };

    const normalized = header.toLowerCase().replace(/[^a-z0-9]/g, '');
    return headerMap[normalized] || header.toLowerCase().replace(/[^a-z0-9_]/g, '');
}

function parsePrice(priceStr) {
    if (!priceStr || typeof priceStr !== 'string') return null;
    
    // Remove currency symbols and whitespace
    let cleaned = priceStr.replace(/[$€£¥₹,\s]/g, '').trim();
    
    // Handle negative prices (invalid)
    if (cleaned.startsWith('-')) return null;
    
    // Parse as float
    const price = parseFloat(cleaned);
    
    // Validate
    if (isNaN(price) || !isFinite(price) || price < 0) {
        return null;
    }
    
    // Round to 2 decimal places
    return Math.round(price * 100) / 100;
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    // Handle null/undefined/empty lines
    if (!line || typeof line !== 'string') {
        return result;
    }
    
    // Trim BOM and whitespace
    line = line.replace(/^\uFEFF/, '').trim();
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if ((char === ',' || char === '\t' || char === ';') && !inQuotes) {
            result.push(sanitizeField(current));
            current = '';
        } else {
            current += char;
        }
    }
    
    result.push(sanitizeField(current));
    return result;
}

function sanitizeField(field) {
    if (field === null || field === undefined) return '';
    
    // Trim whitespace and quotes
    let sanitized = String(field).trim();
    
    // Remove surrounding quotes
    if ((sanitized.startsWith('"') && sanitized.endsWith('"')) ||
        (sanitized.startsWith("'") && sanitized.endsWith("'"))) {
        sanitized = sanitized.slice(1, -1);
    }
    
    // Handle common null representations
    const nullValues = ['null', 'NULL', 'undefined', 'UNDEFINED', 'N/A', 'n/a', 'NA', 'na', '-', ''];
    if (nullValues.includes(sanitized)) {
        return '';
    }
    
    return sanitized;
}

async function exportTemplate() {
    // Generate CSV template with filenames and empty metadata columns
    let csv = 'filename,category,sku,name,price,price_history,performance_history\n';

    state.products.forEach(product => {
        csv += `${product.filename},${product.category || ''},,,,\n`;
    });

    const filename = `product-template_${new Date().toISOString().slice(0, 10)}.csv`;

    // Check if running in pywebview
    if (window.pywebview) {
        try {
            const result = await window.pywebview.api.save_file_auto(csv, filename);
            if (result) {
                showToast(`Template saved to Downloads folder: ${filename}. Fill it in Excel and re-import!`, 'success');
            } else {
                showToast('Export failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Export failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('Template exported! Fill it in Excel and re-import.', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showToast('Export failed', 'error');
        } finally {
            // Always revoke the URL to prevent memory leak
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    }
}

function importCompletedTemplate() {
    document.getElementById('importCompletedInput').click();
}

function toggleImportHelp() {
    const helpDiv = document.getElementById('importHelp');
    helpDiv.style.display = helpDiv.style.display === 'none' ? 'block' : 'none';
}

function handleImportCompletedFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        processCompletedTemplate(content);
    };
    reader.readAsText(file);
}

function processCompletedTemplate(content) {
    // Handle null/undefined content
    if (!content || typeof content !== 'string') {
        showToast('No data found or invalid format', 'error');
        return;
    }
    
    // Clean content - handle different line endings and BOM
    content = content.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    
    const lines = content.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
        showToast('No data found in file', 'error');
        return;
    }
    
    // Parse CSV with error handling
    const rows = lines.map((line, index) => {
        try {
            return parseCSVLine(line);
        } catch (e) {
            console.warn(`Parse error on line ${index + 1}:`, e);
            return [];
        }
    }).filter(row => row.length > 0);
    
    if (rows.length < 2) {
        showToast('CSV must have headers and at least one data row', 'error');
        return;
    }
    
    // First row should be headers - normalize them
    const headers = rows[0].map(h => normalizeHeaderName((h || '').toLowerCase().trim()));
    const dataRows = rows.slice(1);
    
    // Validate required column - check for filename or file
    const filenameIndex = headers.findIndex(h => h === 'filename' || h === 'file');
    if (filenameIndex === -1) {
        showToast('CSV must have a "filename" column', 'error');
        return;
    }
    
    // Map to objects with validation
    const importedProducts = dataRows.map((row, index) => {
        // Skip empty rows
        if (row.every(cell => !cell || !cell.trim())) {
            return null;
        }
        
        const obj = {};
        headers.forEach((header, i) => {
            obj[header] = sanitizeField(row[i]);
        });
        return obj;
    }).filter(obj => obj !== null && obj.filename);
    
    // Match by filename and update products
    let matched = 0;
    let notFound = 0;
    const notFoundList = [];
    let hasPriceHistory = false;
    let hasPerformanceHistory = false;

    importedProducts.forEach(imported => {
        const product = state.products.find(p =>
            p.filename.toLowerCase() === imported.filename.toLowerCase()
        );

        if (product) {
            // Update product with imported data
            if (imported.category) product.category = imported.category;
            if (imported.sku) product.sku = imported.sku;
            if (imported.name) product.name = imported.name;
            if (imported.price) product.price = imported.price;

            // Parse history if present
            if (imported.price_history) {
                product.priceHistory = parsePriceHistory(imported.price_history);
                hasPriceHistory = true;
            }
            if (imported.performance_history) {
                product.performanceHistory = parsePerformanceHistory(imported.performance_history);
                hasPerformanceHistory = true;
            }

            matched++;
        } else {
            notFound++;
            notFoundList.push(imported.filename);
        }
    });
    
    // Validate data
    const validationWarnings = validateImportedData(state.products);

    // Show results
    let message = `Imported metadata for ${matched} product(s)`;

    // Show history field detection
    const historyInfo = [];
    if (hasPriceHistory) {
        historyInfo.push('price_history data imported');
    }
    if (hasPerformanceHistory) {
        historyInfo.push('performance_history data imported');
    }
    if (historyInfo.length > 0) {
        message += `\n<span style="color: #48bb78;">${historyInfo.join('<br>')}</span>`;
    }

    if (notFound > 0) {
        message += `\n${notFound} filename(s) not found in uploaded images`;
    }
    if (validationWarnings.length > 0) {
        message += `\n${validationWarnings.length} validation warning(s)`;
    }

    document.getElementById('importStatus').style.display = 'block';
    document.getElementById('importStatusText').innerHTML = message.replace(/\n/g, '<br>');
    
    // Show validation warnings if any
    if (validationWarnings.length > 0) {
        const warningDiv = document.createElement('div');
        warningDiv.style.marginTop = '10px';
        warningDiv.style.fontSize = '12px';
        warningDiv.style.color = '#666';
        warningDiv.innerHTML = '<strong>Validation Warnings:</strong><br>' + 
            validationWarnings.slice(0, 5).join('<br>') +
            (validationWarnings.length > 5 ? `<br>... and ${validationWarnings.length - 5} more` : '');
        document.getElementById('importStatus').appendChild(warningDiv);
    }
    
    // Show not found list if any
    if (notFoundList.length > 0 && notFoundList.length <= 10) {
        const notFoundDiv = document.createElement('div');
        notFoundDiv.style.marginTop = '10px';
        notFoundDiv.style.fontSize = '12px';
        notFoundDiv.style.color = '#666';
        notFoundDiv.innerHTML = '<strong>Not Found:</strong><br>' + 
            notFoundList.map(f => escapeHtml(f)).join('<br>');
        document.getElementById('importStatus').appendChild(notFoundDiv);
    }
    
    saveState();
    showToast(`Imported metadata for ${matched} products`, 'success');
    
    // Offer to skip to metadata step
    if (matched > 0) {
        const skipBtn = document.createElement('button');
        skipBtn.className = 'btn btn-primary';
        skipBtn.textContent = 'SKIP TO REVIEW';
        skipBtn.style.marginTop = '15px';
        skipBtn.onclick = () => goToStep(3);
        document.getElementById('importStatus').appendChild(skipBtn);
    }
}

function validateImportedData(products) {
    const warnings = [];
    
    products.forEach((product, index) => {
        // Check for negative prices
        if (product.price && parseFloat(product.price) < 0) {
            warnings.push(`Row ${index + 1}: Negative price (${product.price})`);
        }
        
        // Check for invalid dates in price history
        if (product.priceHistory) {
            product.priceHistory.forEach((entry, i) => {
                if (entry.date && isNaN(Date.parse(entry.date))) {
                    warnings.push(`Row ${index + 1}: Invalid date in price history (${entry.date})`);
                }
            });
        }
        
        // Check for invalid dates in performance history
        if (product.performanceHistory) {
            product.performanceHistory.forEach((entry, i) => {
                if (entry.date && isNaN(Date.parse(entry.date))) {
                    warnings.push(`Row ${index + 1}: Invalid date in performance history (${entry.date})`);
                }
            });
        }
        
        // Check for duplicate SKUs
        if (product.sku) {
            const duplicates = products.filter(p => p.sku === product.sku);
            if (duplicates.length > 1 && duplicates[0] === product) {
                warnings.push(`Duplicate SKU: ${product.sku} (${duplicates.length} products)`);
            }
        }
    });
    
    return warnings;
}

function previewLinking() {
    const strategy = document.querySelector('input[name="linkStrategy"]:checked').value;
    state.linkingStrategy = strategy;
    
    // Show/hide pattern config
    const patternConfig = document.getElementById('patternConfig');
    if (strategy === 'filename_contains_sku') {
        patternConfig.style.display = 'block';
        state.skuPattern = document.getElementById('skuPattern').value;
    } else {
        patternConfig.style.display = 'none';
    }
    
    // Perform linking
    const matches = performLinking(strategy);
    
    // Update stats
    document.getElementById('linkedCount').textContent = `Linked: ${matches.linked}`;
    document.getElementById('unlinkedCount').textContent = `Unlinked: ${matches.unlinked}`;
    
    // Show preview
    const previewList = document.getElementById('previewList');
    const previewItems = matches.results.slice(0, 10).map(result => {
        if (result.matched) {
            return `<div class="preview-item success">
                <span class="preview-image">${escapeHtml(result.image)}</span>
                <span class="preview-arrow">-></span>
                <span class="preview-data">${escapeHtml(result.data.sku || result.data.name || 'Matched')}</span>
            </div>`;
        } else {
            return `<div class="preview-item warning">
                <span class="preview-image">${escapeHtml(result.image)}</span>
                <span class="preview-arrow">X</span>
                <span class="preview-data">No match</span>
            </div>`;
        }
    }).join('');
    
    const moreCount = matches.results.length - 10;
    previewList.innerHTML = previewItems + 
        (moreCount > 0 ? `<div class="preview-more">... and ${moreCount} more</div>` : '');
}

function performLinking(strategy) {
    const results = [];
    let linked = 0;
    let unlinked = 0;
    
    state.products.forEach(product => {
        let matchedData = null;
        
        switch (strategy) {
            case 'filename_equals_sku':
                matchedData = linkByFilenameEqualsSKU(product);
                break;
            case 'filename_contains_sku':
                matchedData = linkByFilenameContainsSKU(product);
                break;
            case 'folder_equals_sku':
                matchedData = linkByFolderEqualsSKU(product);
                break;
            case 'fuzzy_name':
                matchedData = linkByFuzzyName(product);
                break;
            case 'sku_equals_filename':
                matchedData = linkBySKUEqualsFilename(product);
                break;
            case 'metadata_filename':
                matchedData = linkByMetadataFilename(product);
                break;
            case 'name_equals_filename':
                matchedData = linkByNameEqualsFilename(product);
                break;
            case 'search_all_fields':
                matchedData = linkBySearchAllFields(product);
                break;
        }
        
        if (matchedData) {
            linked++;
            results.push({ image: product.filename, data: matchedData, matched: true });
        } else {
            unlinked++;
            results.push({ image: product.filename, data: {}, matched: false });
        }
    });
    
    return { linked, unlinked, results };
}

function linkByFilenameEqualsSKU(product) {
    if (!product || !product.filename) return null;

    // Get both versions of filename
    const filenameWithExt = product.filename.trim().toLowerCase();
    const filenameNoExt = product.filename.replace(/\.[^.]+$/, '').trim().toLowerCase();

    if (!filenameNoExt) return null;

    return state.importedData.find(data => {
        if (!data || !data.sku) return false;
        const dataSKU = String(data.sku).trim().toLowerCase();

        // Try both with and without extension automatically
        return dataSKU === filenameNoExt || dataSKU === filenameWithExt;
    });
}

function linkByFilenameContainsSKU(product) {
    if (!product || !product.filename) return null;

    try {
        const pattern = new RegExp(state.skuPattern, 'i');
        const match = product.filename.match(pattern);
        if (match) {
            const extractedSKU = match[0].trim().toLowerCase();
            return state.importedData.find(data => {
                if (!data || !data.sku) return false;

                // Try matching SKU with and without extension
                const dataSKU = String(data.sku).trim().toLowerCase();
                const dataSKUNoExt = dataSKU.replace(/\.[^.]+$/, '').toLowerCase();

                return dataSKU === extractedSKU ||
                       dataSKUNoExt === extractedSKU ||
                       dataSKU === extractedSKU.replace(/\.[^.]+$/, '');
            });
        }
    } catch (e) {
        console.error('Invalid regex pattern:', e);
        showToast('Invalid SKU pattern - check regex syntax', 'warning');
    }
    return null;
}

function linkByFolderEqualsSKU(product) {
    if (!product || !product.category) return null;

    const folderName = String(product.category).trim().toLowerCase();
    const folderNameNoExt = folderName.replace(/\.[^.]+$/, '').toLowerCase();

    if (!folderName) return null;

    return state.importedData.find(data => {
        if (!data || !data.sku) return false;
        const dataSKU = String(data.sku).trim().toLowerCase();
        const dataSKUNoExt = dataSKU.replace(/\.[^.]+$/, '').toLowerCase();

        // Try all combinations automatically
        return dataSKU === folderName ||
               dataSKU === folderNameNoExt ||
               dataSKUNoExt === folderName ||
               dataSKUNoExt === folderNameNoExt;
    });
}

function linkByFuzzyName(product) {
    if (!product || !product.filename) return null;

    // Clean and normalize filename
    const cleanFilename = normalizeForFuzzyMatch(
        product.filename.replace(/\.[^.]+$/, '') // Remove extension
    );

    if (!cleanFilename || cleanFilename.length < 2) return null;

    // OPTIMIZATION: Use fuzzy index to get candidate products instead of scanning all
    const candidates = getFuzzyIndexCandidates(cleanFilename);

    // Find best match with scoring
    let bestMatch = null;
    let bestScore = 0;

    for (const data of candidates) {
        if (!data || !data.name) continue;

        const cleanName = normalizeForFuzzyMatch(data.name);
        if (!cleanName) continue;

        // Calculate similarity score
        const score = calculateFuzzyScore(cleanFilename, cleanName);

        if (score > bestScore && score >= 0.5) { // Minimum 50% match
            bestScore = score;
            bestMatch = data;
        }
    }

    return bestMatch;
}

function normalizeForFuzzyMatch(str) {
    if (!str || typeof str !== 'string') return '';

    return str
        .toLowerCase()
        .replace(/[_\-\.]/g, ' ')  // Replace separators with spaces
        .replace(/[^a-z0-9\s]/g, '') // Remove special chars
        .replace(/\s+/g, ' ')  // Normalize whitespace
        .trim();
}

// NEW EDGE CASE STRATEGIES

/**
 * Reverse matching: Metadata SKU matches image filename
 * Use case: User named images differently than SKU (e.g., IMG001.jpg but SKU is ABC-123)
 * CSV has SKU column, we match SKU value to image filename
 * Automatically handles with/without extensions
 */
function linkBySKUEqualsFilename(product) {
    if (!product || !product.filename) return null;

    // Get both versions of the filename
    const filenameWithExt = product.filename.trim().toLowerCase();
    const filenameNoExt = product.filename.replace(/\.[^.]+$/, '').trim().toLowerCase();

    if (!filenameNoExt) return null;

    // Find metadata where SKU matches the image filename (try both versions)
    return state.importedData.find(data => {
        if (!data || !data.sku) return false;
        const dataSKU = String(data.sku).trim().toLowerCase();
        const dataSKUNoExt = dataSKU.replace(/\.[^.]+$/, '').toLowerCase();

        // Try all combinations - be smart about extensions
        return dataSKU === filenameNoExt ||
               dataSKU === filenameWithExt ||
               dataSKUNoExt === filenameNoExt ||
               dataSKUNoExt === filenameWithExt;
    });
}

/**
 * Direct filename column matching
 * Use case: CSV has a "filename" or "image" column with exact filenames
 * Example: CSV row has filename="ABC-123.jpg", image is "ABC-123.jpg"
 */
function linkByMetadataFilename(product) {
    if (!product || !product.filename) return null;

    const imageFilename = product.filename.toLowerCase();

    // Try common column names for filename
    const filenameFields = ['filename', 'image', 'image_name', 'file', 'photo', 'picture'];

    return state.importedData.find(data => {
        if (!data) return false;

        // Check all possible filename fields
        for (const field of filenameFields) {
            if (data[field]) {
                let metadataFilename = String(data[field]).trim().toLowerCase();

                // Try exact match
                if (metadataFilename === imageFilename) {
                    return true;
                }

                // Try without extension on metadata side
                const metadataWithoutExt = metadataFilename.replace(/\.[^.]+$/, '');
                const imageWithoutExt = imageFilename.replace(/\.[^.]+$/, '');
                if (metadataWithoutExt === imageWithoutExt) {
                    return true;
                }
            }
        }

        return false;
    });
}

/**
 * Product name matches image filename (exact match, not fuzzy)
 * Use case: User named images exactly like product names
 * Example: "Blue Widget.jpg" matches product name "Blue Widget"
 * Automatically handles with/without extensions
 */
function linkByNameEqualsFilename(product) {
    if (!product || !product.filename) return null;

    // Normalize filename both with and without extension
    const normalizeString = (str) => str
        .trim()
        .toLowerCase()
        .replace(/[_-]/g, ' ')
        .replace(/\s+/g, ' ');

    const filenameWithExt = normalizeString(product.filename);
    const filenameNoExt = normalizeString(product.filename.replace(/\.[^.]+$/, ''));

    if (!filenameNoExt) return null;

    return state.importedData.find(data => {
        if (!data || !data.name) return false;

        const cleanName = normalizeString(String(data.name));
        const cleanNameNoExt = normalizeString(String(data.name).replace(/\.[^.]+$/, ''));

        // Try all combinations automatically
        return cleanName === filenameNoExt ||
               cleanName === filenameWithExt ||
               cleanNameNoExt === filenameNoExt ||
               cleanNameNoExt === filenameWithExt;
    });
}

/**
 * Search all metadata fields for match
 * Use case: Flexible matching when you don't know which field has the identifier
 * Checks all CSV columns for a match with image filename (without extension)
 */
function linkBySearchAllFields(product) {
    if (!product || !product.filename) return null;

    const cleanFilename = product.filename
        .replace(/\.[^.]+$/, '')
        .trim()
        .toLowerCase();

    if (!cleanFilename) return null;

    // Find first metadata row where ANY field matches the filename
    return state.importedData.find(data => {
        if (!data) return false;

        // Check all fields in this metadata row
        for (const [key, value] of Object.entries(data)) {
            if (!value) continue;

            const fieldValue = String(value).trim().toLowerCase();

            // Exact match
            if (fieldValue === cleanFilename) {
                return true;
            }

            // Without extension match
            const fieldWithoutExt = fieldValue.replace(/\.[^.]+$/, '');
            if (fieldWithoutExt === cleanFilename) {
                return true;
            }

            // Contains match (for cases like "IMG_ABC-123_final.jpg" matching "ABC-123")
            if (fieldValue.includes(cleanFilename) || cleanFilename.includes(fieldValue)) {
                // Only match if substantial overlap (avoid false positives)
                const minLength = Math.min(fieldValue.length, cleanFilename.length);
                if (minLength >= 3) {  // Require at least 3 chars
                    return true;
                }
            }
        }

        return false;
    });
}

function calculateFuzzyScore(str1, str2) {
    if (!str1 || !str2) return 0;
    
    // Exact match
    if (str1 === str2) return 1.0;
    
    // Contains match
    if (str1.includes(str2) || str2.includes(str1)) {
        const shorter = str1.length < str2.length ? str1 : str2;
        const longer = str1.length < str2.length ? str2 : str1;
        return shorter.length / longer.length;
    }
    
    // Word overlap
    const words1 = str1.split(' ').filter(w => w.length > 1);
    const words2 = str2.split(' ').filter(w => w.length > 1);
    
    if (words1.length === 0 || words2.length === 0) return 0;
    
    let matchingWords = 0;
    for (const w1 of words1) {
        for (const w2 of words2) {
            if (w1 === w2 || w1.includes(w2) || w2.includes(w1)) {
                matchingWords++;
                break;
            }
        }
    }
    
    return matchingWords / Math.max(words1.length, words2.length);
}

/**
 * Build word index for faster fuzzy matching
 * Pre-indexes all words in imported data to avoid scanning all products
 * Reduces fuzzy matching from O(n²) to approximately O(n log n)
 */
function buildFuzzyIndex() {
    state.fuzzyIndex = {};

    state.importedData.forEach((data, index) => {
        if (!data || !data.name) return;

        const cleanName = normalizeForFuzzyMatch(data.name);
        if (!cleanName) return;

        const words = cleanName.split(' ').filter(w => w.length > 1);

        words.forEach(word => {
            if (!state.fuzzyIndex[word]) {
                state.fuzzyIndex[word] = [];
            }

            if (!state.fuzzyIndex[word].includes(index)) {
                state.fuzzyIndex[word].push(index);
            }
        });
    });

    console.log(`✓ Built fuzzy index with ${Object.keys(state.fuzzyIndex).length} unique words`);
}

/**
 * Get candidate products from fuzzy index based on word overlap
 * Scores products by number of matching words
 * Returns top candidates to avoid scoring all products
 */
function getFuzzyIndexCandidates(cleanFilename, limit = 10) {
    // Fallback if index not built
    if (!state.fuzzyIndex || Object.keys(state.fuzzyIndex).length === 0) {
        return state.importedData;
    }

    const fileWords = cleanFilename.split(' ').filter(w => w.length > 1);
    if (fileWords.length === 0) return state.importedData;

    const candidateScores = {};

    // Score products by word overlap
    fileWords.forEach(word => {
        if (state.fuzzyIndex[word]) {
            state.fuzzyIndex[word].forEach(index => {
                candidateScores[index] = (candidateScores[index] || 0) + 1;
            });
        }
    });

    // If no word matches, return random sample
    if (Object.keys(candidateScores).length === 0) {
        return state.importedData.slice(0, limit);
    }

    // Get top candidates by word overlap score
    const topIndices = Object.entries(candidateScores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(limit, Math.ceil(state.importedData.length * 0.1)))
        .map(([index]) => parseInt(index));

    return topIndices.map(i => state.importedData[i]);
}

function applyLinking() {
    const strategy = state.linkingStrategy;
    const totalProducts = state.products.length;
    
    // For large datasets, show progress
    if (totalProducts > 100) {
        showToast(`Processing ${totalProducts} products...`, 'info');
    }
    
    // Use chunked processing for large datasets
    if (totalProducts > 500) {
        applyLinkingChunked(strategy);
        return;
    }
    
    const matches = performLinking(strategy);
    finalizeLinking(matches);
}

function applyLinkingChunked(strategy) {
    const chunkSize = 100;
    const totalProducts = state.products.length;
    let processedCount = 0;
    const allResults = [];
    
    function processChunk(startIndex) {
        const endIndex = Math.min(startIndex + chunkSize, totalProducts);
        const chunk = state.products.slice(startIndex, endIndex);
        
        // Process chunk
        chunk.forEach((product, i) => {
            const globalIndex = startIndex + i;
            let matchedData = null;
            
            switch (strategy) {
                case 'filename_equals_sku':
                    matchedData = linkByFilenameEqualsSKU(product);
                    break;
                case 'filename_contains_sku':
                    matchedData = linkByFilenameContainsSKU(product);
                    break;
                case 'folder_equals_sku':
                    matchedData = linkByFolderEqualsSKU(product);
                    break;
                case 'fuzzy_name':
                    matchedData = linkByFuzzyName(product);
                    break;
                case 'sku_equals_filename':
                    matchedData = linkBySKUEqualsFilename(product);
                    break;
                case 'metadata_filename':
                    matchedData = linkByMetadataFilename(product);
                    break;
                case 'name_equals_filename':
                    matchedData = linkByNameEqualsFilename(product);
                    break;
                case 'search_all_fields':
                    matchedData = linkBySearchAllFields(product);
                    break;
            }
            
            allResults[globalIndex] = {
                image: product.filename,
                data: matchedData || {},
                matched: !!matchedData
            };
        });
        
        processedCount = endIndex;
        
        // Update progress
        const progress = Math.round((processedCount / totalProducts) * 100);
        document.getElementById('importStatusText').innerHTML = 
            `Processing: ${progress}% (${processedCount}/${totalProducts})`;
        
        if (endIndex < totalProducts) {
            // Process next chunk asynchronously
            setTimeout(() => processChunk(endIndex), 0);
        } else {
            // All done
            const linked = allResults.filter(r => r.matched).length;
            const unlinked = allResults.filter(r => !r.matched).length;
            finalizeLinking({ linked, unlinked, results: allResults });
        }
    }
    
    // Start processing
    processChunk(0);
}

function finalizeLinking(matches) {
    // Apply matched data to products
    matches.results.forEach((result, index) => {
        if (result.matched && state.products[index]) {
            const product = state.products[index];
            const data = result.data;
            
            // Merge data - only overwrite if data exists
            if (data.sku) product.sku = data.sku;
            if (data.name) product.name = data.name;
            if (data.price) product.price = data.price;
            if (data.category) product.category = data.category;
            
            // Parse history if present
            if (data.price_history) {
                const parsed = parsePriceHistory(data.price_history);
                if (parsed && parsed.length > 0) {
                    product.priceHistory = parsed;
                }
            }
            if (data.performance_history) {
                const parsed = parsePerformanceHistory(data.performance_history);
                if (parsed && parsed.length > 0) {
                    product.performanceHistory = parsed;
                }
            }
        }
    });
    
    // Store unmatched for manual linking
    state.unmatchedImages = matches.results
        .map((result, index) => ({ ...state.products[index], index }))
        .filter((_, i) => !matches.results[i].matched);
    
    state.unmatchedData = state.importedData.filter(data => {
        return !matches.results.some(result => result.matched && result.data === data);
    });
    
    if (state.unmatchedImages.length > 0) {
        // Show manual linking panel
        showManualLinking();
    } else {
        // All matched, go to next step
        showToast(`All ${matches.linked} products linked successfully!`, 'success');
        goToStep(3);
    }
    
    saveState();
}

function showManualLinking() {
    document.getElementById('linkingPanel').style.display = 'none';
    document.getElementById('manualLinkingPanel').style.display = 'block';
    
    renderUnmatchedImages();
    renderAvailableProducts();
}

function renderUnmatchedImages() {
    const container = document.getElementById('unmatchedImagesList');
    container.innerHTML = state.unmatchedImages.map((product, i) => `
        <div class="unmatched-item" data-image-index="${i}" onclick="selectUnmatchedImage(${i})">
            <div class="item-name">${escapeHtml(product.filename || 'Unknown')}</div>
            <div class="item-category">${product.category ? `[${escapeHtml(product.category)}]` : ''}</div>
            <button class="btn-small" onclick="event.stopPropagation(); linkManually(${i}, null)">SKIP</button>
        </div>
    `).join('');
}

function selectUnmatchedImage(index) {
    // Remove selection from all items
    document.querySelectorAll('.unmatched-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Select clicked item
    const item = document.querySelector(`.unmatched-item[data-image-index="${index}"]`);
    if (item) {
        item.classList.add('selected');
    }
}

function renderAvailableProducts() {
    const container = document.getElementById('availableProductsList');
    container.innerHTML = state.unmatchedData.map((data, i) => `
        <div class="available-item" data-product-index="${i}" onclick="selectProductForLinking(${i})">
            <div class="item-sku">${escapeHtml(data.sku || 'No SKU')}</div>
            <div class="item-name">${escapeHtml(data.name || 'No name')}</div>
            <div class="item-price">${data.price ? '$' + data.price : ''}</div>
        </div>
    `).join('');
}

function filterAvailableProducts() {
    const query = document.getElementById('productSearchInput').value.toLowerCase();
    const items = document.querySelectorAll('.available-item');
    
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(query) ? 'block' : 'none';
    });
}

let selectedImageIndex = null;

function selectProductForLinking(productIndex) {
    // Get currently selected image
    const selectedImage = document.querySelector('.unmatched-item.selected');
    if (!selectedImage) {
        showToast('Select an image first', 'warning');
        return;
    }
    
    const imageIndex = parseInt(selectedImage.dataset.imageIndex);
    linkManually(imageIndex, productIndex);
}

function linkManually(imageIndex, productIndex) {
    const product = state.unmatchedImages[imageIndex];
    
    if (productIndex !== null) {
        const data = state.unmatchedData[productIndex];
        
        // Apply data to product
        const actualProduct = state.products[product.index];
        actualProduct.sku = data.sku || actualProduct.sku;
        actualProduct.name = data.name || actualProduct.name;
        actualProduct.price = data.price || actualProduct.price;
        if (data.category) actualProduct.category = data.category;
        
        // Parse history
        if (data.price_history) {
            actualProduct.priceHistory = parsePriceHistory(data.price_history);
        }
        if (data.performance_history) {
            actualProduct.performanceHistory = parsePerformanceHistory(data.performance_history);
        }
        
        // Remove from unmatched
        state.unmatchedData.splice(productIndex, 1);
    }
    
    // Remove image from unmatched
    state.unmatchedImages.splice(imageIndex, 1);
    
    // Re-render
    renderUnmatchedImages();
    renderAvailableProducts();
    
    if (state.unmatchedImages.length === 0) {
        showToast('All images linked!', 'success');
        setTimeout(() => finishLinking(), 500);
    }
    
    saveState();
}

function parsePriceHistory(historyStr) {
    if (!historyStr) return [];
    return historyStr.split(';').map(entry => {
        const [date, price] = entry.split(':');
        return { date, price: parseFloat(price) || 0 };
    }).filter(e => e.date && e.price);
}

function parsePerformanceHistory(historyStr) {
    if (!historyStr) return [];
    
    // Parse simplified format (JSON array of sales numbers)
    try {
        const parsed = JSON.parse(historyStr);
        if (Array.isArray(parsed)) {
            // Convert simple numbers to internal format with auto-generated dates
            const today = new Date();
            return parsed.map((sales, i) => {
                const date = new Date(today);
                date.setMonth(date.getMonth() - (parsed.length - 1 - i));
                return {
                    date: date.toISOString().split('T')[0],
                    sales: parseInt(sales) || 0
                };
            });
        }
    } catch (e) {
        console.error('Failed to parse performance_history:', e);
    }
    
    return [];
}

function backToLinkingStrategy() {
    document.getElementById('manualLinkingPanel').style.display = 'none';
    document.getElementById('linkingPanel').style.display = 'block';
}

function finishLinking() {
    showToast('Linking complete!', 'success');
    goToStep(3);
}

function skipLinking() {
    // User wants to enter metadata manually
    goToStep(3);
}


// ===== STEP 3: Add Metadata =====
function renderProductsTable() {
    const tbody = document.getElementById('productsTableBody');
    
    tbody.innerHTML = state.products.map((product, index) => `
        <tr data-index="${index}" class="${product.selected ? 'row-selected' : ''}">
            <td><input type="checkbox" ${product.selected ? 'checked' : ''} onchange="toggleProductSelection(${index})"></td>
            <td>${escapeHtml(product.filename)}</td>
            <td><input type="text" value="${escapeHtml(product.category)}" onchange="updateProduct(${index}, 'category', this.value)" placeholder="Auto-detected"></td>
            <td><input type="text" value="${escapeHtml(product.sku)}" onchange="updateProduct(${index}, 'sku', this.value)" placeholder="Optional"></td>
            <td><input type="text" value="${escapeHtml(product.name)}" onchange="updateProduct(${index}, 'name', this.value)" placeholder="Optional"></td>
            <td><input type="number" value="${product.price}" onchange="updateProduct(${index}, 'price', this.value)" placeholder="0.00" step="0.01" min="0"></td>
            <td>
                <button class="btn-icon" onclick="duplicateProduct(${index})" title="Duplicate">DUPLICATE</button>
                <button class="btn-icon delete" onclick="deleteProduct(${index})" title="Delete">DELETE</button>
            </td>
        </tr>
    `).join('');
}

function updateProduct(index, field, value) {
    saveStateForUndo();
    state.products[index][field] = value;
    saveState();
    scheduleAutoSave();
}

function toggleProductSelection(index) {
    state.products[index].selected = !state.products[index].selected;
    renderProductsTable();
}

function toggleSelectAll() {
    const checked = document.getElementById('selectAllCheckbox').checked;
    state.products.forEach(p => p.selected = checked);
    renderProductsTable();
}

function selectAll() {
    state.products.forEach(p => p.selected = true);
    document.getElementById('selectAllCheckbox').checked = true;
    renderProductsTable();
}

function deselectAll() {
    state.products.forEach(p => p.selected = false);
    document.getElementById('selectAllCheckbox').checked = false;
    renderProductsTable();
}

function applyBulkEdit(field) {
    const selectedProducts = state.products.filter(p => p.selected);
    
    if (selectedProducts.length === 0) {
        showToast('No products selected', 'warning');
        return;
    }

    saveStateForUndo();

    let value;
    if (field === 'category') {
        value = document.getElementById('bulkCategory').value.trim();
    } else if (field === 'price') {
        value = document.getElementById('bulkPrice').value;
    }

    if (!value) {
        showToast('Please enter a value', 'warning');
        return;
    }

    selectedProducts.forEach(product => {
        product[field] = value;
    });

    renderProductsTable();
    saveState();
    showToast(`Applied ${field} to ${selectedProducts.length} product(s)`, 'success');
}


function pasteFromExcel() {
    showToast('Paste your Excel data (Ctrl+V or Cmd+V) into the table cells directly', 'info');
    // Note: Direct paste from Excel works natively in the input fields
}

function duplicateProduct(index) {
    saveStateForUndo();
    const product = { ...state.products[index] };
    product.filename = product.filename.replace(/(\.[^.]+)$/, '_copy$1');
    state.products.splice(index + 1, 0, product);
    renderProductsTable();
    saveState();
    showToast('Product duplicated', 'success');
}

function deleteProduct(index) {
    if (confirm('Delete this product?')) {
        saveStateForUndo();
        state.products.splice(index, 1);
        renderProductsTable();
        saveState();
        showToast('Product deleted', 'success');
    }
}

// ===== STEP 3: Price & Performance History =====
function populateProductSelector() {
    const selector = document.getElementById('productSelector');
    selector.innerHTML = '<option value="">-- Select a product --</option>' +
        state.products.map((p, i) => 
            `<option value="${i}">${escapeHtml(p.filename)}</option>`
        ).join('');
}

function loadProductHistory() {
    const index = parseInt(document.getElementById('productSelector').value);
    
    if (isNaN(index)) {
        document.getElementById('historyEditor').style.display = 'none';
        return;
    }

    state.selectedProductIndex = index;
    document.getElementById('historyEditor').style.display = 'block';
    
    updateProductProgress();
    renderPriceHistory();
    renderPerformanceHistory();
}

function updateProductProgress() {
    const withHistory = state.products.filter(p => 
        p.priceHistory.length > 0 || p.performanceHistory.length > 0
    ).length;
    
    document.getElementById('productProgress').textContent = 
        `${withHistory} of ${state.products.length} products have history data`;
}

function switchTab(tab) {
    // Remove active class from all tabs and panels
    document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and panel
    if (tab === 'price') {
        document.querySelector('.tabs .tab:nth-child(1)').classList.add('active');
        document.getElementById('priceTab').classList.add('active');
    } else if (tab === 'performance') {
        document.querySelector('.tabs .tab:nth-child(2)').classList.add('active');
        document.getElementById('performanceTab').classList.add('active');
    }
}


// Price History Functions
function renderPriceHistory() {
    const product = state.products[state.selectedProductIndex];
    const container = document.getElementById('priceEntries');
    
    if (!product.priceHistory || product.priceHistory.length === 0) {
        container.innerHTML = '<p class="empty-state">No price history entries. Click "Add Price Entry" to start.</p>';
        return;
    }

    container.innerHTML = product.priceHistory.map((entry, i) => `
        <div class="history-entry">
            <div class="history-entry-fields">
                <div class="field-group">
                    <label>Date</label>
                    <input type="date" value="${entry.date}" onchange="updatePriceEntry(${i}, 'date', this.value)">
                </div>
                <div class="field-group">
                    <label>Price</label>
                    <input type="number" value="${entry.price}" onchange="updatePriceEntry(${i}, 'price', this.value)" step="0.01" min="0">
                </div>
            </div>
            <div class="history-entry-actions">
                <button class="btn-icon delete" onclick="deletePriceEntry(${i})" title="Delete">DELETE</button>
            </div>
        </div>
    `).join('');
}

function addPriceEntry() {
    const product = state.products[state.selectedProductIndex];
    const today = new Date().toISOString().split('T')[0];
    
    if (!product.priceHistory) {
        product.priceHistory = [];
    }

    product.priceHistory.push({
        date: today,
        price: product.price || 0
    });

    renderPriceHistory();
    saveState();
    updateProductProgress();
}

function updatePriceEntry(index, field, value) {
    const product = state.products[state.selectedProductIndex];
    product.priceHistory[index][field] = field === 'price' ? parseFloat(value) || 0 : value;
    saveState();
}

function deletePriceEntry(index) {
    const product = state.products[state.selectedProductIndex];
    product.priceHistory.splice(index, 1);
    renderPriceHistory();
    saveState();
    updateProductProgress();
}

function clearPriceHistory() {
    if (confirm('Clear all price history for this product?')) {
        const product = state.products[state.selectedProductIndex];
        product.priceHistory = [];
        renderPriceHistory();
        saveState();
        updateProductProgress();
    }
}

function importPriceFromClipboard() {
    navigator.clipboard.readText().then(text => {
        const product = state.products[state.selectedProductIndex];
        const lines = text.trim().split('\n');
        
        product.priceHistory = [];
        
        lines.forEach(line => {
            const parts = line.split(/[\t,;]/).map(s => s.trim());
            
            if (parts.length >= 2) {
                const date = parts[0];
                const price = parseFloat(parts[1]);
                
                if (date && !isNaN(price)) {
                    product.priceHistory.push({ date, price });
                }
            } else if (parts.length === 1) {
                const price = parseFloat(parts[0]);
                if (!isNaN(price)) {
                    const today = new Date();
                    today.setMonth(today.getMonth() - product.priceHistory.length);
                    product.priceHistory.push({
                        date: today.toISOString().split('T')[0],
                        price
                    });
                }
            }
        });

        renderPriceHistory();
        saveState();
        updateProductProgress();
        showToast(`Imported ${product.priceHistory.length} price entries`, 'success');
    }).catch(() => {
        showToast('Failed to read clipboard. Please grant permission.', 'error');
    });
}


// Performance History Functions (Simplified - Sales Only)
function renderPerformanceHistory() {
    const product = state.products[state.selectedProductIndex];
    const container = document.getElementById('performanceEntries');
    
    if (!product.performanceHistory || product.performanceHistory.length === 0) {
        container.innerHTML = '<p class="empty-state">No performance history entries. Click "Add Performance Entry" to start.</p>';
        return;
    }

    container.innerHTML = product.performanceHistory.map((entry, i) => `
        <div class="history-entry">
            <div class="history-entry-fields">
                <div class="field-group">
                    <label>Month ${i + 1} (${entry.date})</label>
                    <input type="number" value="${entry.sales}" onchange="updatePerformanceEntry(${i}, 'sales', this.value)" min="0" placeholder="Sales count">
                </div>
            </div>
            <div class="history-entry-actions">
                <button class="btn-icon delete" onclick="deletePerformanceEntry(${i})" title="Delete">DELETE</button>
            </div>
        </div>
    `).join('');
}

function addPerformanceEntry() {
    const product = state.products[state.selectedProductIndex];
    
    if (!product.performanceHistory) {
        product.performanceHistory = [];
    }

    // Auto-generate date going backwards monthly
    const today = new Date();
    today.setMonth(today.getMonth() - product.performanceHistory.length);
    
    product.performanceHistory.push({
        date: today.toISOString().split('T')[0],
        sales: 0
    });

    renderPerformanceHistory();
    saveState();
    updateProductProgress();
}

function updatePerformanceEntry(index, field, value) {
    const product = state.products[state.selectedProductIndex];
    product.performanceHistory[index].sales = parseInt(value) || 0;
    saveState();
}

function deletePerformanceEntry(index) {
    const product = state.products[state.selectedProductIndex];
    product.performanceHistory.splice(index, 1);
    renderPerformanceHistory();
    saveState();
    updateProductProgress();
}

function clearPerformanceHistory() {
    if (confirm('Clear all performance history for this product?')) {
        const product = state.products[state.selectedProductIndex];
        product.performanceHistory = [];
        renderPerformanceHistory();
        saveState();
        updateProductProgress();
    }
}

function importPerformanceFromClipboard() {
    navigator.clipboard.readText().then(text => {
        const product = state.products[state.selectedProductIndex];
        const lines = text.trim().split('\n');
        
        product.performanceHistory = [];
        
        lines.forEach((line, index) => {
            const parts = line.split(/[\t,;]/).map(s => s.trim());
            
            // Parse just sales numbers (simplified format)
            const sales = parseInt(parts[0]) || 0;
            
            if (sales > 0 || parts[0] === '0') {
                // Auto-generate monthly dates going backwards
                const today = new Date();
                today.setMonth(today.getMonth() - index);
                
                product.performanceHistory.push({
                    date: today.toISOString().split('T')[0],
                    sales: sales
                });
            }
        });

        renderPerformanceHistory();
        saveState();
        updateProductProgress();
        showToast(`Imported ${product.performanceHistory.length} performance entries`, 'success');
    }).catch(() => {
        showToast('Failed to read clipboard. Please grant permission.', 'error');
    });
}

function skipHistory() {
    goToStep(5);
}


// ===== STEP 4: Export =====
function refreshPreview() {
    const csv = generateCSV();
    document.getElementById('csvPreviewContent').textContent = csv;
    
    const lines = csv.split('\n').length;
    const includeHeaders = document.getElementById('includeHeaders').checked;
    const dataLines = includeHeaders ? lines - 1 : lines;
    
    document.getElementById('previewStats').textContent = 
        `${dataLines} data row${dataLines !== 1 ? 's' : ''}, ${state.products.length} product${state.products.length !== 1 ? 's' : ''}`;
}

function generateCSV() {
    const separator = document.getElementById('separatorSelect').value.replace('\\t', '\t');
    const includeHeaders = document.getElementById('includeHeaders').checked;
    const includeEmpty = document.getElementById('includeEmptyFields').checked;
    
    let csv = '';
    
    // Headers
    if (includeHeaders) {
        csv += ['filename', 'category', 'sku', 'name', 'price', 'price_history', 'performance_history'].join(separator) + '\n';
    }
    
    // Data rows
    state.products.forEach(product => {
        const row = [];
        
        // Basic fields
        row.push(quoteCSVField(product.filename, separator));
        row.push(quoteCSVField(product.category || (includeEmpty ? '' : ''), separator));
        row.push(quoteCSVField(product.sku || (includeEmpty ? '' : ''), separator));
        row.push(quoteCSVField(product.name || (includeEmpty ? '' : ''), separator));
        row.push(product.price || (includeEmpty ? '' : ''));
        
        // Price history
        const priceHistory = formatPriceHistory(product.priceHistory);
        row.push(quoteCSVField(priceHistory, separator));
        
        // Performance history
        const performanceHistory = formatPerformanceHistory(product.performanceHistory);
        row.push(quoteCSVField(performanceHistory, separator));
        
        csv += row.join(separator) + '\n';
    });
    
    return csv;
}

function quoteCSVField(field, separator) {
    if (!field) return '';
    
    const str = String(field);
    
    // Quote if contains separator, quotes, or newlines
    if (str.includes(separator) || str.includes('"') || str.includes('\n')) {
        return '"' + str.replace(/"/g, '""') + '"';
    }
    
    return str;
}

function formatPriceHistory(priceHistory) {
    if (!priceHistory || priceHistory.length === 0) return '';
    
    return priceHistory
        .map(entry => `${entry.date}:${entry.price}`)
        .join(';');
}

function formatPerformanceHistory(performanceHistory) {
    if (!performanceHistory || performanceHistory.length === 0) return '';
    
    // Simplified format: JSON array of sales numbers only
    // Backend auto-generates dates (monthly intervals) and sets views/conversion/revenue to 0
    const salesNumbers = performanceHistory.map(entry => entry.sales || 0);
    return JSON.stringify(salesNumbers);
}

async function downloadCSV() {
    const csv = generateCSV();
    const filename = `products_${new Date().toISOString().slice(0, 10)}.csv`;

    // Check if running in pywebview
    if (window.pywebview) {
        try {
            const result = await window.pywebview.api.save_file_auto(csv, filename);
            if (result) {
                showToast(`CSV saved to Downloads folder: ${filename}`, 'success');
            } else {
                showToast('Save failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Save failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('CSV downloaded successfully!', 'success');
        } catch (error) {
            console.error('Download failed:', error);
            showToast('Download failed', 'error');
        } finally {
            // Always revoke the URL to prevent memory leak
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    }
}

function copyToClipboard() {
    const csv = generateCSV();
    navigator.clipboard.writeText(csv).then(() => {
        showToast('CSV copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy to clipboard', 'error');
    });
}

function saveAsTemplate() {
    const templateName = prompt('Enter template name:');
    if (!templateName) return;
    
    const templates = JSON.parse(localStorage.getItem('csvTemplates') || '{}');
    templates[templateName] = {
        products: state.products,
        timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('csvTemplates', JSON.stringify(templates));
    showToast(`Template "${templateName}" saved!`, 'success');
}


// ===== Column Preview Functions =====

/**
 * Update the column preview panel with detected columns
 */
function updateColumnPreview() {
    const container = document.getElementById('detectedColumnsContainer');
    if (!container) return;

    // Get columns from imported data or products
    let columns = [];
    if (state.importedData.length > 0) {
        // Get all unique keys from imported data
        const keySet = new Set();
        state.importedData.forEach(item => {
            Object.keys(item).forEach(key => keySet.add(key));
        });
        columns = Array.from(keySet);
    } else if (state.products.length > 0) {
        // Get columns from products
        columns = Object.keys(state.products[0] || {}).filter(k => k !== 'priceHistory' && k !== 'performanceHistory');
    }

    if (columns.length === 0) {
        container.innerHTML = '<p style="color: #888; font-style: italic;">Import a CSV to see detected columns.</p>';
        return;
    }

    // Detect column types
    const columnInfo = columns.map(col => {
        const values = (state.importedData.length > 0 ? state.importedData : state.products)
            .map(item => item[col])
            .filter(v => v !== null && v !== undefined && v !== '');

        const dataType = detectColumnTypeFromValues(values);
        return { name: col, type: dataType, sampleCount: values.length };
    });

    // Render column chips
    let html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';
    columnInfo.forEach(col => {
        const typeIcon = col.type === 'numeric' ? '#' : 'Aa';
        const typeColor = col.type === 'numeric' ? '#4CAF50' : '#2196F3';
        html += `
            <div style="display: inline-flex; align-items: center; gap: 5px; padding: 8px 12px; background: white; border: 2px solid ${typeColor}; border-radius: 4px;">
                <span style="font-size: 11px; color: ${typeColor}; font-weight: bold;">${typeIcon}</span>
                <span style="font-weight: bold; text-transform: uppercase;">${col.name}</span>
                <span style="font-size: 11px; color: #888;">(${col.sampleCount})</span>
            </div>
        `;
    });
    html += '</div>';
    html += `<p style="margin-top: 10px; font-size: 0.85em; color: #666;">${columnInfo.length} columns detected. <span style="color: #4CAF50;"># = numeric</span>, <span style="color: #2196F3;">Aa = text</span></p>`;

    container.innerHTML = html;
}

/**
 * Detect column type from values
 */
function detectColumnTypeFromValues(values) {
    if (!values || values.length === 0) return 'string';

    const sampleSize = Math.min(50, values.length);
    const sample = values.slice(0, sampleSize);

    let numericCount = 0;
    sample.forEach(val => {
        const cleaned = String(val).replace(/[$,]/g, '').trim();
        if (cleaned && !isNaN(parseFloat(cleaned))) {
            numericCount++;
        }
    });

    return (numericCount / sample.length) >= 0.8 ? 'numeric' : 'string';
}

/**
 * Update the data preview table
 */
function updateDataPreviewTable() {
    const panel = document.getElementById('dataPreviewPanel');
    const thead = document.getElementById('previewTableHead');
    const tbody = document.getElementById('previewTableBody');
    const rowCount = document.getElementById('previewRowCount');

    if (!panel || !thead || !tbody) return;

    // Use imported data or products
    const data = state.importedData.length > 0 ? state.importedData : state.products;

    if (data.length === 0) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';

    // Get all columns
    const columns = [];
    const keySet = new Set();
    data.forEach(item => {
        Object.keys(item).forEach(key => {
            if (key !== 'priceHistory' && key !== 'performanceHistory') {
                keySet.add(key);
            }
        });
    });
    columns.push(...Array.from(keySet));

    // Render header
    thead.innerHTML = `<tr>${columns.map(col => `<th style="text-transform: uppercase;">${col}</th>`).join('')}</tr>`;

    // Render first 10 rows as preview
    const previewRows = data.slice(0, 10);
    tbody.innerHTML = previewRows.map(item => {
        return `<tr>${columns.map(col => {
            const val = item[col] || '';
            const displayVal = String(val).length > 30 ? String(val).substring(0, 30) + '...' : val;
            return `<td>${displayVal}</td>`;
        }).join('')}</tr>`;
    }).join('');

    // Show row count
    if (rowCount) {
        rowCount.textContent = `Showing ${previewRows.length} of ${data.length} rows`;
    }
}


// ===== Navigation & State Management =====
function goToStep(step) {
    // Hide all sections (now only 3 steps)
    for (let i = 1; i <= 3; i++) {
        const stepEl = document.getElementById(`step${i}`);
        const progressEl = document.querySelector(`.progress-step[data-step="${i}"]`);
        if (stepEl) stepEl.style.display = 'none';
        if (progressEl) progressEl.classList.remove('active', 'completed');
    }

    // Show current step
    const currentStepEl = document.getElementById(`step${step}`);
    const currentProgressEl = document.querySelector(`.progress-step[data-step="${step}"]`);
    if (currentStepEl) currentStepEl.style.display = 'block';
    if (currentProgressEl) currentProgressEl.classList.add('active');

    // Mark previous steps as completed
    for (let i = 1; i < step; i++) {
        const progressEl = document.querySelector(`.progress-step[data-step="${i}"]`);
        if (progressEl) progressEl.classList.add('completed');
    }

    state.currentStep = step;

    // Initialize step-specific content
    if (step === 2) {
        // Link & Preview step
        const linkingPanel = document.getElementById('linkingPanel');
        if (linkingPanel) linkingPanel.style.display = 'none';

        // Show import panel if we have images but no imported data
        const importPanel = document.getElementById('importPanel');
        if (importPanel) {
            importPanel.style.display = state.products.length > 0 && state.importedData.length === 0 ? 'block' : 'none';
        }

        // Update column preview
        updateColumnPreview();

        // Update data preview table
        updateDataPreviewTable();
    } else if (step === 3) {
        // Export step
        refreshPreview();
    }

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Undo/Redo functionality
function saveStateForUndo() {
    try {
        const stateString = JSON.stringify(state.products);

        // Check size before adding to undo stack
        if (stateString.length > 1024 * 1024) { // 1MB limit per undo state
            console.warn('State too large for undo, skipping');
            return;
        }

        state.undoStack.push(stateString);
        state.redoStack = []; // Clear redo stack on new action

        // MEMORY OPTIMIZATION: Limit undo/redo stack to prevent state arrays growth (200-500MB possible)
        if (state.undoStack.length > MAX_UNDO_STACK) {
            state.undoStack.shift();
        }
    } catch (e) {
        console.error('Failed to save undo state:', e);
    }
}

// MEMORY OPTIMIZATION: Enforce state size limits to prevent unbounded growth
function enforceStateLimits() {
    // Limit products array size
    if (state.products.length > MAX_PRODUCTS) {
        console.warn(`Products exceed limit (${state.products.length} > ${MAX_PRODUCTS}), removing oldest items`);
        state.products = state.products.slice(-MAX_PRODUCTS);
    }

    // Limit imported data
    if (state.importedData.length > MAX_PRODUCTS) {
        state.importedData = state.importedData.slice(-MAX_PRODUCTS);
    }
}

function undo() {
    if (state.undoStack.length === 0) return;
    
    state.redoStack.push(JSON.stringify(state.products));
    state.products = JSON.parse(state.undoStack.pop());
    
    renderProductsTable();
    saveState();
    showToast('Undo successful', 'info');
}

function redo() {
    if (state.redoStack.length === 0) return;
    
    state.undoStack.push(JSON.stringify(state.products));
    state.products = JSON.parse(state.redoStack.pop());
    
    renderProductsTable();
    saveState();
    showToast('Redo successful', 'info');
}

// Save/Load Draft
function saveDraft() {
    const draftName = prompt('Enter draft name:', `Draft_${new Date().toISOString().slice(0, 10)}`);
    if (!draftName) return;
    
    const drafts = JSON.parse(localStorage.getItem('csvDrafts') || '{}');
    drafts[draftName] = {
        products: state.products,
        timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('csvDrafts', JSON.stringify(drafts));
    showToast(`Draft "${draftName}" saved!`, 'success');
}

function loadDraft() {
    const drafts = JSON.parse(localStorage.getItem('csvDrafts') || '{}');
    const draftNames = Object.keys(drafts);
    
    if (draftNames.length === 0) {
        showToast('No saved drafts found', 'info');
        return;
    }
    
    const draftName = prompt(`Available drafts:\n${draftNames.join('\n')}\n\nEnter draft name to load:`);
    if (!draftName || !drafts[draftName]) {
        showToast('Draft not found', 'error');
        return;
    }
    
    if (confirm('Loading a draft will replace current data. Continue?')) {
        state.products = drafts[draftName].products;
        renderProductsTable();
        saveState();
        showToast(`Draft "${draftName}" loaded!`, 'success');
    }
}

// Auto-save to localStorage with size limit
function saveState() {
    try {
        const stateData = {
            products: state.products,
            currentStep: state.currentStep,
            timestamp: new Date().toISOString()
        };
        
        const stateString = JSON.stringify(stateData);
        
        // Check size (5MB limit to prevent localStorage overflow)
        if (stateString.length > 5 * 1024 * 1024) {
            console.warn('State too large to save, truncating history data');
            // Save without history data if too large
            const truncatedState = {
                products: state.products.map(p => ({
                    filename: p.filename,
                    category: p.category,
                    sku: p.sku,
                    name: p.name,
                    price: p.price,
                    priceHistory: [],
                    performanceHistory: []
                })),
                currentStep: state.currentStep,
                timestamp: new Date().toISOString()
            };
            localStorage.setItem('csvBuilderState', JSON.stringify(truncatedState));
        } else {
            localStorage.setItem('csvBuilderState', stateString);
        }
    } catch (e) {
        console.error('Failed to save state:', e);
        // If localStorage is full, try to clear old data
        if (e.name === 'QuotaExceededError') {
            try {
                localStorage.removeItem('csvBuilderState');
                console.warn('Cleared old state due to quota exceeded');
            } catch (clearError) {
                console.error('Failed to clear state:', clearError);
            }
        }
    }
}

function loadFromLocalStorage() {
    const saved = localStorage.getItem('csvBuilderState');
    if (saved) {
        try {
            const data = JSON.parse(saved);
            if (data.products && data.products.length > 0) {
                const age = Date.now() - new Date(data.timestamp).getTime();
                const hours = Math.floor(age / (1000 * 60 * 60));
                
                // Show custom modal instead of browser confirm
                showLoadSavedWorkModal(hours, data);
            }
        } catch (e) {
            console.error('Failed to load saved state:', e);
        }
    }
}

function scheduleAutoSave() {
    if (state.autoSaveTimer) {
        clearTimeout(state.autoSaveTimer);
        state.autoSaveTimer = null;
    }
    
    state.autoSaveTimer = setTimeout(() => {
        saveState();
        showAutoSaveIndicator();
        state.autoSaveTimer = null;
    }, 2000);
}

function showAutoSaveIndicator() {
    // Could add a visual indicator here
}


// ===== Utility Functions =====
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    const timeout = (type === 'error' || type === 'warning') ? 5000 : 3000;

    setTimeout(() => {
        toast.classList.remove('show');
    }, timeout);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
const keyboardShortcutHandler = (e) => {
    // Ctrl/Cmd + Z for undo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
    }
    
    // Ctrl/Cmd + Shift + Z for redo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
    }
    
    // Ctrl/Cmd + S for save draft
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveDraft();
    }
};

document.addEventListener('keydown', keyboardShortcutHandler);
eventListeners.push({ element: document, event: 'keydown', handler: keyboardShortcutHandler });

// Handle page unload - save state (cleanup is handled by first beforeunload)
// NOTE: Only one beforeunload listener is needed; the first one handles cleanup
const beforeUnloadSaveHandler = (e) => {
    if (state.products.length > 0) {
        saveState();
    }
};
window.addEventListener('beforeunload', beforeUnloadSaveHandler);
eventListeners.push({ element: window, event: 'beforeunload', handler: beforeUnloadSaveHandler });


// Toggle help text in CSV builder
function toggleHelp(helpId) {
    const helpElement = document.getElementById(helpId);
    if (helpElement) {
        helpElement.style.display = helpElement.style.display === 'none' ? 'block' : 'none';
    }
}


// ===== MAIN APP INTEGRATION =====

// Check if data was sent from main app
function checkForMainAppData() {
    const fileData = sessionStorage.getItem('csvBuilderFiles');
    const source = sessionStorage.getItem('csvBuilderSource');
    
    if (fileData && source) {
        const files = JSON.parse(fileData);
        
        // Auto-populate products from main app
        state.products = files.map(file => ({
            filename: file.filename,
            category: file.category || '',
            sku: '',
            name: '',
            price: '',
            priceHistory: [],
            performanceHistory: [],
            selected: false
        }));
        
        // Update UI
        const info = document.getElementById('imageInfo');
        const categoryCount = {};
        state.products.forEach(p => {
            if (p.category) {
                categoryCount[p.category] = (categoryCount[p.category] || 0) + 1;
            }
        });
        
        const categorySummary = Object.keys(categoryCount).length > 0
            ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
            : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';
        
        const displayLimit = 50;
        const hasMore = files.length > displayLimit;
        
        info.innerHTML = `
            <button class="btn clear-btn" onclick="clearCsvBuilderUpload()" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
            <h4>${files.length} images loaded from Main App</h4>
            <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px;">
                <strong>Destination:</strong> <span id="destinationLabel">${source === 'historical' ? 'Historical Products' : 'New Products'}</span>
                <button class="btn-small" onclick="changeDestination()" style="margin-left: 10px;">CHANGE</button>
            </div>
            ${categorySummary}
            <div class="file-list" id="csvBuilderFileList">
                ${state.products.slice(0, displayLimit).map(p => 
                    `<div>${escapeHtml(p.filename)}${p.category ? ` <span style="color: #667eea;">[${p.category}]</span>` : ''}</div>`
                ).join('')}
            </div>
            ${hasMore ? `
                <div style="text-align: center; margin-top: 10px;">
                    <button class="btn" onclick="showAllCsvBuilderFiles(${files.length})" style="font-size: 12px; padding: 5px 15px;">
                        SHOW ALL ${files.length} FILES
                    </button>
                </div>
            ` : ''}
        `;
        info.classList.add('show');
        
        document.getElementById('nextToLink').disabled = false;
        
        // Add "Send to App" button in Step 5
        addSendToAppButton(source);
        
        showToast(`${files.length} images loaded from Main App. Add metadata and send back!`, 'success');
        
        // Clear sessionStorage to prevent memory leaks
        sessionStorage.removeItem('csvBuilderFiles');
        sessionStorage.removeItem('csvBuilderSource');
        
        // Store source for later
        state.mainAppSource = source;
    }
}

// Add "Send to App" button in export step
function addSendToAppButton(source) {
    // Wait for DOM to be ready
    setTimeout(() => {
        const actionsDiv = document.querySelector('#step5 .actions');
        if (actionsDiv && !document.getElementById('sendToAppBtn')) {
            const sendBtn = document.createElement('button');
            sendBtn.id = 'sendToAppBtn';
            sendBtn.className = 'btn btn-primary';
            sendBtn.textContent = 'SEND TO APP';
            sendBtn.onclick = sendToMainApp;
            
            // Insert before download button
            const downloadBtn = actionsDiv.querySelector('button[onclick="downloadCSV()"]');
            if (downloadBtn) {
                actionsDiv.insertBefore(sendBtn, downloadBtn);
            } else {
                actionsDiv.appendChild(sendBtn);
            }
        }
    }, 100);
}

// Show destination selector modal
function showDestinationSelector() {
    // Remove any existing destination modal to prevent accumulation
    const existingModal = document.getElementById('destinationModal');
    if (existingModal) {
        existingModal.remove();
    }

    const modal = document.createElement('div');
    modal.id = 'destinationModal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;

    modal.innerHTML = `
        <div style="background: white; padding: 30px; border-radius: 8px; max-width: 400px; text-align: center;">
            <h3>Select Destination</h3>
            <p>Where should this CSV be sent?</p>
            <div style="display: flex; gap: 10px; margin-top: 20px; justify-content: center;">
                <button class="btn" onclick="setDestination('historical'); document.getElementById('destinationModal').remove();">
                    Historical Products
                </button>
                <button class="btn" onclick="setDestination('new'); document.getElementById('destinationModal').remove();">
                    New Products
                </button>
            </div>
            <button class="btn" onclick="document.getElementById('destinationModal').remove();" style="margin-top: 10px; width: 100%;">
                Cancel
            </button>
        </div>
    `;

    document.body.appendChild(modal);
}

// Set destination and update UI
function setDestination(section) {
    state.mainAppSource = section;
    const label = section === 'historical' ? 'Historical Products' : 'New Products';
    const destinationLabel = document.getElementById('destinationLabel');
    if (destinationLabel) {
        destinationLabel.textContent = label;
    }
    showToast(`Destination set to: ${label}`, 'success');
}

// Change destination (for main app uploads)
function changeDestination() {
    showDestinationSelector();
}

// Send CSV data back to main app with confirmation
function sendToMainApp() {
    // Check if in webview (no opener) or browser (has opener)
    const isWebview = !window.opener;

    // If no destination set, ask user
    if (!state.mainAppSource) {
        showToast('Please select a destination first', 'warning');
        showDestinationSelector();
        return;
    }

    // Show confirmation modal
    // Remove any existing confirm modal to prevent accumulation
    const existingConfirmModal = document.getElementById('confirmSendModal');
    if (existingConfirmModal) {
        existingConfirmModal.remove();
    }

    const modal = document.createElement('div');
    modal.id = 'confirmSendModal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;

    const destinationLabel = state.mainAppSource === 'historical' ? 'Historical Products' : 'New Products';

    modal.innerHTML = `
        <div style="background: white; padding: 30px; border-radius: 8px; max-width: 400px; text-align: center;">
            <h3>Send CSV to Main App?</h3>
            <p>Destination: <strong>${destinationLabel}</strong></p>
            <p style="font-size: 14px; color: #666;">This will populate the ${destinationLabel.toLowerCase()} section in the main app.</p>
            <div style="display: flex; gap: 10px; margin-top: 20px; justify-content: center;">
                <button class="btn" onclick="confirmSendToMainApp();">
                    SEND
                </button>
                <button class="btn" onclick="document.getElementById('confirmSendModal').remove(); showDestinationSelector();">
                    CHANGE
                </button>
                <button class="btn" onclick="document.getElementById('confirmSendModal').remove();">
                    CANCEL
                </button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

// Confirm and actually send
function confirmSendToMainApp() {
    const modal = document.getElementById('confirmSendModal');
    if (modal) modal.remove();

    const csv = generateCSV();
    const source = state.mainAppSource;

    // Check if in webview or browser
    if (window.opener) {
        // Browser mode: use postMessage to parent window
        window.opener.postMessage({
            type: 'CSV_BUILDER_COMPLETE',
            csvContent: csv,
            section: source
        }, '*');

        showToast('CSV sent to Main App! You can close this window.', 'success');

        // Close window after 2 seconds
        setTimeout(() => {
            window.close();
        }, 2000);
    } else {
        // Webview mode: use sessionStorage and navigate back
        sessionStorage.setItem('csvBuilderResult', JSON.stringify({
            type: 'CSV_BUILDER_COMPLETE',
            csvContent: csv,
            section: source
        }));

        showToast('Returning to Main App...', 'success');

        // Navigate back to main app after brief delay
        setTimeout(() => {
            window.location.href = '/';
        }, 500);
    }
}


// Clear CSV Builder Upload
function clearCsvBuilderUpload() {
    if (!confirm('Clear uploaded folder? This will reset all data.')) {
        return;
    }
    
    // Clear state
    state.products = [];
    state.selectedProductIndex = null;
    state.currentStep = 1;
    state.mainAppSource = null;
    
    // Clear UI
    document.getElementById('imageInfo').innerHTML = '';
    document.getElementById('imageInfo').classList.remove('show');
    document.getElementById('nextToLink').disabled = true;
    
    // Reset file input
    document.getElementById('imageInput').value = '';
    
    // Go back to step 1
    goToStep(1);
    
    showToast('Folder cleared', 'success');
}


// Show custom modal for loading saved work
function showLoadSavedWorkModal(hours, data) {
    // Remove any existing saved work modal to prevent accumulation
    const existingModal = document.getElementById('loadSavedWorkModal');
    if (existingModal) {
        existingModal.remove();
    }

    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.id = 'loadSavedWorkModal';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <h2>Saved Work Found</h2>
            <p>Found saved work from <strong>${hours} hour(s) ago</strong> with <strong>${data.products.length} products</strong>.</p>
            <p>Would you like to load it?</p>
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px;">
                <button class="btn" onclick="loadSavedWork()">YES, LOAD IT</button>
                <button class="btn" onclick="dismissSavedWork()">NO, START FRESH</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Store data for loading
    window.savedWorkData = data;
}

function loadSavedWork() {
    const data = window.savedWorkData;
    if (data && data.products) {
        state.products = data.products;
        if (state.products.length > 0) {
            document.getElementById('nextToLink').disabled = false;
            const info = document.getElementById('imageInfo');
            info.innerHTML = `
                <button class="btn clear-btn" onclick="clearCsvBuilderUpload()" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
                <h4>${state.products.length} products loaded from saved session</h4>
            `;
            info.classList.add('show');
            showToast('Saved work loaded successfully', 'success');
        }
    }
    
    // Close modal and clean up
    const modal = document.getElementById('loadSavedWorkModal');
    if (modal) {
        modal.remove();
    }
    
    // Clean up global reference
    if (window.savedWorkData) {
        window.savedWorkData = null;
        delete window.savedWorkData;
    }
}

function dismissSavedWork() {
    // Clear saved state
    localStorage.removeItem('csvBuilderState');
    
    // Close modal and clean up
    const modal = document.getElementById('loadSavedWorkModal');
    if (modal) {
        modal.remove();
    }
    
    // Clean up global reference
    if (window.savedWorkData) {
        window.savedWorkData = null;
        delete window.savedWorkData;
    }
    
    showToast('Starting fresh', 'success');
}


// Show all files in CSV builder
function showAllCsvBuilderFiles(totalCount) {
    const list = document.getElementById('csvBuilderFileList');
    
    if (!list) return;
    
    // Show all files
    list.innerHTML = state.products.map(p => 
        `<div>${escapeHtml(p.filename)}${p.category ? ` <span style="color: #667eea;">[${p.category}]</span>` : ''}</div>`
    ).join('');
    
    // Remove the "Show All" button
    const button = list.nextElementSibling;
    if (button && button.querySelector('button')) {
        button.remove();
    }
    
    showToast(`Showing all ${totalCount} files`, 'success');
}

/**
 * Download CSV template from CSV Builder Step 1
 * Generates CSV with pre-filled filenames and categories for user to complete
 * Uses filename-based linking by default
 */
function downloadCsvTemplateFromBuilder(totalCount) {
    try {
        // Generate CSV header
        const headers = ['filename', 'category', 'sku', 'name', 'price', 'price_history', 'performance_history'];

        // Generate CSV rows from state.products
        const rows = state.products.map(product =>
            [
                `"${product.filename.replace(/"/g, '""')}"`,  // Filename (required for linking)
                `"${product.category.replace(/"/g, '""')}"`,  // Category (pre-filled)
                '',  // SKU (user fills)
                '',  // Name (user fills)
                '',  // Price (user fills)
                '',  // Price history (user fills)
                ''   // Performance history (user fills)
            ].join(',')
        );

        const csv = [
            headers.join(','),
            ...rows
        ].join('\n');

        const filename = `csv-template-${new Date().toISOString().split('T')[0]}.csv`;

        // Try pywebview first
        if (window.pywebview && window.pywebview.api && window.pywebview.api.save_file_auto) {
            window.pywebview.api.save_file_auto(csv, filename)
                .then(() => {
                    showToast('CSV template downloaded! Fill in the empty columns (sku, name, price, etc.) and upload in Step 2.', 'success');
                })
                .catch(error => {
                    console.error('Pywebview save failed:', error);
                    downloadCsvBrowser(csv, filename);
                });
        } else {
            // Browser download fallback
            downloadCsvBrowser(csv, filename);
        }
    } catch (error) {
        console.error('Error downloading CSV template:', error);
        showToast('Error downloading CSV template', 'error');
    }
}

/**
 * Download CSV via browser (fallback)
 */
function downloadCsvBrowser(csvContent, filename) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    try {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        showToast('CSV template downloaded! Fill in the empty columns (sku, name, price, etc.) and upload in Step 2.', 'success');
    } catch (error) {
        console.error('Browser download failed:', error);
        showToast('Download failed', 'error');
    } finally {
        URL.revokeObjectURL(url);
    }
}
