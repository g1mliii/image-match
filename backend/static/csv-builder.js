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

// Track intervals and channels for cleanup (Fix #11, #12)
let catalogPollingInterval = null;
let catalogChannel = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeStep1();
    loadFromLocalStorage();
    checkForMainAppData();
    initCatalogChangeListener();
});

// Listen for catalog changes from Catalog Manager
function initCatalogChangeListener() {
    // Listen via BroadcastChannel
    try {
        catalogChannel = new BroadcastChannel('catalog_changes');
        catalogChannel.onmessage = (event) => {
            handleCatalogChange(event.data);
        };
    } catch (e) {
        // BroadcastChannel not supported, use polling
        catalogPollingInterval = setInterval(checkCatalogChanges, 2000);
    }
    
    // Also check on visibility change (when user switches back to this tab)
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            checkCatalogChanges();
        }
    });
}

// Cleanup on window unload (Fix #11, #12)
window.addEventListener('beforeunload', () => {
    // Clear polling interval
    if (catalogPollingInterval) {
        clearInterval(catalogPollingInterval);
        catalogPollingInterval = null;
    }
    
    // Close BroadcastChannel
    if (catalogChannel) {
        try {
            catalogChannel.close();
            catalogChannel = null;
        } catch (e) {
            console.warn('Failed to close BroadcastChannel:', e);
        }
    }
});

// Check for catalog changes via sessionStorage
function checkCatalogChanges() {
    const changeData = sessionStorage.getItem('catalogManagerChange');
    if (changeData) {
        try {
            const change = JSON.parse(changeData);
            // Only process recent changes (within last 30 seconds)
            if (Date.now() - change.timestamp < 30000) {
                handleCatalogChange(change);
            }
        } catch (e) {
            console.error('Error parsing catalog change:', e);
        }
    }
}

// Handle catalog change notification
function handleCatalogChange(change) {
    if (!change || !change.action) return;
    
    // Show notification to user
    const actions = {
        'delete': 'A product was deleted',
        'bulk_delete': `${change.details?.count || 'Multiple'} products were deleted`,
        'cleanup': 'Database cleanup was performed',
        'category_cleanup': 'Categories were deleted',
        'date_cleanup': 'Old products were deleted',
        'snapshot_change': 'Catalog snapshot was changed',
        'snapshot_deleted': 'A snapshot was deleted',
        'catalog_loaded': `Catalog "${change.details?.name || 'snapshot'}" was loaded (${change.details?.productCount || 0} products)`
    };
    
    const message = actions[change.action] || 'Catalog was modified';
    
    // If catalog was loaded, this is a major change
    if (change.action === 'catalog_loaded') {
        showToast(`${message}. Refresh if you need updated data.`, 'info');
    } else if (state.products.length > 0) {
        // If we have products loaded that might be affected, warn user
        showToast(`${message} in Catalog Manager. Your CSV data may be out of sync.`, 'warning');
    }
}

// ===== STEP 1: Upload Images =====
function initializeStep1() {
    const dropZone = document.getElementById('imageDropZone');
    const input = document.getElementById('imageInput');
    const browseBtn = document.getElementById('imageBrowseBtn');
    const nextBtn = document.getElementById('nextToLink');

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        input.click();
    });

    dropZone.addEventListener('click', () => input.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleImageFiles(Array.from(e.dataTransfer.files));
    });

    input.addEventListener('change', (e) => {
        handleImageFiles(Array.from(e.target.files));
    });

    nextBtn.addEventListener('click', () => {
        goToStep(2);
    });
    
    // Initialize import file input
    const importInput = document.getElementById('importFileInput');
    if (importInput) {
        importInput.addEventListener('change', handleImportFile);
    }
    
    // Initialize import completed input
    const importCompletedInput = document.getElementById('importCompletedInput');
    if (importCompletedInput) {
        importCompletedInput.addEventListener('change', handleImportCompletedFile);
    }
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
    
    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearCsvBuilderUpload()" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>✓ ${imageFiles.length} images loaded</h4>
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
        } else {
            // Assume order: filename, category, sku, name, price
            obj = {
                filename: sanitizeField(row[0]),
                category: sanitizeField(row[1]),
                sku: sanitizeField(row[2]),
                name: sanitizeField(row[3]),
                price: sanitizeField(row[4])
            };
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
    
    // Build status message
    let statusMessage = `✓ Imported ${state.importedData.length} products from ${source}`;
    const warnings = [];
    
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
    
    if (warnings.length > 0) {
        statusMessage += `<br><span style="color: #ed8936;">⚠️ ${warnings.join(', ')}</span>`;
    }
    
    // Show import status
    document.getElementById('importStatus').style.display = 'block';
    document.getElementById('importStatusText').innerHTML = statusMessage;
    
    // Show linking panel
    document.getElementById('linkingPanel').style.display = 'block';
    document.getElementById('skipLinkingActions').style.display = 'none';
    
    // Auto-preview with default strategy
    previewLinking();
    
    showToast(`Imported ${state.importedData.length} products`, 'success');
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
        'amount': 'price'
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

function exportTemplate() {
    // Generate CSV template with filenames and empty metadata columns
    let csv = 'filename,category,sku,name,price,price_history,performance_history\n';
    
    state.products.forEach(product => {
        csv += `${product.filename},${product.category || ''},,,,\n`;
    });
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `product-template_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Template exported! Fill it in Excel and re-import.', 'success');
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
            }
            if (imported.performance_history) {
                product.performanceHistory = parsePerformanceHistory(imported.performance_history);
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
    let message = `✓ Imported metadata for ${matched} product(s)`;
    if (notFound > 0) {
        message += `\n⚠️ ${notFound} filename(s) not found in uploaded images`;
    }
    if (validationWarnings.length > 0) {
        message += `\n⚠️ ${validationWarnings.length} validation warning(s)`;
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
        skipBtn.textContent = 'SKIP TO REVIEW →';
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
    document.getElementById('linkedCount').textContent = `Linked: ${matches.linked} ✓`;
    document.getElementById('unlinkedCount').textContent = `Unlinked: ${matches.unlinked} ⚠️`;
    
    // Show preview
    const previewList = document.getElementById('previewList');
    const previewItems = matches.results.slice(0, 10).map(result => {
        if (result.matched) {
            return `<div class="preview-item success">
                <span class="preview-image">${escapeHtml(result.image)}</span>
                <span class="preview-arrow">→</span>
                <span class="preview-data">${escapeHtml(result.data.sku || result.data.name || 'Matched')}</span>
            </div>`;
        } else {
            return `<div class="preview-item warning">
                <span class="preview-image">${escapeHtml(result.image)}</span>
                <span class="preview-arrow">✗</span>
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
    
    // Remove extension and clean filename
    const filenameSKU = product.filename
        .replace(/\.[^.]+$/, '') // Remove extension
        .trim();
    
    if (!filenameSKU) return null;
    
    return state.importedData.find(data => {
        if (!data || !data.sku) return false;
        const dataSKU = String(data.sku).trim();
        return dataSKU.toLowerCase() === filenameSKU.toLowerCase();
    });
}

function linkByFilenameContainsSKU(product) {
    if (!product || !product.filename) return null;
    
    try {
        const pattern = new RegExp(state.skuPattern, 'i');
        const match = product.filename.match(pattern);
        if (match) {
            const extractedSKU = match[0];
            return state.importedData.find(data => {
                if (!data || !data.sku) return false;
                const dataSKU = String(data.sku).trim();
                return dataSKU.toLowerCase() === extractedSKU.toLowerCase();
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
    
    const folderSKU = String(product.category).trim();
    if (!folderSKU) return null;
    
    return state.importedData.find(data => {
        if (!data || !data.sku) return false;
        const dataSKU = String(data.sku).trim();
        return dataSKU.toLowerCase() === folderSKU.toLowerCase();
    });
}

function linkByFuzzyName(product) {
    if (!product || !product.filename) return null;
    
    // Clean and normalize filename
    const cleanFilename = normalizeForFuzzyMatch(
        product.filename.replace(/\.[^.]+$/, '') // Remove extension
    );
    
    if (!cleanFilename || cleanFilename.length < 2) return null;
    
    // Find best match with scoring
    let bestMatch = null;
    let bestScore = 0;
    
    for (const data of state.importedData) {
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
    return historyStr.split(';').map(entry => {
        const [date, sales, views, conversion_rate, revenue] = entry.split(':');
        return {
            date,
            sales: parseInt(sales) || 0,
            views: parseInt(views) || 0,
            conversion_rate: parseFloat(conversion_rate) || 0,
            revenue: parseFloat(revenue) || 0
        };
    }).filter(e => e.date);
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
                <button class="btn-icon" onclick="duplicateProduct(${index})" title="Duplicate">📋</button>
                <button class="btn-icon delete" onclick="deleteProduct(${index})" title="Delete">🗑️</button>
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
                <button class="btn-icon delete" onclick="deletePriceEntry(${i})" title="Delete">🗑️</button>
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


// Performance History Functions
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
                    <label>Date</label>
                    <input type="date" value="${entry.date}" onchange="updatePerformanceEntry(${i}, 'date', this.value)">
                </div>
                <div class="field-group">
                    <label>Sales</label>
                    <input type="number" value="${entry.sales}" onchange="updatePerformanceEntry(${i}, 'sales', this.value)" min="0">
                </div>
                <div class="field-group">
                    <label>Views</label>
                    <input type="number" value="${entry.views}" onchange="updatePerformanceEntry(${i}, 'views', this.value)" min="0">
                </div>
                <div class="field-group">
                    <label>Conversion %</label>
                    <input type="number" value="${entry.conversion_rate}" onchange="updatePerformanceEntry(${i}, 'conversion_rate', this.value)" step="0.1" min="0" max="100">
                </div>
                <div class="field-group">
                    <label>Revenue</label>
                    <input type="number" value="${entry.revenue}" onchange="updatePerformanceEntry(${i}, 'revenue', this.value)" step="0.01" min="0">
                </div>
            </div>
            <div class="history-entry-actions">
                <button class="btn-icon delete" onclick="deletePerformanceEntry(${i})" title="Delete">🗑️</button>
            </div>
        </div>
    `).join('');
}

function addPerformanceEntry() {
    const product = state.products[state.selectedProductIndex];
    const today = new Date().toISOString().split('T')[0];
    
    if (!product.performanceHistory) {
        product.performanceHistory = [];
    }

    product.performanceHistory.push({
        date: today,
        sales: 0,
        views: 0,
        conversion_rate: 0,
        revenue: 0
    });

    renderPerformanceHistory();
    saveState();
    updateProductProgress();
}

function updatePerformanceEntry(index, field, value) {
    const product = state.products[state.selectedProductIndex];
    product.performanceHistory[index][field] = parseFloat(value) || 0;
    
    // Auto-calculate conversion rate if sales and views are provided
    if (field === 'sales' || field === 'views') {
        const entry = product.performanceHistory[index];
        if (entry.views > 0) {
            entry.conversion_rate = ((entry.sales / entry.views) * 100).toFixed(2);
        }
    }
    
    renderPerformanceHistory();
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
        
        lines.forEach(line => {
            const parts = line.split(/[\t,;]/).map(s => s.trim());
            
            if (parts.length >= 5) {
                // Format: date, sales, views, conversion, revenue
                product.performanceHistory.push({
                    date: parts[0],
                    sales: parseInt(parts[1]) || 0,
                    views: parseInt(parts[2]) || 0,
                    conversion_rate: parseFloat(parts[3]) || 0,
                    revenue: parseFloat(parts[4]) || 0
                });
            } else if (parts.length >= 4) {
                // Format without date: sales, views, conversion, revenue
                const today = new Date();
                today.setMonth(today.getMonth() - product.performanceHistory.length);
                product.performanceHistory.push({
                    date: today.toISOString().split('T')[0],
                    sales: parseInt(parts[0]) || 0,
                    views: parseInt(parts[1]) || 0,
                    conversion_rate: parseFloat(parts[2]) || 0,
                    revenue: parseFloat(parts[3]) || 0
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
    
    return performanceHistory
        .map(entry => `${entry.date}:${entry.sales}:${entry.views}:${entry.conversion_rate}:${entry.revenue}`)
        .join(';');
}

function downloadCSV() {
    const csv = generateCSV();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `products_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('CSV downloaded successfully!', 'success');
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


// ===== Navigation & State Management =====
function goToStep(step) {
    // Hide all sections
    for (let i = 1; i <= 5; i++) {
        document.getElementById(`step${i}`).style.display = 'none';
        document.querySelector(`.progress-step[data-step="${i}"]`).classList.remove('active', 'completed');
    }
    
    // Show current step
    document.getElementById(`step${step}`).style.display = 'block';
    document.querySelector(`.progress-step[data-step="${step}"]`).classList.add('active');
    
    // Mark previous steps as completed
    for (let i = 1; i < step; i++) {
        document.querySelector(`.progress-step[data-step="${i}"]`).classList.add('completed');
    }
    
    state.currentStep = step;
    
    // Initialize step-specific content
    if (step === 2) {
        // Link step - reset panels
        document.getElementById('linkingPanel').style.display = 'none';
        document.getElementById('manualLinkingPanel').style.display = 'none';
        document.getElementById('skipLinkingActions').style.display = 'block';
    } else if (step === 3) {
        renderProductsTable();
    } else if (step === 4) {
        populateProductSelector();
        updateProductProgress();
    } else if (step === 5) {
        refreshPreview();
    }
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Undo/Redo functionality
function saveStateForUndo() {
    state.undoStack.push(JSON.stringify(state.products));
    state.redoStack = []; // Clear redo stack on new action
    
    // Limit undo stack size
    if (state.undoStack.length > 50) {
        state.undoStack.shift();
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

// Auto-save to localStorage
function saveState() {
    localStorage.setItem('csvBuilderState', JSON.stringify({
        products: state.products,
        currentStep: state.currentStep,
        timestamp: new Date().toISOString()
    }));
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
    }
    
    state.autoSaveTimer = setTimeout(() => {
        saveState();
        showAutoSaveIndicator();
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
document.addEventListener('keydown', (e) => {
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
});

// Handle page unload
window.addEventListener('beforeunload', (e) => {
    if (state.products.length > 0) {
        saveState();
    }
});


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
            <h4>✓ ${files.length} images loaded from Main App</h4>
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
        
        // Add "Send to App" button in Step 4
        addSendToAppButton(source);
        
        showToast(`${files.length} images loaded from Main App. Add metadata and send back!`, 'success');
        
        // Clear sessionStorage
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

// Send CSV data back to main app
function sendToMainApp() {
    if (!window.opener) {
        showToast('Main app window not found. Please download CSV and upload manually.', 'error');
        return;
    }
    
    const csv = generateCSV();
    const source = state.mainAppSource;
    
    // Send message to parent window
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
                <h4>✓ ${state.products.length} products loaded from saved session</h4>
            `;
            info.classList.add('show');
            showToast('Saved work loaded successfully', 'success');
        }
    }
    
    // Close modal
    const modal = document.getElementById('loadSavedWorkModal');
    if (modal) modal.remove();
    window.savedWorkData = null;
}

function dismissSavedWork() {
    // Clear saved state
    localStorage.removeItem('csvBuilderState');
    
    // Close modal
    const modal = document.getElementById('loadSavedWorkModal');
    if (modal) modal.remove();
    window.savedWorkData = null;
    
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
