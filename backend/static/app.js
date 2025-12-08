
// State
let historicalFiles = [];
let newFiles = [];
let historicalCsv = null;
let newCsv = null;
let historicalProducts = [];
let newProducts = [];
let matchResults = [];

// Mode state
let historicalAdvancedMode = false;
let newAdvancedMode = false;
let historicalMode = 'visual'; // 'visual', 'metadata', or 'hybrid'
let newMode = 'visual'; // 'visual', 'metadata', or 'hybrid'

// Advanced features state
// Note: CLIP mode doesn't need color/shape/texture weights - handled by AI
let searchQuery = '';
let filterCategory = 'all';
let filterDuplicatesOnly = false;
let sortBy = 'similarity'; // similarity, price, performance
let sortOrder = 'desc';

// Dynamic result filters
let dynamicThreshold = 30;  // Minimum 30% threshold
let dynamicLimit = 10;

// Similarity weights for matching (default values for CLIP)
let similarityWeights = {
    color: 0.33,
    shape: 0.33,
    texture: 0.34
};

// Pagination state
let currentPage = 1;
const RESULTS_PER_PAGE = 20; // Show 20 products per page

// Undo/Redo state
let historyStack = [];
let historyIndex = -1;
const MAX_HISTORY = 50;

// Retry configuration
const RETRY_CONFIG = {
    maxRetries: 3,
    initialDelay: 1000, // 1 second
    maxDelay: 10000, // 10 seconds
    backoffMultiplier: 2
};

// ============================================================================
// MEMORY LEAK FIX #3 & #4: Event Listener and Blob URL Management
// ============================================================================

// Track event listeners for cleanup (Fix #3)
const eventListeners = {
    historical: [],
    new: [],
    matching: [],
    results: [],
    tooltips: []
};

// Track blob URLs for cleanup (Fix #4)
const blobUrls = new Set();

// Track IntersectionObserver for cleanup
let lazyLoadObserver = null;

// Track intervals and channels for cleanup (Fix #8, #9, #10)
let stateCheckInterval = null;
let catalogPollingInterval = null;
let catalogChannel = null;

/**
 * MEMORY OPTIMIZATION: Clear all operation data after processing
 * Prevents state arrays from growing unbounded (50-100MB+ with large catalogs)
 */
function clearOperationData() {
    historicalProducts = [];
    newProducts = [];
    matchResults = [];
    currentPage = 1;
    console.log('✓ Operation data cleared (freed ~50-100MB)');
}

/**
 * Add event listener with tracking for cleanup
 * @param {Element} element - DOM element
 * @param {string} event - Event name
 * @param {Function} handler - Event handler
 * @param {string} category - Category for cleanup (historical, new, matching, results, tooltips)
 */
function addTrackedListener(element, event, handler, category = 'general') {
    if (!element) return;
    
    element.addEventListener(event, handler);
    
    // Store for cleanup
    if (!eventListeners[category]) {
        eventListeners[category] = [];
    }
    eventListeners[category].push({ element, event, handler });
}

/**
 * Remove all tracked event listeners for a category
 * @param {string} category - Category to clean up
 */
function removeTrackedListeners(category) {
    if (!eventListeners[category]) return;
    
    eventListeners[category].forEach(({ element, event, handler }) => {
        try {
            element.removeEventListener(event, handler);
        } catch (e) {
            console.warn('Failed to remove listener:', e);
        }
    });
    
    eventListeners[category] = [];
}

/**
 * Create blob URL from fetch response and track for cleanup
 * @param {string} url - URL to fetch
 * @returns {Promise<string>} Blob URL
 */
async function createTrackedBlobUrl(url) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        
        // Track for cleanup
        blobUrls.add(blobUrl);
        
        return blobUrl;
    } catch (error) {
        console.error('Failed to create blob URL:', error);
        throw error;
    }
}

/**
 * Revoke all tracked blob URLs to free memory
 */
function revokeAllBlobUrls() {
    let count = 0;
    blobUrls.forEach(url => {
        try {
            URL.revokeObjectURL(url);
            count++;
        } catch (e) {
            console.warn('Failed to revoke blob URL:', e);
        }
    });
    
    blobUrls.clear();
    
    if (count > 0) {
        console.log(`✓ Revoked ${count} blob URLs, freed ~${(count * 0.5).toFixed(1)}MB`);
    }
}

/**
 * Cleanup all memory leaks (event listeners, blob URLs, observers, intervals, channels)
 */
function cleanupMemory() {
    console.log('🧹 Starting memory cleanup...');
    
    // Stop state checking interval (Fix #8)
    if (typeof stopStateChecking === 'function') {
        stopStateChecking();
    }
    
    // Clear catalog polling interval (Fix #9)
    if (catalogPollingInterval) {
        clearInterval(catalogPollingInterval);
        catalogPollingInterval = null;
        console.log('✓ Cleared catalog polling interval');
    }
    
    // Close BroadcastChannel (Fix #10)
    if (catalogChannel) {
        try {
            catalogChannel.close();
            catalogChannel = null;
            console.log('✓ Closed BroadcastChannel');
        } catch (e) {
            console.warn('Failed to close BroadcastChannel:', e);
        }
    }
    
    // Remove all tracked event listeners
    Object.keys(eventListeners).forEach(category => {
        removeTrackedListeners(category);
    });
    
    // Revoke all blob URLs
    revokeAllBlobUrls();
    
    // Disconnect lazy load observer
    if (lazyLoadObserver) {
        lazyLoadObserver.disconnect();
        lazyLoadObserver = null;
    }
    
    // Clear state arrays
    matchResults = [];
    
    console.log('✓ Memory cleanup complete');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initHistoricalUpload();
    initNewUpload();
    initMatching();
    initResults();
    initTooltips();
    initGPUStatus();
    initCatalogOptions();
});

// Cleanup on page unload - CRITICAL FIX
window.addEventListener('beforeunload', async () => {
    cleanupMemory();
    
    // Clean up session data (delete matches) on app close
    try {
        await fetch('/api/session/cleanup', { method: 'POST' });
        console.log('[CLEANUP] Session data cleaned up');
    } catch (error) {
        console.error('[CLEANUP] Error cleaning up session:', error);
    }
});

// Historical Catalog Upload
function initHistoricalUpload() {
    const dropZone = document.getElementById('historicalDropZone');
    const input = document.getElementById('historicalInput');
    const browseBtn = document.getElementById('historicalBrowseBtn');
    const csvInput = document.getElementById('historicalCsvInput');
    const processBtn = document.getElementById('processHistoricalBtn');

    // Use tracked listeners to prevent memory leaks
    addTrackedListener(browseBtn, 'click', (e) => {
        e.stopPropagation();
        input.click();
    }, 'historical');

    addTrackedListener(dropZone, 'click', () => input.click(), 'historical');

    addTrackedListener(dropZone, 'dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    }, 'historical');

    addTrackedListener(dropZone, 'dragleave', () => {
        dropZone.classList.remove('drag-over');
    }, 'historical');

    addTrackedListener(dropZone, 'dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    }, 'historical');

    addTrackedListener(dropZone, 'drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        dropZone.classList.add('drop-success');
        setTimeout(() => dropZone.classList.remove('drop-success'), 500);
        handleHistoricalFiles(Array.from(e.dataTransfer.files));
    }, 'historical');

    addTrackedListener(input, 'change', (e) => {
        handleHistoricalFiles(Array.from(e.target.files));
    }, 'historical');

    addTrackedListener(csvInput, 'change', (e) => {
        if (e.target.files.length) {
            historicalCsv = e.target.files[0];
            showToast('CSV loaded for historical products', 'success');
            
            // Enable process button in advanced mode when CSV is uploaded
            if (historicalAdvancedMode) {
                processBtn.disabled = false;
            }
        }
    }, 'historical');

    // Don't set up the click handler here - it's handled by processHistoricalCatalogWithOptions
    // processBtn.addEventListener('click', processHistoricalCatalog);
}

function handleHistoricalFiles(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));

    if (imageFiles.length === 0) {
        showToast('No image files found in folder', 'error');
        return;
    }

    // Extract categories from folder structure
    const filesWithCategories = imageFiles.map(file => {
        const category = extractCategoryFromPath(file.webkitRelativePath || file.name);
        return { file, category };
    });

    historicalFiles = filesWithCategories;

    // Count categories
    const categoryCount = {};
    filesWithCategories.forEach(({ category }) => {
        if (category) {
            categoryCount[category] = (categoryCount[category] || 0) + 1;
        }
    });

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';

    const info = document.getElementById('historicalInfo');
    const displayLimit = 50;
    const hasMore = imageFiles.length > displayLimit;
    
    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearFolderUpload('historical')" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>✓ ${imageFiles.length} images loaded</h4>
        ${categorySummary}
        <div class="file-list" id="historicalFileList">
            ${filesWithCategories.slice(0, displayLimit).map(({ file, category }) => 
                `<div>${escapeHtml(file.name)}${category ? ` <span style="color: #667eea;">[${category}]</span>` : ''}</div>`
            ).join('')}
        </div>
        ${hasMore ? `
            <div style="text-align: center; margin-top: 10px;">
                <button class="btn" onclick="showAllFiles('historical', ${imageFiles.length})" style="font-size: 12px; padding: 5px 15px;">
                    SHOW ALL ${imageFiles.length} FILES
                </button>
            </div>
        ` : ''}
    `;
    info.classList.add('show');

    // Enable process button based on mode
    if (historicalAdvancedMode) {
        // In advanced mode, only enable if CSV is uploaded (images optional)
        document.getElementById('processHistoricalBtn').disabled = !historicalCsv;
        
        // If CSV not uploaded yet, prompt user
        if (!historicalCsv) {
            setTimeout(() => promptCsvBuilder('historical'), 500);
        }
    } else {
        // In simple mode, enable immediately
        document.getElementById('processHistoricalBtn').disabled = false;
    }
    
    showToast(`${imageFiles.length} historical images loaded from ${Object.keys(categoryCount).length || 0} categories`, 'success');
}

async function processHistoricalCatalog() {
    const statusDiv = document.getElementById('historicalStatus');
    const processBtn = document.getElementById('processHistoricalBtn');

    statusDiv.classList.add('show');
    processBtn.disabled = true;
    showLoadingSpinner(processBtn, true);

    // Parse CSV if provided
    let categoryMap = {};
    if (historicalCsv) {
        try {
            categoryMap = await parseCsv(historicalCsv);
        } catch (error) {
            showToast('Failed to parse CSV file. Please check the format.', 'error');
            processBtn.disabled = false;
            showLoadingSpinner(processBtn, false);
            return;
        }
    }

    // In Mode 2 (CSV only), process CSV rows instead of image files
    const csvOnlyMode = historicalAdvancedMode && historicalFiles.length === 0 && Object.keys(categoryMap).length > 0;
    const itemsToProcess = csvOnlyMode ? Object.keys(categoryMap) : historicalFiles;
    const totalItems = csvOnlyMode ? Object.keys(categoryMap).length : historicalFiles.length;
    
    statusDiv.innerHTML = '<h4>Processing historical catalog...</h4><div class="progress-bar"><div class="progress-fill" id="historicalProgress"></div></div><p id="historicalProgressText">0 of ' + totalItems + ' processed</p><div class="spinner-inline"></div>';

    // Load existing products from DB if using "add_to_existing" option
    const loadOption = getCatalogLoadOption();
    if (loadOption === 'add_to_existing') {
        try {
            console.log('[ADD_TO_EXISTING] Loading existing historical products from DB...');
            const response = await fetch('/api/catalog/products?type=historical&limit=10000');
            if (response.ok) {
                const data = await response.json();
                historicalProducts = data.products.map(p => ({
                    id: p.id,
                    filename: p.filename,
                    category: p.category,
                    sku: p.sku,
                    name: p.product_name,
                    is_historical: true,
                    hasFeatures: p.has_features
                }));
                console.log(`[ADD_TO_EXISTING] Loaded ${historicalProducts.length} existing products`);
            } else {
                console.warn('[ADD_TO_EXISTING] Failed to load existing products, starting fresh');
                historicalProducts = [];
            }
        } catch (error) {
            console.warn('[ADD_TO_EXISTING] Error loading existing products:', error);
            historicalProducts = [];
        }
    } else {
        // For 'replace' option, start with empty array (products already deleted)
        historicalProducts = [];
    }
    const progressFill = document.getElementById('historicalProgress');
    const progressText = document.getElementById('historicalProgressText');
    
    let successCount = 0;
    let failedCount = 0;
    const failedItems = [];

    // Separate Mode 1/3 (images) from Mode 2 (CSV only)
    const imageItems = [];
    const csvOnlyItems = [];
    
    for (let i = 0; i < itemsToProcess.length; i++) {
        if (csvOnlyMode) {
            csvOnlyItems.push(i);
        } else {
            imageItems.push(i);
        }
    }
    
    // Process CSV-only items first (Mode 2) - STREAM in batches of 100
    if (csvOnlyItems.length > 0) {
        console.log(`[BATCH-METADATA] Preparing to stream create ${csvOnlyItems.length} metadata products`);
        
        try {
            // Step 1: Validate all items and collect into batch
            const productsToCreate = [];
            const itemIndexMap = []; // Map batch index back to original item index
            
            for (const i of csvOnlyItems) {
                const fileName = itemsToProcess[i];
                const metadata = categoryMap[fileName];
                const category = metadata.category;
                
                // Validate required fields
                const hasValidSku = metadata.sku && metadata.sku.trim() !== '';
                const hasValidName = metadata.name && metadata.name.trim() !== '';
                
                if (!hasValidSku || !hasValidName) {
                    console.warn(`Skipping row ${i + 1} (${fileName}): Missing required fields (SKU or Name)`);
                    failedCount++;
                    failedItems.push({ row: i + 1, fileName, reason: 'Missing SKU or Name' });
                    continue;
                }
                
                // Add to batch
                productsToCreate.push({
                    sku: metadata.sku,
                    product_name: metadata.name || fileName,
                    category: category,
                    is_historical: true,
                    performance_history: metadata.performanceHistory
                });
                itemIndexMap.push({ i, fileName, metadata, category });
            }
            
            if (productsToCreate.length === 0) {
                console.warn('[BATCH-METADATA] No valid products to create');
            } else {
                // Step 2: Stream batch create in chunks of 100
                const STREAM_BATCH_SIZE = 100;
                const totalBatches = Math.ceil(productsToCreate.length / STREAM_BATCH_SIZE);
                
                console.log(`[BATCH-METADATA] Streaming ${productsToCreate.length} products in ${totalBatches} batch(es) of ${STREAM_BATCH_SIZE}`);
                
                for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                    const batchStart = batchIdx * STREAM_BATCH_SIZE;
                    const batchEnd = Math.min(batchStart + STREAM_BATCH_SIZE, productsToCreate.length);
                    const batchProducts = productsToCreate.slice(batchStart, batchEnd);
                    
                    console.log(`[BATCH-METADATA] Batch ${batchIdx + 1}/${totalBatches}: Creating ${batchProducts.length} products`);
                    progressText.textContent = `Creating batch ${batchIdx + 1}/${totalBatches} (${batchProducts.length} products)...`;
                    
                    const response = await fetchWithRetry('/api/products/metadata/batch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ products: batchProducts })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok && data.product_ids) {
                        // Process results for this batch
                        successCount += data.product_ids.length;
                        
                        for (let j = 0; j < data.product_ids.length; j++) {
                            const productId = data.product_ids[j];
                            const itemInfo = itemIndexMap[batchStart + j];
                            
                            historicalProducts.push({
                                id: productId,
                                filename: itemInfo.fileName,
                                category: itemInfo.category,
                                sku: itemInfo.metadata.sku,
                                name: itemInfo.metadata.name,
                                hasFeatures: false,
                                hasPriceHistory: false
                            });
                        }
                        
                        console.log(`[BATCH-METADATA] Batch ${batchIdx + 1}/${totalBatches} successful: ${data.product_ids.length} created`);
                    } else {
                        failedCount += batchProducts.length;
                        const errorMsg = getUserFriendlyError(data.error_code || 'BATCH_ERROR', data.error, data.suggestion);
                        failedItems.push({ row: 'batch', fileName: 'all', reason: errorMsg });
                        console.error(`[BATCH-METADATA] Batch ${batchIdx + 1}/${totalBatches} failed:`, data);
                    }
                }
                
                console.log(`[BATCH-METADATA] ✓ Successfully created ${successCount} products in ${totalBatches} batches`);
            }
        } catch (error) {
            failedCount += csvOnlyItems.length;
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            failedItems.push({ row: 'batch', fileName: 'all', reason: error.message });
            console.error('[BATCH-METADATA] Batch creation error:', error);
        }
        
        const progress = ((successCount + failedCount) / totalItems) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${successCount + failedCount} of ${totalItems} processed (${successCount} success, ${failedCount} failed)`;
    }
    
    // Process image items in batch (Mode 1/3) - GPU batch processing
    if (imageItems.length > 0) {
        console.log(`[BATCH-UPLOAD] Preparing to batch upload ${imageItems.length} images`);
        
        try {
            // OPTIMIZATION: Stream batch uploads every 100 images
            // This overlaps file I/O with network requests instead of waiting for all files to load
            const STREAM_BATCH_SIZE = 100;
            const totalBatches = Math.ceil(imageItems.length / STREAM_BATCH_SIZE);
            
            console.log(`[BATCH-UPLOAD] Streaming ${imageItems.length} images in ${totalBatches} batch(es) of ${STREAM_BATCH_SIZE}`);
            progressText.textContent = `Uploading ${imageItems.length} images in streaming batches...`;
            
            // Process each batch
            for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                const batchStart = batchIdx * STREAM_BATCH_SIZE;
                const batchEnd = Math.min(batchStart + STREAM_BATCH_SIZE, imageItems.length);
                const batchItems = imageItems.slice(batchStart, batchEnd);
                
                console.log(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches}: Preparing ${batchItems.length} images`);
                
                // Collect image data for this batch
                const batchFormData = new FormData();
                const categories = [];
                const productNames = [];
                const skus = [];
                
                for (const i of batchItems) {
                    const fileObj = historicalFiles[i];
                    const file = fileObj.file;
                    const category = fileObj.category;
                    const metadata = categoryMap[file.name] || {};
                    
                    // Append image file
                    batchFormData.append('images', file);
                    
                    // Collect metadata
                    const finalCategory = metadata.category || category;
                    categories.push(finalCategory || null);
                    productNames.push(metadata.name || file.name);
                    skus.push(metadata.sku || null);
                }
                
                // Append metadata as JSON arrays
                batchFormData.append('categories', JSON.stringify(categories));
                batchFormData.append('product_names', JSON.stringify(productNames));
                batchFormData.append('skus', JSON.stringify(skus));
                batchFormData.append('is_historical', 'true');
                
                console.log(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches}: Sending ${batchItems.length} images`);
                progressText.textContent = `Uploading batch ${batchIdx + 1}/${totalBatches} (${batchItems.length} images)...`;
                
                // Send this batch
                const response = await fetchWithRetry('/api/products/batch-upload', {
                    method: 'POST',
                    body: batchFormData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    console.log(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches} successful: ${data.successful} uploaded`);
                    
                    // Process results for this batch
                    for (let resultIdx = 0; resultIdx < data.results.length; resultIdx++) {
                        const result = data.results[resultIdx];
                        const batchItemIdx = batchItems[resultIdx];
                        const fileObj = historicalFiles[batchItemIdx];
                        const file = fileObj.file;
                        const category = fileObj.category;
                        const metadata = categoryMap[file.name] || {};
                        const finalCategory = metadata.category || category;
                        
                        if (result.status === 'success') {
                            successCount++;
                            historicalProducts.push({
                                id: result.product_id,
                                filename: file.name,
                                category: finalCategory,
                                sku: metadata.sku,
                                name: metadata.name,
                                hasFeatures: true,  // Batch upload extracts features
                                hasPriceHistory: false
                            });
                            
                            // Upload price history if present
                            if (metadata.priceHistory && metadata.priceHistory.length > 0) {
                                try {
                                    const priceResponse = await fetchWithRetry(`/api/products/${result.product_id}/price-history`, {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ prices: metadata.priceHistory })
                                    });
                                    
                                    if (!priceResponse.ok) {
                                        console.warn(`Failed to upload price history for ${file.name}: ${priceResponse.status}`);
                                    }
                                } catch (error) {
                                    console.warn(`Failed to upload price history for ${file.name}:`, error);
                                }
                            }
                        } else {
                            failedCount++;
                            const batchItemIdx = batchItems[resultIdx];
                            failedItems.push({ row: batchItemIdx + 1, fileName: file.name, reason: result.error || 'Unknown error' });
                            console.error(`Failed to process ${file.name}:`, result);
                        }
                    }
                } else {
                    // Batch upload failed - mark all items in this batch as failed
                    console.error(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches} failed:`, data);
                    for (const i of batchItems) {
                        const fileObj = historicalFiles[i];
                        failedCount++;
                        failedItems.push({ row: i + 1, fileName: fileObj.file.name, reason: data.error || 'Batch upload failed' });
                    }
                }
            }
        } catch (error) {
            // Network error - mark all as failed
            console.error(`[BATCH-UPLOAD] Network error:`, error);
            for (const i of imageItems) {
                const fileObj = historicalFiles[i];
                failedCount++;
                failedItems.push({ row: i + 1, fileName: fileObj.file.name, reason: error.message });
            }
        }
        
        const progress = ((successCount + failedCount) / totalItems) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${successCount + failedCount} of ${totalItems} processed (${successCount} success, ${failedCount} failed)`;
    }

    const catalogOption = getCatalogLoadOption();
    const existingCount = catalogOption === 'add_to_existing' ? historicalProducts.filter(p => p.id).length - totalItems : 0;
    const newlyUploaded = historicalProducts.length - existingCount;
    const withoutMetadata = historicalProducts.filter(p => !p.category && !p.sku).length;
    
    let statusMsg = `<h4>✓ Historical catalog processed</h4>`;
    statusMsg += `<p><strong>${successCount} successful</strong>, ${failedCount} failed</p>`;
    
    if (catalogOption === 'add_to_existing' && existingCount > 0) {
        statusMsg += `<p>${successCount} total products ready for matching (${existingCount} existing + ${newlyUploaded} newly added)</p>`;
    } else {
        statusMsg += `<p>${successCount} products ready for matching</p>`;
    }
    
    // Show failed items summary if any
    if (failedItems.length > 0 && failedItems.length <= 10) {
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>Failed items:</strong><ul style="margin: 5px 0; padding-left: 20px;">`;
        failedItems.forEach(item => {
            statusMsg += `<li>Row ${item.row} (${item.fileName}): ${item.reason}</li>`;
        });
        statusMsg += `</ul></div>`;
    } else if (failedItems.length > 10) {
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>${failedItems.length} items failed</strong> - check console for details</div>`;
        console.log('Failed items:', failedItems);
    }
    
    statusDiv.innerHTML = statusMsg;

    showToast(`Historical catalog ready: ${successCount} products`, 'success');
    showLoadingSpinner(processBtn, false);
    
    // MEMORY OPTIMIZATION: Clear operation data to free 50-100MB
    clearOperationData();

    // Show next step - force display
    const newSection = document.getElementById('newSection');
    console.log('[DEBUG] Attempting to show newSection:', newSection);
    if (newSection) {
        newSection.style.display = 'block';
        newSection.style.visibility = 'visible';
        console.log('[DEBUG] newSection display set to block, current style:', newSection.style.display);
        setTimeout(() => {
            newSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    } else {
        console.error('[ERROR] newSection element not found in DOM!');
    }
}

// New Products Upload
function initNewUpload() {
    const dropZone = document.getElementById('newDropZone');
    const input = document.getElementById('newInput');
    const browseBtn = document.getElementById('newBrowseBtn');
    const csvInput = document.getElementById('newCsvInput');
    const processBtn = document.getElementById('processNewBtn');

    // Use tracked listeners to prevent memory leaks
    addTrackedListener(browseBtn, 'click', (e) => {
        e.stopPropagation();
        input.click();
    }, 'new');

    addTrackedListener(dropZone, 'click', () => input.click(), 'new');

    addTrackedListener(dropZone, 'dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    }, 'new');

    addTrackedListener(dropZone, 'dragleave', () => {
        dropZone.classList.remove('drag-over');
    }, 'new');

    addTrackedListener(dropZone, 'dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    }, 'new');

    addTrackedListener(dropZone, 'drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        dropZone.classList.add('drop-success');
        setTimeout(() => dropZone.classList.remove('drop-success'), 500);
        handleNewFiles(Array.from(e.dataTransfer.files));
    }, 'new');

    addTrackedListener(input, 'change', (e) => {
        handleNewFiles(Array.from(e.target.files));
    }, 'new');

    addTrackedListener(csvInput, 'change', (e) => {
        if (e.target.files.length) {
            newCsv = e.target.files[0];
            showToast('CSV loaded for new products', 'success');
            
            // Enable process button in advanced mode when CSV is uploaded
            if (newAdvancedMode) {
                processBtn.disabled = false;
            }
        }
    }, 'new');

    // NOTE: processBtn click handler is set up later in processNewCatalogWithOptions (line ~4349)
    // Don't add duplicate handler here!
}

function handleNewFiles(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));

    if (imageFiles.length === 0) {
        showToast('No image files found in folder', 'error');
        return;
    }

    // Extract categories from folder structure
    const filesWithCategories = imageFiles.map(file => {
        const category = extractCategoryFromPath(file.webkitRelativePath || file.name);
        return { file, category };
    });

    newFiles = filesWithCategories;

    // Count categories
    const categoryCount = {};
    filesWithCategories.forEach(({ category }) => {
        if (category) {
            categoryCount[category] = (categoryCount[category] || 0) + 1;
        }
    });

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';

    const info = document.getElementById('newInfo');
    const displayLimit = 50;
    const hasMore = imageFiles.length > displayLimit;
    
    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearFolderUpload('new')" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>✓ ${imageFiles.length} images loaded</h4>
        ${categorySummary}
        <div class="file-list" id="newFileList">
            ${filesWithCategories.slice(0, displayLimit).map(({ file, category }) => 
                `<div>${escapeHtml(file.name)}${category ? ` <span style="color: #667eea;">[${category}]</span>` : ''}</div>`
            ).join('')}
        </div>
        ${hasMore ? `
            <div style="text-align: center; margin-top: 10px;">
                <button class="btn" onclick="showAllFiles('new', ${imageFiles.length})" style="font-size: 12px; padding: 5px 15px;">
                    SHOW ALL ${imageFiles.length} FILES
                </button>
            </div>
        ` : ''}
    `;
    info.classList.add('show');

    // Enable process button based on mode
    if (newAdvancedMode) {
        // In advanced mode, only enable if CSV is uploaded (images optional)
        document.getElementById('processNewBtn').disabled = !newCsv;
        
        // If CSV not uploaded yet, prompt user
        if (!newCsv) {
            setTimeout(() => promptCsvBuilder('new'), 500);
        }
    } else {
        // In simple mode, enable immediately
        document.getElementById('processNewBtn').disabled = false;
    }
    
    showToast(`${imageFiles.length} new product images loaded from ${Object.keys(categoryCount).length || 0} categories`, 'success');
}

async function processNewProducts() {
    const statusDiv = document.getElementById('newStatus');
    const processBtn = document.getElementById('processNewBtn');

    statusDiv.classList.add('show');
    processBtn.disabled = true;
    showLoadingSpinner(processBtn, true);

    // Parse CSV if provided
    let categoryMap = {};
    if (newCsv) {
        try {
            categoryMap = await parseCsv(newCsv);
        } catch (error) {
            showToast('Failed to parse CSV file. Please check the format.', 'error');
            processBtn.disabled = false;
            showLoadingSpinner(processBtn, false);
            return;
        }
    }

    // In Mode 2 (CSV only), process CSV rows instead of image files
    // IMPORTANT: Only use CSV-only mode if we have NO image files AND we have CSV data
    // If we have image files, always process them (regardless of mode selection)
    const hasImageFiles = newFiles && newFiles.length > 0;
    const hasCsvData = Object.keys(categoryMap).length > 0;
    const csvOnlyMode = !hasImageFiles && hasCsvData && newAdvancedMode;
    
    const itemsToProcess = hasImageFiles ? newFiles : (csvOnlyMode ? Object.keys(categoryMap) : []);
    const totalItems = itemsToProcess.length;
    
    statusDiv.innerHTML = '<h4>Processing new products...</h4><div class="progress-bar"><div class="progress-fill" id="newProgress"></div></div><p id="newProgressText">0 of ' + totalItems + ' processed</p><div class="spinner-inline"></div>';

    // Load existing products from DB if using "add_to_existing" option
    const newLoadOption = getNewCatalogLoadOption();
    if (newLoadOption === 'add_to_existing') {
        try {
            console.log('[ADD_TO_EXISTING] Loading existing new products from DB...');
            const response = await fetch('/api/catalog/products?type=new&limit=10000');
            if (response.ok) {
                const data = await response.json();
                newProducts = data.products.map(p => ({
                    id: p.id,
                    filename: p.filename,
                    category: p.category,
                    sku: p.sku,
                    name: p.product_name,
                    is_historical: false,
                    hasFeatures: p.has_features
                }));
                console.log(`[ADD_TO_EXISTING] Loaded ${newProducts.length} existing products`);
            } else {
                console.warn('[ADD_TO_EXISTING] Failed to load existing products, starting fresh');
                newProducts = [];
            }
        } catch (error) {
            console.warn('[ADD_TO_EXISTING] Error loading existing products:', error);
            newProducts = [];
        }
    } else {
        // For 'replace' option, start with empty array (products already deleted)
        newProducts = [];
    }
    const progressFill = document.getElementById('newProgress');
    const progressText = document.getElementById('newProgressText');
    
    let successCount = 0;
    let failedCount = 0;
    const failedItems = [];

    // Separate Mode 1/3 (images) from Mode 2 (CSV only)
    const imageItems = [];
    const csvOnlyItems = [];
    
    for (let i = 0; i < itemsToProcess.length; i++) {
        if (csvOnlyMode) {
            csvOnlyItems.push(i);
        } else {
            imageItems.push(i);
        }
    }
    
    // Process CSV-only items first (Mode 2) - BATCH all at once for 80-90% speedup
    if (csvOnlyItems.length > 0) {
        console.log(`[BATCH-METADATA] Preparing to batch create ${csvOnlyItems.length} metadata products`);
        
        try {
            // Step 1: Validate all items and collect into batch
            const productsToCreate = [];
            const itemIndexMap = []; // Map batch index back to original item index
            
            for (const i of csvOnlyItems) {
                const fileName = itemsToProcess[i];
                const metadata = categoryMap[fileName];
                const category = metadata.category;
                
                // Validate required fields
                const hasValidSku = metadata.sku && metadata.sku.trim() !== '';
                const hasValidName = metadata.name && metadata.name.trim() !== '';
                
                if (!hasValidSku || !hasValidName) {
                    console.warn(`Skipping row ${i + 1} (${fileName}): Missing required fields (SKU or Name)`);
                    failedCount++;
                    failedItems.push({ row: i + 1, fileName, reason: 'Missing SKU or Name' });
                    continue;
                }
                
                // Add to batch
                productsToCreate.push({
                    sku: metadata.sku,
                    product_name: metadata.name || fileName,
                    category: category,
                    is_historical: false,
                    performance_history: metadata.performanceHistory
                });
                itemIndexMap.push({ i, fileName, metadata, category });
            }
            
            if (productsToCreate.length === 0) {
                console.warn('[BATCH-METADATA] No valid products to create');
            } else {
                // Step 2: Batch create all products in one API call
                console.log(`[BATCH-METADATA] Batch creating ${productsToCreate.length} products...`);
                
                const response = await fetchWithRetry('/api/products/metadata/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ products: productsToCreate })
                });
                
                const data = await response.json();
                
                if (response.ok && data.product_ids) {
                    // Step 3: Process results
                    successCount += data.product_ids.length;
                    
                    for (let j = 0; j < data.product_ids.length; j++) {
                        const productId = data.product_ids[j];
                        const itemInfo = itemIndexMap[j];
                        
                        newProducts.push({
                            id: productId,
                            filename: itemInfo.fileName,
                            category: itemInfo.category,
                            sku: itemInfo.metadata.sku,
                            name: itemInfo.metadata.name,
                            hasFeatures: false,
                            hasPriceHistory: false
                        });
                    }
                    
                    console.log(`[BATCH-METADATA] ✓ Successfully created ${data.product_ids.length} products`);
                } else {
                    failedCount += productsToCreate.length;
                    const errorMsg = getUserFriendlyError(data.error_code || 'BATCH_ERROR', data.error, data.suggestion);
                    failedItems.push({ row: 'batch', fileName: 'all', reason: errorMsg });
                    console.error('[BATCH-METADATA] Batch creation failed:', data);
                }
            }
        } catch (error) {
            failedCount += csvOnlyItems.length;
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            failedItems.push({ row: 'batch', fileName: 'all', reason: error.message });
            console.error('[BATCH-METADATA] Batch creation error:', error);
        }
        
        const progress = ((successCount + failedCount) / totalItems) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${successCount + failedCount} of ${totalItems} processed (${successCount} success, ${failedCount} failed)`;
    }
    
    // Process image items in batch (Mode 1/3) - GPU batch processing
    if (imageItems.length > 0) {
        console.log(`[BATCH-UPLOAD] Preparing to batch upload ${imageItems.length} images`);
        
        try {
            // Collect all image data
            const batchFormData = new FormData();
            const categories = [];
            const productNames = [];
            const skus = [];
            
            for (const i of imageItems) {
                const fileObj = newFiles[i];
                const file = fileObj.file;
                const category = fileObj.category;
                const metadata = categoryMap[file.name] || {};
                
                // Append image file
                batchFormData.append('images', file);
                
                // Collect metadata
                const finalCategory = metadata.category || category;
                categories.push(finalCategory || null);
                productNames.push(metadata.name || file.name);
                skus.push(metadata.sku || null);
            }
            
            // Append metadata as JSON arrays
            batchFormData.append('categories', JSON.stringify(categories));
            batchFormData.append('product_names', JSON.stringify(productNames));
            batchFormData.append('skus', JSON.stringify(skus));
            batchFormData.append('is_historical', 'false');
            
            console.log(`[BATCH-UPLOAD] Sending ${imageItems.length} images to batch-upload endpoint`);
            progressText.textContent = `Uploading ${imageItems.length} images in batch...`;
            
            // Send all images at once to batch-upload endpoint
            const response = await fetchWithRetry('/api/products/batch-upload', {
                method: 'POST',
                body: batchFormData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                console.log(`[BATCH-UPLOAD] Batch upload successful:`, data);
                
                // Process results
                for (let resultIdx = 0; resultIdx < data.results.length; resultIdx++) {
                    const result = data.results[resultIdx];
                    const originalIdx = imageItems[resultIdx];
                    const fileObj = newFiles[originalIdx];
                    const file = fileObj.file;
                    const category = fileObj.category;
                    const metadata = categoryMap[file.name] || {};
                    const finalCategory = metadata.category || category;
                    
                    if (result.status === 'success') {
                        successCount++;
                        newProducts.push({
                            id: result.product_id,
                            filename: file.name,
                            category: finalCategory,
                            sku: metadata.sku,
                            name: metadata.name,
                            hasFeatures: true,  // Batch upload extracts features
                            hasPriceHistory: false
                        });
                        
                        // Upload price history if present
                        if (metadata.priceHistory && metadata.priceHistory.length > 0) {
                            try {
                                const priceResponse = await fetchWithRetry(`/api/products/${result.product_id}/price-history`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ prices: metadata.priceHistory })
                                });
                                
                                if (!priceResponse.ok) {
                                    console.warn(`Failed to upload price history for ${file.name}: ${priceResponse.status}`);
                                }
                            } catch (error) {
                                console.warn(`Failed to upload price history for ${file.name}:`, error);
                            }
                        }
                    } else {
                        failedCount++;
                        failedItems.push({ row: originalIdx + 1, fileName: file.name, reason: result.error || 'Unknown error' });
                        console.error(`Failed to process ${file.name}:`, result);
                    }
                }
            } else {
                // Batch upload failed - mark all as failed
                console.error(`[BATCH-UPLOAD] Batch upload failed:`, data);
                for (const i of imageItems) {
                    const fileObj = newFiles[i];
                    failedCount++;
                    failedItems.push({ row: i + 1, fileName: fileObj.file.name, reason: data.error || 'Batch upload failed' });
                }
            }
        } catch (error) {
            // Network error - mark all as failed
            console.error(`[BATCH-UPLOAD] Network error:`, error);
            for (const i of imageItems) {
                const fileObj = newFiles[i];
                failedCount++;
                failedItems.push({ row: i + 1, fileName: fileObj.file.name, reason: error.message });
            }
        }
        
        const progress = ((successCount + failedCount) / totalItems) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${successCount + failedCount} of ${totalItems} processed (${successCount} success, ${failedCount} failed)`;
    }

    // Continue with the rest of the function (status display, etc.)
    const existingCount = newLoadOption === 'add_to_existing' ? newProducts.filter(p => p.id).length - totalItems : 0;
    const newlyUploaded = newProducts.length - existingCount;
    const withoutMetadata = newProducts.filter(p => !p.category && !p.sku).length;
    
    let statusMsg = `<h4>✓ New products processed</h4>`;
    statusMsg += `<p><strong>${successCount} successful</strong>, ${failedCount} failed</p>`;
    
    if (newLoadOption === 'add_to_existing' && existingCount > 0) {
        statusMsg += `<p>${successCount} total products ready for matching (${existingCount} existing + ${newlyUploaded} newly added)</p>`;
    } else {
        statusMsg += `<p>${successCount} products ready for matching</p>`;
    }
    
    // Show failed items summary if any
    if (failedItems.length > 0 && failedItems.length <= 10) {
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>Failed items:</strong><ul style="margin: 5px 0; padding-left: 20px;">`;
        failedItems.forEach(item => {
            statusMsg += `<li>Row ${item.row} (${item.fileName}): ${item.reason}</li>`;
        });
        statusMsg += `</ul></div>`;
    } else if (failedItems.length > 10) {
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>${failedItems.length} items failed</strong> - check console for details</div>`;
        console.log('Failed items:', failedItems);
    }
    
    statusDiv.innerHTML = statusMsg;

    showToast(`New products ready: ${successCount} products`, 'success');
    showLoadingSpinner(processBtn, false);
    
    // MEMORY OPTIMIZATION: Clear operation data to free 50-100MB
    if (newLoadOption === 'replace') {
        newFiles = [];
        newCsv = null;
        categoryMap = {};
    }
    
    // Re-enable button
    processBtn.disabled = false;

    showToast(`New products ready: ${successCount} products`, 'success');
    showLoadingSpinner(processBtn, false);
    
    // MEMORY OPTIMIZATION: Clear operation data to free 50-100MB
    clearOperationData();

    // Show matching section - force display
    const matchSection = document.getElementById('matchSection');
    console.log('[DEBUG] Attempting to show matchSection:', matchSection);
    if (matchSection) {
        matchSection.style.display = 'block';
        matchSection.style.visibility = 'visible';
        console.log('[DEBUG] matchSection display set to block, current style:', matchSection.style.display);
        setTimeout(() => {
            matchSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    } else {
        console.error('[ERROR] matchSection element not found in DOM!');
    }
}

// Matching
function initMatching() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const matchBtn = document.getElementById('matchBtn');

    // Use tracked listeners to prevent memory leaks
    addTrackedListener(thresholdSlider, 'input', (e) => {
        thresholdValue.textContent = e.target.value;
    }, 'matching');

    addTrackedListener(matchBtn, 'click', startMatching, 'matching');
}

async function startMatching() {
    const threshold = parseInt(document.getElementById('thresholdSlider').value);
    const limit = parseInt(document.getElementById('limitSelect').value);
    const progressDiv = document.getElementById('matchProgress');
    const matchBtn = document.getElementById('matchBtn');
    
    // Initialize dynamic filters with the matching parameters
    dynamicThreshold = threshold;
    dynamicLimit = limit;

    progressDiv.classList.add('show');
    matchBtn.disabled = true;
    showLoadingSpinner(matchBtn, true);

    // Load new products from database if not in memory
    if (!newProducts || newProducts.length === 0) {
        console.log('[MATCHING] newProducts array empty, loading from database...');
        try {
            // CRITICAL FIX: Only load NEW products (is_historical=false), not historical ones
            const response = await fetch('/api/catalog/products?type=new&limit=10000');
            if (!response.ok) throw new Error('Failed to load new products');
            
            const data = await response.json();
            // Filter to ensure we only get new products (is_historical=false)
            newProducts = data.products
                .filter(p => !p.is_historical)  // CRITICAL: Filter out historical products
                .map(p => ({
                    id: p.id,
                    filename: p.product_name || p.filename,  // Use product_name if available (for metadata-only products)
                    product_name: p.product_name,  // Include product_name explicitly
                    category: p.category,
                    sku: p.sku,
                    name: p.product_name,
                    hasFeatures: p.has_features || false,
                    is_historical: false
                }));
            console.log(`[MATCHING] Loaded ${newProducts.length} new products from database (filtered out historical)`);
        } catch (error) {
            console.error('[MATCHING] Failed to load new products:', error);
            showToast('Failed to load new products from database', 'error');
            showLoadingSpinner(matchBtn, false);
            matchBtn.disabled = false;
            return;
        }
    }

    if (newProducts.length === 0) {
        showToast('No new products to match. Upload new products first.', 'warning');
        showLoadingSpinner(matchBtn, false);
        matchBtn.disabled = false;
        return;
    }

    progressDiv.innerHTML = '<h4>Finding matches...</h4><div class="progress-bar"><div class="progress-fill" id="matchProgressFill"></div></div><p id="matchProgressText">Batch matching 0 of ' + newProducts.length + ' products...</p><div class="spinner-inline"></div>';

    matchResults = [];
    const progressFill = document.getElementById('matchProgressFill');
    const progressText = document.getElementById('matchProgressText');

    // BATCH MATCHING OPTIMIZATION: Collect all product IDs and send one batch request
    // Instead of looping and making N requests, make 1 request for all products
    // This enables batch insert of all matches in 1-2 DB transactions instead of N
    try {
        console.log(`[BATCH-MATCHING] Starting batch matching for ${newProducts.length} products`);
        
        // AUTO-DETECT MODE: Check if products have features to determine matching mode
        const productsWithFeatures = newProducts.filter(p => p.hasFeatures).length;
        const productsWithoutFeatures = newProducts.length - productsWithFeatures;
        
        let effectiveMode = newMode;
        
        // If NO products have features, force metadata mode
        if (productsWithFeatures === 0) {
            console.log(`[BATCH-MATCHING] Auto-detected: All ${newProducts.length} products are metadata-only (no features)`);
            console.log(`[BATCH-MATCHING] Forcing Mode 2 (Metadata) instead of ${newMode}`);
            effectiveMode = 'metadata';
        }
        // If SOME products have features, use hybrid mode for best results
        else if (productsWithFeatures < newProducts.length && newMode !== 'metadata') {
            console.log(`[BATCH-MATCHING] Auto-detected: ${productsWithFeatures}/${newProducts.length} products have features`);
            console.log(`[BATCH-MATCHING] Switching to Mode 3 (Hybrid) for mixed data`);
            effectiveMode = 'hybrid';
        }
        
        console.log(`[BATCH-MATCHING] Effective mode: ${effectiveMode} (selected: ${newMode})`);
        
        // Collect all product IDs
        const productIds = newProducts.map(p => p.id);
        
        // Determine weights based on effective mode
        let visualWeight = 0;
        let metadataWeight = 0;
        
        if (effectiveMode === 'visual') {
            // Mode 1: Pure visual matching
            visualWeight = 1.0;
            metadataWeight = 0;
        } else if (effectiveMode === 'metadata') {
            // Mode 2: Pure metadata matching
            visualWeight = 0;
            metadataWeight = 1.0;
        } else if (effectiveMode === 'hybrid') {
            // Mode 3: Hybrid matching
            visualWeight = 0.5;
            metadataWeight = 0.5;
        }
        
        // Prepare batch request
        const batchPayload = {
            product_ids: productIds,
            threshold: threshold,
            limit: limit,
            match_against_all: false,
            visual_weight: visualWeight,
            metadata_weight: metadataWeight
        };
        
        console.log(`[BATCH-MATCHING] Step 1: Prepare batch request`);
        console.log(`[BATCH-MATCHING] Product IDs: ${productIds.length} products`);
        console.log(`[BATCH-MATCHING] Weights: visual=${visualWeight}, metadata=${metadataWeight}`);
        console.log(`[BATCH-MATCHING] Threshold: ${threshold}, Limit: ${limit}`);
        console.log(`[BATCH-MATCHING] Sending batch request for ${productIds.length} products (Effective Mode: ${effectiveMode})`);
        progressText.textContent = `Batch matching ${productIds.length} products...`;
        
        // Send batch request
        console.log(`[BATCH-MATCHING] Step 2: Send POST request to /api/products/batch-match`);
        const response = await fetchWithRetry('/api/products/batch-match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(batchPayload)
        });
        
        const data = await response.json();
        console.log(`[BATCH-MATCHING] Step 3: Received response - status: ${response.status}, ok: ${response.ok}`);
        console.log(`[BATCH-MATCHING] Response data:`, data);
        
        if (!response.ok) {
            console.error(`[BATCH-MATCHING] Error response:`, data);
            showToast('Batch matching failed: ' + (data.error || 'Unknown error'), 'error');
            showLoadingSpinner(matchBtn, false);
            matchBtn.disabled = false;
            return;
        }
        
        // Process batch results
        const batchResults = data.results || [];
        console.log(`[BATCH-MATCHING] Step 4: Process results - Received ${batchResults.length} results from batch`);
        
        // Create a map of product ID to product for quick lookup
        const productMap = {};
        newProducts.forEach(p => {
            productMap[p.id] = p;
        });
        
        // Process each result
        for (let i = 0; i < batchResults.length; i++) {
            const result = batchResults[i];
            const product = productMap[result.product_id];
            
            if (!product) {
                console.warn(`[BATCH-MATCHING] Product ${result.product_id} not found in product map`);
                continue;
            }
            
            const matches = result.matches || [];
            console.log(`[BATCH-MATCHING] Product ${product.id}: ${matches.length} matches found`);
            
            // Fetch price and performance history for each match
            for (const match of matches) {
                // Fetch price history
                try {
                    const priceResponse = await fetchWithRetry(`/api/products/${match.product_id}/price-history`);
                    if (priceResponse.ok) {
                        const priceData = await priceResponse.json();
                        match.priceHistory = priceData.price_history || [];
                        match.priceStatistics = priceData.statistics;
                    }
                } catch (error) {
                    console.warn(`Failed to fetch price history for match ${match.product_id}:`, error);
                    match.priceHistory = [];
                    match.priceStatistics = null;
                }
                
                // Fetch performance history (simple format)
                try {
                    const perfResponse = await fetchWithRetry(`/api/products/${match.product_id}/performance-history`);
                    if (perfResponse.ok) {
                        const perfData = await perfResponse.json();
                        // Extract just sales numbers for simple display
                        const perfHistory = perfData.performance_history || [];
                        // CRITICAL: Filter out invalid values (null, undefined, NaN) to prevent sparkline errors
                        match.performanceHistory = perfHistory
                            .map(p => p.sales)
                            .filter(s => typeof s === 'number' && !isNaN(s) && isFinite(s));
                        
                        // Calculate simple statistics
                        if (match.performanceHistory.length > 0) {
                            const total = match.performanceHistory.reduce((sum, val) => sum + val, 0);
                            const avg = total / match.performanceHistory.length;
                            const latest = match.performanceHistory[0]; // Most recent
                            const oldest = match.performanceHistory[match.performanceHistory.length - 1];
                            const trend = latest > oldest ? 'up' : latest < oldest ? 'down' : 'stable';
                            
                            match.performanceStatistics = {
                                total_sales: total,
                                average_sales: Math.round(avg),
                                sales_trend: trend
                            };
                        } else {
                            match.performanceStatistics = null;
                        }
                    }
                } catch (error) {
                    console.warn(`Failed to fetch performance history for match ${match.product_id}:`, error);
                    match.performanceHistory = [];
                    match.performanceStatistics = null;
                }
            }
            
            matchResults.push({
                product: product,
                matches: matches,
                error: result.status === 'success' ? null : result.error
            });
            
            const progress = ((i + 1) / batchResults.length) * 100;
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `${i + 1} of ${batchResults.length} products matched`;
        }
        
        console.log(`[BATCH-MATCHING] ✓ Complete! Processed ${matchResults.length} products`);
        
    } catch (error) {
        console.error(`[BATCH-MATCHING] Error:`, error);
        const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
        showToast('Batch matching failed: ' + errorMsg, 'error');
        showLoadingSpinner(matchBtn, false);
        matchBtn.disabled = false;
        return;
    }
    
    const progress = 100;
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${matchResults.length} of ${newProducts.length} products matched`;
    
    console.log(`[BATCH-MATCHING] Matching complete. Total matchResults: ${matchResults.length}`);

    progressDiv.innerHTML = '<h4>✓ Matching complete!</h4>';
    showToast('Matching complete!', 'success');
    showLoadingSpinner(matchBtn, false);

    // Show results
    displayResults();
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

// Results
function initResults() {
    // Use tracked listeners to prevent memory leaks
    addTrackedListener(document.getElementById('exportCsvBtn'), 'click', exportResults, 'results');
    addTrackedListener(document.getElementById('resetBtn'), 'click', resetApp, 'results');
    addTrackedListener(document.getElementById('modalClose'), 'click', closeModal, 'results');
}

function displayResults(resetPage = true) {
    console.log('[DISPLAY] displayResults called');
    console.log('[DISPLAY] matchResults length:', matchResults.length);
    console.log('[DISPLAY] matchResults:', matchResults);
    
    const summaryDiv = document.getElementById('resultsSummary');
    const listDiv = document.getElementById('resultsList');
    
    if (!summaryDiv || !listDiv) {
        console.error('[DISPLAY] ERROR: resultsSummary or resultsList div not found!');
        return;
    }
    
    // MEMORY OPTIMIZATION: Clear DOM containers before rendering (frees 10-30MB)
    summaryDiv.innerHTML = '';
    listDiv.innerHTML = '';

    // Reset to page 1 when filters change
    if (resetPage) {
        currentPage = 1;
    }

    // Populate category filter
    populateCategoryFilter();
    
    // Apply filters and sorting
    const filteredResults = filterAndSortResults(matchResults);
    console.log('[DISPLAY] After filtering - filteredResults length:', filteredResults.length);
    console.log('[DISPLAY] Filtered results:', filteredResults);
    
    const totalProducts = matchResults.length;
    const totalMatches = matchResults.reduce((sum, r) => sum + r.matches.length, 0);
    const productsWithMatches = matchResults.filter(r => r.matches.length > 0).length;
    const avgMatches = productsWithMatches > 0 ? (totalMatches / productsWithMatches).toFixed(1) : 0;
    
    const filteredCount = filteredResults.length;
    console.log('[DISPLAY] Stats - Total:', totalProducts, 'With matches:', productsWithMatches, 'Total matches:', totalMatches);

    // Calculate pagination
    const totalPages = Math.ceil(filteredCount / RESULTS_PER_PAGE);
    const startIndex = (currentPage - 1) * RESULTS_PER_PAGE;
    const endIndex = Math.min(startIndex + RESULTS_PER_PAGE, filteredCount);
    const paginatedResults = filteredResults.slice(startIndex, endIndex);

    summaryDiv.innerHTML = `
        <h3>Match Results Summary</h3>
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-value">${totalProducts}</span>
                <span class="stat-label">New Products</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${productsWithMatches}</span>
                <span class="stat-label">With Matches</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${totalMatches}</span>
                <span class="stat-label">Total Matches</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${avgMatches}</span>
                <span class="stat-label">Avg Matches/Product</span>
            </div>
            ${filteredCount < totalProducts ? `
            <div class="stat-item" style="background: rgba(102, 126, 234, 0.15);">
                <span class="stat-value">${filteredCount}</span>
                <span class="stat-label">Filtered Results</span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f7fafc; border: 2px solid #000; display: flex; gap: 30px; align-items: center; justify-content: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">Min Similarity:</label>
                <input type="range" id="dynamicThresholdSlider" min="30" max="100" value="${dynamicThreshold}" 
                       style="width: 150px;" oninput="updateDynamicThreshold(this.value)">
                <span id="dynamicThresholdValue" style="font-weight: 600; min-width: 40px;">${dynamicThreshold}%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">Max Matches:</label>
                <select id="dynamicLimitSelect" onchange="updateDynamicLimit(this.value)" 
                        style="padding: 5px 10px; border: 2px solid #000; background: white; font-weight: 600;">
                    <option value="5" ${dynamicLimit === 5 ? 'selected' : ''}>5</option>
                    <option value="10" ${dynamicLimit === 10 ? 'selected' : ''}>10</option>
                    <option value="20" ${dynamicLimit === 20 ? 'selected' : ''}>20</option>
                    <option value="50" ${dynamicLimit === 50 ? 'selected' : ''}>50</option>
                    <option value="0" ${dynamicLimit === 0 ? 'selected' : ''}>All</option>
                </select>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">Category:</label>
                <select id="dynamicCategoryFilter" onchange="updateDynamicCategory(this.value)" 
                        style="padding: 5px 10px; border: 2px solid #000; background: white; font-weight: 600;">
                    <option value="">All Categories</option>
                    ${[...new Set(matchResults.map(m => m.category || 'Uncategorized'))].map(cat => 
                        `<option value="${cat}">${cat}</option>`
                    ).join('')}
                </select>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">Search:</label>
                <input type="text" id="dynamicSearchFilter" placeholder="Search by name/SKU..." 
                       style="padding: 5px 10px; border: 2px solid #000; font-weight: 600;" 
                       oninput="updateDynamicSearch(this.value)">
            </div>
        </div>
        
        ${filteredCount > RESULTS_PER_PAGE ? `
            <div style="text-align: center; margin-top: 15px; color: #718096;">
                Showing ${startIndex + 1}-${endIndex} of ${filteredCount} products
            </div>
        ` : ''}
    `;

    if (filteredResults.length === 0) {
        listDiv.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <h3>No Results Found</h3>
                <p>Try adjusting your filters or search query</p>
            </div>
        `;
        return;
    }

    // Detect if we're in Mode 2 (metadata-only) using global mode tracker
    const isMetadataMode = newMode === 'metadata';
    
    listDiv.innerHTML = paginatedResults.map((result, index) => {
        const product = result.product;
        const matches = result.matches;
        
        // Use product_name for display if available (especially for metadata-only products)
        // Fall back to filename if product_name is not available
        const displayName = product.product_name || product.filename;

        return `
            <div class="result-item">
                <div class="result-header">
                    ${!isMetadataMode ? `<img data-src="/api/products/${product.id}/image" class="result-image lazy-load" 
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><rect fill='%23e2e8f0' width='120' height='120'/></svg>"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22120%22 height=%22120%22/></svg>'"
                         alt="${displayName}">` : ''}
                    <div class="result-info">
                        <h3>${escapeHtml(displayName)}</h3>
                        <div class="result-meta">
                            Category: ${product.category || 'Uncategorized'} | 
                            ${matches.length} match${matches.length !== 1 ? 'es' : ''} found
                        </div>
                    </div>
                </div>
                ${matches.length > 0 ? `
                    <div class="matches-grid">
                        ${matches.map(match => `
                            <div class="match-card" onclick="showDetailedComparison(${product.id}, ${match.product_id})">
                                ${!isMetadataMode ? `<img data-src="/api/products/${match.product_id}/image" class="match-image lazy-load"
                                     src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='120'><rect fill='%23e2e8f0' width='180' height='120'/></svg>"
                                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22180%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22180%22 height=%22120%22/></svg>'"
                                     alt="Match">` : ''}
                                <div class="match-score ${getScoreClass(match.similarity_score)}">
                                    ${match.similarity_score.toFixed(1)}%
                                </div>
                                ${match.similarity_score > 90 ? '<span class="duplicate-badge">DUPLICATE?</span>' : ''}
                                <div class="match-info">
                                    ${escapeHtml(match.product_name || 'Unknown')}
                                    ${match.priceStatistics ? `
                                        <div class="price-sparkline" title="Price trend: ${match.priceStatistics.trend}">
                                            ${generateSparkline(match.priceHistory)}
                                            <span class="price-current">$${match.priceStatistics.current}</span>
                                            <span class="price-trend price-trend-${match.priceStatistics.trend}">${getTrendIcon(match.priceStatistics.trend)}</span>
                                        </div>
                                    ` : ''}
                                    ${match.performanceStatistics ? `
                                        <div class="performance-sparkline" title="Sales trend: ${match.performanceStatistics.sales_trend}">
                                            ${generatePerformanceSparkline(match.performanceHistory)}
                                            <span class="performance-sales" style="font-weight: bold;">${match.performanceStatistics.total_sales} SALES</span>
                                            <span class="performance-trend performance-trend-${match.performanceStatistics.sales_trend}">${getTrendIcon(match.performanceStatistics.sales_trend)}</span>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : '<div class="no-matches">No matches found</div>'}
            </div>
        `;
    }).join('');
    
    // Add pagination controls if needed
    if (filteredCount > RESULTS_PER_PAGE) {
        const hasMore = currentPage < totalPages;
        const hasPrevious = currentPage > 1;
        
        listDiv.innerHTML += `
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 30px; padding: 20px;">
                ${hasPrevious ? `
                    <button class="btn" onclick="loadPreviousPage()" style="min-width: 120px;">
                        ← Previous
                    </button>
                ` : ''}
                <div style="display: flex; align-items: center; color: #718096; font-weight: 500;">
                    Page ${currentPage} of ${totalPages}
                </div>
                ${hasMore ? `
                    <button class="btn" onclick="loadNextPage()" style="min-width: 120px;">
                        Next →
                    </button>
                ` : ''}
            </div>
        `;
    }
    
    // Initialize lazy loading for images
    initLazyLoading();
}

function loadNextPage() {
    currentPage++;
    displayResults(false);
    document.getElementById('resultsList').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function loadPreviousPage() {
    currentPage--;
    displayResults(false);
    document.getElementById('resultsList').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateDynamicThreshold(value) {
    dynamicThreshold = parseInt(value);
    document.getElementById('dynamicThresholdValue').textContent = value + '%';
    displayResults(true); // Reset to page 1 when filter changes
}

function updateDynamicLimit(value) {
    dynamicLimit = parseInt(value);
    displayResults(true); // Reset to page 1 when filter changes
}

let dynamicCategory = '';
let dynamicSearch = '';
let dynamicSearchResults = new Map(); // Cache search results

function updateDynamicCategory(value) {
    dynamicCategory = value;
    displayResults(true); // Reset to page 1 when filter changes
}

async function updateDynamicSearch(value) {
    dynamicSearch = value.toLowerCase().trim();
    
    // Clear cache if search is empty
    if (!dynamicSearch) {
        dynamicSearchResults.clear();
        displayResults(true);
        return;
    }
    
    // Debounce search - wait 300ms before searching
    if (window.searchTimeout) {
        clearTimeout(window.searchTimeout);
    }
    
    window.searchTimeout = setTimeout(async () => {
        try {
            // Call backend search API
            const response = await fetch(`/api/products/search?q=${encodeURIComponent(dynamicSearch)}&limit=1000`);
            const data = await response.json();
            
            if (data.success) {
                // Build a map of product IDs for fast lookup
                dynamicSearchResults.clear();
                data.results.forEach(product => {
                    dynamicSearchResults.set(product.id, product);
                });
                console.log(`[SEARCH] Found ${data.results.length} products matching "${dynamicSearch}"`);
            }
        } catch (error) {
            console.error('[SEARCH] Error:', error);
        }
        
        displayResults(true);
    }, 300);
}

function getScoreClass(score) {
    if (score >= 70) return 'score-high';
    if (score >= 50) return 'score-medium';
    return 'score-low';
}

async function showDetailedComparison(newProductId, matchedProductId) {
    const modal = document.getElementById('detailModal');
    const modalBody = document.getElementById('modalBody');

    // Show loading state
    modalBody.innerHTML = '<div class="modal-loading"><div class="spinner"></div><p>Loading comparison...</p></div>';
    modal.classList.add('show');

    try {
        // Fetch both products with retry logic
        const [newResp, matchResp] = await Promise.all([
            fetchWithRetry(`/api/products/${newProductId}`),
            fetchWithRetry(`/api/products/${matchedProductId}`)
        ]);

        if (!newResp.ok || !matchResp.ok) {
            throw new Error('Failed to load product details');
        }

        const newData = await newResp.json();
        const matchData = await matchResp.json();

        // Find the match details
        const matchResult = matchResults.find(r => r.product.id === newProductId);
        const matchDetails = matchResult?.matches.find(m => m.product_id === matchedProductId);

        // Detect if we're in Mode 2 (metadata-only)
        const isMetadataMode = newMode === 'metadata';
        
        modalBody.innerHTML = `
            <h2>Detailed Comparison</h2>
            <div class="comparison-view">
                <div class="comparison-item">
                    <h3>New Product</h3>
                    ${!isMetadataMode ? `<img data-src="/api/products/${newProductId}/image" class="lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect fill='%23e2e8f0' width='300' height='300'/></svg>"
                         alt="New Product">` : ''}
                    <div class="comparison-details">
                        <p><strong>Product:</strong> ${escapeHtml(newData.product.product_name || 'Unknown')}</p>
                        <p><strong>SKU:</strong> ${escapeHtml(newData.product.sku || 'N/A')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(newData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
                <div class="comparison-item">
                    <h3>Matched Product</h3>
                    ${!isMetadataMode ? `<img data-src="/api/products/${matchedProductId}/image" class="lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect fill='%23e2e8f0' width='300' height='300'/></svg>"
                         alt="Matched Product">` : ''}
                    <div class="comparison-details">
                        <p><strong>Product:</strong> ${escapeHtml(matchData.product.product_name || 'Unknown')}</p>
                        <p><strong>SKU:</strong> ${escapeHtml(matchData.product.sku || 'N/A')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(matchData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
            </div>
            ${matchDetails ? `
                <div class="score-breakdown">
                    <h4>Similarity Score</h4>
                    <div class="score-bar">
                        <div class="score-bar-label">
                            <span>Overall Similarity</span>
                            <span>${matchDetails.similarity_score.toFixed(1)}%</span>
                        </div>
                        <div class="score-bar-fill">
                            <div style="width: ${matchDetails.similarity_score}%"></div>
                        </div>
                    </div>
                    
                    <!-- Mode 3: Hybrid Score Breakdown -->
                    ${matchDetails.visual_score !== undefined && matchDetails.metadata_score !== undefined ? `
                        <div style="margin-top: 20px; padding-top: 20px; border-top: 2px solid #e2e8f0;">
                            <h5 style="margin-bottom: 15px; color: #2d3748;">Score Breakdown (Hybrid Mode)</h5>
                            
                            <!-- Visual Score Component -->
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span style="font-weight: 600; color: #2d3748;">Visual Similarity (CLIP)</span>
                                    <span style="font-weight: 600; color: #667eea;">${matchDetails.visual_score.toFixed(1)}%</span>
                                </div>
                                <div style="width: 100%; height: 8px; background: #e2e8f0; border: 1px solid #cbd5e0; border-radius: 4px; overflow: hidden;">
                                    <div style="width: ${matchDetails.visual_score}%; height: 100%; background: #667eea;"></div>
                                </div>
                            </div>
                            
                            <!-- Metadata Score Component -->
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span style="font-weight: 600; color: #2d3748;">Metadata Similarity</span>
                                    <span style="font-weight: 600; color: #f6ad55;">${matchDetails.metadata_score.toFixed(1)}%</span>
                                </div>
                                <div style="width: 100%; height: 8px; background: #e2e8f0; border: 1px solid #cbd5e0; border-radius: 4px; overflow: hidden;">
                                    <div style="width: ${matchDetails.metadata_score}%; height: 100%; background: #f6ad55;"></div>
                                </div>
                            </div>
                            
                            <!-- Metadata Sub-Scores -->
                            ${matchDetails.sku_score !== undefined || matchDetails.name_score !== undefined || matchDetails.category_score !== undefined || matchDetails.price_score !== undefined || matchDetails.performance_score !== undefined ? `
                                <div style="margin-top: 15px; padding: 12px; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px;">
                                    <h6 style="margin: 0 0 10px 0; color: #2d3748; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Metadata Components</h6>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                        ${matchDetails.sku_score !== undefined ? `
                                            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                                <span style="color: #4a5568;">SKU Match</span>
                                                <span style="font-weight: 600; color: #2d3748;">${matchDetails.sku_score.toFixed(1)}%</span>
                                            </div>
                                        ` : ''}
                                        ${matchDetails.name_score !== undefined ? `
                                            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                                <span style="color: #4a5568;">Name Match</span>
                                                <span style="font-weight: 600; color: #2d3748;">${matchDetails.name_score.toFixed(1)}%</span>
                                            </div>
                                        ` : ''}
                                        ${matchDetails.category_score !== undefined ? `
                                            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                                <span style="color: #4a5568;">Category Match</span>
                                                <span style="font-weight: 600; color: #2d3748;">${matchDetails.category_score.toFixed(1)}%</span>
                                            </div>
                                        ` : ''}
                                        ${matchDetails.price_score !== undefined ? `
                                            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                                <span style="color: #4a5568;">Price Similarity</span>
                                                <span style="font-weight: 600; color: #2d3748;">${matchDetails.price_score.toFixed(1)}%</span>
                                            </div>
                                        ` : ''}
                                        ${matchDetails.performance_score !== undefined ? `
                                            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                                <span style="color: #4a5568;">Performance Similarity</span>
                                                <span style="font-weight: 600; color: #2d3748;">${matchDetails.performance_score.toFixed(1)}%</span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
            ` : ''}
            ${matchDetails?.priceStatistics ? `
                <div class="price-history-section">
                    <h4>PRICE HISTORY</h4>
                    <div class="price-statistics">
                        <div class="price-stat">
                            <span class="price-stat-label">Current</span>
                            <span class="price-stat-value">$${matchDetails.priceStatistics.current}</span>
                        </div>
                        <div class="price-stat">
                            <span class="price-stat-label">Average</span>
                            <span class="price-stat-value">$${matchDetails.priceStatistics.average}</span>
                        </div>
                        <div class="price-stat">
                            <span class="price-stat-label">Min</span>
                            <span class="price-stat-value">$${matchDetails.priceStatistics.min}</span>
                        </div>
                        <div class="price-stat">
                            <span class="price-stat-label">Max</span>
                            <span class="price-stat-value">$${matchDetails.priceStatistics.max}</span>
                        </div>
                        <div class="price-stat">
                            <span class="price-stat-label">Trend</span>
                            <span class="price-stat-value price-trend-${matchDetails.priceStatistics.trend}">
                                ${getTrendIcon(matchDetails.priceStatistics.trend)} ${matchDetails.priceStatistics.trend}
                            </span>
                        </div>
                    </div>
                    <div class="price-chart-container">
                        ${generatePriceChart(matchDetails.priceHistory, 'modalPriceChart')}
                    </div>
                </div>
            ` : ''}
            ${matchDetails?.performanceStatistics ? `
                <div class="performance-history-section">
                    <h4>PERFORMANCE HISTORY</h4>
                    <div class="performance-statistics">
                        <div class="performance-stat">
                            <span class="performance-stat-label">Total Sales</span>
                            <span class="performance-stat-value">${matchDetails.performanceStatistics.total_sales}</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Average</span>
                            <span class="performance-stat-value">${matchDetails.performanceStatistics.average_sales}</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Trend</span>
                            <span class="performance-stat-value performance-trend-${matchDetails.performanceStatistics.sales_trend}">
                                ${getTrendIcon(matchDetails.performanceStatistics.sales_trend)} ${matchDetails.performanceStatistics.sales_trend}
                            </span>
                        </div>
                    </div>
                    <div class="performance-chart-container">
                        ${generatePerformanceChart(matchDetails.performanceHistory, 'modalPerformanceChart')}
                    </div>
                </div>
            ` : ''}
        `;

        modal.classList.add('show');
        
        // Initialize lazy loading for modal images
        initLazyLoading();
    } catch (error) {
        showToast('Failed to load comparison details', 'error');
    }
}

function closeModal() {
    document.getElementById('detailModal').classList.remove('show');
}

function exportResults() {
    let csv = 'New Product,Category,Match Count,Top Match,Top Score,Price Current,Price Avg,Price Min,Price Max,Price Trend,Total Sales,Avg Sales,Sales Trend\n';

    matchResults.forEach(result => {
        const product = result.product;
        const topMatch = result.matches[0];

        csv += `"${product.filename}","${product.category || 'Uncategorized'}",${result.matches.length}`;

        if (topMatch) {
            csv += `,"${topMatch.product_name || 'Unknown'}",${topMatch.similarity_score.toFixed(1)}`;
            
            // Add price history data if available
            if (topMatch.priceStatistics) {
                csv += `,${topMatch.priceStatistics.current}`;
                csv += `,${topMatch.priceStatistics.average}`;
                csv += `,${topMatch.priceStatistics.min}`;
                csv += `,${topMatch.priceStatistics.max}`;
                csv += `,"${topMatch.priceStatistics.trend}"`;
            } else {
                csv += ',,,,,';
            }
            
            // Add performance history data if available (SIMPLE FORMAT: just sales)
            if (topMatch.performanceStatistics) {
                csv += `,${topMatch.performanceStatistics.total_sales}`;
                csv += `,${topMatch.performanceStatistics.average_sales}`;
                csv += `,"${topMatch.performanceStatistics.sales_trend}"`;
            } else {
                csv += ',,,';
            }
        } else {
            csv += ',"No matches",0,,,,,,,,';
        }

        csv += '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    try {
        const a = document.createElement('a');
        a.href = url;
        a.download = `match_results_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        showToast('Results exported to CSV', 'success');
    } catch (error) {
        console.error('Export failed:', error);
        showToast('Export failed', 'error');
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 100);
    }
}

function resetApp() {
    if (confirm('Start over? This will clear all data.')) {
        // Clean up memory before reload
        cleanupMemory();
        
        // Small delay to ensure cleanup completes
        setTimeout(() => {
            location.reload();
        }, 100);
    }
}

// Utilities
// Helper function to parse a CSV line properly handling quoted fields
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote
                current += '"';
                i++; // Skip next quote
            } else {
                // Toggle quote state
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // Field separator
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add last field
    result.push(current.trim());
    
    return result;
}

async function parseCsv(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim());
            const errors = [];

            // Check if first line is a header
            const firstLine = lines[0];
            const hasHeader = firstLine.toLowerCase().includes('filename') ||
                firstLine.toLowerCase().includes('category') ||
                firstLine.toLowerCase().includes('sku');

            // Validate header order if present
            if (hasHeader) {
                const headerParts = parseCSVLine(firstLine.toLowerCase());
                const expectedOrder = ['filename', 'category', 'sku', 'name', 'price', 'price_history', 'performance_history'];
                
                // Check if headers match expected order (at least first 4 columns)
                if (headerParts.length >= 4) {
                    const actualOrder = headerParts.slice(0, 4).map(h => h.trim());
                    const expectedFirst4 = expectedOrder.slice(0, 4);
                    
                    // Check if order matches
                    const orderMatches = actualOrder.every((header, i) => {
                        return header === expectedFirst4[i] || 
                               header.replace(/_/g, '') === expectedFirst4[i].replace(/_/g, '');
                    });
                    
                    if (!orderMatches) {
                        const headerWarning = `CSV headers in wrong order! Expected: ${expectedOrder.slice(0, 4).join(', ')}. Found: ${headerParts.slice(0, 4).join(', ')}. Data may be mapped incorrectly.`;
                        showToast(headerWarning, 'warning');
                        errors.push(`Header order mismatch - Expected: ${expectedOrder.slice(0, 4).join(', ')}`);
                    }
                }
            }

            // Use Web Worker for parallel CSV parsing (non-blocking)
            console.log('[CSV-PARSER] Starting Web Worker for CSV parsing');
            
            if (typeof(Worker) !== 'undefined') {
                // Web Workers supported - use parallel parsing
                const worker = new Worker('/static/csv-parser-worker.js');
                
                worker.onmessage = function(event) {
                    const result = event.data;
                    
                    if (result.success) {
                        console.log(`[CSV-PARSER] ✓ Web Worker parsed ${result.lineCount} lines`);
                        resolve(result.map);
                    } else {
                        console.error('[CSV-PARSER] Web Worker error:', result.error);
                        showToast('CSV parsing error: ' + result.error, 'error');
                        resolve({});
                    }
                    
                    worker.terminate();
                };
                
                worker.onerror = function(error) {
                    console.error('[CSV-PARSER] Web Worker error:', error.message);
                    showToast('CSV parsing error: ' + error.message, 'error');
                    resolve({});
                    worker.terminate();
                };
                
                // Send CSV data to worker
                console.log('[CSV-PARSER] Sending CSV data to Web Worker');
                worker.postMessage({
                    csvText: text,
                    hasHeader: hasHeader
                });
            } else {
                // Web Workers not supported - fallback to main thread parsing
                console.warn('[CSV-PARSER] Web Workers not supported, falling back to main thread parsing');
                
                const map = {};
                const dataLines = hasHeader ? lines.slice(1) : lines;

                dataLines.forEach((line, index) => {
                    try {
                        const parts = parseCSVLine(line);

                        if (parts.length >= 1) {
                            const filename = parts[0];
                            if (!filename) return;
                            
                            const category = parts[1] || null;
                            const sku = parts[2] || null;
                            const name = parts[3] || null;
                            
                            let priceHistory = null;
                            const priceHistoryStr = parts[4] || parts[5] || null;
                            if (priceHistoryStr && (priceHistoryStr.includes(':') || priceHistoryStr.includes(';'))) {
                                try {
                                    const parsed = parsePriceHistory(priceHistoryStr);
                                    if (parsed && parsed.length > 0) {
                                        priceHistory = parsed;
                                    }
                                } catch (error) {
                                    errors.push(`Row ${index + 2}: Failed to parse price history for ${filename}`);
                                }
                            }
                            
                            // Parse performance history
                            let performanceHistory = null;
                            const performanceHistoryStr = parts[5] || parts[6] || null;
                            
                            if (performanceHistoryStr) {
                                try {
                                    if (performanceHistoryStr.includes(':')) {
                                        const parsed = parsePerformanceHistory(performanceHistoryStr);
                                        if (parsed && parsed.length > 0) {
                                            performanceHistory = parsed;
                                        }
                                    } else {
                                        const numbers = performanceHistoryStr.split(',')
                                            .map(s => parseFloat(s.trim()))
                                            .filter(n => !isNaN(n) && n >= 0);
                                        if (numbers.length > 0) {
                                            performanceHistory = numbers;
                                        }
                                    }
                                } catch (error) {
                                    errors.push(`Row ${index + 2}: Failed to parse performance history for ${filename}`);
                                }
                            }

                            if (filename) {
                                map[filename] = {
                                    category: category,
                                    sku: sku,
                                    name: name,
                                    priceHistory: priceHistory,
                                    performanceHistory: performanceHistory
                                };
                            }
                        }
                    } catch (error) {
                        errors.push(`Row ${index + 2}: ${error.message}`);
                    }
                });

                console.log(`[CSV-PARSER] ✓ Fallback parsing complete: ${dataLines.length} lines parsed`);
                resolve(map);
            }
        };
        
        reader.onerror = () => {
            showToast('Failed to read CSV file. Please check the file format.', 'error');
            resolve({});
        };
        
        reader.readAsText(file);
    });
}

function parsePriceHistory(priceHistoryStr) {
    // Parse price history string - FLEXIBLE FORMATS:
    // Format 1: "2024-01-15:29.99;2024-02-15:31.50" (semicolon separated)
    // Format 2: "2024-01-15:29.99,2024-02-15:31.50" (comma separated)
    // Format 3: "29.99;31.50;28.75" (prices only, auto-generate monthly dates)
    // Returns array of {date, price} objects
    
    if (!priceHistoryStr || priceHistoryStr.trim() === '') {
        return null;
    }
    
    const str = priceHistoryStr.trim();
    const priceHistory = [];
    
    // Check if it contains dates (has colons)
    if (str.includes(':')) {
        // Format with dates
        const entries = str.split(/[;,]/).filter(e => e.trim());
        
        for (const entry of entries) {
            const parts = entry.split(':').map(s => s.trim());
            if (parts.length >= 2) {
                const date = parts[0];
                const price = parseFloat(parts[1]);
                
                // Validate date format (YYYY-MM-DD or MM/DD/YYYY or similar)
                if (date && !isNaN(price) && price >= 0) {
                    // Try to normalize date to YYYY-MM-DD
                    const normalizedDate = normalizeDateString(date);
                    if (normalizedDate) {
                        priceHistory.push({
                            date: normalizedDate,
                            price: price
                        });
                    }
                }
            }
        }
    } else {
        // Format without dates - just prices
        // Generate monthly dates going backwards from today
        const prices = str.split(/[;,]/).filter(e => e.trim()).map(p => parseFloat(p.trim()));
        const today = new Date();
        
        prices.forEach((price, index) => {
            if (!isNaN(price) && price >= 0) {
                const date = new Date(today);
                date.setMonth(date.getMonth() - (prices.length - 1 - index));
                priceHistory.push({
                    date: date.toISOString().split('T')[0],
                    price: price
                });
            }
        });
    }
    
    // Limit to 12 months and sort by date
    if (priceHistory.length > 0) {
        priceHistory.sort((a, b) => new Date(a.date) - new Date(b.date));
        return priceHistory.slice(-12); // Keep most recent 12
    }
    
    return null;
}

function normalizeDateString(dateStr) {
    // Try to parse various date formats and return YYYY-MM-DD
    try {
        // Already in YYYY-MM-DD format
        if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
            return dateStr;
        }
        
        // MM/DD/YYYY or M/D/YYYY
        if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
            const [month, day, year] = dateStr.split('/');
            return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
        }
        
        // DD/MM/YYYY or D/M/YYYY (European format)
        if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
            const [day, month, year] = dateStr.split('/');
            // Ambiguous - assume MM/DD/YYYY (US format) by default
            return `${year}-${day.padStart(2, '0')}-${month.padStart(2, '0')}`;
        }
        
        // Try parsing with Date constructor
        const date = new Date(dateStr);
        if (!isNaN(date.getTime())) {
            return date.toISOString().split('T')[0];
        }
        
        return null;
    } catch (error) {
        return null;
    }
}

function parsePerformanceHistory(performanceHistoryStr) {
    // Parse performance history string - FLEXIBLE FORMATS:
    // Format 1: "2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160"
    //           (date:sales:views:conversion:revenue)
    // Format 2: "150:1200:12.5:1800;180:1500:12.0:2160" (no dates, auto-generate)
    // Returns array of {date, sales, views, conversion_rate, revenue} objects
    
    if (!performanceHistoryStr || performanceHistoryStr.trim() === '') {
        return null;
    }
    
    const str = performanceHistoryStr.trim();
    const performanceHistory = [];
    
    // Split by semicolon or comma
    const entries = str.split(/[;,]/).filter(e => e.trim());
    
    for (const entry of entries) {
        const parts = entry.split(':').map(s => s.trim());
        
        let date, sales, views, conversion_rate, revenue;
        
        // Check if first part is a date
        if (parts.length >= 5 && /^\d{4}-\d{2}-\d{2}$/.test(parts[0])) {
            // Format with date
            date = parts[0];
            sales = parseInt(parts[1]) || 0;
            views = parseInt(parts[2]) || 0;
            conversion_rate = parseFloat(parts[3]) || 0.0;
            revenue = parseFloat(parts[4]) || 0.0;
        } else if (parts.length >= 4) {
            // Format without date - generate monthly dates backwards
            const today = new Date();
            const monthsBack = performanceHistory.length;
            const dateObj = new Date(today);
            dateObj.setMonth(dateObj.getMonth() - monthsBack);
            date = dateObj.toISOString().split('T')[0];
            
            sales = parseInt(parts[0]) || 0;
            views = parseInt(parts[1]) || 0;
            conversion_rate = parseFloat(parts[2]) || 0.0;
            revenue = parseFloat(parts[3]) || 0.0;
        } else {
            continue; // Skip invalid entries
        }
        
        // Validate values
        if (sales >= 0 && views >= 0 && conversion_rate >= 0 && conversion_rate <= 100 && revenue >= 0) {
            performanceHistory.push({
                date: date,
                sales: sales,
                views: views,
                conversion_rate: conversion_rate,
                revenue: revenue
            });
        }
    }
    
    // Limit to 12 months and sort by date
    if (performanceHistory.length > 0) {
        performanceHistory.sort((a, b) => new Date(a.date) - new Date(b.date));
        return performanceHistory.slice(-12); // Keep most recent 12
    }
    
    return null;
}

function showToast(message, type = 'info') {
    if (!message) return;

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

function extractCategoryFromPath(path) {
    // Extract category from folder structure
    // Examples:
    // "MainFolder/Placemats/image1.jpg" -> "Placemats" (subfolder = category)
    // "MainFolder/image1.jpg" -> null (no subfolder = no category)
    // "image1.jpg" -> null (no folder)
    
    if (!path) return null;
    
    const parts = path.split('/');
    
    // If only filename (no folders), return null
    if (parts.length === 1) return null;
    
    // If only one folder level (MainFolder/image.jpg), return null (no category)
    // Categories should only come from subfolders INSIDE the main upload folder
    if (parts.length === 2) return null;
    
    // Get the immediate parent folder (last folder before filename)
    const category = parts[parts.length - 2];
    
    // Ignore common root folder names
    const ignoredFolders = ['historical_products', 'new_products', 'products', 'images', 'uploads'];
    if (ignoredFolders.includes(category.toLowerCase())) {
        // If there's another folder level, use that
        if (parts.length > 2) {
            return parts[parts.length - 3];
        }
        return null;
    }
    
    return category;
}

// Lazy Loading Implementation for Performance Optimization
function initLazyLoading() {
    // Disconnect previous observer to prevent memory leaks
    if (lazyLoadObserver) {
        lazyLoadObserver.disconnect();
    }
    
    // Use Intersection Observer API for efficient lazy loading
    lazyLoadObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                const src = img.getAttribute('data-src');
                
                if (src) {
                    // Check if it's an API endpoint (needs blob URL for tracking)
                    if (src.startsWith('/api/products/')) {
                        // Create tracked blob URL to prevent memory leaks
                        createTrackedBlobUrl(src)
                            .then(blobUrl => {
                                img.src = blobUrl;
                                img.removeAttribute('data-src');
                            })
                            .catch(error => {
                                console.error('Failed to load image:', error);
                                // Fallback to direct URL
                                img.src = src;
                                img.removeAttribute('data-src');
                            });
                    } else {
                        // For non-API images, load directly
                        img.src = src;
                        img.removeAttribute('data-src');
                    }
                    
                    // Stop observing this image
                    observer.unobserve(img);
                }
            }
        });
    }, {
        // Load images slightly before they enter viewport
        rootMargin: '50px 0px',
        threshold: 0.01
    });

    // Observe all images with lazy-load class
    const lazyImages = document.querySelectorAll('img.lazy-load');
    lazyImages.forEach(img => {
        lazyLoadObserver.observe(img);
    });
}

// Call lazy loading on page load for any existing images
document.addEventListener('DOMContentLoaded', () => {
    initLazyLoading();
});

// Retry Logic with Exponential Backoff
async function fetchWithRetry(url, options = {}, retryCount = 0) {
    try {
        const response = await fetch(url, options);
        
        // Only retry rate limit (429) - don't retry 500 errors as they're usually application errors
        if (response.status === 429 && retryCount < RETRY_CONFIG.maxRetries) {
            const delay = Math.min(
                RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, retryCount),
                RETRY_CONFIG.maxDelay
            );
            
            showToast(`Rate limited. Retrying in ${delay / 1000} seconds... (Attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries})`, 'warning');
            
            await sleep(delay);
            return fetchWithRetry(url, options, retryCount + 1);
        }
        
        return response;
    } catch (error) {
        // Network error - retry
        if (retryCount < RETRY_CONFIG.maxRetries) {
            const delay = Math.min(
                RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, retryCount),
                RETRY_CONFIG.maxDelay
            );
            
            showToast(`Network error. Retrying in ${delay / 1000} seconds... (Attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries})`, 'warning');
            
            await sleep(delay);
            return fetchWithRetry(url, options, retryCount + 1);
        }
        
        throw error;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// User-Friendly Error Messages
function getUserFriendlyError(errorCode, originalError, suggestion) {
    const errorMessages = {
        'NETWORK_ERROR': 'Unable to connect to the server. Please check your connection and try again.',
        'INVALID_IMAGE': 'This image file is corrupted or in an unsupported format. Please use JPEG, PNG, or WebP.',
        'FILE_TOO_LARGE': 'This image file is too large. Please use images under 10MB.',
        'MISSING_FEATURES': 'Could not extract features from this image. The image may be corrupted or too simple.',
        'NO_HISTORICAL_PRODUCTS': 'No historical products found in this category. Please add historical products first.',
        'DATABASE_ERROR': 'A database error occurred. Please try again or restart the application.',
        'PROCESSING_ERROR': 'Failed to process this image. Please try a different image.',
        'UNKNOWN_ERROR': 'An unexpected error occurred. Please try again.'
    };

    let message = errorMessages[errorCode] || originalError || errorMessages['UNKNOWN_ERROR'];
    
    if (suggestion) {
        message += ` Suggestion: ${suggestion}`;
    }
    
    return message;
}

// Loading Spinner for Buttons
function showLoadingSpinner(button, show) {
    if (show) {
        if (!button.querySelector('.btn-spinner')) {
            const spinner = document.createElement('span');
            spinner.className = 'btn-spinner';
            button.appendChild(spinner);
        }
        button.classList.add('loading');
    } else {
        const spinner = button.querySelector('.btn-spinner');
        if (spinner) {
            spinner.remove();
        }
        button.classList.remove('loading');
    }
}

// Tooltip Initialization
function initTooltips() {
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-popup';
    tooltip.style.display = 'none';
    document.body.appendChild(tooltip);

    // Add tooltips to key UI elements
    const tooltipElements = [
        { selector: '#thresholdSlider', text: 'Set the minimum similarity score (0-100) for matches. Higher values show only very similar products.' },
        { selector: '#limitSelect', text: 'Maximum number of matches to show for each new product.' },
        { selector: '#historicalBrowseBtn', text: 'Select a folder containing images of products you\'ve sold before.' },
        { selector: '#newBrowseBtn', text: 'Select a folder containing images of new products to match.' },
        { selector: '#matchBtn', text: 'Start comparing new products against your historical catalog.' },
        { selector: '#exportBtn', text: 'Download all match results as a CSV file for further analysis.' },
        { selector: '#resetBtn', text: 'Clear all data and start over with new products.' }
    ];

    // Set tooltip attributes
    tooltipElements.forEach(({ selector, text }) => {
        const element = document.querySelector(selector);
        if (element) {
            element.setAttribute('data-tooltip', text);
        }
    });

    // Use event delegation to prevent memory leaks (Fix #5)
    // Single set of listeners for all tooltips instead of per-element
    addTrackedListener(document.body, 'mouseenter', (e) => {
        const element = e.target.closest('[data-tooltip]');
        if (element) {
            const tooltipText = element.getAttribute('data-tooltip');
            if (tooltipText) {
                tooltip.textContent = tooltipText;
                tooltip.style.display = 'block';
                positionTooltip(element, tooltip);
            }
        }
    }, 'tooltips');

    addTrackedListener(document.body, 'mouseleave', (e) => {
        const element = e.target.closest('[data-tooltip]');
        if (element) {
            tooltip.style.display = 'none';
        }
    }, 'tooltips');

    addTrackedListener(document.body, 'mousemove', (e) => {
        const element = e.target.closest('[data-tooltip]');
        if (element && tooltip.style.display === 'block') {
            positionTooltip(element, tooltip);
        }
    }, 'tooltips');
}

function positionTooltip(element, tooltip) {
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let top = rect.bottom + 10;
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
    
    // Adjust if tooltip goes off screen
    if (left < 10) left = 10;
    if (left + tooltipRect.width > window.innerWidth - 10) {
        left = window.innerWidth - tooltipRect.width - 10;
    }
    
    if (top + tooltipRect.height > window.innerHeight - 10) {
        top = rect.top - tooltipRect.height - 10;
    }
    
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
}

// CSV Help Functions
function showCsvHelp(type) {
    const modal = document.getElementById('csvHelpModal');
    modal.classList.add('show');
}

function closeCsvHelp() {
    const modal = document.getElementById('csvHelpModal');
    modal.classList.remove('show');
}

function downloadSampleCsv() {
    const csv = `filename,category,sku,name,price,price_history,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160;2024-03-15:200:1800:11.1:2400
product2.jpg,dinnerware,DW-002,White Plate Set,45.00,2024-01-15:45.00;2024-02-15:42.50;2024-03-15:44.00,2024-01-15:200:2000:10.0:9000;2024-02-15:220:2200:10.0:9900;2024-03-15:240:2400:10.0:10800
product3.jpg,textiles,TX-003,Cotton Napkins,15.99,15.99;16.50;15.75,100:800:12.5:1200;120:900:13.3:1440;110:850:12.9:1320
product4.jpg,placemats,PM-004,Red Placemat,32.00,,
product5.jpg,dinnerware,DW-005,Ceramic Bowl,22.50,2024-01-15:22.50;2024-02-15:23.00,80:600:13.3:960;90:650:13.8:1080`;

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    try {
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sample_product_data.csv';
        a.click();
        showToast('Sample CSV downloaded! Open it in Excel or any text editor.', 'success');
    } catch (error) {
        console.error('Download failed:', error);
        showToast('Download failed', 'error');
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 100);
    }
}

// Enhanced Toast with Action Button
function showToastWithAction(message, type, actionText, actionCallback) {
    const toast = document.getElementById('toast');
    
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;
    
    const actionBtn = document.createElement('button');
    actionBtn.className = 'toast-action';
    actionBtn.textContent = actionText;
    actionBtn.onclick = () => {
        toast.classList.remove('show');
        actionCallback();
    };
    
    toast.innerHTML = '';
    toast.appendChild(messageSpan);
    toast.appendChild(actionBtn);
    toast.className = `toast ${type} show`;

    const timeout = 10000; // Longer timeout for action toasts

    setTimeout(() => {
        toast.classList.remove('show');
    }, timeout);
}

// Price History Visualization Functions

// Chart color management
let chartColor = localStorage.getItem('chartColor') || '#0066FF';

function getChartColor() {
    return chartColor;
}

function setChartColor(color) {
    chartColor = color;
    localStorage.setItem('chartColor', color);
    // Refresh any visible charts
    if (document.getElementById('resultsSection').style.display !== 'none') {
        displayResults();
    }
}

function generateSparkline(priceHistory) {
    // Generate a simple SVG sparkline chart
    if (!priceHistory || priceHistory.length === 0) {
        return '';
    }
    
    const prices = priceHistory.map(p => p.price).reverse(); // Oldest to newest
    const max = Math.max(...prices);
    const min = Math.min(...prices);
    const range = max - min || 1;
    
    const width = 60;
    const height = 20;
    const points = prices.map((price, i) => {
        const x = (i / (prices.length - 1)) * width;
        const y = height - ((price - min) / range) * height;
        return `${x},${y}`;
    }).join(' ');
    
    return `<svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" oncontextmenu="showColorPicker(event); return false;">
        <polyline points="${points}" fill="none" stroke="${getChartColor()}" stroke-width="2"/>
    </svg>`;
}

function generatePerformanceSparkline(performanceHistory) {
    // Generate a simple SVG sparkline chart for sales
    if (!performanceHistory || performanceHistory.length === 0) {
        return '';
    }
    
    // performanceHistory is already an array of numbers (simple format)
    // Filter out invalid values (NaN, null, undefined)
    const validSales = performanceHistory.filter(s => typeof s === 'number' && !isNaN(s) && isFinite(s));
    
    if (validSales.length === 0) {
        return ''; // No valid data to display
    }
    
    const sales = [...validSales].reverse(); // Oldest to newest
    const max = Math.max(...sales);
    const min = Math.min(...sales);
    const range = max - min || 1;
    
    const width = 60;
    const height = 20;
    const points = sales.map((sale, i) => {
        const x = sales.length > 1 ? (i / (sales.length - 1)) * width : width / 2;
        const y = height - ((sale - min) / range) * height;
        // Ensure no NaN values in output
        return `${isFinite(x) ? x : 0},${isFinite(y) ? y : height / 2}`;
    }).join(' ');
    
    return `<svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" oncontextmenu="showColorPicker(event); return false;">
        <polyline points="${points}" fill="none" stroke="${getChartColor()}" stroke-width="2"/>
    </svg>`;
}

function getTrendIcon(trend) {
    switch (trend) {
        case 'up':
            return '↑';
        case 'down':
            return '↓';
        case 'stable':
        default:
            return '→';
    }
}

function generatePerformanceChart(performanceHistory, containerId) {
    // Generate a more detailed performance chart for the modal
    if (!performanceHistory || performanceHistory.length === 0) {
        return '<p>No performance history available</p>';
    }
    
    // CRITICAL: Filter out invalid values to prevent NaN errors
    const validSales = performanceHistory.filter(s => typeof s === 'number' && !isNaN(s) && isFinite(s));
    
    if (validSales.length === 0) {
        return '<p>No valid performance data available</p>';
    }
    
    // performanceHistory is already an array of numbers (simple format)
    const sales = [...validSales].reverse(); // Oldest to newest
    // Generate simple month labels
    const dates = sales.map((_, i) => `Month ${i + 1}`);
    const max = Math.max(...sales);
    const min = Math.min(...sales);
    const range = max - min || 1;
    
    const width = 400;
    const height = 200;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    
    // Generate points for the line
    const points = sales.map((sale, i) => {
        const x = sales.length > 1 ? padding + (i / (sales.length - 1)) * chartWidth : padding + chartWidth / 2;
        const y = padding + chartHeight - ((sale - min) / range) * chartHeight;
        // Ensure no NaN values
        return { 
            x: isFinite(x) ? x : padding, 
            y: isFinite(y) ? y : padding + chartHeight / 2, 
            sale, 
            date: dates[i] 
        };
    });
    
    const linePoints = points.map(p => `${p.x},${p.y}`).join(' ');
    
    // Generate circles for data points
    const circles = points.map(p => 
        `<circle cx="${p.x}" cy="${p.y}" r="4" fill="${getChartColor()}" class="performance-point" data-sales="${p.sale}" data-date="${p.date}"/>`
    ).join('');
    
    // Generate axis labels
    const minLabel = `<text x="${padding}" y="${padding + chartHeight + 20}" font-size="12" fill="#666">${min}</text>`;
    const maxLabel = `<text x="${padding}" y="${padding - 10}" font-size="12" fill="#666">${max}</text>`;
    
    return `
        <svg class="performance-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background: #fff; border: 3px solid #000;" oncontextmenu="showColorPicker(event); return false;">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#000" stroke-width="2"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#000" stroke-width="2"/>
            
            <!-- Sales line -->
            <polyline points="${linePoints}" fill="none" stroke="#000" stroke-width="3"/>
            
            <!-- Data points -->
            ${circles.replace(/fill="[^"]*"/g, 'fill="#000"')}
            
            <!-- Labels -->
            ${minLabel.replace(/fill="#666"/g, 'fill="#000" font-weight="bold"')}
            ${maxLabel.replace(/fill="#666"/g, 'fill="#000" font-weight="bold"')}
        </svg>
        <div class="performance-chart-legend" style="color: #000; font-weight: bold; text-transform: uppercase; font-size: 11px; margin-top: 8px;">
            <span>${sales.length} DATA POINT${sales.length !== 1 ? 'S' : ''}</span>
        </div>
    `;
}

function generatePriceChart(priceHistory, containerId) {
    // Generate a more detailed price chart for the modal
    if (!priceHistory || priceHistory.length === 0) {
        return '<p>No price history available</p>';
    }
    
    const prices = priceHistory.map(p => p.price).reverse(); // Oldest to newest
    const dates = priceHistory.map(p => p.date).reverse();
    const max = Math.max(...prices);
    const min = Math.min(...prices);
    const range = max - min || 1;
    
    const width = 400;
    const height = 200;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    
    // Generate points for the line
    const points = prices.map((price, i) => {
        // Handle single point case - center it horizontally
        const x = prices.length === 1 
            ? padding + chartWidth / 2 
            : padding + (i / (prices.length - 1)) * chartWidth;
        const y = padding + chartHeight - ((price - min) / range) * chartHeight;
        return { x, y, price, date: dates[i] };
    });
    
    const linePoints = points.map(p => `${p.x},${p.y}`).join(' ');
    
    // Generate circles for data points
    const circles = points.map(p => 
        `<circle cx="${p.x}" cy="${p.y}" r="4" fill="${getChartColor()}" class="price-point" data-price="${p.price}" data-date="${p.date}"/>`
    ).join('');
    
    // Generate axis labels
    const minLabel = `<text x="${padding}" y="${padding + chartHeight + 20}" font-size="12" fill="#666">$${min.toFixed(2)}</text>`;
    const maxLabel = `<text x="${padding}" y="${padding - 10}" font-size="12" fill="#666">$${max.toFixed(2)}</text>`;
    
    return `
        <svg class="price-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background: #fff; border: 3px solid #000;" oncontextmenu="showColorPicker(event); return false;">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#000" stroke-width="2"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#000" stroke-width="2"/>
            
            <!-- Price line -->
            <polyline points="${linePoints}" fill="none" stroke="#000" stroke-width="3"/>
            
            <!-- Data points -->
            ${circles.replace(/fill="[^"]*"/g, 'fill="#000"')}
            
            <!-- Labels -->
            ${minLabel.replace(/fill="#666"/g, 'fill="#000" font-weight="bold"')}
            ${maxLabel.replace(/fill="#666"/g, 'fill="#000" font-weight="bold"')}
        </svg>
        <div class="price-chart-legend" style="color: #000; font-weight: bold; text-transform: uppercase; font-size: 11px; margin-top: 8px;">
            <span>${prices.length} PRICE POINT${prices.length !== 1 ? 'S' : ''}</span>
        </div>
    `;
}

// Advanced Features Implementation

// Toggle Advanced Settings
function toggleAdvancedSettings() {
    const panel = document.getElementById('advancedSettings');
    const btn = document.getElementById('advancedSettingsBtn');
    
    if (panel.style.display === 'none') {
        // Detect mode and show appropriate weight section
        detectAndShowWeightSection();
        panel.style.display = 'block';
        btn.textContent = '⚙️ Hide Advanced Settings';
    } else {
        panel.style.display = 'none';
        btn.textContent = '⚙️ Advanced Settings';
    }
}

// Detect matching mode and show appropriate weight section
function detectAndShowWeightSection() {
    const visualSection = document.getElementById('visualWeightsSection');
    const metadataSection = document.getElementById('metadataWeightsSection');
    const hybridSection = document.getElementById('hybridWeightsSection');
    
    // Hide all sections first
    visualSection.style.display = 'none';
    metadataSection.style.display = 'none';
    hybridSection.style.display = 'none';
    
    // Detect mode based on what's uploaded
    const hasHistoricalImages = historicalFiles.length > 0;
    const hasNewImages = newFiles.length > 0;
    const hasImages = hasHistoricalImages || hasNewImages;
    
    const hasHistoricalCsv = historicalCsv !== null;
    const hasNewCsv = newCsv !== null;
    const hasCsv = hasHistoricalCsv || hasNewCsv;
    
    const isAdvancedMode = historicalAdvancedMode || newAdvancedMode;
    
    if (isAdvancedMode) {
        if (hasImages && hasCsv) {
            // Mode 3: Hybrid (Images + CSV)
            hybridSection.style.display = 'block';
        } else if (hasCsv && !hasImages) {
            // Mode 2: Metadata only (CSV, no images)
            metadataSection.style.display = 'block';
        } else {
            // Fallback to visual (shouldn't happen in advanced mode without CSV)
            visualSection.style.display = 'block';
        }
    } else {
        // Mode 1: Visual only (Simple mode)
        visualSection.style.display = 'block';
    }
}

// Update Similarity Weights
function updateWeights() {
    const colorWeight = parseInt(document.getElementById('colorWeightSlider').value);
    const shapeWeight = parseInt(document.getElementById('shapeWeightSlider').value);
    const textureWeight = parseInt(document.getElementById('textureWeightSlider').value);
    
    document.getElementById('colorWeightValue').textContent = colorWeight;
    document.getElementById('shapeWeightValue').textContent = shapeWeight;
    document.getElementById('textureWeightValue').textContent = textureWeight;
    
    const total = colorWeight + shapeWeight + textureWeight;
    document.getElementById('weightTotal').textContent = total;
    
    const warning = document.getElementById('weightWarning');
    const totalDiv = document.querySelector('.weight-total');
    
    if (total !== 100) {
        warning.style.display = 'inline';
        totalDiv.classList.add('invalid');
    } else {
        warning.style.display = 'none';
        totalDiv.classList.remove('invalid');
        
        // Update state
        similarityWeights = {
            color: colorWeight / 100,
            shape: shapeWeight / 100,
            texture: textureWeight / 100
        };
        
        // Save to history
        saveToHistory('weights_changed', { weights: similarityWeights });
    }
}

// Update Metadata Weights (Mode 2)
function updateMetadataWeights() {
    const skuWeight = parseInt(document.getElementById('skuWeightSlider').value);
    const nameWeight = parseInt(document.getElementById('nameWeightSlider').value);
    const categoryWeight = parseInt(document.getElementById('categoryWeightSlider').value);
    const priceWeight = parseInt(document.getElementById('priceWeightSlider').value);
    const performanceWeight = parseInt(document.getElementById('performanceWeightSlider').value);
    
    document.getElementById('skuWeightValue').textContent = skuWeight;
    document.getElementById('nameWeightValue').textContent = nameWeight;
    document.getElementById('categoryWeightValue').textContent = categoryWeight;
    document.getElementById('priceWeightValue').textContent = priceWeight;
    document.getElementById('performanceWeightValue').textContent = performanceWeight;
    
    const total = skuWeight + nameWeight + categoryWeight + priceWeight + performanceWeight;
    document.getElementById('metadataWeightTotal').textContent = total;
    
    const warning = document.getElementById('metadataWeightWarning');
    
    if (total !== 100) {
        warning.style.display = 'inline';
    } else {
        warning.style.display = 'none';
    }
}

// Update Hybrid Weights (Mode 3)
function updateHybridWeights() {
    const visualWeight = parseInt(document.getElementById('hybridVisualWeightSlider').value);
    const metadataWeight = parseInt(document.getElementById('hybridMetadataWeightSlider').value);
    
    document.getElementById('hybridVisualWeightValue').textContent = visualWeight;
    document.getElementById('hybridMetadataWeightValue').textContent = metadataWeight;
    
    const total = visualWeight + metadataWeight;
    document.getElementById('hybridWeightTotal').textContent = total;
    
    const warning = document.getElementById('hybridWeightWarning');
    
    if (total !== 100) {
        warning.style.display = 'inline';
    } else {
        warning.style.display = 'none';
    }
}

// Update Hybrid Visual Sub-Weights
function updateHybridVisualSubWeights() {
    const colorWeight = parseInt(document.getElementById('hybridColorWeightSlider').value);
    const shapeWeight = parseInt(document.getElementById('hybridShapeWeightSlider').value);
    const textureWeight = parseInt(document.getElementById('hybridTextureWeightSlider').value);
    
    document.getElementById('hybridColorWeightValue').textContent = colorWeight;
    document.getElementById('hybridShapeWeightValue').textContent = shapeWeight;
    document.getElementById('hybridTextureWeightValue').textContent = textureWeight;
    
    const total = colorWeight + shapeWeight + textureWeight;
    document.getElementById('hybridVisualSubWeightTotal').textContent = total;
    
    const warning = document.getElementById('hybridVisualSubWeightWarning');
    
    if (total !== 100) {
        warning.style.display = 'inline';
    } else {
        warning.style.display = 'none';
    }
}

// Update Hybrid Metadata Sub-Weights
function updateHybridMetadataSubWeights() {
    const skuWeight = parseInt(document.getElementById('hybridSkuWeightSlider').value);
    const nameWeight = parseInt(document.getElementById('hybridNameWeightSlider').value);
    const categoryWeight = parseInt(document.getElementById('hybridCategoryWeightSlider').value);
    const priceWeight = parseInt(document.getElementById('hybridPriceWeightSlider').value);
    const performanceWeight = parseInt(document.getElementById('hybridPerformanceWeightSlider').value);
    
    document.getElementById('hybridSkuWeightValue').textContent = skuWeight;
    document.getElementById('hybridNameWeightValue').textContent = nameWeight;
    document.getElementById('hybridCategoryWeightValue').textContent = categoryWeight;
    document.getElementById('hybridPriceWeightValue').textContent = priceWeight;
    document.getElementById('hybridPerformanceWeightValue').textContent = performanceWeight;
    
    const total = skuWeight + nameWeight + categoryWeight + priceWeight + performanceWeight;
    document.getElementById('hybridMetadataSubWeightTotal').textContent = total;
    
    const warning = document.getElementById('hybridMetadataSubWeightWarning');
    
    if (total !== 100) {
        warning.style.display = 'inline';
    } else {
        warning.style.display = 'none';
    }
}

// Reset Weights to Default
function resetWeights() {
    // Detect which mode is active and reset accordingly
    const visualSection = document.getElementById('visualWeightsSection');
    const metadataSection = document.getElementById('metadataWeightsSection');
    const hybridSection = document.getElementById('hybridWeightsSection');
    
    if (visualSection.style.display !== 'none') {
        // Reset Mode 1 (Visual)
        document.getElementById('colorWeightSlider').value = 50;
        document.getElementById('shapeWeightSlider').value = 30;
        document.getElementById('textureWeightSlider').value = 20;
        updateWeights();
    }
    
    if (metadataSection.style.display !== 'none') {
        // Reset Mode 2 (Metadata)
        document.getElementById('skuWeightSlider').value = 30;
        document.getElementById('nameWeightSlider').value = 25;
        document.getElementById('categoryWeightSlider').value = 20;
        document.getElementById('priceWeightSlider').value = 15;
        document.getElementById('performanceWeightSlider').value = 10;
        updateMetadataWeights();
    }
    
    if (hybridSection.style.display !== 'none') {
        // Reset Mode 3 (Hybrid)
        document.getElementById('hybridVisualWeightSlider').value = 60;
        document.getElementById('hybridMetadataWeightSlider').value = 40;
        updateHybridWeights();
        
        // Reset sub-weights
        document.getElementById('hybridColorWeightSlider').value = 50;
        document.getElementById('hybridShapeWeightSlider').value = 30;
        document.getElementById('hybridTextureWeightSlider').value = 20;
        updateHybridVisualSubWeights();
        
        document.getElementById('hybridSkuWeightSlider').value = 30;
        document.getElementById('hybridNameWeightSlider').value = 25;
        document.getElementById('hybridCategoryWeightSlider').value = 20;
        document.getElementById('hybridPriceWeightSlider').value = 15;
        document.getElementById('hybridPerformanceWeightSlider').value = 10;
        updateHybridMetadataSubWeights();
    }
    
    showToast('Weights reset to default values', 'success');
}

// Initialize Advanced Features
function initAdvancedFeatures() {
    // Advanced Settings Button
    const advancedSettingsBtn = document.getElementById('advancedSettingsBtn');
    const resetWeightsBtn = document.getElementById('resetWeightsBtn');
    
    if (advancedSettingsBtn) advancedSettingsBtn.addEventListener('click', toggleAdvancedSettings);
    if (resetWeightsBtn) resetWeightsBtn.addEventListener('click', resetWeights);
    
    // Helper function to safely add event listener
    const safeAddListener = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, handler);
    };
    
    // Weight Sliders - Mode 1 (Visual)
    safeAddListener('colorWeightSlider', 'input', updateWeights);
    safeAddListener('shapeWeightSlider', 'input', updateWeights);
    safeAddListener('textureWeightSlider', 'input', updateWeights);
    
    // Weight Sliders - Mode 2 (Metadata)
    safeAddListener('skuWeightSlider', 'input', updateMetadataWeights);
    safeAddListener('nameWeightSlider', 'input', updateMetadataWeights);
    safeAddListener('categoryWeightSlider', 'input', updateMetadataWeights);
    safeAddListener('priceWeightSlider', 'input', updateMetadataWeights);
    safeAddListener('performanceWeightSlider', 'input', updateMetadataWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Main)
    safeAddListener('hybridVisualWeightSlider', 'input', updateHybridWeights);
    safeAddListener('hybridMetadataWeightSlider', 'input', updateHybridWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Visual Sub)
    safeAddListener('hybridColorWeightSlider', 'input', updateHybridVisualSubWeights);
    safeAddListener('hybridShapeWeightSlider', 'input', updateHybridVisualSubWeights);
    safeAddListener('hybridTextureWeightSlider', 'input', updateHybridVisualSubWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Metadata Sub)
    safeAddListener('hybridSkuWeightSlider', 'input', updateHybridMetadataSubWeights);
    safeAddListener('hybridNameWeightSlider', 'input', updateHybridMetadataSubWeights);
    safeAddListener('hybridCategoryWeightSlider', 'input', updateHybridMetadataSubWeights);
    safeAddListener('hybridPriceWeightSlider', 'input', updateHybridMetadataSubWeights);
    safeAddListener('hybridPerformanceWeightSlider', 'input', updateHybridMetadataSubWeights);
    
    // Export Buttons
    safeAddListener('exportCsvBtn', 'click', exportResults);
    safeAddListener('exportWithImagesBtn', 'click', exportWithImages);
    safeAddListener('duplicateReportBtn', 'click', showDuplicateReport);
    
    // Session Management
    safeAddListener('saveSessionBtn', 'click', saveSession);
    safeAddListener('loadSessionBtn', 'click', loadSession);
    
    // Search and Filter
    safeAddListener('searchInput', 'input', applyFilters);
    safeAddListener('categoryFilter', 'change', applyFilters);
    safeAddListener('sortBySelect', 'change', applyFilters);
    safeAddListener('duplicatesOnlyCheckbox', 'change', applyFilters);
    safeAddListener('clearFiltersBtn', 'click', clearFilters);
}

// Export with Images
async function exportWithImages() {
    if (matchResults.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }
    
    showToast('Preparing export with images... This may take a moment.', 'info');
    
    try {
        // Create a zip file using JSZip (we'll need to include this library)
        // For now, we'll create a folder structure guide
        
        let exportData = {
            timestamp: new Date().toISOString(),
            weights: similarityWeights,
            threshold: parseInt(document.getElementById('thresholdSlider').value),
            results: []
        };
        
        for (const result of matchResults) {
            const product = result.product;
            const matches = result.matches;
            
            exportData.results.push({
                product: {
                    id: product.id,
                    filename: product.filename,
                    category: product.category,
                    sku: product.sku,
                    name: product.name
                },
                matches: matches.map(m => ({
                    product_id: m.product_id,
                    product_name: m.product_name,
                    similarity_score: m.similarity_score,
                    color_score: m.color_score,
                    shape_score: m.shape_score,
                    texture_score: m.texture_score,
                    priceStatistics: m.priceStatistics,
                    performanceStatistics: m.performanceStatistics
                }))
            });
        }
        
        // Export JSON with instructions
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = `match_results_full_${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            showToast('Export complete! JSON file includes all match data. Images can be downloaded separately via API.', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showToast('Export failed', 'error');
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    } catch (error) {
        showToast('Failed to export with images: ' + error.message, 'error');
    }
}

// Show Duplicate Report
function showDuplicateReport() {
    if (matchResults.length === 0) {
        showToast('No results to analyze', 'warning');
        return;
    }
    
    // Find all duplicates (similarity > 90%)
    const duplicates = [];
    
    matchResults.forEach(result => {
        const product = result.product;
        const highMatches = result.matches.filter(m => m.similarity_score > 90);
        
        if (highMatches.length > 0) {
            duplicates.push({
                product: product,
                matches: highMatches
            });
        }
    });
    
    if (duplicates.length === 0) {
        showToast('No potential duplicates found (similarity > 90%)', 'info');
        return;
    }
    
    // Create modal content
    const modal = document.getElementById('detailModal');
    const modalBody = document.getElementById('modalBody');
    
    let html = `
        <div class="duplicate-report-modal">
            <div class="duplicate-report-header">
                <h2>🔍 Duplicate Detection Report</h2>
                <p style="color: #64748b; font-size: 16px;">${duplicates.length} product(s) with potential duplicates found</p>
            </div>
            
            <div class="rank-filters">
                <div class="rank-filter-group">
                    <label>Sort By:</label>
                    <select id="duplicateSortSelect" onchange="sortDuplicates()">
                        <option value="similarity">Highest Similarity</option>
                        <option value="price">Price (if available)</option>
                        <option value="performance">Performance (if available)</option>
                    </select>
                </div>
                
                <div class="rank-filter-group">
                    <label>Min Similarity:</label>
                    <select id="duplicateThresholdSelect" onchange="filterDuplicates()">
                        <option value="90">90%+</option>
                        <option value="95">95%+</option>
                        <option value="98">98%+</option>
                    </select>
                </div>
            </div>
            
            <div id="duplicatesList">
    `;
    
    duplicates.forEach(dup => {
        const product = dup.product;
        
        dup.matches.forEach(match => {
            html += `
                <div class="duplicate-item" data-similarity="${match.similarity_score}">
                    <div class="duplicate-images">
                        <img src="/api/products/${product.id}/image" alt="${product.filename}">
                        <img src="/api/products/${match.product_id}/image" alt="${match.product_name}">
                    </div>
                    <div class="duplicate-info">
                        <h4>Potential Duplicate Detected</h4>
                        <div class="duplicate-score">${match.similarity_score.toFixed(1)}% Similar</div>
                        <div class="duplicate-details">
                            <p><strong>New Product:</strong> ${escapeHtml(product.filename)} ${product.category ? `(${product.category})` : ''}</p>
                            <p><strong>Matched Product:</strong> ${escapeHtml(match.product_name || 'Unknown')} ${match.category ? `(${match.category})` : ''}</p>
                            ${match.priceStatistics ? `
                                <p><strong>Price:</strong> ${match.priceStatistics.current} (Trend: ${match.priceStatistics.trend})</p>
                            ` : ''}
                            ${match.performanceStatistics ? `
                                <p><strong>Performance:</strong> ${match.performanceStatistics.total_sales} sales, ${match.performanceStatistics.total_revenue} revenue</p>
                            ` : ''}
                            <p><strong>Recommendation:</strong> ${match.similarity_score > 95 ? 'Very likely duplicate - review carefully' : 'Possible duplicate - manual review recommended'}</p>
                        </div>
                    </div>
                </div>
            `;
        });
    });
    
    html += `
            </div>
            
            <div style="margin-top: 24px; text-align: center;">
                <button class="btn btn-primary" onclick="exportDuplicateReport()">📄 Export Duplicate Report</button>
            </div>
        </div>
    `;
    
    modalBody.innerHTML = html;
    modal.classList.add('show');
}

// Export Duplicate Report
function exportDuplicateReport() {
    const duplicates = [];
    
    matchResults.forEach(result => {
        const product = result.product;
        const highMatches = result.matches.filter(m => m.similarity_score > 90);
        
        if (highMatches.length > 0) {
            highMatches.forEach(match => {
                duplicates.push({
                    new_product: product.filename,
                    new_category: product.category || 'Uncategorized',
                    matched_product: match.product_name || 'Unknown',
                    matched_category: match.category || 'Uncategorized',
                    similarity_score: match.similarity_score.toFixed(1),
                    price_current: match.priceStatistics?.current || 'N/A',
                    price_trend: match.priceStatistics?.trend || 'N/A',
                    total_sales: match.performanceStatistics?.total_sales || 'N/A',
                    total_revenue: match.performanceStatistics?.total_revenue || 'N/A',
                    recommendation: match.similarity_score > 95 ? 'Very likely duplicate' : 'Possible duplicate'
                });
            });
        }
    });
    
    let csv = 'New Product,New Category,Matched Product,Matched Category,Similarity Score,Price Current,Price Trend,Total Sales,Avg Sales,Sales Trend,Recommendation\n';
    
    duplicates.forEach(dup => {
        csv += `"${dup.new_product}","${dup.new_category}","${dup.matched_product}","${dup.matched_category}",${dup.similarity_score},"${dup.price_current}","${dup.price_trend}","${dup.total_sales}","${dup.total_revenue}","${dup.recommendation}"\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    try {
        const a = document.createElement('a');
        a.href = url;
        a.download = `duplicate_report_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        showToast('Duplicate report exported to CSV', 'success');
    } catch (error) {
        console.error('Export failed:', error);
        showToast('Export failed', 'error');
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 100);
    }
}

// Save Session
function saveSession() {
    if (matchResults.length === 0) {
        showToast('No session data to save', 'warning');
        return;
    }
    
    const sessionData = {
        version: '1.0',
        timestamp: new Date().toISOString(),
        weights: similarityWeights,
        threshold: parseInt(document.getElementById('thresholdSlider').value),
        limit: parseInt(document.getElementById('limitSelect').value),
        historicalProducts: historicalProducts,
        newProducts: newProducts,
        matchResults: matchResults
    };
    
    const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    try {
        const a = document.createElement('a');
        a.href = url;
        a.download = `matching_session_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        showToast('Session saved successfully', 'success');
    } catch (error) {
        console.error('Save failed:', error);
        showToast('Save failed', 'error');
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 100);
    }
}

// Load Session
function loadSession() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        try {
            const text = await file.text();
            const sessionData = JSON.parse(text);
            
            // Validate session data
            if (!sessionData.version || !sessionData.matchResults) {
                throw new Error('Invalid session file format');
            }
            
            // Restore state
            similarityWeights = sessionData.weights || { color: 0.5, shape: 0.3, texture: 0.2 };
            historicalProducts = sessionData.historicalProducts || [];
            newProducts = sessionData.newProducts || [];
            matchResults = sessionData.matchResults || [];
            
            // Update UI
            if (sessionData.threshold) {
                document.getElementById('thresholdSlider').value = sessionData.threshold;
                document.getElementById('thresholdValue').textContent = sessionData.threshold;
            }
            
            if (sessionData.limit) {
                document.getElementById('limitSelect').value = sessionData.limit;
            }
            
            // Update weights UI
            document.getElementById('colorWeightSlider').value = Math.round(similarityWeights.color * 100);
            document.getElementById('shapeWeightSlider').value = Math.round(similarityWeights.shape * 100);
            document.getElementById('textureWeightSlider').value = Math.round(similarityWeights.texture * 100);
            updateWeights();
            
            // Display results
            displayResults();
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
            
            showToast(`Session loaded: ${matchResults.length} products with matches`, 'success');
            
            // Save to history
            saveToHistory('session_loaded', { timestamp: sessionData.timestamp });
        } catch (error) {
            showToast('Failed to load session: ' + error.message, 'error');
        }
    };
    
    input.click();
}

// Apply Filters
function applyFilters() {
    searchQuery = document.getElementById('searchInput').value.toLowerCase();
    filterCategory = document.getElementById('categoryFilter').value;
    filterDuplicatesOnly = document.getElementById('duplicatesOnlyCheckbox').checked;
    sortBy = document.getElementById('sortBySelect').value;
    
    // Re-render results with filters
    displayResults();
    
    // Save to history
    saveToHistory('filters_applied', { searchQuery, filterCategory, filterDuplicatesOnly, sortBy });
}

// Clear Filters
function clearFilters() {
    document.getElementById('searchInput').value = '';
    document.getElementById('categoryFilter').value = 'all';
    document.getElementById('duplicatesOnlyCheckbox').checked = false;
    document.getElementById('sortBySelect').value = 'similarity';
    
    applyFilters();
    showToast('Filters cleared', 'success');
}

// Populate Category Filter
function populateCategoryFilter() {
    const categories = new Set();
    
    matchResults.forEach(result => {
        if (result.product.category) {
            categories.add(result.product.category);
        }
    });
    
    const select = document.getElementById('categoryFilter');
    select.innerHTML = '<option value="all">All Categories</option>';
    
    Array.from(categories).sort().forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    });
}

// Filter and Sort Results
function filterAndSortResults(results) {
    let filtered = results.map(result => {
        // Apply dynamic threshold and limit to matches
        let filteredMatches = result.matches.filter(m => m.similarity_score >= dynamicThreshold);
        
        // Apply dynamic category filter to matches
        if (dynamicCategory) {
            filteredMatches = filteredMatches.filter(m => (m.category || 'Uncategorized') === dynamicCategory);
        }
        
        // Apply dynamic search filter to matches (using backend search results)
        if (dynamicSearch && dynamicSearchResults.size > 0) {
            filteredMatches = filteredMatches.filter(m => {
                return dynamicSearchResults.has(m.matched_product_id);
            });
        }
        
        // Apply dynamic limit
        if (dynamicLimit > 0) {
            filteredMatches = filteredMatches.slice(0, dynamicLimit);
        }
        
        return {
            ...result,
            matches: filteredMatches
        };
    }).filter(result => {
        const product = result.product;
        
        // Search filter
        if (searchQuery) {
            const searchText = `${product.filename} ${product.name || ''} ${product.sku || ''} ${product.category || ''}`.toLowerCase();
            if (!searchText.includes(searchQuery)) {
                return false;
            }
        }
        
        // Category filter
        if (filterCategory !== 'all' && product.category !== filterCategory) {
            return false;
        }
        
        // Duplicates only filter
        if (filterDuplicatesOnly) {
            const hasDuplicate = result.matches.some(m => m.similarity_score > 90);
            if (!hasDuplicate) {
                return false;
            }
        }
        
        return true;
    });
    
    // Sort results
    filtered.sort((a, b) => {
        const aMatch = a.matches[0];
        const bMatch = b.matches[0];
        
        if (!aMatch && !bMatch) return 0;
        if (!aMatch) return 1;
        if (!bMatch) return -1;
        
        switch (sortBy) {
            case 'similarity':
                return bMatch.similarity_score - aMatch.similarity_score;
            
            case 'price':
                const aPrice = aMatch.priceStatistics?.current_numeric || 0;
                const bPrice = bMatch.priceStatistics?.current_numeric || 0;
                return bPrice - aPrice;
            
            case 'performance':
                const aPerf = aMatch.performanceStatistics?.total_sales || 0;
                const bPerf = bMatch.performanceStatistics?.total_sales || 0;
                return bPerf - aPerf;
            
            case 'name':
                return (a.product.filename || '').localeCompare(b.product.filename || '');
            
            default:
                return 0;
        }
    });
    
    return filtered;
}

// History Management (Undo/Redo)
function saveToHistory(action, data) {
    // Remove any history after current index
    historyStack = historyStack.slice(0, historyIndex + 1);
    
    // Add new history entry
    historyStack.push({
        action: action,
        data: data,
        timestamp: Date.now(),
        state: {
            weights: { ...similarityWeights },
            searchQuery: searchQuery,
            filterCategory: filterCategory,
            filterDuplicatesOnly: filterDuplicatesOnly,
            sortBy: sortBy
        }
    });
    
    // Limit history size
    if (historyStack.length > MAX_HISTORY) {
        historyStack.shift();
    } else {
        historyIndex++;
    }
    
    updateUndoRedoButtons();
}

function undo() {
    if (historyIndex > 0) {
        historyIndex--;
        const entry = historyStack[historyIndex];
        restoreState(entry.state);
        showToast(`Undo: ${entry.action}`, 'info');
    }
}

function redo() {
    if (historyIndex < historyStack.length - 1) {
        historyIndex++;
        const entry = historyStack[historyIndex];
        restoreState(entry.state);
        showToast(`Redo: ${entry.action}`, 'info');
    }
}

function restoreState(state) {
    similarityWeights = { ...state.weights };
    searchQuery = state.searchQuery;
    filterCategory = state.filterCategory;
    filterDuplicatesOnly = state.filterDuplicatesOnly;
    sortBy = state.sortBy;
    
    // Update UI
    document.getElementById('colorWeightSlider').value = Math.round(similarityWeights.color * 100);
    document.getElementById('shapeWeightSlider').value = Math.round(similarityWeights.shape * 100);
    document.getElementById('textureWeightSlider').value = Math.round(similarityWeights.texture * 100);
    updateWeights();
    
    document.getElementById('searchInput').value = searchQuery;
    document.getElementById('categoryFilter').value = filterCategory;
    document.getElementById('duplicatesOnlyCheckbox').checked = filterDuplicatesOnly;
    document.getElementById('sortBySelect').value = sortBy;
    
    applyFilters();
}

function updateUndoRedoButtons() {
    // This would update undo/redo button states if we add them to the UI
    // For now, we'll just track the state
}

// Initialize advanced features on page load
document.addEventListener('DOMContentLoaded', () => {
    initAdvancedFeatures();
});

// Toggle help text in CSV format modal
function toggleHelp(helpId) {
    const helpElement = document.getElementById(helpId);
    if (helpElement) {
        helpElement.style.display = helpElement.style.display === 'none' ? 'block' : 'none';
    }
}

// Color picker for charts
function showColorPicker(event) {
    event.preventDefault();
    
    // Remove existing picker if any
    const existing = document.getElementById('chartColorPicker');
    if (existing) existing.remove();
    
    // Create color picker popup
    const picker = document.createElement('div');
    picker.id = 'chartColorPicker';
    picker.style.position = 'fixed';
    picker.style.left = event.clientX + 'px';
    picker.style.top = event.clientY + 'px';
    picker.style.background = '#fff';
    picker.style.border = '3px solid #000';
    picker.style.padding = '15px';
    picker.style.zIndex = '10000';
    
    picker.innerHTML = `
        <div style="font-family: 'Courier New', monospace; font-weight: bold; margin-bottom: 10px;">CHART COLOR</div>
        <input type="color" id="colorInput" value="${getChartColor()}" style="width: 100px; height: 40px; border: 2px solid #000; cursor: pointer;">
        <div style="margin-top: 10px; display: flex; gap: 5px; flex-wrap: wrap;">
            <button onclick="setChartColor('#FF0000'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #FF0000; border: 2px solid #000; cursor: pointer;"></button>
            <button onclick="setChartColor('#0066FF'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #0066FF; border: 2px solid #000; cursor: pointer;"></button>
            <button onclick="setChartColor('#00FF00'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #00FF00; border: 2px solid #000; cursor: pointer;"></button>
            <button onclick="setChartColor('#FF00FF'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #FF00FF; border: 2px solid #000; cursor: pointer;"></button>
            <button onclick="setChartColor('#FFFF00'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #FFFF00; border: 2px solid #000; cursor: pointer;"></button>
            <button onclick="setChartColor('#FF6600'); document.getElementById('chartColorPicker').remove();" style="width: 30px; height: 30px; background: #FF6600; border: 2px solid #000; cursor: pointer;"></button>
        </div>
        <button onclick="document.getElementById('chartColorPicker').remove();" style="margin-top: 10px; padding: 8px 15px; background: #000; color: #fff; border: none; font-family: 'Courier New', monospace; font-weight: bold; cursor: pointer; width: 100%;">CLOSE</button>
    `;
    
    document.body.appendChild(picker);
    
    // Handle color input change
    document.getElementById('colorInput').addEventListener('change', (e) => {
        setChartColor(e.target.value);
        picker.remove();
    });
    
    // Close on click outside
    const closePickerOutside = (e) => {
        if (!picker.contains(e.target)) {
            picker.remove();
            document.removeEventListener('click', closePickerOutside);
        }
    };
    
    setTimeout(() => {
        addTrackedListener(document, 'click', closePickerOutside, 'results');
    }, 100);
}

// Update file input label when file is selected
function updateFileLabel(input, labelId) {
    const label = document.getElementById(labelId);
    if (input.files && input.files.length > 0) {
        label.textContent = '✓ ' + input.files[0].name;
        label.classList.add('has-file');
    } else {
        label.textContent = 'Use BUILD CSV or see CSV FORMAT to create your file';
        label.classList.remove('has-file');
    }
}

// Mode Toggle Functions
function setMode(section, mode) {
    console.log(`setMode called: section=${section}, mode=${mode}`);
    
    // Mode can be: 'visual', 'metadata', or 'hybrid'
    const isMetadataMode = mode === 'metadata';
    const isHybridMode = mode === 'hybrid';
    const isVisualMode = mode === 'visual';
    
    if (section === 'historical') {
        // Update mode state (keep backward compatibility with 'advanced')
        historicalAdvancedMode = isMetadataMode || isHybridMode;
        historicalMode = mode; // Track current mode globally
        
        const toggle = document.getElementById('historicalModeToggle');
        const csvBox = document.getElementById('historicalCsvBox');
        const dropZone = document.getElementById('historicalDropZone');
        const processBtn = document.getElementById('processHistoricalBtn');
        
        // Update toggle buttons
        const buttons = toggle.querySelectorAll('.mode-option');
        buttons.forEach(btn => {
            btn.classList.remove('active');
            const btnText = btn.textContent.trim().toUpperCase();
            if ((btnText === 'VISUAL' && isVisualMode) || 
                (btnText === 'METADATA' && isMetadataMode) ||
                (btnText === 'HYBRID' && isHybridMode)) {
                btn.classList.add('active');
            }
        });
        
        // Show/hide UI elements based on mode
        const catalogOptions = document.getElementById('catalogOptions');
        const folderTip = document.querySelector('#historicalSection .folder-tip');
        
        if (isMetadataMode) {
            // Metadata mode: CSV only, hide image upload and catalog management
            csvBox.style.display = 'block';
            dropZone.style.display = 'none';
            if (catalogOptions) catalogOptions.style.display = 'none';
            if (folderTip) folderTip.style.display = 'none';
            // In metadata mode, default to "replace" since we're uploading new data
            const replaceRadio = document.querySelector('input[name="catalogLoadOption"][value="replace"]');
            if (replaceRadio) replaceRadio.checked = true;
            processBtn.disabled = !historicalCsv;
            if (!historicalCsv) {
                showToast('Metadata Mode: Upload CSV file (no images needed)', 'info');
            }
        } else if (isHybridMode) {
            // Hybrid mode: Both CSV and images
            csvBox.style.display = 'block';
            dropZone.style.display = 'block';
            if (catalogOptions) catalogOptions.style.display = 'block';
            if (folderTip) folderTip.style.display = 'block';
            processBtn.disabled = !historicalCsv;
            if (!historicalCsv) {
                showToast('Hybrid Mode: Upload CSV + images for combined matching', 'info');
            }
        } else {
            // Visual mode: Images only, hide CSV
            csvBox.style.display = 'none';
            dropZone.style.display = 'block';
            if (catalogOptions) catalogOptions.style.display = 'block';
            if (folderTip) folderTip.style.display = 'block';
            processBtn.disabled = historicalFiles.length === 0;
        }
    } else if (section === 'new') {
        // Update mode state (keep backward compatibility with 'advanced')
        newAdvancedMode = isMetadataMode || isHybridMode;
        newMode = mode; // Track current mode globally
        
        const toggle = document.getElementById('newModeToggle');
        const csvBox = document.getElementById('newCsvBox');
        const dropZone = document.getElementById('newDropZone');
        const processBtn = document.getElementById('processNewBtn');
        
        // Update toggle buttons
        const buttons = toggle.querySelectorAll('.mode-option');
        buttons.forEach(btn => {
            btn.classList.remove('active');
            const btnText = btn.textContent.trim().toUpperCase();
            if ((btnText === 'VISUAL' && isVisualMode) || 
                (btnText === 'METADATA' && isMetadataMode) ||
                (btnText === 'HYBRID' && isHybridMode)) {
                btn.classList.add('active');
            }
        });
        
        // Show/hide UI elements based on mode
        const newCatalogOptions = document.getElementById('newCatalogOptions');
        const newFolderTip = document.querySelector('#newSection .folder-tip');
        
        if (isMetadataMode) {
            // Metadata mode: CSV only, hide image upload and catalog management
            csvBox.style.display = 'block';
            dropZone.style.display = 'none';
            if (newCatalogOptions) newCatalogOptions.style.display = 'none';
            if (newFolderTip) newFolderTip.style.display = 'none';
            // In metadata mode, default to "replace" since we're uploading new data
            const replaceRadio = document.querySelector('input[name="newCatalogLoadOption"][value="replace"]');
            if (replaceRadio) replaceRadio.checked = true;
            processBtn.disabled = !newCsv;
            if (!newCsv) {
                showToast('Metadata Mode: Upload CSV file (no images needed)', 'info');
            }
        } else if (isHybridMode) {
            // Hybrid mode: Both CSV and images
            csvBox.style.display = 'block';
            dropZone.style.display = 'block';
            if (newCatalogOptions) newCatalogOptions.style.display = 'block';
            if (newFolderTip) newFolderTip.style.display = 'block';
            processBtn.disabled = !newCsv;
            if (!newCsv) {
                showToast('Hybrid Mode: Upload CSV + images for combined matching', 'info');
            }
        } else {
            // Visual mode: Images only, hide CSV
            csvBox.style.display = 'none';
            dropZone.style.display = 'block';
            if (newCatalogOptions) newCatalogOptions.style.display = 'block';
            if (newFolderTip) newFolderTip.style.display = 'block';
            processBtn.disabled = newFiles.length === 0;
        }
    }
    
    // Sync mode to the other section (historical <-> new)
    if (section === 'historical') {
        // Also update new section to match
        const newToggle = document.getElementById('newModeToggle');
        if (newToggle) {
            const newButtons = newToggle.querySelectorAll('.mode-option');
            newButtons.forEach(btn => {
                btn.classList.remove('active');
                const btnText = btn.textContent.trim().toUpperCase();
                if ((btnText === 'VISUAL' && isVisualMode) || 
                    (btnText === 'METADATA' && isMetadataMode) ||
                    (btnText === 'HYBRID' && isHybridMode)) {
                    btn.classList.add('active');
                }
            });
        }
        // Update new section mode state
        newAdvancedMode = isMetadataMode || isHybridMode;
    } else if (section === 'new') {
        // Also update historical section to match
        const histToggle = document.getElementById('historicalModeToggle');
        if (histToggle) {
            const histButtons = histToggle.querySelectorAll('.mode-option');
            histButtons.forEach(btn => {
                btn.classList.remove('active');
                const btnText = btn.textContent.trim().toUpperCase();
                if ((btnText === 'VISUAL' && isVisualMode) || 
                    (btnText === 'METADATA' && isMetadataMode) ||
                    (btnText === 'HYBRID' && isHybridMode)) {
                    btn.classList.add('active');
                }
            });
        }
        // Update historical section mode state
        historicalAdvancedMode = isMetadataMode || isHybridMode;
    }
    
    // Save state to localStorage
    saveMainAppState();
}

// Prompt for CSV Builder after folder upload in advanced mode
async function promptCsvBuilder(section) {
    const files = section === 'historical' ? historicalFiles : newFiles;
    
    if (files.length === 0) return;
    
    // Create custom modal for prompt
    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <h2>Add Product Metadata?</h2>
            <p>You've uploaded ${files.length} images in Advanced Mode.</p>
            <p><strong>Would you like to add metadata now?</strong></p>
            <ul style="text-align: left; margin: 20px 0;">
                <li>SKU and product names</li>
                <li>Price history</li>
                <li>Sales performance data</li>
            </ul>
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px;">
                <button class="btn" onclick="openIntegratedCsvBuilder('${section}')">YES, ADD METADATA</button>
                <button class="btn" onclick="closePromptModal()">NO, I'LL UPLOAD CSV</button>
            </div>
            <p style="margin-top: 15px; font-size: 0.9em; color: #718096;">You can also upload a CSV file manually or click "BUILD CSV" button.</p>
        </div>
    `;
    document.body.appendChild(modal);
    
    // Store modal reference for cleanup
    window.currentPromptModal = modal;
}

function closePromptModal() {
    if (window.currentPromptModal) {
        window.currentPromptModal.remove();
        window.currentPromptModal = null;
    }
}

// Open Integrated CSV Builder
function openIntegratedCsvBuilder(section) {
    closePromptModal();
    
    const files = section === 'historical' ? historicalFiles : newFiles;
    
    // Store files in sessionStorage for CSV builder
    const fileData = files.map(({ file, category }) => ({
        filename: file.name,
        category: category || '',
        size: file.size,
        type: file.type
    }));
    
    sessionStorage.setItem('csvBuilderFiles', JSON.stringify(fileData));
    sessionStorage.setItem('csvBuilderSource', section);
    
    // Open CSV builder in new tab
    const builderWindow = window.open('/csv-builder', '_blank');
    
    // Listen for CSV data from builder
    const csvBuilderListener = (event) => {
        if (event.data && event.data.type === 'CSV_BUILDER_COMPLETE') {
            const csvContent = event.data.csvContent;
            const targetSection = event.data.section;
            
            // Create a Blob and File object from CSV content
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const file = new File([blob], 'products.csv', { type: 'text/csv' });
            
            // Set the CSV in the appropriate section
            if (targetSection === 'historical') {
                historicalCsv = file;
                document.getElementById('historicalFileLabel').textContent = 'products.csv (from CSV Builder)';
                document.getElementById('processHistoricalBtn').disabled = false;
                removeWorkflowIndicators('historical');
                showToast('CSV loaded from builder! Ready to process.', 'success');
            } else if (targetSection === 'new') {
                newCsv = file;
                document.getElementById('newFileLabel').textContent = 'products.csv (from CSV Builder)';
                document.getElementById('processNewBtn').disabled = false;
                removeWorkflowIndicators('new');
                showToast('CSV loaded from builder! Ready to process.', 'success');
            }
            
            // Remove listener after receiving data
            window.removeEventListener('message', csvBuilderListener);
        }
    };
    
    addTrackedListener(window, 'message', csvBuilderListener, 'general');
    
    showToast('CSV Builder opened. Complete the form and click "SEND TO APP".', 'info');
}

// Update file label when CSV is selected
function updateFileLabel(input, labelId) {
    const label = document.getElementById(labelId);
    if (input.files && input.files[0]) {
        label.textContent = input.files[0].name;
        
        // Update CSV state
        if (labelId === 'historicalFileLabel') {
            historicalCsv = input.files[0];
            if (historicalAdvancedMode && historicalFiles.length > 0) {
                document.getElementById('processHistoricalBtn').disabled = false;
                // Remove workflow indicators since CSV is now uploaded
                removeWorkflowIndicators('historical');
            }
        } else if (labelId === 'newFileLabel') {
            newCsv = input.files[0];
            if (newAdvancedMode && newFiles.length > 0) {
                document.getElementById('processNewBtn').disabled = false;
                // Remove workflow indicators since CSV is now uploaded
                removeWorkflowIndicators('new');
            }
        }
    } else {
        label.textContent = 'Use BUILD CSV or see CSV FORMAT to create your file';
    }
}

// Clear Folder Upload
function clearFolderUpload(section) {
    if (!confirm('Clear uploaded folder? This will reset all data for this section.')) {
        return;
    }
    
    if (section === 'historical') {
        // Clear state
        historicalFiles = [];
        historicalCsv = null;
        historicalProducts = [];
        
        // Clear UI
        document.getElementById('historicalInfo').innerHTML = '';
        document.getElementById('historicalInfo').classList.remove('show');
        document.getElementById('historicalFileLabel').textContent = 'Use BUILD CSV or see CSV FORMAT to create your file';
        document.getElementById('processHistoricalBtn').disabled = true;
        document.getElementById('historicalStatus').innerHTML = '';
        document.getElementById('historicalStatus').classList.remove('show');
        
        // Reset file input
        document.getElementById('historicalInput').value = '';
        document.getElementById('historicalCsvInput').value = '';
        
        showToast('Historical folder cleared', 'success');
        
    } else if (section === 'new') {
        // Clear state
        newFiles = [];
        newCsv = null;
        newProducts = [];
        
        // Clear UI
        document.getElementById('newInfo').innerHTML = '';
        document.getElementById('newInfo').classList.remove('show');
        document.getElementById('newFileLabel').textContent = 'Use BUILD CSV or see CSV FORMAT to create your file';
        document.getElementById('processNewBtn').disabled = true;
        document.getElementById('newStatus').innerHTML = '';
        document.getElementById('newStatus').classList.remove('show');
        
        // Reset file input
        document.getElementById('newInput').value = '';
        document.getElementById('newCsvInput').value = '';
        
        showToast('New products folder cleared', 'success');
    }
}

// Save state to localStorage
function saveMainAppState() {
    const state = {
        historicalMode,
        newMode,
        timestamp: new Date().toISOString()
    };
    localStorage.setItem('mainAppState', JSON.stringify(state));
}

// Load state from localStorage
function loadMainAppState() {
    const saved = localStorage.getItem('mainAppState');
    if (saved) {
        try {
            const state = JSON.parse(saved);
            
            // Restore mode settings with actual mode values
            if (state.historicalMode && ['visual', 'metadata', 'hybrid'].includes(state.historicalMode)) {
                setMode('historical', state.historicalMode);
            }
            if (state.newMode && ['visual', 'metadata', 'hybrid'].includes(state.newMode)) {
                setMode('new', state.newMode);
            }
        } catch (e) {
            console.error('Failed to load main app state:', e);
        }
    }
}

// Call on page load
document.addEventListener('DOMContentLoaded', () => {
    loadMainAppState();
});

// Show all files in the list
function showAllFiles(section, totalCount) {
    const files = section === 'historical' ? historicalFiles : newFiles;
    const listId = section === 'historical' ? 'historicalFileList' : 'newFileList';
    const list = document.getElementById(listId);
    
    if (!list) return;
    
    // Show all files
    list.innerHTML = files.map(({ file, category }) => 
        `<div>${escapeHtml(file.name)}${category ? ` <span style="color: #667eea;">[${category}]</span>` : ''}</div>`
    ).join('');
    
    // Remove the "Show All" button
    const button = list.nextElementSibling;
    if (button && button.querySelector('button')) {
        button.remove();
    }
    
    showToast(`Showing all ${totalCount} files`, 'success');
}

// Add visual workflow indicators for advanced mode
function addWorkflowIndicators(section) {
    const dropZoneId = section === 'historical' ? 'historicalDropZone' : 'newDropZone';
    const csvBoxId = section === 'historical' ? 'historicalCsvBox' : 'newCsvBox';
    
    // Dim the upload area (files already uploaded)
    const dropZone = document.getElementById(dropZoneId);
    if (dropZone) {
        dropZone.classList.add('upload-area-completed');
    }
    
    // Highlight the CSV box
    const csvBox = document.getElementById(csvBoxId);
    if (csvBox && !document.querySelector(`#${csvBoxId} .next-step-indicator`)) {
        // Add next step indicator
        const indicator = document.createElement('div');
        indicator.className = 'next-step-indicator';
        indicator.innerHTML = 'NEXT STEP: Add product metadata using CSV Builder or upload CSV file';
        csvBox.insertBefore(indicator, csvBox.firstChild);
        
        // Add highlight animation
        csvBox.classList.add('csv-box-highlight');
        
        // Remove animation after it completes
        setTimeout(() => {
            csvBox.classList.remove('csv-box-highlight');
        }, 6000);
    }
}

// Remove workflow indicators when CSV is uploaded
function removeWorkflowIndicators(section) {
    const dropZoneId = section === 'historical' ? 'historicalDropZone' : 'newDropZone';
    const csvBoxId = section === 'historical' ? 'historicalCsvBox' : 'newCsvBox';
    
    // Remove dim from upload area
    const dropZone = document.getElementById(dropZoneId);
    if (dropZone) {
        dropZone.classList.remove('upload-area-completed');
    }
    
    // Remove next step indicator
    const csvBox = document.getElementById(csvBoxId);
    if (csvBox) {
        const indicator = csvBox.querySelector('.next-step-indicator');
        if (indicator) {
            indicator.remove();
        }
        csvBox.classList.remove('csv-box-highlight');
    }
}


// GPU Status Initialization
function initGPUStatus() {
    const gpuStatusEl = document.getElementById('gpuStatus');
    if (!gpuStatusEl) return;
    
    // Fetch GPU status from backend
    fetch('/api/gpu/status')
        .then(response => response.json())
        .then(data => {
            updateGPUStatus(data);
        })
        .catch(error => {
            console.error('Failed to fetch GPU status:', error);
            updateGPUStatus({
                available: false,
                device: 'cpu',
                error: 'Failed to check GPU status'
            });
        });
}

function updateGPUStatus(status) {
    const gpuStatusEl = document.getElementById('gpuStatus');
    if (!gpuStatusEl) return;
    
    const statusIcon = gpuStatusEl.querySelector('.status-icon');
    const statusText = gpuStatusEl.querySelector('.status-text');
    
    // Remove all status classes
    gpuStatusEl.classList.remove('gpu-active', 'gpu-cpu', 'gpu-error');
    
    if (status.available && status.device !== 'cpu') {
        // GPU is active
        gpuStatusEl.classList.add('gpu-active');
        statusIcon.textContent = '⚡';
        
        let deviceName = 'GPU';
        let tooltip = 'GPU acceleration active';
        
        if (status.device === 'cuda') {
            deviceName = 'NVIDIA GPU';
            tooltip = `GPU: ${status.gpu_name || 'NVIDIA'} (CUDA) - ${status.throughput || 'N/A'} img/s`;
        } else if (status.device === 'rocm') {
            deviceName = 'AMD GPU';
            tooltip = `GPU: ${status.gpu_name || 'AMD'} (ROCm) - ${status.throughput || 'N/A'} img/s`;
        } else if (status.device === 'mps') {
            deviceName = 'Apple Silicon';
            tooltip = `GPU: ${status.gpu_name || 'Apple Silicon'} (MPS) - ${status.throughput || 'N/A'} img/s`;
        } else if (status.device === 'xpu') {
            deviceName = 'Intel GPU';
            tooltip = `GPU: ${status.gpu_name || 'Intel GPU'} (Intel Extension) - ${status.throughput || '30-80'} img/s`;
        }
        
        statusText.textContent = `${deviceName} Active`;
        gpuStatusEl.setAttribute('data-tooltip', tooltip);
        
        // Model is now pre-cached, no download notification needed
    } else if (status.error) {
        // GPU error
        gpuStatusEl.classList.add('gpu-error');
        statusIcon.textContent = '⚠️';
        statusText.textContent = 'GPU Error';
        gpuStatusEl.setAttribute('data-tooltip', `GPU initialization failed: ${status.error}. Using CPU mode.`);
    } else {
        // CPU mode
        gpuStatusEl.classList.add('gpu-cpu');
        statusIcon.textContent = '💻';
        statusText.textContent = 'CPU Mode';
        
        let tooltip = 'Running on CPU - ';
        if (status.throughput) {
            tooltip += `${status.throughput} img/s. `;
        }
        tooltip += 'For faster processing, see GPU Setup Guide.';
        
        gpuStatusEl.setAttribute('data-tooltip', tooltip);
    }
}

// Add processing speed display during batch operations
function updateProcessingSpeed(imagesProcessed, timeElapsed) {
    const gpuStatusEl = document.getElementById('gpuStatus');
    if (!gpuStatusEl) return;
    
    const statusText = gpuStatusEl.querySelector('.status-text');
    const speed = (imagesProcessed / (timeElapsed / 1000)).toFixed(1);
    
    // Temporarily show processing speed
    const originalText = statusText.textContent;
    statusText.textContent = `${speed} img/s`;
    
    // Restore original text after 3 seconds
    setTimeout(() => {
        statusText.textContent = originalText;
    }, 3000);
}

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
                statsText += ` | 📂 <strong>${data.loaded_snapshot.name}</strong>`;
            }
            
            statsEl.innerHTML = statsText;
            
            // Check for large database warning
            if (data.database_size_mb && data.database_size_mb > 500) {
                showToast('⚠️ Database is large (' + data.database_size_mb.toFixed(0) + ' MB). Consider cleaning up old products.', 'warning', 8000);
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

function handleCatalogOptionChange(e) {
    const option = e.target.value;
    const dropZone = document.getElementById('historicalDropZone');
    const processBtn = document.getElementById('processHistoricalBtn');
    
    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING CATALOG';
    } else if (option === 'replace') {
        // Replace catalog - show warning
        if (existingCatalogStats && existingCatalogStats.historical_products > 0) {
            const confirmed = confirm(
                `⚠️ WARNING: This will DELETE all ${existingCatalogStats.historical_products.toLocaleString()} existing historical products and create a NEW catalog!\n\n` +
                `A backup snapshot will be created automatically.\n\n` +
                `Are you sure you want to replace with a new catalog?`
            );
            if (!confirmed) {
                // Revert to use_existing
                document.querySelector('input[name="catalogLoadOption"][value="use_existing"]').checked = true;
                return;
            }
        }
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'REPLACE & PROCESS';
    } else {
        // Add to existing
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'ADD & PROCESS';
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
                `<p class="success">✓ Loaded ${historicalProducts.length} products from existing catalog</p>`;
            
            // Show next section
            document.getElementById('newSection').style.display = 'block';
            document.getElementById('newSection').scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error loading existing catalog:', error);
            showToast('Failed to load existing catalog', 'error');
        }
        return;
    }
    
    if (option === 'replace') {
        // Create automatic backup snapshot before replacing
        try {
            console.log('[REPLACE] Creating automatic backup snapshot...');
            showToast('Creating backup snapshot...', 'info');
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const snapshotName = `auto-backup-before-replace-historical-${timestamp}`;
            
            const snapshotResponse = await fetch('/api/catalogs/save-current', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name: snapshotName,
                    description: 'Automatic backup created before replacing historical catalog'
                })
            });
            
            if (snapshotResponse.ok) {
                console.log('[REPLACE] Backup snapshot created:', snapshotName);
                showToast('Backup snapshot created', 'success');
            } else {
                console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                showToast('Warning: Could not create backup snapshot', 'warning');
            }
            
            // Wait a moment to ensure snapshot is complete
            await new Promise(resolve => setTimeout(resolve, 300));
        } catch (error) {
            console.warn('[REPLACE] Error creating backup snapshot:', error);
            showToast('Warning: Could not create backup snapshot', 'warning');
            // Continue with replace even if snapshot fails
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

// ============ NEW PRODUCTS CATALOG OPTIONS ============

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
    } catch (error) {
        console.error('Error checking existing new catalog:', error);
    }
}

function handleNewCatalogOptionChange() {
    const option = getNewCatalogLoadOption();
    const dropZone = document.getElementById('newDropZone');
    const processBtn = document.getElementById('processNewBtn');
    
    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING NEW PRODUCTS';
    } else if (option === 'replace') {
        // Replacing - enable upload, show warning
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
        
        // Show warning
        if (existingCatalogStats && existingCatalogStats.new_products > 0) {
            const confirmed = confirm(
                `⚠️ WARNING: This will DELETE all ${existingCatalogStats.new_products} existing new products and create a NEW catalog!\n\n` +
                `A backup snapshot will be created automatically.\n\n` +
                `Are you sure you want to replace with a new catalog?`
            );
            if (!confirmed) {
                // Revert to add_to_existing
                document.querySelector('input[name="newCatalogLoadOption"][value="add_to_existing"]').checked = true;
                handleNewCatalogOptionChange();
                return;
            }
        }
    } else {
        // add_to_existing - enable upload
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
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
                `<p class="success">✓ Loaded ${newProducts.length} new products from existing catalog</p>`;
            
            // Show next section
            document.getElementById('matchSection').style.display = 'block';
            document.getElementById('matchSection').scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error loading existing new products:', error);
            showToast('Failed to load existing new products', 'error');
        }
        return;
    }
    
    if (option === 'replace') {
        // Create automatic backup snapshot before replacing
        try {
            console.log('[REPLACE] Creating automatic backup snapshot...');
            showToast('Creating backup snapshot...', 'info');
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const snapshotName = `auto-backup-before-replace-new-${timestamp}`;
            
            const snapshotResponse = await fetch('/api/catalogs/save-current', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name: snapshotName,
                    description: 'Automatic backup created before replacing new products catalog'
                })
            });
            
            if (snapshotResponse.ok) {
                console.log('[REPLACE] Backup snapshot created:', snapshotName);
                showToast('Backup snapshot created', 'success');
            } else {
                console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                showToast('Warning: Could not create backup snapshot', 'warning');
            }
            
            // Wait a moment to ensure snapshot is complete
            await new Promise(resolve => setTimeout(resolve, 300));
        } catch (error) {
            console.warn('[REPLACE] Error creating backup snapshot:', error);
            showToast('Warning: Could not create backup snapshot', 'warning');
            // Continue with replace even if snapshot fails
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
document.addEventListener('DOMContentLoaded', () => {
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


// ============ Catalog State Synchronization ============
// Ensures main app state stays in sync with database changes from Catalog Manager

// Store last known catalog state for change detection
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
function resetAppState(reason = 'Catalog data has changed') {
    console.log('Resetting app state:', reason);
    
    // Clear in-memory product data
    historicalProducts = [];
    newProducts = [];
    matchResults = [];
    historicalFiles = [];
    newFiles = [];
    historicalCsv = null;
    newCsv = null;
    
    // Reset UI to initial state
    const historicalSection = document.getElementById('historicalSection');
    const newSection = document.getElementById('newSection');
    const matchSection = document.getElementById('matchSection');
    const resultsSection = document.getElementById('resultsSection');
    
    if (newSection) newSection.style.display = 'none';
    if (matchSection) matchSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    
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

// Listen for visibility changes (user returns from Catalog Manager tab)
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
document.addEventListener('DOMContentLoaded', () => {
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
                text += ` | 📂 Loaded from: "${data.loaded_snapshot.name}"`;
            }
            
            summary.textContent = text;
            infoBar.style.display = 'block';
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

// Open Catalog Manager
function openCatalogManager() {
    window.open('/catalog-manager', '_blank');
}

// Listen for catalog changes from Catalog Manager
function initCatalogChangeListener() {
    // Listen via BroadcastChannel
    try {
        catalogChannel = new BroadcastChannel('catalog_changes');
        catalogChannel.onmessage = (event) => {
            handleCatalogChangeInMainApp(event.data);
        };
    } catch (e) {
        // BroadcastChannel not supported, use polling
        catalogPollingInterval = setInterval(checkCatalogChangesInMainApp, 2000);
    }
    
    // Also check on visibility change (when user switches back to this tab)
    const catalogVisibilityHandler = () => {
        if (!document.hidden) {
            checkCatalogChangesInMainApp();
            
            // Restart polling if it was stopped and BroadcastChannel not available
            if (!catalogChannel && !catalogPollingInterval) {
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

// Check for catalog changes via sessionStorage
function checkCatalogChangesInMainApp() {
    const changeData = sessionStorage.getItem('catalogManagerChange');
    if (changeData) {
        try {
            const change = JSON.parse(changeData);
            // Only process recent changes (within last 30 seconds)
            if (Date.now() - change.timestamp < 30000) {
                handleCatalogChangeInMainApp(change);
            }
        } catch (e) {
            console.error('Error parsing catalog change:', e);
        }
    }
}

// Handle catalog change notification in main app
function handleCatalogChangeInMainApp(change) {
    if (!change || !change.action) return;
    
    // Refresh catalog info
    loadCatalogInfo();
    
    // If a new catalog was loaded, show prominent notification
    if (change.action === 'catalog_loaded') {
        showToast(`Catalog "${change.details?.name || 'snapshot'}" loaded with ${change.details?.productCount || 0} products`, 'success');
        
        // If user has already processed products, warn them
        if (historicalProducts.length > 0 || newProducts.length > 0) {
            showToast('Note: Your current session data may be from the previous catalog. Consider resetting.', 'warning');
        }
    } else if (change.action === 'cleanup' || change.action === 'bulk_delete') {
        showToast('Catalog was modified in Catalog Manager', 'info');
    }
}
