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

// Advanced features state
// Note: CLIP mode doesn't need color/shape/texture weights - handled by AI
let searchQuery = '';
let filterCategory = 'all';
let filterDuplicatesOnly = false;
let sortBy = 'similarity'; // similarity, price, performance
let sortOrder = 'desc';

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

// Historical Catalog Upload
function initHistoricalUpload() {
    const dropZone = document.getElementById('historicalDropZone');
    const input = document.getElementById('historicalInput');
    const browseBtn = document.getElementById('historicalBrowseBtn');
    const csvInput = document.getElementById('historicalCsvInput');
    const processBtn = document.getElementById('processHistoricalBtn');

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

    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        dropZone.classList.add('drop-success');
        setTimeout(() => dropZone.classList.remove('drop-success'), 500);
        handleHistoricalFiles(Array.from(e.dataTransfer.files));
    });

    input.addEventListener('change', (e) => {
        handleHistoricalFiles(Array.from(e.target.files));
    });

    csvInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            historicalCsv = e.target.files[0];
            showToast('CSV loaded for historical products', 'success');
        }
    });

    processBtn.addEventListener('click', processHistoricalCatalog);
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

    statusDiv.innerHTML = '<h4>Processing historical catalog...</h4><div class="progress-bar"><div class="progress-fill" id="historicalProgress"></div></div><p id="historicalProgressText">0 of ' + historicalFiles.length + ' processed</p><div class="spinner-inline"></div>';

    historicalProducts = [];
    const progressFill = document.getElementById('historicalProgress');
    const progressText = document.getElementById('historicalProgressText');

    for (let i = 0; i < historicalFiles.length; i++) {
        const { file, category } = historicalFiles[i];
        const metadata = categoryMap[file.name] || {};

        try {
            const formData = new FormData();
            formData.append('image', file);
            // Use folder-based category, or CSV override if provided
            const finalCategory = metadata.category || category;
            if (finalCategory) formData.append('category', finalCategory);
            if (metadata.sku) formData.append('sku', metadata.sku);
            if (metadata.name) formData.append('product_name', metadata.name);
            else formData.append('product_name', file.name); // Fallback to filename
            formData.append('is_historical', 'true');

            const response = await fetchWithRetry('/api/products/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const productId = data.product_id;
                
                historicalProducts.push({
                    id: productId,
                    filename: file.name,
                    category: finalCategory,
                    sku: metadata.sku,
                    name: metadata.name,
                    hasFeatures: data.feature_extraction_status === 'success',
                    hasPriceHistory: false
                });
                
                // Upload price history if present
                if (metadata.priceHistory && metadata.priceHistory.length > 0) {
                    try {
                        const priceResponse = await fetchWithRetry(`/api/products/${productId}/price-history`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ prices: metadata.priceHistory })
                        });
                        
                        if (priceResponse.ok) {
                            const priceData = await priceResponse.json();
                            // Update product to indicate it has price history
                            const product = historicalProducts.find(p => p.id === productId);
                            if (product) {
                                product.hasPriceHistory = true;
                            }
                            
                            // Show validation warnings if any
                            if (priceData.validation_errors && priceData.validation_errors.length > 0) {
                                console.warn(`Price validation warnings for ${file.name}:`, priceData.validation_errors);
                            }
                        } else {
                            // Non-critical error - log but continue
                            console.warn(`Failed to upload price history for ${file.name}: ${priceResponse.status}`);
                        }
                    } catch (error) {
                        // Non-critical error - log but continue
                        console.warn(`Failed to upload price history for ${file.name}:`, error);
                    }
                }
                
                // Upload performance history if present
                if (metadata.performanceHistory && metadata.performanceHistory.length > 0) {
                    try {
                        const perfResponse = await fetchWithRetry(`/api/products/${productId}/performance-history`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ performance: metadata.performanceHistory })
                        });
                        
                        if (perfResponse.ok) {
                            const perfData = await perfResponse.json();
                            // Update product to indicate it has performance history
                            const product = historicalProducts.find(p => p.id === productId);
                            if (product) {
                                product.hasPerformanceHistory = true;
                            }
                            
                            // Show validation warnings if any
                            if (perfData.validation_errors && perfData.validation_errors.length > 0) {
                                console.warn(`Performance validation warnings for ${file.name}:`, perfData.validation_errors);
                            }
                        } else {
                            // Non-critical error - log but continue
                            console.warn(`Failed to upload performance history for ${file.name}: ${perfResponse.status}`);
                        }
                    } catch (error) {
                        // Non-critical error - log but continue
                        console.warn(`Failed to upload performance history for ${file.name}:`, error);
                    }
                }

                // Show warnings if any
                if (data.warning) {
                    showToast(`${file.name}: ${data.warning}`, 'warning');
                }
                if (data.warning_sku) {
                    showToast(`${file.name}: ${data.warning_sku}`, 'warning');
                }
                if (data.warning_category) {
                    showToast(`${file.name}: ${data.warning_category}`, 'warning');
                }
            } else {
                // Show error from backend
                const errorMsg = getUserFriendlyError(data.error_code || 'UNKNOWN_ERROR', data.error, data.suggestion);
                showToast(`${file.name}: ${errorMsg}`, 'error');
                console.error(`Failed to process ${file.name}:`, data);
            }
        } catch (error) {
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            showToast(`${file.name}: ${errorMsg}`, 'error');
            console.error(`Failed to process ${file.name}:`, error);
        }

        const progress = ((i + 1) / historicalFiles.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${i + 1} of ${historicalFiles.length} processed`;
    }

    const successful = historicalProducts.filter(p => p.hasFeatures).length;
    const failed = historicalFiles.length - historicalProducts.length;
    const withoutMetadata = historicalProducts.filter(p => !p.category && !p.sku).length;
    
    let statusMsg = `<h4>✓ Historical catalog processed</h4><p>${successful} products ready for matching`;
    if (failed > 0) statusMsg += ` (${failed} failed)`;
    if (withoutMetadata > 0) statusMsg += ` (${withoutMetadata} without CSV metadata)`;
    statusMsg += `</p>`;
    statusDiv.innerHTML = statusMsg;

    showToast(`Historical catalog ready: ${successful} products`, 'success');
    showLoadingSpinner(processBtn, false);

    // Show next step
    document.getElementById('newSection').style.display = 'block';
    document.getElementById('newSection').scrollIntoView({ behavior: 'smooth' });
}

// New Products Upload
function initNewUpload() {
    const dropZone = document.getElementById('newDropZone');
    const input = document.getElementById('newInput');
    const browseBtn = document.getElementById('newBrowseBtn');
    const csvInput = document.getElementById('newCsvInput');
    const processBtn = document.getElementById('processNewBtn');

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

    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        dropZone.classList.add('drop-success');
        setTimeout(() => dropZone.classList.remove('drop-success'), 500);
        handleNewFiles(Array.from(e.dataTransfer.files));
    });

    input.addEventListener('change', (e) => {
        handleNewFiles(Array.from(e.target.files));
    });

    csvInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            newCsv = e.target.files[0];
            showToast('CSV loaded for new products', 'success');
        }
    });

    processBtn.addEventListener('click', processNewProducts);
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

    statusDiv.innerHTML = '<h4>Processing new products...</h4><div class="progress-bar"><div class="progress-fill" id="newProgress"></div></div><p id="newProgressText">0 of ' + newFiles.length + ' processed</p><div class="spinner-inline"></div>';

    newProducts = [];
    const progressFill = document.getElementById('newProgress');
    const progressText = document.getElementById('newProgressText');

    for (let i = 0; i < newFiles.length; i++) {
        const { file, category } = newFiles[i];
        const metadata = categoryMap[file.name] || {};

        try {
            const formData = new FormData();
            formData.append('image', file);
            // Use folder-based category, or CSV override if provided
            const finalCategory = metadata.category || category;
            if (finalCategory) formData.append('category', finalCategory);
            if (metadata.sku) formData.append('sku', metadata.sku);
            if (metadata.name) formData.append('product_name', metadata.name);
            else formData.append('product_name', file.name); // Fallback to filename
            formData.append('is_historical', 'false');

            const response = await fetchWithRetry('/api/products/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const productId = data.product_id;
                
                newProducts.push({
                    id: productId,
                    filename: file.name,
                    category: finalCategory,
                    sku: metadata.sku,
                    name: metadata.name,
                    hasFeatures: data.feature_extraction_status === 'success',
                    hasPriceHistory: false
                });
                
                // Upload price history if present
                if (metadata.priceHistory && metadata.priceHistory.length > 0) {
                    try {
                        const priceResponse = await fetchWithRetry(`/api/products/${productId}/price-history`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ prices: metadata.priceHistory })
                        });
                        
                        if (priceResponse.ok) {
                            const priceData = await priceResponse.json();
                            // Update product to indicate it has price history
                            const product = newProducts.find(p => p.id === productId);
                            if (product) {
                                product.hasPriceHistory = true;
                            }
                            
                            // Show validation warnings if any
                            if (priceData.validation_errors && priceData.validation_errors.length > 0) {
                                console.warn(`Price validation warnings for ${file.name}:`, priceData.validation_errors);
                            }
                        } else {
                            // Non-critical error - log but continue
                            console.warn(`Failed to upload price history for ${file.name}: ${priceResponse.status}`);
                        }
                    } catch (error) {
                        // Non-critical error - log but continue
                        console.warn(`Failed to upload price history for ${file.name}:`, error);
                    }
                }
                
                // Upload performance history if present
                if (metadata.performanceHistory && metadata.performanceHistory.length > 0) {
                    try {
                        const perfResponse = await fetchWithRetry(`/api/products/${productId}/performance-history`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ performance: metadata.performanceHistory })
                        });
                        
                        if (perfResponse.ok) {
                            const perfData = await perfResponse.json();
                            // Update product to indicate it has performance history
                            const product = newProducts.find(p => p.id === productId);
                            if (product) {
                                product.hasPerformanceHistory = true;
                            }
                            
                            // Show validation warnings if any
                            if (perfData.validation_errors && perfData.validation_errors.length > 0) {
                                console.warn(`Performance validation warnings for ${file.name}:`, perfData.validation_errors);
                            }
                        } else {
                            // Non-critical error - log but continue
                            console.warn(`Failed to upload performance history for ${file.name}: ${perfResponse.status}`);
                        }
                    } catch (error) {
                        // Non-critical error - log but continue
                        console.warn(`Failed to upload performance history for ${file.name}:`, error);
                    }
                }

                // Show warnings if any
                if (data.warning) {
                    showToast(`${file.name}: ${data.warning}`, 'warning');
                }
                if (data.warning_sku) {
                    showToast(`${file.name}: ${data.warning_sku}`, 'warning');
                }
                if (data.warning_category) {
                    showToast(`${file.name}: ${data.warning_category}`, 'warning');
                }
            } else {
                // Show error from backend
                const errorMsg = getUserFriendlyError(data.error_code || 'UNKNOWN_ERROR', data.error, data.suggestion);
                showToast(`${file.name}: ${errorMsg}`, 'error');
                console.error(`Failed to process ${file.name}:`, data);
            }
        } catch (error) {
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            showToast(`${file.name}: ${errorMsg}`, 'error');
            console.error(`Failed to process ${file.name}:`, error);
        }

        const progress = ((i + 1) / newFiles.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${i + 1} of ${newFiles.length} processed`;
    }

    const successful = newProducts.filter(p => p.hasFeatures).length;
    const failed = newFiles.length - newProducts.length;
    const withoutMetadata = newProducts.filter(p => !p.category && !p.sku).length;
    
    let statusMsg = `<h4>✓ New products processed</h4><p>${successful} products ready for matching`;
    if (failed > 0) statusMsg += ` (${failed} failed)`;
    if (withoutMetadata > 0) statusMsg += ` (${withoutMetadata} without CSV metadata)`;
    statusMsg += `</p>`;
    statusDiv.innerHTML = statusMsg;

    showToast(`New products ready: ${successful} products`, 'success');
    showLoadingSpinner(processBtn, false);

    // Show matching section
    document.getElementById('matchSection').style.display = 'block';
    document.getElementById('matchSection').scrollIntoView({ behavior: 'smooth' });
}

// Matching
function initMatching() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const matchBtn = document.getElementById('matchBtn');

    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = e.target.value;
    });

    matchBtn.addEventListener('click', startMatching);
}

async function startMatching() {
    const threshold = parseInt(document.getElementById('thresholdSlider').value);
    const limit = parseInt(document.getElementById('limitSelect').value);
    const progressDiv = document.getElementById('matchProgress');
    const matchBtn = document.getElementById('matchBtn');

    progressDiv.classList.add('show');
    matchBtn.disabled = true;
    showLoadingSpinner(matchBtn, true);

    progressDiv.innerHTML = '<h4>Finding matches...</h4><div class="progress-bar"><div class="progress-fill" id="matchProgressFill"></div></div><p id="matchProgressText">0 of ' + newProducts.length + ' products matched</p><div class="spinner-inline"></div>';

    matchResults = [];
    const progressFill = document.getElementById('matchProgressFill');
    const progressText = document.getElementById('matchProgressText');

    for (let i = 0; i < newProducts.length; i++) {
        const product = newProducts[i];

        if (!product.hasFeatures) {
            matchResults.push({
                product: product,
                matches: [],
                error: 'No features extracted'
            });
            continue;
        }

        try {
            const response = await fetchWithRetry('/api/products/match', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_id: product.id,
                    threshold: threshold,
                    limit: limit,
                    color_weight: similarityWeights.color,
                    shape_weight: similarityWeights.shape,
                    texture_weight: similarityWeights.texture
                })
            });

            const data = await response.json();
            
            const matches = data.matches || [];
            
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
                
                // Fetch performance history
                try {
                    const perfResponse = await fetchWithRetry(`/api/products/${match.product_id}/performance-history`);
                    if (perfResponse.ok) {
                        const perfData = await perfResponse.json();
                        match.performanceHistory = perfData.performance_history || [];
                        match.performanceStatistics = perfData.statistics;
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
                error: response.ok ? null : data.error
            });
        } catch (error) {
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            matchResults.push({
                product: product,
                matches: [],
                error: errorMsg
            });
        }

        const progress = ((i + 1) / newProducts.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${i + 1} of ${newProducts.length} products matched`;
    }

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
    document.getElementById('exportCsvBtn').addEventListener('click', exportResults);
    document.getElementById('resetBtn').addEventListener('click', resetApp);
    document.getElementById('modalClose').addEventListener('click', closeModal);
}

function displayResults() {
    const summaryDiv = document.getElementById('resultsSummary');
    const listDiv = document.getElementById('resultsList');

    // Populate category filter
    populateCategoryFilter();
    
    // Apply filters and sorting
    const filteredResults = filterAndSortResults(matchResults);
    
    const totalProducts = matchResults.length;
    const totalMatches = matchResults.reduce((sum, r) => sum + r.matches.length, 0);
    const productsWithMatches = matchResults.filter(r => r.matches.length > 0).length;
    const avgMatches = productsWithMatches > 0 ? (totalMatches / productsWithMatches).toFixed(1) : 0;
    
    const filteredCount = filteredResults.length;

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

    listDiv.innerHTML = filteredResults.map((result, index) => {
        const product = result.product;
        const matches = result.matches;

        return `
            <div class="result-item">
                <div class="result-header">
                    <img data-src="/api/products/${product.id}/image" class="result-image lazy-load" 
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><rect fill='%23e2e8f0' width='120' height='120'/></svg>"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22120%22 height=%22120%22/></svg>'"
                         alt="${product.filename}">
                    <div class="result-info">
                        <h3>${escapeHtml(product.filename)}</h3>
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
                                <img data-src="/api/products/${match.product_id}/image" class="match-image lazy-load"
                                     src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='120'><rect fill='%23e2e8f0' width='180' height='120'/></svg>"
                                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22180%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22180%22 height=%22120%22/></svg>'"
                                     alt="Match">
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
                                            <span class="performance-sales">📊 ${match.performanceStatistics.total_sales} sales</span>
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
    
    // Initialize lazy loading for images
    initLazyLoading();
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

        modalBody.innerHTML = `
            <h2>Detailed Comparison</h2>
            <div class="comparison-view">
                <div class="comparison-item">
                    <h3>New Product</h3>
                    <img data-src="/api/products/${newProductId}/image" class="lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect fill='%23e2e8f0' width='300' height='300'/></svg>"
                         alt="New Product">
                    <div class="comparison-details">
                        <p><strong>Filename:</strong> ${escapeHtml(newData.product.product_name || 'Unknown')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(newData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
                <div class="comparison-item">
                    <h3>Matched Product</h3>
                    <img data-src="/api/products/${matchedProductId}/image" class="lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect fill='%23e2e8f0' width='300' height='300'/></svg>"
                         alt="Matched Product">
                    <div class="comparison-details">
                        <p><strong>Filename:</strong> ${escapeHtml(matchData.product.product_name || 'Unknown')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(matchData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
            </div>
            ${matchDetails ? `
                <div class="score-breakdown">
                    <h4>Similarity Breakdown</h4>
                    <div class="score-bar">
                        <div class="score-bar-label">
                            <span>Overall Similarity</span>
                            <span>${matchDetails.similarity_score.toFixed(1)}%</span>
                        </div>
                        <div class="score-bar-fill">
                            <div style="width: ${matchDetails.similarity_score}%"></div>
                        </div>
                    </div>
                    <div class="score-bar">
                        <div class="score-bar-label">
                            <span>Color Similarity</span>
                            <span>${matchDetails.color_score?.toFixed(1) || 'N/A'}%</span>
                        </div>
                        <div class="score-bar-fill">
                            <div style="width: ${matchDetails.color_score || 0}%"></div>
                        </div>
                    </div>
                    <div class="score-bar">
                        <div class="score-bar-label">
                            <span>Shape Similarity</span>
                            <span>${matchDetails.shape_score?.toFixed(1) || 'N/A'}%</span>
                        </div>
                        <div class="score-bar-fill">
                            <div style="width: ${matchDetails.shape_score || 0}%</div>
                        </div>
                    </div>
                    <div class="score-bar">
                        <div class="score-bar-label">
                            <span>Texture Similarity</span>
                            <span>${matchDetails.texture_score?.toFixed(1) || 'N/A'}%</span>
                        </div>
                        <div class="score-bar-fill">
                            <div style="width: ${matchDetails.texture_score || 0}%"></div>
                        </div>
                    </div>
                </div>
            ` : ''}
            ${matchDetails?.priceStatistics ? `
                <div class="price-history-section">
                    <h4>💰 Price History</h4>
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
                    <h4>📊 Performance History</h4>
                    <div class="performance-statistics">
                        <div class="performance-stat">
                            <span class="performance-stat-label">Total Sales</span>
                            <span class="performance-stat-value">${matchDetails.performanceStatistics.total_sales}</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Total Views</span>
                            <span class="performance-stat-value">${matchDetails.performanceStatistics.total_views.toLocaleString()}</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Avg Conversion</span>
                            <span class="performance-stat-value">${matchDetails.performanceStatistics.avg_conversion}%</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Total Revenue</span>
                            <span class="performance-stat-value">$${matchDetails.performanceStatistics.total_revenue.toLocaleString()}</span>
                        </div>
                        <div class="performance-stat">
                            <span class="performance-stat-label">Sales Trend</span>
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
    let csv = 'New Product,Category,Match Count,Top Match,Top Score,Price Current,Price Avg,Price Min,Price Max,Price Trend,Total Sales,Total Revenue,Avg Conversion,Sales Trend\n';

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
            
            // Add performance history data if available
            if (topMatch.performanceStatistics) {
                csv += `,${topMatch.performanceStatistics.total_sales}`;
                csv += `,${topMatch.performanceStatistics.total_revenue}`;
                csv += `,${topMatch.performanceStatistics.avg_conversion}`;
                csv += `,"${topMatch.performanceStatistics.sales_trend}"`;
            } else {
                csv += ',,,,';
            }
        } else {
            csv += ',"No matches",0,,,,,,,,,';
        }

        csv += '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `match_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Results exported to CSV with price & performance history', 'success');
}

function resetApp() {
    if (confirm('Start over? This will clear all data.')) {
        location.reload();
    }
}

// Utilities
async function parseCsv(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim());
            const map = {};
            const duplicates = [];
            const errors = [];

            // Check if first line is a header
            const firstLine = lines[0];
            const hasHeader = firstLine.toLowerCase().includes('filename') ||
                firstLine.toLowerCase().includes('category') ||
                firstLine.toLowerCase().includes('sku');

            const dataLines = hasHeader ? lines.slice(1) : lines;

            dataLines.forEach((line, index) => {
                try {
                    // Handle both comma and tab separated values
                    const parts = line.split(/[,\t]/).map(s => s.trim().replace(/^"|"$/g, '')); // Remove quotes

                    if (parts.length >= 1) {
                        const filename = parts[0];
                        if (!filename) return; // Skip empty lines
                        
                        const category = parts[1] || null;
                        const sku = parts[2] || null;
                        const name = parts[3] || null;
                        
                        // Parse price - can be in column 4 OR 5 (flexible)
                        // Format 1: Single price in column 4
                        // Format 2: Price history in column 5
                        let priceHistory = null;
                        let singlePrice = null;
                        
                        // Try to parse column 4 as single price
                        if (parts[4] && !parts[4].includes(':') && !parts[4].includes(';')) {
                            const price = parseFloat(parts[4]);
                            if (!isNaN(price) && price >= 0) {
                                singlePrice = price;
                                // Convert single price to price history with today's date
                                const today = new Date().toISOString().split('T')[0];
                                priceHistory = [{ date: today, price: price }];
                            }
                        }
                        
                        // Try to parse column 4 or 5 as price history
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
                        
                        // Parse performance history - can be in column 5 or 6
                        // Format: "2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160"
                        // (date:sales:views:conversion:revenue)
                        let performanceHistory = null;
                        const performanceHistoryStr = parts[5] || parts[6] || null;
                        
                        if (performanceHistoryStr && performanceHistoryStr.includes(':')) {
                            try {
                                const parsed = parsePerformanceHistory(performanceHistoryStr);
                                if (parsed && parsed.length > 0) {
                                    performanceHistory = parsed;
                                }
                            } catch (error) {
                                errors.push(`Row ${index + 2}: Failed to parse performance history for ${filename}`);
                            }
                        }

                        if (filename) {
                            // Check for duplicate filename
                            if (map[filename]) {
                                duplicates.push(filename);
                            }
                            
                            // Store metadata (last entry wins if duplicate)
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

            // Show warnings
            if (duplicates.length > 0) {
                const uniqueDuplicates = [...new Set(duplicates)];
                showToast(`CSV Warning: ${uniqueDuplicates.length} duplicate filename(s) found. Using last entry.`, 'warning');
            }
            
            if (errors.length > 0 && errors.length < 10) {
                errors.forEach(err => console.warn(err));
                showToast(`CSV parsed with ${errors.length} warning(s). Check console for details.`, 'warning');
            } else if (errors.length >= 10) {
                showToast(`CSV parsed with ${errors.length} warnings. Check console for details.`, 'warning');
            }

            resolve(map);
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
    // "historical_products/placemats/image1.jpg" -> "placemats"
    // "placemats/image1.jpg" -> "placemats"
    // "image1.jpg" -> null (no subfolder)
    
    if (!path) return null;
    
    const parts = path.split('/');
    
    // If only filename (no folders), return null
    if (parts.length === 1) return null;
    
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
    // Use Intersection Observer API for efficient lazy loading
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                const src = img.getAttribute('data-src');
                
                if (src) {
                    // Load the actual image
                    img.src = src;
                    img.removeAttribute('data-src');
                    
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
        imageObserver.observe(img);
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
        
        // If server error (5xx) or rate limit (429), retry
        if ((response.status >= 500 || response.status === 429) && retryCount < RETRY_CONFIG.maxRetries) {
            const delay = Math.min(
                RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, retryCount),
                RETRY_CONFIG.maxDelay
            );
            
            showToast(`Request failed. Retrying in ${delay / 1000} seconds... (Attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries})`, 'warning');
            
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

    tooltipElements.forEach(({ selector, text }) => {
        const element = document.querySelector(selector);
        if (element) {
            element.setAttribute('data-tooltip', text);
            
            element.addEventListener('mouseenter', (e) => {
                const tooltipText = e.target.getAttribute('data-tooltip');
                if (tooltipText) {
                    tooltip.textContent = tooltipText;
                    tooltip.style.display = 'block';
                    positionTooltip(e.target, tooltip);
                }
            });

            element.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });

            element.addEventListener('mousemove', (e) => {
                if (tooltip.style.display === 'block') {
                    positionTooltip(e.target, tooltip);
                }
            });
        }
    });
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
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_product_data.csv';
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Sample CSV downloaded! Open it in Excel or any text editor.', 'success');
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
    
    const sales = performanceHistory.map(p => p.sales).reverse(); // Oldest to newest
    const max = Math.max(...sales);
    const min = Math.min(...sales);
    const range = max - min || 1;
    
    const width = 60;
    const height = 20;
    const points = sales.map((sale, i) => {
        const x = (i / (sales.length - 1)) * width;
        const y = height - ((sale - min) / range) * height;
        return `${x},${y}`;
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
    
    const sales = performanceHistory.map(p => p.sales).reverse(); // Oldest to newest
    const dates = performanceHistory.map(p => p.date).reverse();
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
        const x = padding + (i / (sales.length - 1)) * chartWidth;
        const y = padding + chartHeight - ((sale - min) / range) * chartHeight;
        return { x, y, sale, date: dates[i] };
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
        <svg class="performance-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" oncontextmenu="showColorPicker(event); return false;">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            
            <!-- Sales line -->
            <polyline points="${linePoints}" fill="none" stroke="${getChartColor()}" stroke-width="3"/>
            
            <!-- Data points -->
            ${circles}
            
            <!-- Labels -->
            ${minLabel}
            ${maxLabel}
        </svg>
        <div class="performance-chart-legend">
            <span>Showing ${sales.length} data point${sales.length !== 1 ? 's' : ''}</span>
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
        const x = padding + (i / (prices.length - 1)) * chartWidth;
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
        <svg class="price-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" oncontextmenu="showColorPicker(event); return false;">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            
            <!-- Price line -->
            <polyline points="${linePoints}" fill="none" stroke="${getChartColor()}" stroke-width="3"/>
            
            <!-- Data points -->
            ${circles}
            
            <!-- Labels -->
            ${minLabel}
            ${maxLabel}
        </svg>
        <div class="price-chart-legend">
            <span>Showing ${prices.length} price point${prices.length !== 1 ? 's' : ''}</span>
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
    document.getElementById('advancedSettingsBtn').addEventListener('click', toggleAdvancedSettings);
    document.getElementById('resetWeightsBtn').addEventListener('click', resetWeights);
    
    // Weight Sliders - Mode 1 (Visual)
    document.getElementById('colorWeightSlider').addEventListener('input', updateWeights);
    document.getElementById('shapeWeightSlider').addEventListener('input', updateWeights);
    document.getElementById('textureWeightSlider').addEventListener('input', updateWeights);
    
    // Weight Sliders - Mode 2 (Metadata)
    document.getElementById('skuWeightSlider').addEventListener('input', updateMetadataWeights);
    document.getElementById('nameWeightSlider').addEventListener('input', updateMetadataWeights);
    document.getElementById('categoryWeightSlider').addEventListener('input', updateMetadataWeights);
    document.getElementById('priceWeightSlider').addEventListener('input', updateMetadataWeights);
    document.getElementById('performanceWeightSlider').addEventListener('input', updateMetadataWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Main)
    document.getElementById('hybridVisualWeightSlider').addEventListener('input', updateHybridWeights);
    document.getElementById('hybridMetadataWeightSlider').addEventListener('input', updateHybridWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Visual Sub)
    document.getElementById('hybridColorWeightSlider').addEventListener('input', updateHybridVisualSubWeights);
    document.getElementById('hybridShapeWeightSlider').addEventListener('input', updateHybridVisualSubWeights);
    document.getElementById('hybridTextureWeightSlider').addEventListener('input', updateHybridVisualSubWeights);
    
    // Weight Sliders - Mode 3 (Hybrid Metadata Sub)
    document.getElementById('hybridSkuWeightSlider').addEventListener('input', updateHybridMetadataSubWeights);
    document.getElementById('hybridNameWeightSlider').addEventListener('input', updateHybridMetadataSubWeights);
    document.getElementById('hybridCategoryWeightSlider').addEventListener('input', updateHybridMetadataSubWeights);
    document.getElementById('hybridPriceWeightSlider').addEventListener('input', updateHybridMetadataSubWeights);
    document.getElementById('hybridPerformanceWeightSlider').addEventListener('input', updateHybridMetadataSubWeights);
    
    // Export Buttons
    document.getElementById('exportCsvBtn').addEventListener('click', exportResults);
    document.getElementById('exportWithImagesBtn').addEventListener('click', exportWithImages);
    document.getElementById('duplicateReportBtn').addEventListener('click', showDuplicateReport);
    
    // Session Management
    document.getElementById('saveSessionBtn').addEventListener('click', saveSession);
    document.getElementById('loadSessionBtn').addEventListener('click', loadSession);
    
    // Search and Filter
    document.getElementById('searchInput').addEventListener('input', applyFilters);
    document.getElementById('categoryFilter').addEventListener('change', applyFilters);
    document.getElementById('sortBySelect').addEventListener('change', applyFilters);
    document.getElementById('duplicatesOnlyCheckbox').addEventListener('change', applyFilters);
    document.getElementById('clearFiltersBtn').addEventListener('click', clearFilters);
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
        const a = document.createElement('a');
        a.href = url;
        a.download = `match_results_full_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        showToast('Export complete! JSON file includes all match data. Images can be downloaded separately via API.', 'success');
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
    
    let csv = 'New Product,New Category,Matched Product,Matched Category,Similarity Score,Price Current,Price Trend,Total Sales,Total Revenue,Recommendation\n';
    
    duplicates.forEach(dup => {
        csv += `"${dup.new_product}","${dup.new_category}","${dup.matched_product}","${dup.matched_category}",${dup.similarity_score},"${dup.price_current}","${dup.price_trend}","${dup.total_sales}","${dup.total_revenue}","${dup.recommendation}"\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `duplicate_report_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Duplicate report exported to CSV', 'success');
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
    const a = document.createElement('a');
    a.href = url;
    a.download = `matching_session_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Session saved successfully', 'success');
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
    let filtered = results.filter(result => {
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
    setTimeout(() => {
        document.addEventListener('click', function closePickerOutside(e) {
            if (!picker.contains(e.target)) {
                picker.remove();
                document.removeEventListener('click', closePickerOutside);
            }
        });
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
    const isAdvanced = mode === 'advanced';
    
    if (section === 'historical') {
        historicalAdvancedMode = isAdvanced;
        const toggle = document.getElementById('historicalModeToggle');
        const csvBox = document.getElementById('historicalCsvBox');
        const processBtn = document.getElementById('processHistoricalBtn');
        
        // Update toggle buttons
        const buttons = toggle.querySelectorAll('.mode-option');
        buttons.forEach(btn => {
            btn.classList.remove('active');
            if ((btn.textContent === 'SIMPLE' && !isAdvanced) || 
                (btn.textContent === 'ADVANCED' && isAdvanced)) {
                btn.classList.add('active');
            }
        });
        
        // Show/hide CSV box and reorder sections
        csvBox.style.display = isAdvanced ? 'block' : 'none';
        const uploadBox = document.querySelector('#historicalSection .upload-box');
        if (isAdvanced) {
            uploadBox.classList.add('advanced-mode-active');
        } else {
            uploadBox.classList.remove('advanced-mode-active');
        }
        
        // Update process button
        if (isAdvanced) {
            // In advanced mode, require CSV (images optional)
            processBtn.disabled = !historicalCsv;
            if (!historicalCsv) {
                showToast('Advanced mode requires CSV. Upload CSV or use CSV Builder.', 'info');
            }
        } else {
            // In simple mode, enable process if files are loaded
            if (historicalFiles.length > 0) {
                processBtn.disabled = false;
            }
        }
    } else if (section === 'new') {
        newAdvancedMode = isAdvanced;
        const toggle = document.getElementById('newModeToggle');
        const csvBox = document.getElementById('newCsvBox');
        const processBtn = document.getElementById('processNewBtn');
        
        // Update toggle buttons
        const buttons = toggle.querySelectorAll('.mode-option');
        buttons.forEach(btn => {
            btn.classList.remove('active');
            if ((btn.textContent === 'SIMPLE' && !isAdvanced) || 
                (btn.textContent === 'ADVANCED' && isAdvanced)) {
                btn.classList.add('active');
            }
        });
        
        // Show/hide CSV box and reorder sections
        csvBox.style.display = isAdvanced ? 'block' : 'none';
        const uploadBox = document.querySelector('#newSection .upload-box');
        if (isAdvanced) {
            uploadBox.classList.add('advanced-mode-active');
        } else {
            uploadBox.classList.remove('advanced-mode-active');
        }
        
        // Update process button
        if (isAdvanced) {
            // In advanced mode, require CSV (images optional)
            processBtn.disabled = !newCsv;
            if (!newCsv) {
                showToast('Advanced mode requires CSV. Upload CSV or use CSV Builder.', 'info');
            }
        } else {
            // In simple mode, enable process if files are loaded
            if (newFiles.length > 0) {
                processBtn.disabled = false;
            }
        }
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
    window.addEventListener('message', function csvBuilderListener(event) {
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
    });
    
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
        historicalAdvancedMode,
        newAdvancedMode,
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
            
            // Restore mode settings
            if (state.historicalAdvancedMode) {
                setMode('historical', 'advanced');
            }
            if (state.newAdvancedMode) {
                setMode('new', 'advanced');
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
        }
        
        statusText.textContent = `${deviceName} Active`;
        gpuStatusEl.setAttribute('data-tooltip', tooltip);
        
        // Show first-run model download message if applicable (once per session)
        if (status.first_run && !sessionStorage.getItem('clipDownloadShown')) {
            showToast('First run: Downloading AI model (~350MB). This may take 1-2 minutes.', 'info', 10000);
            sessionStorage.setItem('clipDownloadShown', 'true');
        }
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
        const response = await fetch('/api/catalog/stats');
        if (!response.ok) throw new Error('Failed to fetch catalog stats');
        
        const stats = await response.json();
        existingCatalogStats = stats;
        
        const catalogOptions = document.getElementById('catalogOptions');
        const statsEl = document.getElementById('existingCatalogStats');
        
        if (stats.historical_products > 0) {
            // Show catalog options
            catalogOptions.style.display = 'block';
            statsEl.innerHTML = `<strong>${stats.historical_products.toLocaleString()}</strong> historical products | ` +
                               `<strong>${stats.unique_categories}</strong> categories | ` +
                               `<strong>${stats.database_size_mb.toFixed(1)} MB</strong>`;
            
            // Check for large database warning
            if (stats.database_size_mb > 500) {
                showToast('⚠️ Database is large (' + stats.database_size_mb.toFixed(0) + ' MB). Consider cleaning up old products.', 'warning', 8000);
            }
        } else {
            // No existing catalog, hide options
            catalogOptions.style.display = 'none';
        }
    } catch (error) {
        console.error('Error checking existing catalog:', error);
        document.getElementById('catalogOptions').style.display = 'none';
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
                `⚠️ WARNING: This will delete ${existingCatalogStats.historical_products.toLocaleString()} existing products.\n\n` +
                `This action cannot be undone. Continue?`
            );
            if (!confirmed) {
                // Revert to use_existing
                document.querySelector('input[name="catalogLoadOption"][value="use_existing"]').checked = true;
                return;
            }
        }
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0;
        processBtn.textContent = 'REPLACE & PROCESS';
    } else {
        // Add to existing
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0;
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
                is_historical: true
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
        // Clear existing catalog first
        try {
            showToast('Clearing existing catalog...', 'info');
            const response = await fetch('/api/catalog/cleanup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'historical' })
            });
            
            if (!response.ok) throw new Error('Failed to clear catalog');
            
            showToast('Existing catalog cleared', 'success');
        } catch (error) {
            console.error('Error clearing catalog:', error);
            showToast('Failed to clear existing catalog', 'error');
            return;
        }
    }
    
    // Continue with normal processing (add_to_existing or replace after clearing)
    if (originalProcessHistoricalCatalog) {
        originalProcessHistoricalCatalog();
    }
}

// Hook into the process button
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        const processBtn = document.getElementById('processHistoricalBtn');
        if (processBtn) {
            // Store original handler
            const originalHandler = processBtn.onclick;
            
            processBtn.onclick = async (e) => {
                const option = getCatalogLoadOption();
                
                if (option === 'use_existing') {
                    await processHistoricalCatalogWithOptions();
                } else if (option === 'replace') {
                    await processHistoricalCatalogWithOptions();
                    // After clearing, call original handler if files are selected
                    if (historicalFiles.length > 0 && originalHandler) {
                        originalHandler.call(processBtn, e);
                    }
                } else {
                    // add_to_existing - just use original handler
                    if (originalHandler) {
                        originalHandler.call(processBtn, e);
                    }
                }
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
let stateCheckInterval = null;

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
        const channel = new BroadcastChannel('catalog_changes');
        channel.onmessage = (event) => {
            handleCatalogChangeInMainApp(event.data);
        };
    } catch (e) {
        // BroadcastChannel not supported, use polling
        setInterval(checkCatalogChangesInMainApp, 2000);
    }
    
    // Also check on visibility change (when user switches back to this tab)
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            checkCatalogChangesInMainApp();
        }
    });
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
