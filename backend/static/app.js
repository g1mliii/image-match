// State
let historicalFiles = [];
let newFiles = [];
let historicalCsv = null;
let newCsv = null;
let historicalProducts = [];
let newProducts = [];
let matchResults = [];

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
    info.innerHTML = `
        <h4>✓ ${imageFiles.length} images loaded</h4>
        ${categorySummary}
        <div class="file-list">
            ${filesWithCategories.slice(0, 10).map(({ file, category }) => 
                `<div>${escapeHtml(file.name)}${category ? ` <span style="color: #667eea;">[${category}]</span>` : ''}</div>`
            ).join('')}
            ${imageFiles.length > 10 ? `<div>... and ${imageFiles.length - 10} more</div>` : ''}
        </div>
    `;
    info.classList.add('show');

    document.getElementById('processHistoricalBtn').disabled = false;
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
    info.innerHTML = `
        <h4>✓ ${imageFiles.length} images loaded</h4>
        ${categorySummary}
        <div class="file-list">
            ${filesWithCategories.slice(0, 10).map(({ file, category }) => 
                `<div>${escapeHtml(file.name)}${category ? ` <span style="color: #667eea;">[${category}]</span>` : ''}</div>`
            ).join('')}
            ${imageFiles.length > 10 ? `<div>... and ${imageFiles.length - 10} more</div>` : ''}
        </div>
    `;
    info.classList.add('show');

    document.getElementById('processNewBtn').disabled = false;
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
                    limit: limit
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
    document.getElementById('exportBtn').addEventListener('click', exportResults);
    document.getElementById('resetBtn').addEventListener('click', resetApp);
    document.getElementById('modalClose').addEventListener('click', closeModal);
}

function displayResults() {
    const summaryDiv = document.getElementById('resultsSummary');
    const listDiv = document.getElementById('resultsList');

    const totalProducts = matchResults.length;
    const totalMatches = matchResults.reduce((sum, r) => sum + r.matches.length, 0);
    const productsWithMatches = matchResults.filter(r => r.matches.length > 0).length;
    const avgMatches = productsWithMatches > 0 ? (totalMatches / productsWithMatches).toFixed(1) : 0;

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
        </div>
    `;

    listDiv.innerHTML = matchResults.map((result, index) => {
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
    const csv = `filename,category,sku,name,price,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160;2024-03-15:200:1800:11.1:2400
product2.jpg,dinnerware,DW-002,White Plate Set,45.00,2024-01-15:200:2000:10.0:9000;2024-02-15:220:2200:10.0:9900;2024-03-15:240:2400:10.0:10800
product3.jpg,textiles,TX-003,Cotton Napkins,15.99,100:800:12.5:1200;120:900:13.3:1440;110:850:12.9:1320
product4.jpg,placemats,PM-004,Red Placemat,32.00,
product5.jpg,dinnerware,DW-005,Ceramic Bowl,22.50,80:600:13.3:960;90:650:13.8:1080`;

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
    
    return `<svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        <polyline points="${points}" fill="none" stroke="#667eea" stroke-width="2"/>
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
    
    return `<svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        <polyline points="${points}" fill="none" stroke="#48bb78" stroke-width="2"/>
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
        `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#48bb78" class="performance-point" data-sales="${p.sale}" data-date="${p.date}"/>`
    ).join('');
    
    // Generate axis labels
    const minLabel = `<text x="${padding}" y="${padding + chartHeight + 20}" font-size="12" fill="#666">${min}</text>`;
    const maxLabel = `<text x="${padding}" y="${padding - 10}" font-size="12" fill="#666">${max}</text>`;
    
    return `
        <svg class="performance-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            
            <!-- Sales line -->
            <polyline points="${linePoints}" fill="none" stroke="#48bb78" stroke-width="3"/>
            
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
        `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#667eea" class="price-point" data-price="${p.price}" data-date="${p.date}"/>`
    ).join('');
    
    // Generate axis labels
    const minLabel = `<text x="${padding}" y="${padding + chartHeight + 20}" font-size="12" fill="#666">$${min.toFixed(2)}</text>`;
    const maxLabel = `<text x="${padding}" y="${padding - 10}" font-size="12" fill="#666">$${max.toFixed(2)}</text>`;
    
    return `
        <svg class="price-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <!-- Grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#e2e8f0" stroke-width="1"/>
            
            <!-- Price line -->
            <polyline points="${linePoints}" fill="none" stroke="#667eea" stroke-width="3"/>
            
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
