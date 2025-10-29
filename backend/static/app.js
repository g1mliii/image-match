// State
let historicalFiles = [];
let newFiles = [];
let historicalCsv = null;
let newCsv = null;
let historicalProducts = [];
let newProducts = [];
let matchResults = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initHistoricalUpload();
    initNewUpload();
    initMatching();
    initResults();
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

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
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

    historicalFiles = imageFiles;

    const info = document.getElementById('historicalInfo');
    info.innerHTML = `
        <h4>✓ ${imageFiles.length} images loaded</h4>
        <div class="file-list">
            ${imageFiles.slice(0, 10).map(f => `<div>${escapeHtml(f.name)}</div>`).join('')}
            ${imageFiles.length > 10 ? `<div>... and ${imageFiles.length - 10} more</div>` : ''}
        </div>
    `;
    info.classList.add('show');

    document.getElementById('processHistoricalBtn').disabled = false;
    showToast(`${imageFiles.length} historical images loaded`, 'success');
}

async function processHistoricalCatalog() {
    const statusDiv = document.getElementById('historicalStatus');
    const processBtn = document.getElementById('processHistoricalBtn');

    statusDiv.classList.add('show');
    processBtn.disabled = true;

    // Parse CSV if provided
    let categoryMap = {};
    if (historicalCsv) {
        categoryMap = await parseCsv(historicalCsv);
    }

    statusDiv.innerHTML = '<h4>Processing historical catalog...</h4><div class="progress-bar"><div class="progress-fill" id="historicalProgress"></div></div><p id="historicalProgressText">0 of ' + historicalFiles.length + ' processed</p>';

    historicalProducts = [];
    const progressFill = document.getElementById('historicalProgress');
    const progressText = document.getElementById('historicalProgressText');

    for (let i = 0; i < historicalFiles.length; i++) {
        const file = historicalFiles[i];
        const metadata = categoryMap[file.name] || {};

        try {
            const formData = new FormData();
            formData.append('image', file);
            if (metadata.category) formData.append('category', metadata.category);
            if (metadata.sku) formData.append('sku', metadata.sku);
            if (metadata.name) formData.append('product_name', metadata.name);
            else formData.append('product_name', file.name); // Fallback to filename
            formData.append('is_historical', 'true');

            const response = await fetch('/api/products/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                historicalProducts.push({
                    id: data.product_id,
                    filename: file.name,
                    category: metadata.category,
                    sku: metadata.sku,
                    name: metadata.name,
                    hasFeatures: data.feature_extraction_status === 'success'
                });

                // Show warnings if any
                if (data.warning) {
                    showToast(`${file.name}: ${data.warning}`, 'warning');
                }
                if (data.warning_sku) {
                    showToast(`${file.name}: ${data.warning_sku}`, 'warning');
                }
            } else {
                // Show error from backend
                const errorMsg = data.suggestion
                    ? `${file.name}: ${data.error} - ${data.suggestion}`
                    : `${file.name}: ${data.error}`;
                showToast(errorMsg, 'error');
                console.error(`Failed to process ${file.name}:`, data);
            }
        } catch (error) {
            showToast(`${file.name}: Network error - ${error.message}`, 'error');
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

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
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

    newFiles = imageFiles;

    const info = document.getElementById('newInfo');
    info.innerHTML = `
        <h4>✓ ${imageFiles.length} images loaded</h4>
        <div class="file-list">
            ${imageFiles.slice(0, 10).map(f => `<div>${escapeHtml(f.name)}</div>`).join('')}
            ${imageFiles.length > 10 ? `<div>... and ${imageFiles.length - 10} more</div>` : ''}
        </div>
    `;
    info.classList.add('show');

    document.getElementById('processNewBtn').disabled = false;
    showToast(`${imageFiles.length} new product images loaded`, 'success');
}

async function processNewProducts() {
    const statusDiv = document.getElementById('newStatus');
    const processBtn = document.getElementById('processNewBtn');

    statusDiv.classList.add('show');
    processBtn.disabled = true;

    // Parse CSV if provided
    let categoryMap = {};
    if (newCsv) {
        categoryMap = await parseCsv(newCsv);
    }

    statusDiv.innerHTML = '<h4>Processing new products...</h4><div class="progress-bar"><div class="progress-fill" id="newProgress"></div></div><p id="newProgressText">0 of ' + newFiles.length + ' processed</p>';

    newProducts = [];
    const progressFill = document.getElementById('newProgress');
    const progressText = document.getElementById('newProgressText');

    for (let i = 0; i < newFiles.length; i++) {
        const file = newFiles[i];
        const metadata = categoryMap[file.name] || {};

        try {
            const formData = new FormData();
            formData.append('image', file);
            if (metadata.category) formData.append('category', metadata.category);
            if (metadata.sku) formData.append('sku', metadata.sku);
            if (metadata.name) formData.append('product_name', metadata.name);
            else formData.append('product_name', file.name); // Fallback to filename
            formData.append('is_historical', 'false');

            const response = await fetch('/api/products/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                newProducts.push({
                    id: data.product_id,
                    filename: file.name,
                    category: metadata.category,
                    sku: metadata.sku,
                    name: metadata.name,
                    hasFeatures: data.feature_extraction_status === 'success'
                });

                // Show warnings if any
                if (data.warning) {
                    showToast(`${file.name}: ${data.warning}`, 'warning');
                }
                if (data.warning_sku) {
                    showToast(`${file.name}: ${data.warning_sku}`, 'warning');
                }
            } else {
                // Show error from backend
                const errorMsg = data.suggestion
                    ? `${file.name}: ${data.error} - ${data.suggestion}`
                    : `${file.name}: ${data.error}`;
                showToast(errorMsg, 'error');
                console.error(`Failed to process ${file.name}:`, data);
            }
        } catch (error) {
            showToast(`${file.name}: Network error - ${error.message}`, 'error');
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

    progressDiv.innerHTML = '<h4>Finding matches...</h4><div class="progress-bar"><div class="progress-fill" id="matchProgressFill"></div></div><p id="matchProgressText">0 of ' + newProducts.length + ' products matched</p>';

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
            const response = await fetch('/api/products/match', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_id: product.id,
                    threshold: threshold,
                    limit: limit
                })
            });

            const data = await response.json();

            matchResults.push({
                product: product,
                matches: data.matches || [],
                error: response.ok ? null : data.error
            });
        } catch (error) {
            matchResults.push({
                product: product,
                matches: [],
                error: error.message
            });
        }

        const progress = ((i + 1) / newProducts.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${i + 1} of ${newProducts.length} products matched`;
    }

    progressDiv.innerHTML = '<h4>✓ Matching complete!</h4>';
    showToast('Matching complete!', 'success');

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
                    <img src="/api/products/${product.id}/image" class="result-image" 
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
                                <img src="/api/products/${match.product_id}/image" class="match-image"
                                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22180%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22180%22 height=%22120%22/></svg>'"
                                     alt="Match">
                                <div class="match-score ${getScoreClass(match.similarity_score)}">
                                    ${match.similarity_score.toFixed(1)}%
                                </div>
                                ${match.similarity_score > 90 ? '<span class="duplicate-badge">DUPLICATE?</span>' : ''}
                                <div class="match-info">
                                    ${escapeHtml(match.product_name || 'Unknown')}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : '<div class="no-matches">No matches found</div>'}
            </div>
        `;
    }).join('');
}

function getScoreClass(score) {
    if (score >= 70) return 'score-high';
    if (score >= 50) return 'score-medium';
    return 'score-low';
}

async function showDetailedComparison(newProductId, matchedProductId) {
    const modal = document.getElementById('detailModal');
    const modalBody = document.getElementById('modalBody');

    try {
        // Fetch both products
        const [newResp, matchResp] = await Promise.all([
            fetch(`/api/products/${newProductId}`),
            fetch(`/api/products/${matchedProductId}`)
        ]);

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
                    <img src="/api/products/${newProductId}/image" alt="New Product">
                    <div class="comparison-details">
                        <p><strong>Filename:</strong> ${escapeHtml(newData.product.product_name || 'Unknown')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(newData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
                <div class="comparison-item">
                    <h3>Matched Product</h3>
                    <img src="/api/products/${matchedProductId}/image" alt="Matched Product">
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
        `;

        modal.classList.add('show');
    } catch (error) {
        showToast('Failed to load comparison details', 'error');
    }
}

function closeModal() {
    document.getElementById('detailModal').classList.remove('show');
}

function exportResults() {
    let csv = 'New Product,Category,Match Count,Top Match,Top Score\n';

    matchResults.forEach(result => {
        const product = result.product;
        const topMatch = result.matches[0];

        csv += `"${product.filename}","${product.category || 'Uncategorized'}",${result.matches.length}`;

        if (topMatch) {
            csv += `,"${topMatch.product_name || 'Unknown'}",${topMatch.similarity_score.toFixed(1)}`;
        } else {
            csv += ',"No matches",0';
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

    showToast('Results exported to CSV', 'success');
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

            // Check if first line is a header
            const firstLine = lines[0];
            const hasHeader = firstLine.toLowerCase().includes('filename') ||
                firstLine.toLowerCase().includes('category');

            const dataLines = hasHeader ? lines.slice(1) : lines;

            dataLines.forEach((line, index) => {
                const parts = line.split(',').map(s => s.trim().replace(/^"|"$/g, '')); // Remove quotes

                if (parts.length >= 1) {
                    const filename = parts[0];
                    const category = parts[1] || null;
                    const sku = parts[2] || null;
                    const name = parts[3] || null;

                    if (filename) {
                        // Check for duplicate filename
                        if (map[filename]) {
                            duplicates.push(filename);
                        }
                        
                        // Store metadata (last entry wins if duplicate)
                        map[filename] = {
                            category: category,
                            sku: sku,
                            name: name
                        };
                    }
                }
            });

            // Warn about duplicates
            if (duplicates.length > 0) {
                const uniqueDuplicates = [...new Set(duplicates)];
                showToast(`CSV Warning: ${uniqueDuplicates.length} duplicate filename(s) found. Using last entry for: ${uniqueDuplicates.slice(0, 3).join(', ')}${uniqueDuplicates.length > 3 ? '...' : ''}`, 'warning');
            }

            resolve(map);
        };
        reader.readAsText(file);
    });
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
