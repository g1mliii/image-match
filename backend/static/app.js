// State
let currentFile = null;
let batchFiles = [];
let categories = [];
let currentMatches = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUploadView();
    initCatalogView();
    initBatchView();
    loadCategories();
});

// Navigation
function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const viewName = link.dataset.view;
            switchView(viewName);
        });
    });
}

function switchView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    
    document.getElementById(`${viewName}-view`).classList.add('active');
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
    
    if (viewName === 'catalog') {
        loadCatalog();
    }
}

// Upload View
function initUploadView() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    
    dropZone.addEventListener('click', () => fileInput.click());
    
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
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    uploadBtn.addEventListener('click', uploadProduct);
    
    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = e.target.value;
        filterResults();
    });
    
    document.getElementById('limitSelect').addEventListener('change', filterResults);
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }
    
    currentFile = file;
    document.getElementById('uploadBtn').disabled = false;
    
    const dropZone = document.getElementById('dropZone');
    dropZone.innerHTML = `
        <img src="${URL.createObjectURL(file)}" style="max-width: 200px; max-height: 200px; border-radius: 5px;">
        <p>${file.name}</p>
    `;
}

async function uploadProduct() {
    if (!currentFile) return;
    
    const formData = new FormData();
    formData.append('image', currentFile);
    
    const name = document.getElementById('productName').value.trim();
    const sku = document.getElementById('productSku').value.trim();
    const category = document.getElementById('productCategory').value;
    
    // Only append if not empty - backend handles NULL/missing fields
    if (name) formData.append('product_name', name);
    if (sku) formData.append('sku', sku);
    if (category) formData.append('category', category);
    
    try {
        document.getElementById('uploadBtn').disabled = true;
        document.getElementById('uploadBtn').textContent = 'Uploading...';
        
        const response = await fetch('/api/products/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Show detailed error with suggestion if available
            const errorMsg = data.suggestion 
                ? `${data.error}\n\nSuggestion: ${data.suggestion}`
                : data.error || 'Upload failed';
            throw new Error(errorMsg);
        }
        
        // Show warnings if any (e.g., feature extraction failed, duplicate SKU)
        if (data.warning) {
            showToast(data.warning, 'warning');
        }
        if (data.warning_sku) {
            showToast(data.warning_sku, 'warning');
        }
        
        showToast('Product uploaded successfully!', 'success');
        
        // Find matches (handle case where product has no features)
        if (data.feature_extraction_status === 'success') {
            await findMatches(data.product_id);
        } else {
            showToast('Feature extraction failed - cannot find matches. Try re-uploading.', 'error');
        }
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        document.getElementById('uploadBtn').disabled = false;
        document.getElementById('uploadBtn').textContent = 'Upload & Find Matches';
    }
}

async function findMatches(productId) {
    try {
        const response = await fetch('/api/products/match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_id: productId })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Show detailed error with suggestion
            const errorMsg = data.suggestion 
                ? `${data.error}\n\nSuggestion: ${data.suggestion}`
                : data.error || 'Match failed';
            throw new Error(errorMsg);
        }
        
        // Show warnings if any (e.g., empty catalog, partial failures)
        if (data.warnings && data.warnings.length > 0) {
            data.warnings.forEach(w => showToast(w, 'warning'));
        }
        
        if (data.note) {
            showToast(data.note, 'info');
        }
        
        currentMatches = data.matches || [];
        displayResults();
        
        // Show message if no matches found
        if (currentMatches.length === 0 && data.message) {
            showToast(data.message, 'info');
        }
        
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function displayResults() {
    const resultsSection = document.getElementById('matchResults');
    const resultsList = document.getElementById('resultsList');
    
    if (currentMatches.length === 0) {
        resultsList.innerHTML = '<p class="loading">No matches found</p>';
        resultsSection.classList.remove('hidden');
        return;
    }
    
    resultsSection.classList.remove('hidden');
    filterResults();
}

function filterResults() {
    const threshold = parseInt(document.getElementById('thresholdSlider').value);
    const limit = parseInt(document.getElementById('limitSelect').value);
    
    const filtered = currentMatches
        .filter(m => m.similarity_score >= threshold)
        .slice(0, limit);
    
    const resultsList = document.getElementById('resultsList');
    
    if (filtered.length === 0) {
        resultsList.innerHTML = '<p class="loading">No matches above threshold</p>';
        return;
    }
    
    resultsList.innerHTML = filtered.map(match => {
        // Safely handle missing fields
        const productName = match.product_name || 'Unnamed Product';
        const sku = match.sku || null;
        const category = match.category || 'Uncategorized';
        const score = match.similarity_score || 0;
        
        return `
            <div class="result-card" onclick="viewDetails(${match.product_id})">
                <img src="/api/products/${match.product_id}/image" class="result-thumbnail" 
                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23ecf0f1%22 width=%22100%22 height=%22100%22/></svg>'"
                     alt="${productName}">
                <div class="result-info">
                    <h4>${escapeHtml(productName)}</h4>
                    <div class="result-meta">
                        ${sku ? `SKU: ${escapeHtml(sku)} | ` : ''}
                        ${escapeHtml(category)}
                    </div>
                    <span class="similarity-score ${getScoreClass(score)}">
                        ${score.toFixed(1)}% Match
                    </span>
                    ${score > 90 ? '<span class="duplicate-badge">POTENTIAL DUPLICATE</span>' : ''}
                </div>
            </div>
        `;
    }).join('');
}

function getScoreClass(score) {
    if (score >= 70) return 'score-high';
    if (score >= 50) return 'score-medium';
    return 'score-low';
}

// Catalog View
function initCatalogView() {
    document.getElementById('searchInput').addEventListener('input', debounce(loadCatalog, 300));
    document.getElementById('categoryFilter').addEventListener('change', loadCatalog);
    document.getElementById('addHistoricalBtn').addEventListener('click', showAddHistoricalModal);
}

async function loadCatalog() {
    const search = document.getElementById('searchInput').value;
    const category = document.getElementById('categoryFilter').value;
    
    try {
        let url = '/api/products/historical?';
        if (search) url += `search=${encodeURIComponent(search)}&`;
        if (category) url += `category=${encodeURIComponent(category)}&`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load catalog');
        }
        
        displayCatalog(data.products || []);
        displayStats(data.products || []);
        
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function displayCatalog(products) {
    const catalogList = document.getElementById('catalogList');
    
    if (products.length === 0) {
        catalogList.innerHTML = '<p class="loading">No products found</p>';
        return;
    }
    
    catalogList.innerHTML = products.map(product => {
        // Safely handle missing fields
        const productName = product.product_name || 'Unnamed Product';
        const sku = product.sku || null;
        const category = product.category || 'Uncategorized';
        const hasFeatures = product.has_features !== false; // Default to true if not specified
        
        return `
            <div class="catalog-card">
                <img src="/api/products/${product.id}/image" 
                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22150%22><rect fill=%22%23ecf0f1%22 width=%22200%22 height=%22150%22/></svg>'"
                     alt="${productName}">
                <div class="catalog-card-info">
                    <h4>${escapeHtml(productName)}</h4>
                    <p>${sku ? `SKU: ${escapeHtml(sku)}` : 'No SKU'}</p>
                    <p>${escapeHtml(category)}</p>
                    ${!hasFeatures ? '<p style="color: #e74c3c; font-size: 11px;">⚠ Missing features</p>' : ''}
                </div>
            </div>
        `;
    }).join('');
}

function displayStats(products) {
    const stats = document.getElementById('catalogStats');
    const total = products.length;
    const withSku = products.filter(p => p.sku).length;
    const categorized = products.filter(p => p.category).length;
    const withFeatures = products.filter(p => p.has_features !== false).length;
    const missingData = total - withSku;
    
    stats.innerHTML = `
        <strong>Total Products:</strong> ${total} | 
        <strong>With SKU:</strong> ${withSku} | 
        <strong>Categorized:</strong> ${categorized} | 
        <strong>With Features:</strong> ${withFeatures}
        ${missingData > 0 ? ` | <span style="color: #e67e22;">⚠ ${missingData} missing SKU</span>` : ''}
    `;
}

function showAddHistoricalModal() {
    // Simple implementation - reuse upload form
    showToast('Use the Upload tab and mark as historical', 'info');
}

// Batch View
function initBatchView() {
    const batchDropZone = document.getElementById('batchDropZone');
    const batchFileInput = document.getElementById('batchFileInput');
    const batchUploadBtn = document.getElementById('batchUploadBtn');
    
    batchDropZone.addEventListener('click', () => batchFileInput.click());
    
    batchDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        batchDropZone.classList.add('drag-over');
    });
    
    batchDropZone.addEventListener('dragleave', () => {
        batchDropZone.classList.remove('drag-over');
    });
    
    batchDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        batchDropZone.classList.remove('drag-over');
        handleBatchFiles(Array.from(e.dataTransfer.files));
    });
    
    batchFileInput.addEventListener('change', (e) => {
        handleBatchFiles(Array.from(e.target.files));
    });
    
    batchUploadBtn.addEventListener('click', processBatch);
}

function handleBatchFiles(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    batchFiles = [...batchFiles, ...imageFiles];
    displayBatchFiles();
    document.getElementById('batchUploadBtn').disabled = batchFiles.length === 0;
}

function displayBatchFiles() {
    const list = document.getElementById('batchFileList');
    
    if (batchFiles.length === 0) {
        list.innerHTML = '';
        return;
    }
    
    list.innerHTML = batchFiles.map((file, index) => `
        <div class="batch-file-item">
            <img src="${URL.createObjectURL(file)}">
            <span class="file-name">${file.name}</span>
            <button class="remove-btn" onclick="removeBatchFile(${index})">Remove</button>
        </div>
    `).join('');
}

function removeBatchFile(index) {
    batchFiles.splice(index, 1);
    displayBatchFiles();
    document.getElementById('batchUploadBtn').disabled = batchFiles.length === 0;
}

async function processBatch() {
    const category = document.getElementById('batchCategory').value;
    const progressSection = document.getElementById('batchProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('batchResults');
    
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    document.getElementById('batchUploadBtn').disabled = true;
    
    const results = [];
    
    for (let i = 0; i < batchFiles.length; i++) {
        const file = batchFiles[i];
        const formData = new FormData();
        formData.append('image', file);
        if (category) formData.append('category', category);
        formData.append('is_historical', 'true');
        
        try {
            const response = await fetch('/api/products/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            results.push({ file: file.name, success: response.ok, data });
            
        } catch (error) {
            results.push({ file: file.name, success: false, error: error.message });
        }
        
        const progress = ((i + 1) / batchFiles.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${i + 1} of ${batchFiles.length} files processed`;
    }
    
    displayBatchResults(results);
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    
    batchFiles = [];
    displayBatchFiles();
    document.getElementById('batchUploadBtn').disabled = true;
}

function displayBatchResults(results) {
    const resultsSection = document.getElementById('batchResults');
    const successful = results.filter(r => r.success).length;
    const failed = results.length - successful;
    const withWarnings = results.filter(r => r.success && r.data && (r.data.warning || r.data.warning_sku)).length;
    
    resultsSection.innerHTML = `
        <h3>Batch Complete</h3>
        <p>
            <strong>Successful:</strong> ${successful} | 
            <strong>Failed:</strong> ${failed}
            ${withWarnings > 0 ? ` | <strong style="color: #f39c12;">Warnings:</strong> ${withWarnings}` : ''}
        </p>
        <div style="margin-top: 20px;">
            ${results.map(r => {
                const bgColor = r.success ? '#d4edda' : '#f8d7da';
                const hasWarning = r.success && r.data && (r.data.warning || r.data.warning_sku);
                const warningMsg = hasWarning 
                    ? `<br><small style="color: #856404;">${r.data.warning || r.data.warning_sku}</small>`
                    : '';
                
                return `
                    <div style="padding: 10px; margin-bottom: 5px; background: ${bgColor}; border-radius: 5px;">
                        ${escapeHtml(r.file)}: ${r.success ? '✓ Success' : '✗ ' + escapeHtml(r.error || 'Failed')}
                        ${warningMsg}
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// Categories
async function loadCategories() {
    try {
        const response = await fetch('/api/products/historical');
        const data = await response.json();
        
        if (response.ok && data.products) {
            // Filter out null/undefined categories and get unique values
            const uniqueCategories = [...new Set(
                data.products
                    .map(p => p.category)
                    .filter(c => c && c.trim())
            )].sort();
            
            categories = uniqueCategories;
            
            const selects = [
                document.getElementById('productCategory'),
                document.getElementById('batchCategory'),
                document.getElementById('categoryFilter')
            ];
            
            selects.forEach(select => {
                if (!select) return; // Handle missing elements
                
                if (select.id === 'categoryFilter') {
                    select.innerHTML = '<option value="">All Categories</option>' +
                        (categories.length > 0 
                            ? categories.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')
                            : '<option disabled>No categories yet</option>');
                } else {
                    select.innerHTML = '<option value="">No category (optional)</option>' +
                        (categories.length > 0 
                            ? categories.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')
                            : '');
                }
            });
        }
    } catch (error) {
        console.error('Failed to load categories:', error);
        // Don't show error to user - categories are optional
    }
}

// Utilities
function showToast(message, type = 'info') {
    if (!message) return;
    
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    // Longer timeout for errors and warnings
    const timeout = (type === 'error' || type === 'warning') ? 5000 : 3000;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, timeout);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function viewDetails(productId) {
    showToast('Detailed view coming soon', 'info');
}

// Utility to escape HTML and prevent XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
