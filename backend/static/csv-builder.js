// CSV Builder Application State
const state = {
    currentStep: 1,
    products: [],
    selectedProductIndex: null,
    undoStack: [],
    redoStack: [],
    autoSaveTimer: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeStep1();
    loadFromLocalStorage();
});

// ===== STEP 1: Upload Images =====
function initializeStep1() {
    const dropZone = document.getElementById('imageDropZone');
    const input = document.getElementById('imageInput');
    const browseBtn = document.getElementById('imageBrowseBtn');
    const nextBtn = document.getElementById('nextToMetadata');

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
    info.innerHTML = `
        <h4>✓ ${imageFiles.length} images loaded</h4>
        ${categorySummary}
        <div class="file-list">
            ${state.products.slice(0, 10).map(p => 
                `<div>${escapeHtml(p.filename)}${p.category ? ` <span style="color: #667eea;">[${p.category}]</span>` : ''}</div>`
            ).join('')}
            ${imageFiles.length > 10 ? `<div>... and ${imageFiles.length - 10} more</div>` : ''}
        </div>
    `;
    info.classList.add('show');

    document.getElementById('nextToMetadata').disabled = false;
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


// ===== STEP 2: Add Metadata =====
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
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    if (tab === 'price') {
        document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
        document.getElementById('priceTab').classList.add('active');
    } else {
        document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
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
    goToStep(4);
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
    for (let i = 1; i <= 4; i++) {
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
        renderProductsTable();
    } else if (step === 3) {
        populateProductSelector();
        updateProductProgress();
    } else if (step === 4) {
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
                
                if (confirm(`Found saved work from ${hours} hour(s) ago. Load it?`)) {
                    state.products = data.products;
                    if (state.products.length > 0) {
                        document.getElementById('nextToMetadata').disabled = false;
                        const info = document.getElementById('imageInfo');
                        info.innerHTML = `<h4>✓ ${state.products.length} products loaded from saved session</h4>`;
                        info.classList.add('show');
                    }
                }
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
