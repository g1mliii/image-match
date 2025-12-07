/**
 * Catalog Manager JavaScript
 * Handles catalog browsing, product management, and database cleanup
 */

// State
let currentPage = 1;
const pageSize = 50;
let totalProducts = 0;
let selectedProducts = new Set();
let searchTimeout = null;
let catalogStats = null;
let imageObserver = null; // Track IntersectionObserver for cleanup

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadCategories();
    loadSnapshots(); // Load snapshots on page load
    
    // Set snapshots tab as active by default
    switchTab('snapshots');
});

// ============ Tab Navigation ============

function switchTab(tabName) {
    // Map old tab names to new ones
    const tabMap = {
        'catalog': 'overview',
        'cleanup': 'overview'
    };
    const actualTab = tabMap[tabName] || tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`.tab[onclick="switchTab('${actualTab}')"]`)?.classList.add('active');
    
    // Update tab panels
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
    document.getElementById(`${actualTab}Tab`)?.classList.add('active');
    
    // Load data for specific tabs
    if (actualTab === 'browse') {
        loadProducts();
    }
}

// ============ Statistics ============

async function loadStats() {
    try {
        const response = await fetch('/api/catalog/stats');
        if (!response.ok) throw new Error('Failed to load stats');
        
        const data = await response.json();
        catalogStats = data;
        
        // Update overview stats
        document.getElementById('totalProducts').textContent = data.total_products.toLocaleString();
        document.getElementById('historicalProducts').textContent = data.historical_products.toLocaleString();
        document.getElementById('newProducts').textContent = data.new_products.toLocaleString();
        document.getElementById('dbSize').textContent = formatSize(data.database_size_mb);
        document.getElementById('totalMatches').textContent = data.total_matches.toLocaleString();
        document.getElementById('uniqueCategories').textContent = data.unique_categories;
        
        // Data quality
        document.getElementById('missingFeatures').textContent = data.missing_features;
        document.getElementById('missingCategory').textContent = data.missing_category;
        document.getElementById('missingSku').textContent = data.missing_sku;
        document.getElementById('duplicateSkus').textContent = data.duplicate_skus;
        
        // Database info
        document.getElementById('dbPath').textContent = data.database_path || 'Unknown';
        document.getElementById('uploadsSize').textContent = formatSize(data.uploads_size_mb);
        document.getElementById('lastUpdated').textContent = data.last_updated || 'Unknown';
        
        // Category breakdown
        renderCategoryBreakdown(data.category_breakdown);
        
        // Warnings
        updateWarnings(data);
        
    } catch (error) {
        console.error('Error loading stats:', error);
        showToast('Failed to load catalog statistics', 'error');
    }
}

function refreshStats() {
    loadStats();
    showToast('Statistics refreshed', 'success');
}

function renderCategoryBreakdown(categories) {
    const container = document.getElementById('categoryBreakdown');
    
    if (!categories || categories.length === 0) {
        container.innerHTML = '<p>No categories found</p>';
        return;
    }
    
    let html = '';
    for (const cat of categories) {
        const name = cat.category || '(Uncategorized)';
        html += `
            <div class="category-item">
                <span>${escapeHtml(name)}</span>
                <span><strong>${cat.count.toLocaleString()}</strong> products</span>
            </div>
        `;
    }
    container.innerHTML = html;
}

function updateWarnings(stats) {
    // Database size warning
    const dbCard = document.getElementById('dbSizeCard');
    if (stats.database_size_mb > 500) {
        dbCard.classList.add('danger');
    } else if (stats.database_size_mb > 200) {
        dbCard.classList.add('warning');
    } else {
        dbCard.classList.remove('warning', 'danger');
    }
    
    // Total products warning
    const totalCard = document.getElementById('totalProductsCard');
    if (stats.total_products > 10000) {
        totalCard.classList.add('warning');
    } else {
        totalCard.classList.remove('warning');
    }
}

// ============ Product Browsing ============

async function loadCategories() {
    try {
        const response = await fetch('/api/catalog/categories');
        if (!response.ok) throw new Error('Failed to load categories');
        
        const data = await response.json();
        const select = document.getElementById('categoryFilter');
        
        // Keep first option
        select.innerHTML = '<option value="">All Categories</option>';
        
        for (const cat of data.categories) {
            const option = document.createElement('option');
            option.value = cat;
            option.textContent = cat;
            select.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

async function loadProducts() {
    const grid = document.getElementById('productGrid');
    grid.innerHTML = '';
    
    try {
        const params = new URLSearchParams({
            page: currentPage,
            limit: pageSize,
            search: document.getElementById('searchInput').value,
            category: document.getElementById('categoryFilter').value,
            type: document.getElementById('typeFilter').value,
            features: document.getElementById('featureFilter').value,
            sort: document.getElementById('sortBy').value
        });
        
        const response = await fetch(`/api/catalog/products?${params}`);
        if (!response.ok) throw new Error('Failed to load products');
        
        const data = await response.json();
        totalProducts = data.total;
        
        renderProducts(data.products);
        updatePagination();
        
    } catch (error) {
        console.error('Error loading products:', error);
        grid.innerHTML = '<p>Failed to load products. Please try again.</p>';
    }
}

function renderProducts(products) {
    const grid = document.getElementById('productGrid');
    
    if (!products || products.length === 0) {
        grid.innerHTML = '<p style="padding: 20px; text-align: center; color: #666;">No products found matching your filters.</p>';
        return;
    }
    
    let html = '';
    for (const product of products) {
        const name = product.product_name || product.filename || `Product ${product.id}`;
        const category = product.category || 'Uncategorized';
        const sku = product.sku ? `SKU: ${product.sku}` : 'No SKU';
        // Check if image_path exists and is not null/empty
        const hasImage = product.image_path && product.image_path !== 'null' && product.image_path.trim() !== '' && product.image_path !== 'None';
        
        // Debug log for first product
        if (products.indexOf(product) === 0) {
            console.log('First product image_path:', product.image_path, 'hasImage:', hasImage);
        }
        
        html += `
            <div class="product-card" onclick="showProductDetail(${product.id})">
                <div class="product-card-thumbnail ${!hasImage ? 'no-image' : ''}">
                    ${hasImage ? 
                        `<img data-src="/api/products/${product.id}/image" 
                             alt="${escapeHtml(name)}"
                             onerror="this.onerror=null; this.parentElement.classList.add('no-image'); this.parentElement.innerHTML='NO IMAGE';">` 
                        : 'NO IMAGE'}
                </div>
                <div class="product-card-info">
                    <div class="product-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
                    <div class="product-meta">${escapeHtml(category)} • ${escapeHtml(sku)}</div>
                    <div class="product-status">
                        <span class="status-badge">
                            ${product.has_features ? '✓ Features' : '✗ No Features'}
                        </span>
                        <span class="status-badge">
                            ${product.is_historical ? 'Historical' : 'New'}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }
    grid.innerHTML = html;
    
    // Lazy load images only for products that have images
    lazyLoadImages();
}

// Lazy loading for product images
function lazyLoadImages() {
    const images = document.querySelectorAll('img[data-src]');
    
    // Disconnect previous observer if exists (Fix #13: Memory leak)
    if (imageObserver) {
        imageObserver.disconnect();
    }
    
    imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                img.classList.add('loaded');
                observer.unobserve(img);
            }
        });
    }, {
        rootMargin: '50px' // Start loading 50px before image enters viewport
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// Cleanup on page unload (Fix #13: Memory leak)
window.addEventListener('beforeunload', () => {
    if (imageObserver) {
        imageObserver.disconnect();
        imageObserver = null;
    }
    if (searchTimeout) {
        clearTimeout(searchTimeout);
        searchTimeout = null;
    }
});

function updatePagination() {
    const totalPages = Math.ceil(totalProducts / pageSize);
    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages} (${totalProducts} products)`;
    document.getElementById('prevBtn').disabled = currentPage <= 1;
    document.getElementById('nextBtn').disabled = currentPage >= totalPages;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        loadProducts();
    }
}

function nextPage() {
    const totalPages = Math.ceil(totalProducts / pageSize);
    if (currentPage < totalPages) {
        currentPage++;
        loadProducts();
    }
}

function debounceSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        currentPage = 1;
        loadProducts();
    }, 300);
}

// ============ Product Selection ============

function toggleProductSelection(productId, event) {
    if (event) event.stopPropagation();
    
    if (selectedProducts.has(productId)) {
        selectedProducts.delete(productId);
    } else {
        selectedProducts.add(productId);
    }
    
    updateSelectionUI();
    
    // Update card visual
    const cards = document.querySelectorAll('.product-card');
    cards.forEach(card => {
        const checkbox = card.querySelector('.checkbox');
        if (checkbox) {
            const id = parseInt(card.getAttribute('onclick').match(/\d+/)[0]);
            card.classList.toggle('selected', selectedProducts.has(id));
            checkbox.checked = selectedProducts.has(id);
        }
    });
}

function updateSelectionUI() {
    const bulkActions = document.getElementById('bulkActions');
    const count = selectedProducts.size;
    
    if (count > 0) {
        bulkActions.classList.add('visible');
        document.getElementById('selectedCount').textContent = `${count} selected`;
    } else {
        bulkActions.classList.remove('visible');
    }
}

function clearSelection() {
    selectedProducts.clear();
    updateSelectionUI();
    loadProducts();
}

// ============ Product Detail ============

async function showProductDetail(productId) {
    try {
        const response = await fetch(`/api/products/${productId}`);
        if (!response.ok) throw new Error('Failed to load product');
        
        const data = await response.json();
        const product = data.product;
        
        const modal = document.getElementById('productModal');
        const body = document.getElementById('productModalBody');
        
        body.innerHTML = `
            <h2>PRODUCT DETAILS</h2>
            <div class="product-detail">
                <div class="detail-image">
                    <img src="/api/products/${product.id}/image" alt="Product Image"
                         onerror="this.style.display='none'">
                </div>
                <div class="detail-info">
                    <div class="detail-row">
                        <span class="label">ID:</span>
                        <span class="value">${product.id}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Filename:</span>
                        <span class="value">${escapeHtml(getFilename(product.image_path))}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Category:</span>
                        <input type="text" class="value editable" id="editCategory" 
                               value="${escapeHtml(product.category || '')}" placeholder="Enter category">
                    </div>
                    <div class="detail-row">
                        <span class="label">SKU:</span>
                        <input type="text" class="value editable" id="editSku" 
                               value="${escapeHtml(product.sku || '')}" placeholder="Enter SKU">
                    </div>
                    <div class="detail-row">
                        <span class="label">Name:</span>
                        <input type="text" class="value editable" id="editName" 
                               value="${escapeHtml(product.product_name || '')}" placeholder="Enter name">
                    </div>
                    <div class="detail-row">
                        <span class="label">Type:</span>
                        <span class="value">${product.is_historical ? 'Historical' : 'New Product'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Features:</span>
                        <span class="value">${product.feature_extraction_status === 'success' ? '[OK] Extracted' : '[--] Not Extracted'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Created:</span>
                        <span class="value">${formatDate(product.created_at)}</span>
                    </div>
                    
                    <div class="detail-actions">
                        <button class="btn btn-primary" onclick="saveProductChanges(${product.id})">SAVE CHANGES</button>
                        <button class="btn" onclick="reextractFeatures(${product.id})">RE-EXTRACT FEATURES</button>
                        <button class="btn danger" onclick="deleteProduct(${product.id})">DELETE</button>
                    </div>
                </div>
            </div>
        `;
        
        modal.style.display = 'flex';
        
    } catch (error) {
        console.error('Error loading product:', error);
        showToast('Failed to load product details', 'error');
    }
}

function closeProductModal() {
    document.getElementById('productModal').style.display = 'none';
}

function showCleanupModal() {
    document.getElementById('cleanupModal').style.display = 'flex';
}

function closeCleanupModal() {
    document.getElementById('cleanupModal').style.display = 'none';
}

async function saveProductChanges(productId) {
    const category = document.getElementById('editCategory').value.trim() || null;
    const sku = document.getElementById('editSku').value.trim() || null;
    const name = document.getElementById('editName').value.trim() || null;
    
    try {
        const response = await fetch(`/api/catalog/products/${productId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, sku, product_name: name })
        });
        
        if (!response.ok) throw new Error('Failed to update product');
        
        showToast('Product updated successfully', 'success');
        closeProductModal();
        loadProducts();
        loadStats();
        
    } catch (error) {
        console.error('Error updating product:', error);
        showToast('Failed to update product', 'error');
    }
}

async function reextractFeatures(productId) {
    try {
        showToast('Re-extracting features...', 'info');
        
        const response = await fetch(`/api/catalog/products/${productId}/reextract`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to re-extract features');
        
        const data = await response.json();
        showToast(data.message || 'Features re-extracted successfully', 'success');
        showProductDetail(productId);
        
    } catch (error) {
        console.error('Error re-extracting features:', error);
        showToast('Failed to re-extract features', 'error');
    }
}

async function deleteProduct(productId) {
    if (!confirm('Are you sure you want to delete this product? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/catalog/products/${productId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete product');
        
        showToast('Product deleted successfully', 'success');
        closeProductModal();
        loadProducts();
        loadStats();
        
        // Notify main app of database change
        notifyMainAppOfChange('delete', { productId });
        
    } catch (error) {
        console.error('Error deleting product:', error);
        showToast('Failed to delete product', 'error');
    }
}

// ============ Bulk Operations ============

async function bulkDelete() {
    const count = selectedProducts.size;
    if (count === 0) return;
    
    if (!confirm(`Are you sure you want to delete ${count} product(s)? This cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/catalog/products/bulk-delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_ids: Array.from(selectedProducts) })
        });
        
        if (!response.ok) throw new Error('Failed to delete products');
        
        const data = await response.json();
        showToast(`Deleted ${data.deleted_count} product(s)`, 'success');
        clearSelection();
        loadProducts();
        loadStats();
        
        // Notify main app of database change
        notifyMainAppOfChange('bulk_delete', { count: data.deleted_count });
        
    } catch (error) {
        console.error('Error bulk deleting:', error);
        showToast('Failed to delete products', 'error');
    }
}

async function bulkEditCategory() {
    const count = selectedProducts.size;
    if (count === 0) return;
    
    const newCategory = prompt(`Enter new category for ${count} product(s):`);
    if (newCategory === null) return;
    
    try {
        const response = await fetch('/api/catalog/products/bulk-update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                product_ids: Array.from(selectedProducts),
                category: newCategory.trim() || null
            })
        });
        
        if (!response.ok) throw new Error('Failed to update products');
        
        const data = await response.json();
        showToast(`Updated ${data.updated_count} product(s)`, 'success');
        clearSelection();
        loadProducts();
        loadStats();
        loadCategories();
        
    } catch (error) {
        console.error('Error bulk updating:', error);
        showToast('Failed to update products', 'error');
    }
}

async function bulkReextract() {
    const count = selectedProducts.size;
    if (count === 0) return;
    
    if (!confirm(`Re-extract features for ${count} product(s)? This may take a while.`)) {
        return;
    }
    
    try {
        showToast('Re-extracting features...', 'info');
        
        const response = await fetch('/api/catalog/products/bulk-reextract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_ids: Array.from(selectedProducts) })
        });
        
        if (!response.ok) throw new Error('Failed to re-extract features');
        
        const data = await response.json();
        showToast(`Re-extracted features for ${data.success_count} product(s)`, 'success');
        clearSelection();
        loadProducts();
        
    } catch (error) {
        console.error('Error bulk re-extracting:', error);
        showToast('Failed to re-extract features', 'error');
    }
}

// ============ Cleanup Operations ============

function confirmCleanup(type) {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    let title, message, details;
    
    switch (type) {
        case 'all':
            title = 'Clear All Products';
            message = 'This will delete ALL products, features, and matches from the database.';
            details = `${catalogStats?.total_products || 0} products will be deleted`;
            break;
        case 'historical':
            title = 'Clear Historical Products';
            message = 'This will delete all historical products and their features.';
            details = `${catalogStats?.historical_products || 0} historical products will be deleted`;
            break;
        case 'new':
            title = 'Clear New Products';
            message = 'This will delete all new products and their features.';
            details = `${catalogStats?.new_products || 0} new products will be deleted`;
            break;
        case 'matches':
            title = 'Clear Matches';
            message = 'This will delete all stored match results. Products and features will be kept.';
            details = `${catalogStats?.total_matches || 0} matches will be deleted`;
            break;
    }
    
    body.innerHTML = `
        <div class="confirm-modal">
            <div class="warning-icon">[!]</div>
            <h2>${title}</h2>
            <p class="confirm-message">${message}</p>
            <div class="confirm-details">
                <strong>Impact:</strong> ${details}<br>
                <strong>Warning:</strong> This action cannot be undone!
            </div>
            <div class="confirm-actions">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn danger" onclick="executeCleanup('${type}')">DELETE</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

function closeConfirmModal() {
    document.getElementById('confirmModal').style.display = 'none';
}

async function executeCleanup(type) {
    closeConfirmModal();
    
    try {
        showToast('Cleaning up...', 'info');
        
        const response = await fetch('/api/catalog/cleanup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type })
        });
        
        if (!response.ok) throw new Error('Cleanup failed');
        
        const data = await response.json();
        showToast(data.message || 'Cleanup completed', 'success');
        loadStats();
        loadProducts();
        loadCategories();
        
        // Notify main app of database change
        notifyMainAppOfChange('cleanup', type);
        
    } catch (error) {
        console.error('Error during cleanup:', error);
        showToast('Cleanup failed', 'error');
    }
}

// Notify main app that database has changed
function notifyMainAppOfChange(action, details) {
    // Store change notification in sessionStorage for main app to detect
    const changeEvent = {
        action: action,
        details: details,
        timestamp: Date.now()
    };
    sessionStorage.setItem('catalogManagerChange', JSON.stringify(changeEvent));
    
    // Also try to notify via BroadcastChannel if available
    try {
        const channel = new BroadcastChannel('catalog_changes');
        channel.postMessage(changeEvent);
        channel.close();
    } catch (e) {
        // BroadcastChannel not supported, rely on sessionStorage
    }
}

function showCategoryCleanup() {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    const categories = catalogStats?.category_breakdown || [];
    let options = categories.map(cat => 
        `<label style="display: block; margin: 5px 0;">
            <input type="checkbox" value="${escapeHtml(cat.category || '')}" class="category-checkbox">
            ${escapeHtml(cat.category || '(Uncategorized)')} (${cat.count} products)
        </label>`
    ).join('');
    
    body.innerHTML = `
        <div class="confirm-modal">
            <h2>Clear by Category</h2>
            <p>Select categories to delete:</p>
            <div class="confirm-details" style="max-height: 300px; overflow-y: auto;">
                ${options || '<p>No categories found</p>'}
            </div>
            <div class="confirm-actions">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn danger" onclick="executeCategoryCleanup()">DELETE SELECTED</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function executeCategoryCleanup() {
    const checkboxes = document.querySelectorAll('.category-checkbox:checked');
    const categories = Array.from(checkboxes).map(cb => cb.value);
    
    if (categories.length === 0) {
        showToast('Please select at least one category', 'error');
        return;
    }
    
    closeConfirmModal();
    
    try {
        showToast('Deleting categories...', 'info');
        
        const response = await fetch('/api/catalog/cleanup/categories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ categories })
        });
        
        if (!response.ok) throw new Error('Cleanup failed');
        
        const data = await response.json();
        showToast(data.message || 'Categories deleted', 'success');
        loadStats();
        loadProducts();
        loadCategories();
        
        // Notify main app of database change
        notifyMainAppOfChange('category_cleanup', { categories });
        
    } catch (error) {
        console.error('Error deleting categories:', error);
        showToast('Failed to delete categories', 'error');
    }
}

function showDateCleanup() {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    body.innerHTML = `
        <div class="confirm-modal">
            <h2>Clear by Date</h2>
            <p>Delete products older than:</p>
            <div class="confirm-details">
                <select id="daysSelect" style="width: 100%; padding: 10px; font-size: 1em;">
                    <option value="7">7 days</option>
                    <option value="30" selected>30 days</option>
                    <option value="60">60 days</option>
                    <option value="90">90 days</option>
                    <option value="180">180 days</option>
                    <option value="365">1 year</option>
                </select>
            </div>
            <div class="confirm-actions">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn danger" onclick="executeDateCleanup()">DELETE</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function executeDateCleanup() {
    const days = parseInt(document.getElementById('daysSelect').value);
    closeConfirmModal();
    
    try {
        showToast('Deleting old products...', 'info');
        
        const response = await fetch('/api/catalog/cleanup/by-date', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ older_than_days: days })
        });
        
        if (!response.ok) throw new Error('Cleanup failed');
        
        const data = await response.json();
        showToast(data.message || 'Old products deleted', 'success');
        loadStats();
        loadProducts();
        
        // Notify main app of database change
        if (data.products_deleted > 0) {
            notifyMainAppOfChange('date_cleanup', { days, deleted: data.products_deleted });
        }
        
    } catch (error) {
        console.error('Error deleting old products:', error);
        showToast('Failed to delete old products', 'error');
    }
}

async function vacuumDatabase() {
    try {
        showToast('Vacuuming database...', 'info');
        
        const response = await fetch('/api/catalog/vacuum', {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Vacuum failed');
        
        const data = await response.json();
        showToast(data.message || 'Database vacuumed successfully', 'success');
        loadStats();
        
    } catch (error) {
        console.error('Error vacuuming database:', error);
        showToast('Failed to vacuum database', 'error');
    }
}

async function clearUploadedImages() {
    if (!confirm('This will delete all uploaded image files but keep product metadata. Continue?')) {
        return;
    }
    
    try {
        showToast('Clearing images...', 'info');
        
        const response = await fetch('/api/catalog/clear-images', {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to clear images');
        
        const data = await response.json();
        showToast(data.message || 'Images cleared', 'success');
        loadStats();
        
    } catch (error) {
        console.error('Error clearing images:', error);
        showToast('Failed to clear images', 'error');
    }
}

async function exportBackup() {
    try {
        showToast('Generating backup...', 'info');
        
        const response = await fetch('/api/catalog/export');
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = `catalog-backup-${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            showToast('Backup downloaded', 'success');
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
        
    } catch (error) {
        console.error('Error exporting backup:', error);
        showToast('Failed to export backup', 'error');
    }
}

// ============ Utility Functions ============

function formatSize(mb) {
    if (mb === null || mb === undefined) return 'Unknown';
    if (mb < 1) return `${Math.round(mb * 1024)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
}

function formatDate(dateStr) {
    if (!dateStr) return 'Unknown';
    try {
        const date = new Date(dateStr);
        return date.toLocaleString();
    } catch {
        return dateStr;
    }
}

function getFilename(path) {
    if (!path) return 'Unknown';
    return path.split(/[/\\]/).pop();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyDbPath() {
    const path = document.getElementById('dbPath').textContent;
    navigator.clipboard.writeText(path).then(() => {
        showToast('Path copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy path', 'error');
    });
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function toggleHelp(id) {
    const el = document.getElementById(id);
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// Close modals on outside click
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
};

// ============ Snapshot Management ============

let snapshotData = { historical: [], new: [] };
let activeSnapshots = { active_historical: [], active_new: [] };

async function loadSnapshots() {
    try {
        const [listResponse, activeResponse] = await Promise.all([
            fetch('/api/catalogs/list'),
            fetch('/api/catalogs/active')
        ]);
        
        if (!listResponse.ok || !activeResponse.ok) {
            throw new Error('Failed to load snapshots');
        }
        
        snapshotData = await listResponse.json();
        activeSnapshots = await activeResponse.json();
        
        renderSnapshots();
        updateActiveSnapshotSummary();
        
    } catch (error) {
        console.error('Error loading snapshots:', error);
        showToast('Failed to load snapshots', 'error');
    }
}

function renderSnapshots() {
    // Combine all snapshots into one list
    const allSnapshots = [
        ...(snapshotData.historical || []).map(s => ({...s, type: 'Historical'})),
        ...(snapshotData.new || []).map(s => ({...s, type: 'New'}))
    ];
    renderSnapshotList('allSnapshots', allSnapshots);
}

function renderSnapshotList(containerId, snapshots) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (!snapshots || snapshots.length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #666;">No snapshots found. Click "SAVE CURRENT CATALOG" to create one.</p>';
        return;
    }
    
    let html = '';
    for (const snapshot of snapshots) {
        const tags = snapshot.tags?.join(', ') || '';
        
        html += `
            <div class="snapshot-card" data-snapshot="${escapeHtml(snapshot.snapshot_file)}">
                <div class="snapshot-info">
                    <div class="snapshot-name">
                        ${escapeHtml(snapshot.name)} 
                        <span style="font-weight: normal; color: #666;">[${snapshot.type}]</span>
                    </div>
                    <div class="snapshot-meta">
                        ${snapshot.product_count?.toLocaleString() || 0} products | ${snapshot.total_size_mb || 0} MB
                    </div>
                    <div class="snapshot-date">Created: ${formatDate(snapshot.created_at)}</div>
                    ${tags ? `<div class="snapshot-tags">Tags: ${escapeHtml(tags)}</div>` : ''}
                </div>
                <div class="snapshot-actions">
                    <button class="btn-small" onclick="event.stopPropagation(); loadSnapshotToMain('${escapeHtml(snapshot.snapshot_file)}')" title="Load this snapshot">LOAD</button>
                    <button class="btn-small" onclick="event.stopPropagation(); renameSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="Rename">RENAME</button>
                    <button class="btn-small danger" onclick="event.stopPropagation(); deleteSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="Delete">DEL</button>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function updateActiveSnapshotSummary() {
    const nameSpan = document.getElementById('activeSnapshotName');
    if (!nameSpan) return;
    
    // Check if a snapshot is currently loaded
    fetch('/api/catalogs/main-db-stats')
        .then(res => res.json())
        .then(data => {
            if (data.loaded_snapshot && data.loaded_snapshot.loaded) {
                const snapshotFile = data.loaded_snapshot.snapshot_file;
                const snapshotName = data.loaded_snapshot.name || snapshotFile;
                nameSpan.textContent = `${snapshotName} (${data.total_products} products)`;
            } else {
                nameSpan.textContent = 'None (using main database)';
            }
        })
        .catch(err => {
            console.error('Error checking loaded snapshot:', err);
            nameSpan.textContent = 'Unknown';
        });
}

async function toggleSnapshotActive(snapshotFile, isHistorical, isActive) {
    try {
        let historical = [...(activeSnapshots.active_historical || [])];
        let newList = [...(activeSnapshots.active_new || [])];
        
        if (isHistorical) {
            if (isActive && !historical.includes(snapshotFile)) {
                historical.push(snapshotFile);
            } else if (!isActive) {
                historical = historical.filter(f => f !== snapshotFile);
            }
        } else {
            if (isActive && !newList.includes(snapshotFile)) {
                newList.push(snapshotFile);
            } else if (!isActive) {
                newList = newList.filter(f => f !== snapshotFile);
            }
        }
        
        const response = await fetch('/api/catalogs/active', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ historical, new: newList })
        });
        
        if (!response.ok) throw new Error('Failed to update active catalogs');
        
        activeSnapshots = await response.json();
        renderSnapshots();
        updateActiveSnapshotSummary();
        
        showToast(isActive ? 'Snapshot activated' : 'Snapshot deactivated', 'success');
        
        // Notify main app
        notifyMainAppOfChange('snapshot_change', { snapshotFile, isActive });
        
    } catch (error) {
        console.error('Error toggling snapshot:', error);
        showToast('Failed to update active catalogs', 'error');
        loadSnapshots(); // Reload to reset state
    }
}

function showCreateSnapshotModal(isHistorical = true) {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    body.innerHTML = `
        <div class="confirm-modal">
            <h2>CREATE NEW SNAPSHOT</h2>
            <div class="form-group">
                <label>Snapshot Name:</label>
                <input type="text" id="newSnapshotName" placeholder="e.g., Summer 2024 Plates" style="width: 100%; padding: 10px;">
            </div>
            <div class="form-group" style="margin-top: 15px;">
                <label>Type:</label>
                <select id="newSnapshotType" style="width: 100%; padding: 10px;">
                    <option value="true" ${isHistorical ? 'selected' : ''}>Historical Catalog</option>
                    <option value="false" ${!isHistorical ? 'selected' : ''}>New Products</option>
                </select>
            </div>
            <div class="form-group" style="margin-top: 15px;">
                <label>Description (optional):</label>
                <textarea id="newSnapshotDesc" placeholder="Optional description..." style="width: 100%; padding: 10px; height: 60px;"></textarea>
            </div>
            <div class="form-group" style="margin-top: 15px;">
                <label>Tags (comma-separated, optional):</label>
                <input type="text" id="newSnapshotTags" placeholder="e.g., plates, summer, 2024" style="width: 100%; padding: 10px;">
            </div>
            <div class="confirm-actions" style="margin-top: 20px;">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn btn-primary" onclick="createSnapshot()">CREATE</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
    document.getElementById('newSnapshotName').focus();
}

async function createSnapshot() {
    const name = document.getElementById('newSnapshotName').value.trim();
    const isHistorical = document.getElementById('newSnapshotType').value === 'true';
    const description = document.getElementById('newSnapshotDesc').value.trim();
    const tagsStr = document.getElementById('newSnapshotTags').value.trim();
    const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];
    
    if (!name) {
        showToast('Please enter a snapshot name', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/catalogs/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, is_historical: isHistorical, description, tags })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to create snapshot');
        }
        
        closeConfirmModal();
        showToast('Snapshot created successfully', 'success');
        loadSnapshots();
        
    } catch (error) {
        console.error('Error creating snapshot:', error);
        showToast(error.message || 'Failed to create snapshot', 'error');
    }
}

async function deleteSnapshot(snapshotFile) {
    if (!confirm(`Are you sure you want to delete "${snapshotFile}"? This cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/catalogs/${encodeURIComponent(snapshotFile)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to delete snapshot');
        }
        
        showToast('Snapshot deleted', 'success');
        loadSnapshots();
        
        notifyMainAppOfChange('snapshot_deleted', { snapshotFile });
        
    } catch (error) {
        console.error('Error deleting snapshot:', error);
        showToast(error.message || 'Failed to delete snapshot', 'error');
    }
}

async function renameSnapshot(snapshotFile) {
    const newName = prompt('Enter new name for snapshot:');
    if (!newName || !newName.trim()) return;
    
    try {
        const response = await fetch(`/api/catalogs/${encodeURIComponent(snapshotFile)}/rename`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_name: newName.trim() })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to rename snapshot');
        }
        
        showToast('Snapshot renamed', 'success');
        loadSnapshots();
        
    } catch (error) {
        console.error('Error renaming snapshot:', error);
        showToast(error.message || 'Failed to rename snapshot', 'error');
    }
}

async function exportSnapshot(snapshotFile) {
    try {
        showToast('Exporting snapshot...', 'info');
        
        const response = await fetch('/api/catalogs/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ snapshot: snapshotFile })
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = snapshotFile.replace('.db', '-export.zip');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            showToast('Snapshot exported', 'success');
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
        
    } catch (error) {
        console.error('Error exporting snapshot:', error);
        showToast('Failed to export snapshot', 'error');
    }
}

function showImportSnapshotModal() {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    body.innerHTML = `
        <div class="confirm-modal">
            <h2>IMPORT SNAPSHOT</h2>
            <p>Select a .zip file exported from another catalog:</p>
            <div class="form-group" style="margin-top: 15px;">
                <input type="file" id="importSnapshotFile" accept=".zip" style="width: 100%;">
            </div>
            <div class="confirm-actions" style="margin-top: 20px;">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn btn-primary" onclick="importSnapshot()">IMPORT</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function importSnapshot() {
    const fileInput = document.getElementById('importSnapshotFile');
    const file = fileInput?.files[0];
    
    if (!file) {
        showToast('Please select a file', 'error');
        return;
    }
    
    try {
        showToast('Importing snapshot...', 'info');
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/catalogs/import', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Import failed');
        }
        
        closeConfirmModal();
        showToast('Snapshot imported successfully', 'success');
        loadSnapshots();
        
    } catch (error) {
        console.error('Error importing snapshot:', error);
        showToast(error.message || 'Failed to import snapshot', 'error');
    }
}

function showMergeSnapshotsModal() {
    const modal = document.getElementById('confirmModal');
    const body = document.getElementById('confirmModalBody');
    
    const allSnapshots = [...(snapshotData.historical || []), ...(snapshotData.new || [])];
    
    if (allSnapshots.length < 2) {
        showToast('Need at least 2 snapshots to merge', 'error');
        return;
    }
    
    let options = allSnapshots.map(s => `
        <label style="display: block; margin: 5px 0;">
            <input type="checkbox" value="${escapeHtml(s.snapshot_file)}" class="merge-checkbox">
            ${escapeHtml(s.name)} (${s.product_count} products)
        </label>
    `).join('');
    
    body.innerHTML = `
        <div class="confirm-modal">
            <h2>MERGE SNAPSHOTS</h2>
            <p>Select snapshots to merge:</p>
            <div class="confirm-details" style="max-height: 200px; overflow-y: auto;">
                ${options}
            </div>
            <div class="form-group" style="margin-top: 15px;">
                <label>New Snapshot Name:</label>
                <input type="text" id="mergeSnapshotName" placeholder="e.g., Combined Catalog 2024" style="width: 100%; padding: 10px;">
            </div>
            <div class="form-group" style="margin-top: 15px;">
                <label>Type:</label>
                <select id="mergeSnapshotType" style="width: 100%; padding: 10px;">
                    <option value="true">Historical Catalog</option>
                    <option value="false">New Products</option>
                </select>
            </div>
            <div class="confirm-actions" style="margin-top: 20px;">
                <button class="btn" onclick="closeConfirmModal()">CANCEL</button>
                <button class="btn btn-primary" onclick="mergeSnapshots()">MERGE</button>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function mergeSnapshots() {
    const checkboxes = document.querySelectorAll('.merge-checkbox:checked');
    const snapshots = Array.from(checkboxes).map(cb => cb.value);
    const newName = document.getElementById('mergeSnapshotName').value.trim();
    const isHistorical = document.getElementById('mergeSnapshotType').value === 'true';
    
    if (snapshots.length < 2) {
        showToast('Select at least 2 snapshots to merge', 'error');
        return;
    }
    
    if (!newName) {
        showToast('Please enter a name for the merged snapshot', 'error');
        return;
    }
    
    try {
        showToast('Merging snapshots...', 'info');
        
        const response = await fetch('/api/catalogs/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ snapshots, new_name: newName, is_historical: isHistorical })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Merge failed');
        }
        
        const data = await response.json();
        closeConfirmModal();
        showToast(`Merged ${data.products_merged} products into new snapshot`, 'success');
        loadSnapshots();
        
    } catch (error) {
        console.error('Error merging snapshots:', error);
        showToast(error.message || 'Failed to merge snapshots', 'error');
    }
}

async function viewSnapshot(snapshotFile) {
    // For now, show snapshot info in a modal
    // In a full implementation, this would open a product browser for that specific snapshot
    try {
        const response = await fetch(`/api/catalogs/${encodeURIComponent(snapshotFile)}/info`);
        if (!response.ok) throw new Error('Failed to load snapshot info');
        
        const data = await response.json();
        const snapshot = data.snapshot;
        
        const modal = document.getElementById('confirmModal');
        const body = document.getElementById('confirmModalBody');
        
        let categoriesHtml = '';
        if (snapshot.categories && snapshot.categories.length > 0) {
            categoriesHtml = snapshot.categories.map(c => 
                `<div class="category-item">
                    <span>${escapeHtml(c.category || '(Uncategorized)')}</span>
                    <span>${c.count} products</span>
                </div>`
            ).join('');
        } else {
            categoriesHtml = '<p>No categories</p>';
        }
        
        body.innerHTML = `
            <div class="confirm-modal" style="text-align: left;">
                <h2>${escapeHtml(snapshot.name)}</h2>
                <div class="snapshot-detail-grid">
                    <div class="detail-row">
                        <span class="label">File:</span>
                        <span class="value">${escapeHtml(snapshot.snapshot_file)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Version:</span>
                        <span class="value">${snapshot.version || '1.0'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Type:</span>
                        <span class="value">${snapshot.is_historical ? 'Historical Catalog' : 'New Products'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Products:</span>
                        <span class="value">${snapshot.product_count?.toLocaleString() || 0}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Database Size:</span>
                        <span class="value">${formatSize(snapshot.size_mb)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Images Size:</span>
                        <span class="value">${formatSize(snapshot.uploads_size_mb)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Created:</span>
                        <span class="value">${formatDate(snapshot.created_at)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Last Modified:</span>
                        <span class="value">${formatDate(snapshot.last_modified)}</span>
                    </div>
                    ${snapshot.description ? `
                    <div class="detail-row">
                        <span class="label">Description:</span>
                        <span class="value">${escapeHtml(snapshot.description)}</span>
                    </div>
                    ` : ''}
                    ${snapshot.tags?.length ? `
                    <div class="detail-row">
                        <span class="label">Tags:</span>
                        <span class="value">${escapeHtml(snapshot.tags.join(', '))}</span>
                    </div>
                    ` : ''}
                </div>
                <h3 style="margin-top: 20px;">Categories</h3>
                <div class="category-breakdown" style="max-height: 150px;">
                    ${categoriesHtml}
                </div>
                <div class="confirm-actions" style="margin-top: 20px;">
                    <button class="btn" onclick="closeConfirmModal()">CLOSE</button>
                </div>
            </div>
        `;
        
        modal.style.display = 'flex';
        
    } catch (error) {
        console.error('Error viewing snapshot:', error);
        showToast('Failed to load snapshot details', 'error');
    }
}

// Snapshots are now in catalog tab, loaded on page init


// ============ Main Database Integration ============

async function saveCurrentAsSnapshot() {
    const name = prompt('Enter name for this snapshot:');
    if (!name || !name.trim()) return;
    
    const description = prompt('Enter description (optional):') || '';
    
    try {
        showToast('Saving current catalog as snapshot...', 'info');
        
        const response = await fetch('/api/catalogs/save-current', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                name: name.trim(), 
                description,
                tags: ['saved']
            })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to save snapshot');
        }
        
        const data = await response.json();
        showToast(`Saved as "${data.name}" (${data.product_count} products)`, 'success');
        loadSnapshots();
        
    } catch (error) {
        console.error('Error saving snapshot:', error);
        showToast(error.message || 'Failed to save snapshot', 'error');
    }
}

async function loadSnapshotToMain(snapshotFile) {
    if (!confirm(`Load "${snapshotFile}" into main catalog?\n\nThis will REPLACE your current catalog. Make sure to save it first if needed.`)) {
        return;
    }
    
    try {
        showToast('Loading snapshot to main catalog...', 'info');
        
        const response = await fetch(`/api/catalogs/load/${encodeURIComponent(snapshotFile)}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to load snapshot');
        }
        
        const data = await response.json();
        showToast(`Loaded "${data.name}" (${data.product_count} products) to main catalog`, 'success');
        
        // Notify main app and CSV builder of catalog change
        notifyMainAppOfChange('catalog_loaded', { 
            snapshotFile, 
            name: data.name,
            productCount: data.product_count 
        });
        
        loadStats();
        loadSnapshots();
        
    } catch (error) {
        console.error('Error loading snapshot:', error);
        showToast(error.message || 'Failed to load snapshot', 'error');
    }
}

// Update snapshot card rendering to include "Load to Main" button
const originalRenderSnapshotList = renderSnapshotList;
renderSnapshotList = function(containerId, snapshots, isHistorical) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (!snapshots || snapshots.length === 0) {
        container.innerHTML = '<p class="no-snapshots">No snapshots found. Create one to get started.</p>';
        return;
    }
    
    const activeList = isHistorical ? activeSnapshots.active_historical : activeSnapshots.active_new;
    
    let html = '';
    for (const snapshot of snapshots) {
        const isActive = activeList?.includes(snapshot.snapshot_file);
        const tags = snapshot.tags?.join(', ') || '';
        
        html += `
            <div class="snapshot-card ${isActive ? 'active' : ''}" data-snapshot="${escapeHtml(snapshot.snapshot_file)}">
                <div class="snapshot-info" style="flex: 1;">
                    <div class="snapshot-name">${escapeHtml(snapshot.name)} (v${snapshot.version || '1.0'})</div>
                    <div class="snapshot-meta">
                        ${snapshot.product_count?.toLocaleString() || 0} products | ${snapshot.total_size_mb || 0} MB
                    </div>
                    <div class="snapshot-date">Created: ${formatDate(snapshot.created_at)}</div>
                    ${tags ? `<div class="snapshot-tags">Tags: ${escapeHtml(tags)}</div>` : ''}
                </div>
                <div class="snapshot-actions">
                    <button class="btn-small btn-primary" onclick="loadSnapshotToMain('${escapeHtml(snapshot.snapshot_file)}')" title="Load this snapshot into main catalog">LOAD</button>
                    <button class="btn-small" onclick="viewSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="View products">VIEW</button>
                    <button class="btn-small" onclick="renameSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="Rename">RENAME</button>
                    <button class="btn-small" onclick="exportSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="Export as ZIP">EXPORT</button>
                    <button class="btn-small danger" onclick="deleteSnapshot('${escapeHtml(snapshot.snapshot_file)}')" title="Delete">DEL</button>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
};
