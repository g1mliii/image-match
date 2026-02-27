/*
 * Results/filter/history module extracted from app.core.js.
 */

const runWhenDomReadyResults = window.__catalogMatchOnDomReady || function(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback, { once: true });
        return;
    }
    setTimeout(callback, 0);
};

function initResults() {
    // Use tracked listeners to prevent memory leaks
    addTrackedListener(document.getElementById('exportCsvBtn'), 'click', exportResults, 'results');
    addTrackedListener(document.getElementById('resetBtn'), 'click', resetApp, 'results');
    addTrackedListener(document.getElementById('modalClose'), 'click', closeModal, 'results');
}

const debounceMap = new Map();
function debounce(key, func, delay = 300) {
    if (debounceMap.has(key)) {
        clearTimeout(debounceMap.get(key));
    }
    const timeoutId = setTimeout(() => {
        func();
        debounceMap.delete(key);
    }, delay);
    debounceMap.set(key, timeoutId);
}

// Cleanup debounce timers (prevents memory leaks)
function clearAllDebounces() {
    debounceMap.forEach(timeoutId => clearTimeout(timeoutId));
    debounceMap.clear();
}

function yieldToUi() {
    return new Promise(resolve => setTimeout(resolve, 0));
}

let dynamicFiltersDatasetToken = null;
let categoryFilterDatasetToken = null;
let serverSessionFacets = null;
let serverSessionFacetsFor = null;
let serverDynamicFiltersForSession = null;
let serverResultsSummary = null;
let serverResultsTotal = 0;
let serverResultsFilteredCount = 0;
let serverResultsAbortController = null;
let serverResultsRequestToken = 0;

function isSessionServerMode() {
    return typeof serverResultsMode !== 'undefined' &&
           serverResultsMode === true &&
           typeof currentMatchSessionId === 'string' &&
           currentMatchSessionId.length > 0;
}

function resetServerResultsClientState() {
    serverSessionFacets = null;
    serverSessionFacetsFor = null;
    serverDynamicFiltersForSession = null;
    serverResultsSummary = null;
    serverResultsTotal = 0;
    serverResultsFilteredCount = 0;
    if (serverResultsAbortController) {
        try {
            serverResultsAbortController.abort();
        } catch (_) {}
    }
    serverResultsAbortController = null;
    serverResultsRequestToken = 0;

    const dynamicFiltersContainer = document.getElementById('dynamicFiltersContainer');
    if (dynamicFiltersContainer) {
        cleanupDynamicFilterDropdownListeners(dynamicFiltersContainer);
        dynamicFiltersContainer.remove();
    }
}

function setServerResultsSessionState(state) {
    if (!state || !state.sessionId) return;
    serverResultsSummary = state.summary || {};
    serverResultsTotal = Number(state.totalResults || 0);

    if (serverSessionFacetsFor !== state.sessionId) {
        serverSessionFacets = null;
        serverDynamicFiltersForSession = null;
        dynamicFiltersDatasetToken = null;
        categoryFilterDatasetToken = null;
    }
}

function getResultsDatasetToken(results = matchResults) {
    if (!results || results.length === 0) return 'empty';
    const firstId = results[0]?.p?.id || 0;
    const lastId = results[results.length - 1]?.p?.id || 0;
    return `${results.length}:${firstId}:${lastId}`;
}


function createCheckboxFilter(key, values, withSearch) {
    const container = document.createElement('div');
    container.className = 'checkbox-filter-container';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '5px';

    // Search box for 11-50 values
    if (withSearch) {
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = `Search...`;
        searchInput.className = 'input input-sm';
        searchInput.style.width = '150px';
        searchInput.style.marginBottom = '5px';

        // PERFORMANCE FIX #5: Cache DOM queries and implement actual debouncing
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();

            // Debounce to avoid excessive DOM queries on rapid typing
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const checkboxes = container.querySelectorAll('.filter-checkbox-item');
                checkboxes.forEach(item => {
                    // Cache label element to avoid repeated querySelector
                    const labelEl = item.querySelector('label');
                    if (labelEl) {
                        const label = labelEl.textContent.toLowerCase();
                        item.style.display = label.includes(searchTerm) ? 'flex' : 'none';
                    }
                });
            }, 150); // 150ms debounce delay
        });

        container.appendChild(searchInput);
    }

    // Checkbox list container (scrollable for 11-50)
    const listContainer = document.createElement('div');
    listContainer.className = 'checkbox-list';
    listContainer.style.display = 'flex';
    listContainer.style.flexDirection = 'column';
    listContainer.style.gap = '3px';

    if (withSearch) {
        // Scrollable container for 11-50 values
        listContainer.style.maxHeight = '200px';
        listContainer.style.overflowY = 'auto';
        listContainer.style.border = '1px solid #e2e8f0';
        listContainer.style.borderRadius = '4px';
        listContainer.style.padding = '5px';
    }

    // Create checkboxes (PERFORMANCE: use fragment for batch DOM updates)
    const fragment = document.createDocumentFragment();
    values.forEach((val, idx) => {
        const item = document.createElement('div');
        item.className = 'filter-checkbox-item';
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '5px';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `filter-${key}-${idx}`;
        checkbox.value = val;
        checkbox.style.cursor = 'pointer';

        // PERFORMANCE: Event delegation handled via onchange
        checkbox.onchange = (e) => {
            updateMetadataFilterMulti(key, val, e.target.checked);
        };

        const label = document.createElement('label');
        label.htmlFor = checkbox.id;
        label.textContent = val;
        label.style.fontSize = '11px';
        label.style.cursor = 'pointer';
        label.style.userSelect = 'none';

        item.appendChild(checkbox);
        item.appendChild(label);
        fragment.appendChild(item);
    });

    listContainer.appendChild(fragment);
    container.appendChild(listContainer);

    return container;
}


function createSearchableDropdown(key, values) {
    const container = document.createElement('div');
    container.className = 'searchable-dropdown-container';
    container.style.position = 'relative';
    container.style.width = '200px';

    // Search input
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = `Search ${values.length} values...`;
    searchInput.className = 'input input-sm';
    searchInput.style.width = '100%';
    searchInput.style.paddingRight = '30px';

    // Dropdown icon
    const dropdownIcon = document.createElement('span');
    dropdownIcon.innerHTML = '▼';
    dropdownIcon.style.position = 'absolute';
    dropdownIcon.style.right = '10px';
    dropdownIcon.style.top = '8px';
    dropdownIcon.style.fontSize = '10px';
    dropdownIcon.style.pointerEvents = 'none';
    dropdownIcon.style.color = '#718096';

    // Dropdown list container
    const dropdownList = document.createElement('div');
    dropdownList.className = 'dropdown-list';
    dropdownList.style.display = 'none';
    dropdownList.style.position = 'absolute';
    dropdownList.style.top = '100%';
    dropdownList.style.left = '0';
    dropdownList.style.width = '100%';
    dropdownList.style.maxHeight = '250px';
    dropdownList.style.overflowY = 'auto';
    dropdownList.style.backgroundColor = 'white';
    dropdownList.style.border = '1px solid #e2e8f0';
    dropdownList.style.borderRadius = '4px';
    dropdownList.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    dropdownList.style.zIndex = '1000';
    dropdownList.style.marginTop = '2px';

    // Selected values display
    const selectedDisplay = document.createElement('div');
    selectedDisplay.className = 'selected-values';
    selectedDisplay.style.fontSize = '10px';
    selectedDisplay.style.color = '#718096';
    selectedDisplay.style.marginTop = '3px';
    selectedDisplay.style.minHeight = '14px';

    const RENDER_BATCH_SIZE = 200;
    let currentFilteredValues = values;
    let renderedCount = 0;
    const loadState = document.createElement('div');
    loadState.style.padding = '8px 10px';
    loadState.style.fontSize = '10px';
    loadState.style.color = '#718096';
    loadState.style.textAlign = 'center';

    function createDropdownItem(val) {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.style.padding = '8px 10px';
        item.style.cursor = 'pointer';
        item.style.fontSize = '11px';
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '5px';
        item.style.borderBottom = '1px solid #f7fafc';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = val;
        checkbox.style.cursor = 'pointer';

        // Check if already selected
        if (window.metadataFilterCriteria &&
            window.metadataFilterCriteria[key] &&
            window.metadataFilterCriteria[key].values &&
            window.metadataFilterCriteria[key].values.has(val)) {
            checkbox.checked = true;
        }

        checkbox.onclick = (e) => {
            e.stopPropagation(); // Prevent dropdown close
            updateMetadataFilterMulti(key, val, checkbox.checked);
            updateSelectedDisplay();
        };

        const label = document.createElement('span');
        label.textContent = val;
        label.style.flex = '1';

        item.appendChild(checkbox);
        item.appendChild(label);

        // Hover effect
        item.onmouseenter = () => item.style.backgroundColor = '#f7fafc';
        item.onmouseleave = () => item.style.backgroundColor = 'white';

        item.onclick = () => {
            checkbox.checked = !checkbox.checked;
            updateMetadataFilterMulti(key, val, checkbox.checked);
            updateSelectedDisplay();
        };

        return item;
    }

    function updateLoadState() {
        if (currentFilteredValues.length === 0) {
            loadState.textContent = 'No values found';
            if (loadState.parentNode !== dropdownList) dropdownList.appendChild(loadState);
            return;
        }

        if (renderedCount < currentFilteredValues.length) {
            loadState.textContent = `Showing ${renderedCount} of ${currentFilteredValues.length}. Scroll for more.`;
            if (loadState.parentNode !== dropdownList) dropdownList.appendChild(loadState);
        } else if (loadState.parentNode === dropdownList) {
            dropdownList.removeChild(loadState);
        }
    }

    function renderNextBatch() {
        if (renderedCount >= currentFilteredValues.length) {
            updateLoadState();
            return;
        }

        const fragment = document.createDocumentFragment();
        const endIndex = Math.min(renderedCount + RENDER_BATCH_SIZE, currentFilteredValues.length);
        for (let i = renderedCount; i < endIndex; i++) {
            fragment.appendChild(createDropdownItem(currentFilteredValues[i]));
        }

        if (loadState.parentNode === dropdownList) {
            dropdownList.removeChild(loadState);
        }
        dropdownList.appendChild(fragment);
        renderedCount = endIndex;
        updateLoadState();
    }

    function renderItems(filteredValues, reset = false) {
        if (reset) {
            dropdownList.innerHTML = '';
            currentFilteredValues = filteredValues;
            renderedCount = 0;
        }
        renderNextBatch();
    }

    function updateSelectedDisplay() {
        const selected = window.metadataFilterCriteria &&
                        window.metadataFilterCriteria[key] &&
                        window.metadataFilterCriteria[key].values
                        ? Array.from(window.metadataFilterCriteria[key].values)
                        : [];

        if (selected.length > 0) {
            selectedDisplay.textContent = `${selected.length} selected: ${selected.slice(0, 3).join(', ')}${selected.length > 3 ? '...' : ''}`;
        } else {
            selectedDisplay.textContent = '';
        }
    }

    // Show/hide dropdown
    searchInput.addEventListener('focus', () => {
        dropdownList.style.display = 'block';
        renderItems(values, true);
    });

    // PERFORMANCE: Debounced search
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        debounce(`dropdown-search-${key}`, () => {
            const filtered = searchTerm
                ? values.filter(v => v.toLowerCase().includes(searchTerm))
                : values;
            renderItems(filtered, true);
        }, 200);
    });

    // Infinite-scroll style incremental rendering for large value lists.
    dropdownList.addEventListener('scroll', () => {
        const nearBottom = dropdownList.scrollTop + dropdownList.clientHeight >= dropdownList.scrollHeight - 24;
        if (nearBottom) {
            renderNextBatch();
        }
    });

    // Close dropdown when clicking outside (MEMORY LEAK PREVENTION: use named function for cleanup)
    const closeDropdown = (e) => {
        if (!container.contains(e.target)) {
            dropdownList.style.display = 'none';
        }
    };
    document.addEventListener('click', closeDropdown);

    // Store reference for cleanup
    container._closeDropdown = closeDropdown;

    container.appendChild(searchInput);
    container.appendChild(dropdownIcon);
    container.appendChild(dropdownList);
    container.appendChild(selectedDisplay);

    return container;
}


function updateMetadataFilterMulti(key, value, isChecked) {
    if (!window.metadataFilterCriteria) window.metadataFilterCriteria = {};
    if (!window.metadataFilterCriteria[key]) window.metadataFilterCriteria[key] = { values: new Set() };

    if (isChecked) {
        window.metadataFilterCriteria[key].values.add(value);
    } else {
        window.metadataFilterCriteria[key].values.delete(value);
    }

    // Clean up if no values selected
    if (window.metadataFilterCriteria[key].values.size === 0) {
        delete window.metadataFilterCriteria[key];
    }

    displayResults();
}

function generateDynamicFilters() {
    if (isSessionServerMode()) return;

    const container = document.querySelector('.filters');
    // Prevent duplicate generation or generating if no matches
    if (!container || document.getElementById('dynamicFiltersContainer') || !matchResults || matchResults.length === 0) return;

    // Require schema or at least meaningful data
    // If no schema (Mode 1), we might infer from match results, but schema is better
    const schema = window.metadataSchema;
    if (!schema) return;

    // Create container
    const dynContainer = document.createElement('div');
    dynContainer.id = 'dynamicFiltersContainer';
    dynContainer.style.marginTop = '15px';
    dynContainer.style.paddingTop = '15px';
    dynContainer.style.borderTop = '2px solid #e2e8f0';
    dynContainer.style.display = 'flex';
    dynContainer.style.gap = '15px';
    dynContainer.style.flexWrap = 'wrap';
    dynContainer.style.alignItems = 'center';
    dynContainer.style.width = '100%';

    // Toggle button for filters
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'btn btn-sm';
    toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▼</span>';
    toggleBtn.style.marginRight = '10px';
    toggleBtn.onclick = () => {
        const content = dynContainer.querySelector('.dyn-content');
        if (content.style.display === 'none') {
            content.style.display = 'flex';
            toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▲</span>';
        } else {
            content.style.display = 'none';
            toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▼</span>';
        }
    };

    // Content wrapper
    const contentDiv = document.createElement('div');
    contentDiv.className = 'dyn-content';
    contentDiv.style.display = 'none'; // Hidden by default
    contentDiv.style.flexWrap = 'wrap';
    contentDiv.style.gap = '15px';
    contentDiv.style.alignItems = 'flex-start';
    contentDiv.style.width = '100%';

    // Loop schema cols
    let filterCount = 0;
    // PERFORMANCE FIX #4: Use DocumentFragment to batch DOM insertions and avoid multiple reflows
    const fragment = document.createDocumentFragment();
    // PERFORMANCE: Build a single sampled match pool once, then reuse for all non-numeric fields.
    const totalResults = matchResults.length;
    const maxScan = totalResults <= 2000 ? totalResults : Math.min(totalResults, 10000);
    const scanStep = totalResults > maxScan ? Math.max(1, Math.floor(totalResults / maxScan)) : 1;
    const maxMatchesPerProduct = 20;
    const maxSampledMatches = 50000;
    const sampledMatches = [];
    let scannedCount = 0;

    for (let i = 0; i < totalResults; i += scanStep) {
        if (scannedCount >= maxScan || sampledMatches.length >= maxSampledMatches) break;
        scannedCount++;

        const mList = matchResults[i].m || [];
        for (let j = 0; j < Math.min(mList.length, maxMatchesPerProduct); j++) {
            sampledMatches.push(mList[j]);
            if (sampledMatches.length >= maxSampledMatches) break;
        }
    }

    schema.forEach(col => {
        const key = col.column_name;
        if (['id', 'image_path', 'sku', 'name', 'category'].includes(key)) return;

        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.gap = '5px';

        const label = document.createElement('label');
        label.className = 'filter-label';
        label.textContent = col.display_name;
        label.style.fontSize = '12px';
        label.style.fontWeight = '600';
        label.style.color = '#718096';
        wrapper.appendChild(label);

        if (col.data_type === 'numeric') {
            // Range inputs
            const inputs = document.createElement('div');
            inputs.style.display = 'flex';
            inputs.style.gap = '5px';
            inputs.style.alignItems = 'center';
            inputs.innerHTML = `
                <input type="number" placeholder="Min" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'min')">
                <span style="color:#cbd5e0; line-height:1;">-</span>
                <input type="number" placeholder="Max" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'max')">
            `;
            wrapper.appendChild(inputs);
            fragment.appendChild(wrapper);  // PERFORMANCE FIX #4: Append to fragment instead of contentDiv
            filterCount++;
        } else {
            // SMART HYBRID FILTERS: Use sampled matches to get unique values.
            const uniqueVals = new Set();
            const maxUniqueValues = 2000; // Keep dropdowns practical while supporting large catalogs

            for (let i = 0; i < sampledMatches.length; i++) {
                if (uniqueVals.size >= maxUniqueValues) break;
                const m = sampledMatches[i];
                const val = (m.mv && m.mv[key]) || (m.metadata_values && m.metadata_values[key]);
                if (val) {
                    uniqueVals.add(String(val)); // Ensure string for consistency
                    if (uniqueVals.size >= maxUniqueValues) break;
                }
            }

            if (uniqueVals.size > 0) {
                const sortedVals = Array.from(uniqueVals).sort();
                const valueCount = sortedVals.length;

                // OPTION 3: Smart Hybrid UI based on value count
                if (valueCount <= 10) {
                    // ≤10 values: Simple checkbox list
                    wrapper.appendChild(createCheckboxFilter(key, sortedVals, false));
                    fragment.appendChild(wrapper);  // PERFORMANCE FIX #4: Append to fragment instead of contentDiv
                    filterCount++;
                } else if (valueCount <= 50) {
                    // 11-50 values: Scrollable checkbox list with search
                    wrapper.appendChild(createCheckboxFilter(key, sortedVals, true));
                    fragment.appendChild(wrapper);  // PERFORMANCE FIX #4: Append to fragment instead of contentDiv
                    filterCount++;
                } else {
                    // >50 values: Searchable multi-select dropdown
                    wrapper.appendChild(createSearchableDropdown(key, sortedVals));
                    fragment.appendChild(wrapper);  // PERFORMANCE FIX #4: Append to fragment instead of contentDiv
                    filterCount++;
                }
            }
        }
    });


    contentDiv.appendChild(fragment);

    if (filterCount > 0) {
        // Append
        container.appendChild(dynContainer);
        // Only add toggle if many filters
        if (filterCount > 3) {
            dynContainer.appendChild(toggleBtn);
            dynContainer.appendChild(contentDiv);
        } else {
            // Show inline if few
            contentDiv.style.display = 'flex';
            dynContainer.appendChild(contentDiv);
        }

        // Add listener helper
        window.updateMetadataFilter = (key, value, type) => {
            if (!window.metadataFilterCriteria) window.metadataFilterCriteria = {};

            if (value === '') {
                // Clear filter for this key/type
                if (window.metadataFilterCriteria[key]) {
                    delete window.metadataFilterCriteria[key][type];
                    // If no more criteria for this key, delete the key
                    if (Object.keys(window.metadataFilterCriteria[key]).length === 0) {
                        delete window.metadataFilterCriteria[key];
                    }
                }
            } else {
                // Set filter criteria
                if (!window.metadataFilterCriteria[key]) window.metadataFilterCriteria[key] = {};

                if (type === 'min') {
                    window.metadataFilterCriteria[key].min = parseFloat(value);
                } else if (type === 'max') {
                    window.metadataFilterCriteria[key].max = parseFloat(value);
                } else if (type === 'equals') {
                    window.metadataFilterCriteria[key].equals = value;
                }
            }
            displayResults();
        };
    }
}

function populateDynamicSortOptions() {
    const sortSelect = document.getElementById('sortBySelect');

    if (!sortSelect || !window.metadataSchema) return;

    // Get existing option values to avoid duplicates
    const existingValues = Array.from(sortSelect.options).map(opt => opt.value);

    // Add dynamic columns from schema
    window.metadataSchema.forEach(col => {
        const key = col.column_name;

        // Skip core fields and already existing options
        if (['id', 'image_path', 'sku', 'name', 'category'].includes(key)) return;
        if (existingValues.includes(key)) return;

        // Create new option
        const option = document.createElement('option');
        option.value = key;
        option.textContent = (col.display_name || key).toUpperCase();
        sortSelect.appendChild(option);
    });
}


function renderMetadataScoresHtml(metadataScores, metadataValues) {
    if (!metadataScores) return '';

    const entries = Object.entries(metadataScores)
        .filter(([_, score]) => score !== undefined)
        .slice(0, 4);

    if (entries.length === 0) return '';

    // PERFORMANCE: Pre-compute numeric fields once (avoid checking in loop)
    const numericFields = new Map(); // key -> numeric value
    if (metadataValues) {
        for (const [key, value] of Object.entries(metadataValues)) {
            if (value !== undefined) {
                if (typeof value === 'number') {
                    numericFields.set(key, value);
                } else if (!isNaN(parseFloat(value)) && isFinite(parseFloat(value))) {
                    numericFields.set(key, parseFloat(value));
                }
            }
        }
    }

    const tagsHtml = entries.map(([key, score]) => {
        // Check if this field is numeric (pre-computed)
        if (numericFields.has(key)) {
            const numValue = numericFields.get(key);
            // Format numeric values with commas
            const formattedValue = numValue >= 1000
                ? numValue.toLocaleString('en-US', { maximumFractionDigits: 0 })
                : numValue.toFixed(1);
            return `<span class="metadata-tag" title="${key}: ${formattedValue}">${key.substring(0, 3)}: ${formattedValue}</span>`;
        }
        // For non-numeric fields, show similarity percentage
        return `<span class="metadata-tag" title="${key}: ${score.toFixed(0)}% match">${key.substring(0, 3)}: ${score.toFixed(0)}%</span>`;
    }).join('');

    const moreTag = Object.keys(metadataScores).length > 4 ? '<span class="metadata-tag">+more</span>' : '';

    return `<div class="match-metadata-scores">${tagsHtml}${moreTag}</div>`;
}

function renderMatchCard(match, productId, isMetadataMode) {
    const similarityScore = getScore(match, 'similarity');
    // CRITICAL FIX: Use mscores (compact format) with fallback, and pass actual values
    const metadataScoresHtml = renderMetadataScoresHtml(
        match.metadata_scores || match.mscores,
        match.metadata_values || match.mv
    );

    const imageHtml = !isMetadataMode ?
        `<img data-src="/api/products/${match.mid}/image" class="match-image lazy-load"
             src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='120'><rect fill='%23e2e8f0' width='180' height='120'/></svg>"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22180%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22180%22 height=%22120%22/></svg>'"
             alt="Match">` : '';

    const filenameHtml = (match.fn && !match.fn.includes('METADATA_ONLY') && !match.fn.includes('METADATA ONLY')) ?
        `<div style="font-size: 11px; color: #718096; margin-bottom: 2px;">${escapeHtml((match.fn || '').split(/[\\/]/).pop())}</div>` : '';

    const duplicateBadge = similarityScore > 90 ? '<span class="duplicate-badge">DUPLICATE?</span>' : '';

    return `
        <div class="match-card" onclick="showDetailedComparison(${productId}, ${match.mid})">
            ${imageHtml}
            <div class="match-score ${getScoreClass(similarityScore)}">
                ${similarityScore.toFixed(1)}%
            </div>
            ${duplicateBadge}
            <div class="match-info">
                <div style="font-weight: 500;">${escapeHtml(match.name || 'Unknown')}</div>
                ${filenameHtml}
                ${metadataScoresHtml}
            </div>
        </div>
    `;
}

function serializeMetadataFilterCriteriaForServer() {
    const serialized = {};
    const criteria = window.metadataFilterCriteria || {};

    Object.entries(criteria).forEach(([field, fieldCriteria]) => {
        if (!fieldCriteria || typeof fieldCriteria !== 'object') return;
        const entry = {};

        if (fieldCriteria.min !== undefined && fieldCriteria.min !== null && fieldCriteria.min !== '') {
            entry.min = fieldCriteria.min;
        }
        if (fieldCriteria.max !== undefined && fieldCriteria.max !== null && fieldCriteria.max !== '') {
            entry.max = fieldCriteria.max;
        }
        if (fieldCriteria.equals !== undefined && fieldCriteria.equals !== null && fieldCriteria.equals !== '') {
            entry.equals = fieldCriteria.equals;
        }
        if (fieldCriteria.values && fieldCriteria.values.size > 0) {
            entry.values = Array.from(fieldCriteria.values);
        }

        if (Object.keys(entry).length > 0) {
            serialized[field] = entry;
        }
    });

    return serialized;
}

function ensureMetadataRangeFilterUpdater() {
    window.updateMetadataFilter = (key, value, type) => {
        if (!window.metadataFilterCriteria) window.metadataFilterCriteria = {};

        if (value === '') {
            if (window.metadataFilterCriteria[key]) {
                delete window.metadataFilterCriteria[key][type];
                if (Object.keys(window.metadataFilterCriteria[key]).length === 0) {
                    delete window.metadataFilterCriteria[key];
                }
            }
        } else {
            if (!window.metadataFilterCriteria[key]) window.metadataFilterCriteria[key] = {};
            if (type === 'min') {
                window.metadataFilterCriteria[key].min = parseFloat(value);
            } else if (type === 'max') {
                window.metadataFilterCriteria[key].max = parseFloat(value);
            } else if (type === 'equals') {
                window.metadataFilterCriteria[key].equals = value;
            }
        }

        displayResults();
    };
}

function populateCategoryFilterFromServerFacets() {
    const select = document.getElementById('categoryFilter');
    if (!select) return;

    const previousValue = select.value || filterCategory || 'all';
    const categories = (serverSessionFacets && Array.isArray(serverSessionFacets.categories))
        ? serverSessionFacets.categories
        : [];

    select.innerHTML = '<option value="all">All Categories</option>';
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    });

    if (previousValue && Array.from(select.options).some(opt => opt.value === previousValue)) {
        select.value = previousValue;
    } else {
        select.value = 'all';
    }
}

function generateDynamicFiltersFromServerFacets() {
    if (!isSessionServerMode()) return;

    const container = document.querySelector('.filters');
    if (!container) return;

    const sessionId = currentMatchSessionId;
    if (!sessionId) return;

    const existing = document.getElementById('dynamicFiltersContainer');
    if (existing && serverDynamicFiltersForSession === sessionId) {
        return;
    }
    if (existing) {
        cleanupDynamicFilterDropdownListeners(existing);
        existing.remove();
    }

    const schema = window.metadataSchema || [];
    const facets = (serverSessionFacets && serverSessionFacets.metadata_facets) || {};
    if (!schema || schema.length === 0 || !facets || Object.keys(facets).length === 0) {
        return;
    }

    const dynContainer = document.createElement('div');
    dynContainer.id = 'dynamicFiltersContainer';
    dynContainer.style.marginTop = '15px';
    dynContainer.style.paddingTop = '15px';
    dynContainer.style.borderTop = '2px solid #e2e8f0';
    dynContainer.style.display = 'flex';
    dynContainer.style.gap = '15px';
    dynContainer.style.flexWrap = 'wrap';
    dynContainer.style.alignItems = 'center';
    dynContainer.style.width = '100%';

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'btn btn-sm';
    toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▼</span>';
    toggleBtn.style.marginRight = '10px';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'dyn-content';
    contentDiv.style.display = 'none';
    contentDiv.style.flexWrap = 'wrap';
    contentDiv.style.gap = '15px';
    contentDiv.style.alignItems = 'flex-start';
    contentDiv.style.width = '100%';

    toggleBtn.onclick = () => {
        if (contentDiv.style.display === 'none') {
            contentDiv.style.display = 'flex';
            toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▲</span>';
        } else {
            contentDiv.style.display = 'none';
            toggleBtn.innerHTML = 'Filters <span style="font-size: 10px;">▼</span>';
        }
    };

    let filterCount = 0;
    const fragment = document.createDocumentFragment();
    const coreFields = new Set(['id', 'image_path', 'sku', 'name', 'category']);

    schema.forEach(col => {
        const key = col.column_name;
        if (!key || coreFields.has(key)) return;

        const facet = facets[key];
        if (!facet) return;

        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.gap = '5px';

        const label = document.createElement('label');
        label.className = 'filter-label';
        label.textContent = col.display_name || key;
        label.style.fontSize = '12px';
        label.style.fontWeight = '600';
        label.style.color = '#718096';
        wrapper.appendChild(label);

        if (facet.type === 'numeric' || col.data_type === 'numeric') {
            const inputs = document.createElement('div');
            inputs.style.display = 'flex';
            inputs.style.gap = '5px';
            inputs.style.alignItems = 'center';
            inputs.innerHTML = `
                <input type="number" placeholder="Min" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'min')">
                <span style="color:#cbd5e0; line-height:1;">-</span>
                <input type="number" placeholder="Max" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'max')">
            `;
            wrapper.appendChild(inputs);
            fragment.appendChild(wrapper);
            filterCount++;
            return;
        }

        const values = Array.isArray(facet.values) ? facet.values : [];
        if (values.length === 0) return;

        if (values.length <= 10) {
            wrapper.appendChild(createCheckboxFilter(key, values, false));
        } else if (values.length <= 50) {
            wrapper.appendChild(createCheckboxFilter(key, values, true));
        } else {
            wrapper.appendChild(createSearchableDropdown(key, values));
        }
        fragment.appendChild(wrapper);
        filterCount++;
    });

    if (filterCount === 0) return;

    ensureMetadataRangeFilterUpdater();
    contentDiv.appendChild(fragment);
    container.appendChild(dynContainer);

    if (filterCount > 3) {
        dynContainer.appendChild(toggleBtn);
        dynContainer.appendChild(contentDiv);
    } else {
        contentDiv.style.display = 'flex';
        dynContainer.appendChild(contentDiv);
    }

    serverDynamicFiltersForSession = sessionId;
}

async function ensureServerSessionFacetsLoaded() {
    if (!isSessionServerMode()) return null;
    const sessionId = currentMatchSessionId;
    if (!sessionId) return null;

    if (serverSessionFacetsFor === sessionId && serverSessionFacets) {
        return serverSessionFacets;
    }

    const response = await fetch(`/api/products/match-results/session/facets?session_id=${encodeURIComponent(sessionId)}`);
    const data = await response.json();
    if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to load filter facets');
    }

    serverSessionFacets = data;
    serverSessionFacetsFor = sessionId;
    serverResultsSummary = data.summary || serverResultsSummary || {};
    populateCategoryFilterFromServerFacets();
    generateDynamicFiltersFromServerFacets();
    return serverSessionFacets;
}

function buildServerSessionQueryParams(page, limit) {
    const params = new URLSearchParams();
    params.set('session_id', currentMatchSessionId);
    params.set('page', String(page));
    params.set('limit', String(limit));
    params.set('search_query', searchQuery || '');
    params.set('filter_category', filterCategory || 'all');
    params.set('duplicates_only', filterDuplicatesOnly ? 'true' : 'false');
    params.set('sort_by', sortBy || 'similarity');
    params.set('sort_order', sortOrder || 'desc');
    params.set('threshold', String(dynamicThreshold || 30));
    params.set('dynamic_limit', String(dynamicLimit || 0));
    params.set('dynamic_search', dynamicSearch || '');
    params.set('metadata_filters', JSON.stringify(serializeMetadataFilterCriteriaForServer()));
    return params;
}

function compactSessionQueryResult(rawResult) {
    if (!rawResult || typeof rawResult !== 'object') {
        return null;
    }

    let product = rawResult.product_data;
    if (!product) {
        product = {
            id: rawResult.product_id,
            filename: `Product ${rawResult.product_id}`,
            name: `Product ${rawResult.product_id}`,
            category: 'Unknown',
            sku: '',
            hasFeatures: false,
            metadata: {}
        };
    } else {
        product.hasFeatures = product.has_features || false;
        if (!product.name) {
            product.name = product.product_name || product.filename || `Product ${product.id}`;
        }
    }

    const rawMatches = Array.isArray(rawResult.matches) ? rawResult.matches : [];
    const seenMatchIds = new Set();
    const uniqueMatches = [];

    for (let i = 0; i < rawMatches.length; i++) {
        const match = rawMatches[i];
        const matchId = match?.product_id || match?.mid || match?.id;
        if (!matchId || seenMatchIds.has(matchId)) continue;
        seenMatchIds.add(matchId);
        uniqueMatches.push(match);
    }

    const resultObj = {
        p: createCompactProduct(product),
        m: uniqueMatches.map(m => createCompactMatch(m)),
        summary_stats: rawResult.summary_stats
    };

    if (rawResult.status !== 'success' && rawResult.error) {
        resultObj.err = rawResult.error;
    }

    return resultObj;
}

async function fetchAllServerSessionResults(options = {}) {
    const {
        actionLabel = 'Loading results',
        pageSize = 500,
        maxResults = 0,
        showProgress = true
    } = options;

    if (!isSessionServerMode()) {
        const localResults = Array.isArray(matchResults) ? matchResults.slice() : [];
        return {
            results: localResults,
            totalResults: localResults.length,
            fetchedResults: localResults.length,
            truncated: false
        };
    }

    await ensureServerSessionFacetsLoaded();

    const boundedPageSize = Math.max(50, Math.min(parseInt(pageSize, 10) || 500, 1000));
    const maxAllowedResults = Math.max(0, parseInt(maxResults, 10) || 0);
    const compactResults = [];

    let page = 1;
    let totalPages = 1;
    let totalResults = 0;
    let nextProgressAt = boundedPageSize * 5;

    while (page <= totalPages) {
        const params = buildServerSessionQueryParams(page, boundedPageSize);
        const response = await fetch(`/api/products/match-results/session/query?${params.toString()}`);
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Failed to load session results');
        }

        if (page === 1) {
            totalPages = data.total_pages || 0;
            totalResults = data.total_results || 0;
            if (totalResults === 0) {
                break;
            }
        } else if (data.total_pages) {
            totalPages = data.total_pages;
        }

        const pageRows = Array.isArray(data.results) ? data.results : [];
        for (let i = 0; i < pageRows.length; i++) {
            const compact = compactSessionQueryResult(pageRows[i]);
            if (compact) {
                compactResults.push(compact);
            }
            if (maxAllowedResults > 0 && compactResults.length >= maxAllowedResults) {
                break;
            }
        }

        if (showProgress && compactResults.length >= nextProgressAt) {
            showToast(
                `${actionLabel}: loaded ${compactResults.length.toLocaleString()} of ${totalResults.toLocaleString()}...`,
                'info'
            );
            nextProgressAt += boundedPageSize * 5;
        }

        if (maxAllowedResults > 0 && compactResults.length >= maxAllowedResults) {
            break;
        }
        if (page >= totalPages) {
            break;
        }

        page += 1;
        await yieldToUi();
    }

    return {
        results: compactResults,
        totalResults: totalResults || compactResults.length,
        fetchedResults: compactResults.length,
        truncated: maxAllowedResults > 0 && totalResults > compactResults.length
    };
}

async function fetchServerSessionResults(resetPage) {
    if (!isSessionServerMode()) return null;
    if (resetPage) currentPage = 1;

    await ensureServerSessionFacetsLoaded();
    const params = buildServerSessionQueryParams(currentPage, RESULTS_PER_PAGE);

    if (serverResultsAbortController) {
        serverResultsAbortController.abort();
    }
    serverResultsAbortController = new AbortController();
    const requestToken = ++serverResultsRequestToken;

    const response = await fetch(
        `/api/products/match-results/session/query?${params.toString()}`,
        { signal: serverResultsAbortController.signal }
    );
    const data = await response.json();

    if (requestToken !== serverResultsRequestToken) {
        return null;
    }

    if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to load session results');
    }

    matchResults = [];
    appendBatchResults(data.results || []);
    totalMatchCount = data.total_results || 0;
    serverResultsFilteredCount = data.total_results || 0;
    serverResultsTotal = data.total_results || serverResultsTotal;
    serverResultsSummary = data.summary || serverResultsSummary || {};

    return data;
}

function displayResults(resetPage = true) {
    if (isSessionServerMode()) {
        void displayResultsServerMode(resetPage);
        return;
    }

    console.log('[DISPLAY] displayResults called');

    // MEMORY: Clean up selections for products no longer in results
    cleanupMetricSelections();

    // Populate dynamic sort options if schema is available
    populateDynamicSortOptions();

    const currentDatasetToken = getResultsDatasetToken();
    const existingFilters = document.getElementById('dynamicFiltersContainer');

    // Regenerate dynamic filters only when the underlying dataset changes.
    if (existingFilters && dynamicFiltersDatasetToken !== currentDatasetToken) {
        cleanupDynamicFilterDropdownListeners(existingFilters);
        existingFilters.remove();
    }

    // Generate once per dataset to avoid expensive rebuilds on every re-render.
    generateDynamicFilters();
    if (dynamicFiltersDatasetToken !== currentDatasetToken) {
        dynamicFiltersDatasetToken = currentDatasetToken;
    }

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

    // CHUNKING SUPPORT: If dataset > 10K, only process current chunk
    let resultsToFilter = matchResults;
    const chunkInfo = getChunkInfo();

    if (chunkInfo.totalResults > CHUNK_SIZE) {
        // Only filter the current chunk to keep memory low
        resultsToFilter = matchResults.slice(chunkInfo.startIdx, chunkInfo.endIdx);
    }

    // Apply filters and sorting
    const filteredResults = filterAndSortResults(resultsToFilter);
    console.log('[DISPLAY] After filtering - filteredResults length:', filteredResults.length);

    const totalProducts = resultsToFilter.length;  // Products in current chunk
    const totalMatches = resultsToFilter.reduce((sum, r) => sum + r.m.length, 0);
    const productsWithMatches = resultsToFilter.filter(r => r.m.length > 0).length;
    const avgMatches = productsWithMatches > 0 ? (totalMatches / productsWithMatches).toFixed(1) : 0;

    const filteredCount = filteredResults.length;

    // Calculate pagination
    const totalPages = Math.ceil(filteredCount / RESULTS_PER_PAGE);
    const startIndex = (currentPage - 1) * RESULTS_PER_PAGE;
    const endIndex = Math.min(startIndex + RESULTS_PER_PAGE, filteredCount);
    const paginatedResults = filteredResults.slice(startIndex, endIndex);

    summaryDiv.innerHTML = `
        <h3>Match Results Summary</h3>
        ${chunkInfo.totalResults > CHUNK_SIZE ? `
            <div style="margin-bottom: 10px; padding: 8px; background: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; border-radius: 4px;">
                <strong>Large Dataset:</strong> Chunk ${chunkInfo.chunkNumber} (${chunkInfo.startIdx.toLocaleString()}-${chunkInfo.endIdx.toLocaleString()})
            </div>
        ` : ''}
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-value">${totalProducts}</span>
                <span class="stat-label">Products (This Chunk)</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${productsWithMatches}</span>
                <span class="stat-label">With Matches</span>
            </div>
            ${filteredCount < totalProducts ? `
            <div class="stat-item" style="background: rgba(102, 126, 234, 0.15);">
                <span class="stat-value">${filteredCount}</span>
                <span class="stat-label">Filtered Results</span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 12px; padding: 10px; background: #f7fafc; border: 2px solid #000; display: flex; gap: 16px; align-items: center; justify-content: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">Min Similarity:</label>
                <input type="range" id="dynamicThresholdSlider" min="30" max="100" value="${dynamicThreshold}"
                       style="width: 150px;"
                       oninput="updateDynamicThresholdPreview(this.value)"
                       onchange="applyDynamicThreshold(this.value)"
                       onkeyup="applyDynamicThreshold(this.value)"
                       onmouseup="applyDynamicThreshold(this.value)"
                       ontouchend="applyDynamicThreshold(this.value)">
                <span id="dynamicThresholdValue" style="font-weight: 600; min-width: 40px;">${dynamicThreshold}%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: 600; color: #2d3748;">SHOW TOP:</label>
                <select id="dynamicLimitSelect" onchange="updateDynamicLimit(this.value)"
                        style="padding: 4px 6px; border: 2px solid #000; background: white; font-weight: 600;">
                    <option value="5" ${dynamicLimit === 5 ? 'selected' : ''}>5</option>
                    <option value="10" ${dynamicLimit === 10 ? 'selected' : ''}>10</option>
                    <option value="20" ${dynamicLimit === 20 ? 'selected' : ''}>20</option>
                    <option value="50" ${dynamicLimit === 50 ? 'selected' : ''}>50</option>
                    <option value="0" ${dynamicLimit === 0 ? 'selected' : ''}>All</option>
                </select>
            </div>
        </div>
        
        ${filteredCount > RESULTS_PER_PAGE ? `
            <div style="text-align: center; margin-top: 10px; color: #718096;">
                Showing ${startIndex + 1}-${endIndex} of ${filteredCount} products
            </div>
        ` : ''}
    `;

    if (filteredResults.length === 0) {
        listDiv.innerHTML = `
            <div class="empty-state">
                <h3>No Results Found</h3>
                <p>Try adjusting your search or filters.</p>
            </div>
        `;
        return;
    }


    const isMetadataMode = newMode === 'metadata';

    listDiv.innerHTML = paginatedResults.map((result, index) => {
        const product = result.p;  // Compact product object
        const matches = result.m;  // Compact matches array

        // Use name from compact product object
        const displayName = product.name;

        // PERFORMANCE: Use cached metadata statistics (avoids recalculation)
        const metadataStats = getCachedMetadataStats(result);
        const statsHtml = renderMetadataStats(metadataStats, product.id);

        // Dynamic Sort Context
        let sortContextHtml = '';
        if (sortBy !== 'similarity' && sortBy !== 'match_count' && sortBy !== 'avg_similarity' && sortBy !== 'name' && sortBy !== 'category') {
            // It's a custom numeric/string sort
            const val = product[sortBy] || (product.meta && product.meta[sortBy]);
            if (val !== undefined) {
                sortContextHtml = `
                    <div style="margin-top:5px; font-size:12px; color:#4a5568; font-weight:600; background:#edf2f7; display:inline-block; padding:2px 6px; border-radius:4px;">
                        Sorted by ${sortBy}: <span style="color:#2b6cb0;">${val}</span>
                    </div>
                 `;
            }
        }

        return `
            <div class="result-item">
                <div class="result-header">
                    ${!isMetadataMode ? `<img data-src="/api/products/${product.id}/image" class="result-image lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><rect fill='%23e2e8f0' width='120' height='120'/></svg>"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22120%22 height=%22120%22/></svg>'"
                         alt="${displayName}">` : ''}
                    <div class="result-info">
                        <h3>${escapeHtml(displayName)}</h3>
                        ${sortContextHtml}
                        <div class="result-meta">
                            Category: ${product.cat || 'Uncategorized'} |
                            ${matches.length} match${matches.length !== 1 ? 'es' : ''} found
                        </div>
                    </div>
                </div>

                ${statsHtml}

                ${matches.length > 0 ? `
                    <div class="matches-grid">
                        ${matches.slice(0, 12).map(match => renderMatchCard(match, product.id, isMetadataMode)).join('')}
                    </div>
                    ${matches.length > 12 ? `
                        <div style="text-align: center; margin-top: 8px;">
                            <button class="btn btn-sm" onclick="showDetailedComparison(${product.id}, ${matches[0].mid})" style="background: #e2e8f0; color: #4a5568;">
                                Show All ${matches.length} Matches
                            </button>
                        </div>
                    ` : ''}
                ` : '<div class="no-matches">No matches found</div>'}
            </div>
        `;
    }).join('');


    if (filteredCount > RESULTS_PER_PAGE) {
        const hasMore = currentPage < totalPages;
        const hasPrevious = currentPage > 1;

        listDiv.innerHTML += `
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 18px; padding: 10px;">
                ${hasPrevious ? `
                    <button class="btn" onclick="loadPreviousPage()" style="min-width: 120px;">
                        Previous
                    </button>
                ` : ''}
                <div style="display: flex; align-items: center; color: #718096; font-weight: 500;">
                    Page ${currentPage} of ${totalPages}
                </div>
                ${hasMore ? `
                    <button class="btn" onclick="loadNextPage()" style="min-width: 120px;">
                        Next
                    </button>
                ` : ''}
            </div>
        `;
    }

    if (chunkInfo.totalResults > CHUNK_SIZE) {
        const totalChunks = Math.ceil(chunkInfo.totalResults / CHUNK_SIZE);
        const hasPrevious = currentChunk > 0;
        const hasNext = chunkInfo.hasMore;

        listDiv.innerHTML += `
            <div style="margin-top: 18px; padding: 10px; text-align: center; border-top: 1px solid #eee;">
                 <div style="margin-bottom: 8px; color: #718096; font-weight: 500;">
                    Data Chunk ${chunkInfo.chunkNumber} of ${totalChunks} (${CHUNK_SIZE.toLocaleString()} products per chunk)
                 </div>
                 <div style="display: flex; gap: 10px; justify-content: center;">
                     <button class="btn btn-sm" onclick="loadPreviousChunk()" ${!hasPrevious ? 'disabled' : ''}>
                        Previous ${CHUNK_SIZE.toLocaleString()}
                     </button>
                     <button class="btn btn-sm" onclick="loadNextChunk()" ${!hasNext ? 'disabled' : ''}>
                        Next ${CHUNK_SIZE.toLocaleString()}
                     </button>
                 </div>
            </div>
        `;
    }

    if (!lazyLoadObserver) {
        // Initialize global observer if not already created
        initLazyLoading();
    }

    // Observe all lazy-load images in the results list using global observer
    const images = listDiv.querySelectorAll('img.lazy-load');
    images.forEach(img => lazyLoadObserver.observe(img));

    // Re-initialize icons in results section only (scoped for performance)
    IconManager.reinit(50, document.getElementById('resultsSection'));
}

async function displayResultsServerMode(resetPage = true) {
    const summaryDiv = document.getElementById('resultsSummary');
    const listDiv = document.getElementById('resultsList');
    if (!summaryDiv || !listDiv) return;

    populateDynamicSortOptions();
    summaryDiv.innerHTML = '';
    listDiv.innerHTML = '';

    try {
        const data = await fetchServerSessionResults(resetPage);
        if (!data) return;
        currentPage = data.page || currentPage;

        const paginatedResults = matchResults;
        const totalProducts = data.total_results || 0;
        const totalMatches = data.filtered_total_matches || 0;
        const productsWithMatches = data.products_with_matches || 0;
        const filteredCount = data.total_results || 0;
        const totalPages = data.total_pages || 0;
        const startIndex = filteredCount === 0 ? 0 : ((data.page - 1) * data.limit);
        const endIndex = filteredCount === 0 ? 0 : Math.min(startIndex + paginatedResults.length, filteredCount);

        summaryDiv.innerHTML = `
            <h3>Match Results Summary</h3>
            <div style="margin-bottom: 10px; padding: 8px; background: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; border-radius: 4px;">
                <strong>Session Mode:</strong> Server-side filtering for large catalogs
            </div>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-value">${totalProducts}</span>
                    <span class="stat-label">Filtered Products</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${productsWithMatches}</span>
                    <span class="stat-label">With Matches</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${totalMatches}</span>
                    <span class="stat-label">Total Matches</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding: 10px; background: #f7fafc; border: 2px solid #000; display: flex; gap: 16px; align-items: center; justify-content: center; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="font-weight: 600; color: #2d3748;">Min Similarity:</label>
                    <input type="range" id="dynamicThresholdSlider" min="30" max="100" value="${dynamicThreshold}"
                           style="width: 150px;"
                           oninput="updateDynamicThresholdPreview(this.value)"
                           onchange="applyDynamicThreshold(this.value)"
                           onkeyup="applyDynamicThreshold(this.value)"
                           onmouseup="applyDynamicThreshold(this.value)"
                           ontouchend="applyDynamicThreshold(this.value)">
                    <span id="dynamicThresholdValue" style="font-weight: 600; min-width: 40px;">${dynamicThreshold}%</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="font-weight: 600; color: #2d3748;">SHOW TOP:</label>
                    <select id="dynamicLimitSelect" onchange="updateDynamicLimit(this.value)"
                            style="padding: 4px 6px; border: 2px solid #000; background: white; font-weight: 600;">
                        <option value="5" ${dynamicLimit === 5 ? 'selected' : ''}>5</option>
                        <option value="10" ${dynamicLimit === 10 ? 'selected' : ''}>10</option>
                        <option value="20" ${dynamicLimit === 20 ? 'selected' : ''}>20</option>
                        <option value="50" ${dynamicLimit === 50 ? 'selected' : ''}>50</option>
                        <option value="0" ${dynamicLimit === 0 ? 'selected' : ''}>All</option>
                    </select>
                </div>
            </div>
            ${filteredCount > 0 ? `
                <div style="text-align: center; margin-top: 10px; color: #718096;">
                    Showing ${startIndex + 1}-${endIndex} of ${filteredCount} products
                </div>
            ` : ''}
        `;

        if (filteredCount === 0 || paginatedResults.length === 0) {
            listDiv.innerHTML = `
                <div class="empty-state">
                    <h3>No Results Found</h3>
                    <p>Try adjusting your search or filters.</p>
                </div>
            `;
            return;
        }

        const isMetadataMode = newMode === 'metadata';
        listDiv.innerHTML = paginatedResults.map((result) => {
            const product = result.p;
            const matches = result.m;
            const displayName = product.name;
            const metadataStats = getCachedMetadataStats(result);
            const statsHtml = renderMetadataStats(metadataStats, product.id);

            let sortContextHtml = '';
            if (sortBy !== 'similarity' && sortBy !== 'match_count' && sortBy !== 'avg_similarity' && sortBy !== 'name' && sortBy !== 'category') {
                const val = product[sortBy] || (product.meta && product.meta[sortBy]);
                if (val !== undefined) {
                    sortContextHtml = `
                        <div style="margin-top:5px; font-size:12px; color:#4a5568; font-weight:600; background:#edf2f7; display:inline-block; padding:2px 6px; border-radius:4px;">
                            Sorted by ${sortBy}: <span style="color:#2b6cb0;">${val}</span>
                        </div>
                     `;
                }
            }

            return `
                <div class="result-item">
                    <div class="result-header">
                        ${!isMetadataMode ? `<img data-src="/api/products/${product.id}/image" class="result-image lazy-load"
                             src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><rect fill='%23e2e8f0' width='120' height='120'/></svg>"
                             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%22120%22><rect fill=%22%23e2e8f0%22 width=%22120%22 height=%22120%22/></svg>'"
                             alt="${displayName}">` : ''}
                        <div class="result-info">
                            <h3>${escapeHtml(displayName)}</h3>
                            ${sortContextHtml}
                            <div class="result-meta">
                                Category: ${product.cat || 'Uncategorized'} |
                                ${matches.length} match${matches.length !== 1 ? 'es' : ''} found
                            </div>
                        </div>
                    </div>

                    ${statsHtml}

                    ${matches.length > 0 ? `
                        <div class="matches-grid">
                            ${matches.slice(0, 12).map(match => renderMatchCard(match, product.id, isMetadataMode)).join('')}
                        </div>
                        ${matches.length > 12 ? `
                            <div style="text-align: center; margin-top: 8px;">
                                <button class="btn btn-sm" onclick="showDetailedComparison(${product.id}, ${matches[0].mid})" style="background: #e2e8f0; color: #4a5568;">
                                    Show All ${matches.length} Matches
                                </button>
                            </div>
                        ` : ''}
                    ` : '<div class="no-matches">No matches found</div>'}
                </div>
            `;
        }).join('');

        if (totalPages > 1) {
            const hasMore = currentPage < totalPages;
            const hasPrevious = currentPage > 1;
            listDiv.innerHTML += `
                <div style="display: flex; justify-content: center; gap: 10px; margin-top: 18px; padding: 10px;">
                    ${hasPrevious ? `
                        <button class="btn" onclick="loadPreviousPage()" style="min-width: 120px;">
                            Previous
                        </button>
                    ` : ''}
                    <div style="display: flex; align-items: center; color: #718096; font-weight: 500;">
                        Page ${currentPage} of ${totalPages}
                    </div>
                    ${hasMore ? `
                        <button class="btn" onclick="loadNextPage()" style="min-width: 120px;">
                            Next
                        </button>
                    ` : ''}
                </div>
            `;
        }

        if (!lazyLoadObserver) initLazyLoading();
        const images = listDiv.querySelectorAll('img.lazy-load');
        images.forEach(img => lazyLoadObserver.observe(img));
        IconManager.reinit(50, document.getElementById('resultsSection'));
    } catch (error) {
        if (error && error.name === 'AbortError') return;
        console.error('[DISPLAY] Session mode render failed:', error);
        listDiv.innerHTML = `
            <div class="empty-state">
                <h3>Failed to load results</h3>
                <p>${escapeHtml(error.message || 'Unknown error')}</p>
            </div>
        `;
    }
}

// Initialize lazy loading for images
initLazyLoading();


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

async function loadMoreHistoricalProducts() {
    try {
        historicalProductsPage++;
        console.log(`[ADD_TO_EXISTING] Loading page ${historicalProductsPage} of historical products`);

        const response = await fetch(`/api/catalog/products?type=historical&page=${historicalProductsPage}&limit=50`);
        if (response.ok) {
            const data = await response.json();
            const newProducts = data.products.map(p => ({
                id: p.id,
                filename: p.filename,
                category: p.category,
                sku: p.sku,
                name: p.product_name,
                is_historical: true,
                hasFeatures: p.has_features
            }));

            // Append to existing products
            historicalProducts.push(...newProducts);
            console.log(`[ADD_TO_EXISTING] Loaded page ${historicalProductsPage}: ${newProducts.length} products, total now ${historicalProducts.length} `);

            // Update UI to show the new products
            showToast(`Loaded ${newProducts.length} more products`, 'info');
        } else {
            console.error('[ADD_TO_EXISTING] Failed to load more products');
            showToast('Failed to load more products', 'error');
        }
    } catch (error) {
        console.error('[ADD_TO_EXISTING] Error loading more products:', error);
        showToast('Error loading more products', 'error');
    }
}

function updateDynamicThresholdPreview(value) {
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return;

    const thresholdValueEl = document.getElementById('dynamicThresholdValue');
    if (thresholdValueEl) {
        thresholdValueEl.textContent = parsed + '%';
    }
}

function applyDynamicThreshold(value) {
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return;

    if (parsed === dynamicThreshold) {
        return;
    }

    dynamicThreshold = parsed;
    displayResults(true); // Reset to page 1 when filter changes
}

// Backward compatibility for any existing inline handlers
function updateDynamicThreshold(value) {
    updateDynamicThresholdPreview(value);
    applyDynamicThreshold(value);
}

function updateDynamicLimit(value) {
    dynamicLimit = parseInt(value);
    displayResults(true); // Reset to page 1 when filter changes
}

let dynamicSearch = '';
let dynamicSearchResults = new Map(); // Cache search results
let dynamicSearchAbortController = null;

async function updateDynamicSearch(value) {
    dynamicSearch = value.toLowerCase().trim();

    // Show spinner while typing
    const statusEl = document.getElementById('dynamicSearchStatus');
    if (!statusEl) return;

    if (isSessionServerMode()) {
        if (!dynamicSearch) {
            statusEl.innerHTML = '';
            displayResults(true);
            return;
        }

        statusEl.innerHTML = '<span class="search-spinner"></span><span style="font-size: 0.75rem;">SEARCHING...</span>';

        if (window.searchTimeout) {
            clearTimeout(window.searchTimeout);
        }

        window.searchTimeout = setTimeout(() => {
            statusEl.innerHTML = '<span class="search-count">server-filter</span>';
            displayResults(true);
        }, 300);
        return;
    }

    // Clear cache if search is empty
    if (!dynamicSearch) {
        if (dynamicSearchAbortController) {
            dynamicSearchAbortController.abort();
            dynamicSearchAbortController = null;
        }
        dynamicSearchResults.clear();
        statusEl.innerHTML = '';
        displayResults(true);
        return;
    }

    // Show searching spinner
    statusEl.innerHTML = '<span class="search-spinner"></span><span style="font-size: 0.75rem;">SEARCHING...</span>';

    // Debounce search - wait 300ms before searching
    if (window.searchTimeout) {
        clearTimeout(window.searchTimeout);
    }

    window.searchTimeout = setTimeout(async () => {
        const queryAtRequest = dynamicSearch;

        if (dynamicSearchAbortController) {
            dynamicSearchAbortController.abort();
        }
        dynamicSearchAbortController = new AbortController();

        try {
            // Call backend search API
            const response = await fetch(
                `/api/products/search?q=${encodeURIComponent(queryAtRequest)}&limit=5000`,
                { signal: dynamicSearchAbortController.signal }
            );
            if (!response.ok) throw new Error('Search failed: ' + response.status);
            const data = await response.json();

            // Ignore stale responses when query changed while request was in flight.
            if (queryAtRequest !== dynamicSearch) {
                return;
            }

            if (data.success && Array.isArray(data.results)) {
                // Build a map of product IDs for fast lookup
                dynamicSearchResults.clear();
                data.results.forEach(product => {
                    dynamicSearchResults.set(product.id, product);
                });
                console.log(`[SEARCH] Found ${data.results.length} products matching "${queryAtRequest}"`);

                // Update status to show count
                const count = data.results.length;
                statusEl.innerHTML = `<span class="search-count">${count} ${count === 1 ? 'match' : 'matches'}</span>`;
            }
        } catch (error) {
            if (error && error.name === 'AbortError') {
                return;
            }
            console.error('[SEARCH] Error:', error);
            statusEl.innerHTML = '<span style="color: #e53e3e; font-size: 0.75rem;">ERROR</span>';
        } finally {
            if (queryAtRequest === dynamicSearch) {
                displayResults(true);
            }
        }
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
        // OPTIMIZED FLOW: Check if we have product data locally in matchResults first
        let newData = null;
        let matchData = null;

        const mainResult = matchResults.find(r => r.p.id === newProductId);
        if (mainResult) {
            // We have the query product data
            newData = {
                status: 'success',
                product: {
                    id: mainResult.p.id,
                    product_name: mainResult.p.name,
                    sku: mainResult.p.sku,
                    category: mainResult.p.cat,
                    metadata: mainResult.p.meta,
                    image_path: mainResult.p.img || '',
                    filename: mainResult.p.fn || ''
                }
            };

            // Check if the matched product is also in our query list
            const matchedQueryResult = matchResults.find(r => r.p.id === matchedProductId);
            if (matchedQueryResult) {
                matchData = {
                    status: 'success',
                    product: {
                        id: matchedQueryResult.p.id,
                        product_name: matchedQueryResult.p.name,
                        sku: matchedQueryResult.p.sku,
                        category: matchedQueryResult.p.cat,
                        metadata: matchedQueryResult.p.meta,
                        image_path: matchedQueryResult.p.img || '',
                        filename: matchedQueryResult.p.fn || ''
                    }
                };
            } else {
                // Find it in the matches of the query product
                const compactMatch = mainResult.m.find(m => m.mid === matchedProductId);
                if (compactMatch) {
                    matchData = {
                        status: 'success',
                        product: {
                            id: compactMatch.mid,
                            product_name: compactMatch.name,
                            sku: compactMatch.sku,
                            category: compactMatch.cat,
                            metadata: compactMatch.mv || {},
                            image_path: compactMatch.img || '',
                            filename: compactMatch.fn || ''
                        }
                    };
                }
            }
        }

        // Fallback to fetch if not found locally (e.g. browsing historical catalog)
        if (!newData || !matchData) {
            console.log(`[COMPARISON] Data missing locally(New: ${!!newData}, Match: ${!!matchData}), fetching from API...`);
            const [newResp, matchResp] = await Promise.all([
                newData ? Promise.resolve({ ok: true, json: () => Promise.resolve(newData) }) : fetchWithRetry(`/api/products/${newProductId}`),
                matchData ? Promise.resolve({ ok: true, json: () => Promise.resolve(matchData) }) : fetchWithRetry(`/api/products/${matchedProductId}`)
            ]);

            if (!newResp.ok || !matchResp.ok) {
                throw new Error('Failed to load product details');
            }

            if (!newData) newData = await (typeof newResp.json === 'function' ? newResp.json() : newResp);
            if (!matchData) matchData = await (typeof matchResp.json === 'function' ? matchResp.json() : matchResp);
        } else {
            console.log(`[COMPARISON] Loaded both products from local cache`);
        }

        // Find the match details (using compact format)
        const matchResult = matchResults.find(r => r.p.id === newProductId);
        const compactMatch = matchResult?.m.find(m => m.mid === matchedProductId);

        // Expand compact match to full format for display
        let matchDetails = null;
        if (compactMatch) {
            matchDetails = {
                product_id: compactMatch.mid,
                similarity_score: compactMatch.s[0],
                color_score: compactMatch.s[1],
                shape_score: compactMatch.s[2],
                texture_score: compactMatch.s[3],
                category: compactMatch.cat,
                product_name: compactMatch.name,
                sku: compactMatch.sku,
                is_potential_duplicate: compactMatch.dup || false,
                // Copy any additional fields that might exist
                visual_score: compactMatch.vs,
                metadata_score: compactMatch.ms,
                sku_score: compactMatch.skus,
                name_score: compactMatch.ns,
                category_score: compactMatch.cs,
                price_score: compactMatch.ps,
                performance_score: compactMatch.pfs,
                metadata_values: compactMatch.mv,
                metadata_scores: compactMatch.mscores  // CRITICAL: Extract metadata_scores dict for dynamic fields
            };
        }

        if (matchDetails && !matchDetails.priceHistory && !matchDetails.performanceHistory) {
            try {
                const [priceResp, perfResp] = await Promise.all([
                    fetchWithRetry(`/api/products/${matchedProductId}/price-history`).catch(() => null),
                    fetchWithRetry(`/api/products/${matchedProductId}/performance-history`).catch(() => null)
                ]);

                if (priceResp?.ok) {
                    const priceData = await priceResp.json();
                    matchDetails.priceHistory = priceData.history || null;
                    matchDetails.priceStatistics = priceData.statistics || null;
                }

                if (perfResp?.ok) {
                    const perfData = await perfResp.json();
                    matchDetails.performanceHistory = perfData.history || null;
                    matchDetails.performanceStatistics = perfData.statistics || null;
                }
            } catch (error) {
                console.warn('Failed to fetch price/performance history:', error);
            }
        }

        const isMetadataMode = newMode === 'metadata' || newMode === 'hybrid';

        modalBody.innerHTML = `
            <h2>Detailed Comparison</h2>
            <div class="comparison-view">
                <div class="comparison-item">
                    <h3>New Product</h3>
                    ${!isMetadataMode ? `<img data-src="/api/products/${newProductId}/image" class="lazy-load"
                         src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect fill='%23e2e8f0' width='300' height='300'/></svg>"
                         alt="New Product">` : ''}
                    <div class="comparison-details">
                        <p><strong>Product:</strong> ${escapeHtml(newData.product.product_name || 'Unknown')}${newData.product.sku ? ` (${escapeHtml(newData.product.sku)})` : ''}</p>
                        ${(() => {
                // Always extract just the filename, not the full path
                let filename = 'N/A';
                if (newData.product.filename && newData.product.filename !== '[METADATA_ONLY]' && newData.product.filename !== '[METADATA ONLY]') {
                    filename = newData.product.filename.split(PATH_SEPARATOR_REGEX).pop();
                } else if (newData.product.image_path) {
                    filename = newData.product.image_path.split(PATH_SEPARATOR_REGEX).pop();
                }
                return (filename !== 'N/A' && filename !== '[METADATA_ONLY]' && filename !== '[METADATA ONLY]') ? `<p><strong>Filename:</strong> ${escapeHtml(filename)}</p>` : '';
            })()}
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
                        <p><strong>Product:</strong> ${escapeHtml(matchData.product.product_name || 'Unknown')}${matchData.product.sku ? ` (${escapeHtml(matchData.product.sku)})` : ''}</p>
                        ${(() => {
                // Always extract just the filename, not the full path
                let filename = 'N/A';
                if (matchData.product.filename && matchData.product.filename !== '[METADATA_ONLY]' && matchData.product.filename !== '[METADATA ONLY]') {
                    filename = matchData.product.filename.split(PATH_SEPARATOR_REGEX).pop();
                } else if (matchData.product.image_path) {
                    filename = matchData.product.image_path.split(PATH_SEPARATOR_REGEX).pop();
                }
                return (filename !== 'N/A' && filename !== '[METADATA_ONLY]' && filename !== '[METADATA ONLY]') ? `<p><strong>Filename:</strong> ${escapeHtml(filename)}</p>` : '';
            })()}
                        <p><strong>SKU:</strong> ${escapeHtml(matchData.product.sku || 'N/A')}</p>
                        <p><strong>Category:</strong> ${escapeHtml(matchData.product.category || 'Uncategorized')}</p>
                    </div>
                </div>
                </div>
            </div>
            ${(() => {
                // Safe metadata extraction
                let productMeta = matchData.product.metadata || {};
                if (typeof productMeta === 'string') {
                    try { productMeta = JSON.parse(productMeta); } catch (e) { console.error('Failed to parse metadata JSON', e); productMeta = {}; }
                }

                // Fallback for filename
                const newFilename = newData.product.filename || (newData.product.image_path ? newData.product.image_path.split(/[\\/]/).pop() : 'N/A');
                const matchFilename = matchData.product.filename || (matchData.product.image_path ? matchData.product.image_path.split(/[\\/]/).pop() : 'N/A');
                if (!matchDetails) matchDetails = { mv: productMeta };
                if (!matchDetails.mv) matchDetails.mv = productMeta;

                if (matchDetails.mv && matchDetails.mv.metadata && typeof matchDetails.mv.metadata === 'object') {
                    const nested = matchDetails.mv.metadata;
                    // Safely remove the wrapper key
                    delete matchDetails.mv.metadata;
                    // Merge nested fields to top level
                    Object.assign(matchDetails.mv, nested);
                }

                return ''; // This block just executed logic, returns nothing to render
            })()}
            
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
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                        ${matchDetails.sku_score !== undefined ? `
                                            <div style="margin-bottom: 4px;">
                                                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                                    <span style="font-size: 12px; color: #4a5568;">SKU Match</span>
                                                    <span style="font-size: 12px; font-weight: 600; color: #2d3748;">${matchDetails.sku_score.toFixed(1)}%</span>
                                                </div>
                                                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden;">
                                                    <div style="width: ${matchDetails.sku_score}%; height: 100%; background: #48bb78;"></div>
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${matchDetails.name_score !== undefined ? `
                                            <div style="margin-bottom: 4px;">
                                                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                                    <span style="font-size: 12px; color: #4a5568;">Name Match</span>
                                                    <span style="font-size: 12px; font-weight: 600; color: #2d3748;">${matchDetails.name_score.toFixed(1)}%</span>
                                                </div>
                                                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden;">
                                                    <div style="width: ${matchDetails.name_score}%; height: 100%; background: #48bb78;"></div>
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${/* Dynamic Metadata Scores Loop - PERFORMANCE OPTIMIZED */ ''}
                                        ${matchDetails.metadata_scores ? renderMetadataScoreBars(matchDetails.metadata_scores) : ''}
                                        ${matchDetails.category_score !== undefined ? `
                                            <div style="margin-bottom: 4px;">
                                                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                                    <span style="font-size: 12px; color: #4a5568;">Category Match</span>
                                                    <span style="font-size: 12px; font-weight: 600; color: #2d3748;">${matchDetails.category_score.toFixed(1)}%</span>
                                                </div>
                                                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden;">
                                                    <div style="width: ${matchDetails.category_score}%; height: 100%; background: #48bb78;"></div>
                                                </div>
                                            </div>
                                        ` : ''}
                                        <!-- Price and performance actual values shown in Full Metadata Comparison table below -->
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
            ` : ''
            }
            <!-- Full Metadata Detailed Comparison (3-Column Layout) -->
        ${(() => {
                // 1. Prepare New Product Metadata
                let newMeta = newData.product.metadata || {}; // FIX: newData.p -> newData.product
                if (typeof newMeta === 'string') {
                    try { newMeta = JSON.parse(newMeta); } catch (e) { console.error('Parsed newMeta error', e); newMeta = {}; }
                }
                // Flatten if nested (keys inside 'metadata' wrapper)
                if (newMeta.metadata && typeof newMeta.metadata === 'object') {
                    const nested = newMeta.metadata;
                    // Shallow copy to avoid mutation issues if ref is shared
                    newMeta = { ...newMeta, ...nested };
                    delete newMeta.metadata;
                }

                let matchedMeta = matchData.product ? (matchData.product.metadata || {}) : {};
                if (typeof matchedMeta === 'string') {
                    try { matchedMeta = JSON.parse(matchedMeta); } catch (e) { console.error('Parsed matchedMeta error', e); matchedMeta = {}; }
                }
                // Flatten if nested
                if (matchedMeta.metadata && typeof matchedMeta.metadata === 'object') {
                    const nested = matchedMeta.metadata;
                    matchedMeta = { ...matchedMeta, ...nested };
                    delete matchedMeta.metadata;
                }

                // If matchDetails.mv exists and has keys not in matchedMeta (e.g. from compact match), merge them in
                // (Optional, but good for robustness)
                if (matchDetails && matchDetails.mv) {
                    matchedMeta = { ...matchedMeta, ...matchDetails.mv };
                }

                // Match top-level keys if missing from metadata blob
                const CORE_KEYS = ['brand', 'sku', 'name', 'category', 'type', 'description', 'price', 'performance'];
                CORE_KEYS.forEach(ck => {
                    // Map frontend names to potential backend names or just use if available
                    let mappedKey = ck;
                    if (ck === 'name') mappedKey = 'product_name';

                    if (newMeta[ck] === undefined && newData.product[mappedKey] !== undefined) {
                        newMeta[ck] = newData.product[mappedKey];
                    }
                    if (matchedMeta[ck] === undefined && matchData.product[mappedKey] !== undefined) {
                        matchedMeta[ck] = matchData.product[mappedKey];
                    }
                });

                const newKeys = Object.keys(newMeta);
                const matchKeys = Object.keys(matchedMeta);

                // Union of all keys, sorted - prioritize CORE_KEYS at the top
                const allKeys = [...new Set([...CORE_KEYS.filter(k => newMeta[k] !== undefined || matchedMeta[k] !== undefined), ...newKeys, ...matchKeys])];

                // Remove duplicates, filter out redundant fields (already shown in header)
                const REDUNDANT_FIELDS = ['filename', 'image_path']; // Already shown in product card header
                const uniqueKeys = [...new Set(allKeys)].filter(k => !REDUNDANT_FIELDS.includes(k.toLowerCase()));

                return `
                <div style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #e2e8f0;">
                    <h5 style="margin-bottom: 12px; color: #2d3748;">Full Metadata Comparison</h5>
                    ${allKeys.length > 0 ? `
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        ${renderMetadataComparison(uniqueKeys, newMeta, matchedMeta)}
                    </div>
                    ` : `
                    <div style="padding: 20px; background: #f7fafc; border: 2px dashed #cbd5e0; border-radius: 4px; text-align: center; color: #718096;">
                        <p style="margin: 0; font-style: italic;">No metadata available for these products.</p>
                        <p style="margin: 8px 0 0 0; font-size: 12px;">Upload CSV files with product metadata to see detailed comparisons here.</p>
                    </div>
                    `}
                </div>
                `;
            })()
            } 

         `;

        modal.classList.add('show');

        // Initialize lazy loading for modal images
        initLazyLoading();
    } catch (error) {
        showToast('Failed to load comparison details', 'error');
    }
}

function closeModal() {
    const modal = document.getElementById('detailModal');
    modal.classList.remove('show');
    setTimeout(() => {
        if (!modal.classList.contains('show')) {
            document.getElementById('modalBody').innerHTML = '';
        }
    }, 300); // Match CSS transition duration
}

async function exportResults() {
    // In server mode, stream the CSV directly from the backend — no need to
    // pull every page into the browser first.
    if (isSessionServerMode()) {
        try {
            showToast('Generating CSV on server...', 'info');
            const params = buildServerSessionQueryParams(1, 999999);
            const url = `/api/products/match-results/session/export-csv?${params.toString()}`;

            if (window.pywebview) {
                // Desktop: fetch blob then save via pywebview
                const response = await fetch(url);
                if (!response.ok) throw new Error('Server returned ' + response.status);
                const csvText = await response.text();
                const filename = `match_results_${new Date().toISOString().slice(0, 10)}.csv`;
                const result = await window.pywebview.api.save_file_auto(csvText, filename);
                if (result) {
                    showToast(`Results saved: ${filename}`, 'success');
                } else {
                    showToast('Export cancelled', 'info');
                }
            } else {
                // Browser: trigger download via hidden link
                const a = document.createElement('a');
                a.href = url;
                a.download = '';
                document.body.appendChild(a);
                a.click();
                a.remove();
                showToast('CSV download started', 'success');
            }
            return;
        } catch (error) {
            console.error('Server-side CSV export failed, falling back to client-side:', error);
            showToast('Server export failed — trying client-side export...', 'warning');
            // Fall through to client-side export below
        }
    }

    let exportResultsSource = matchResults;

    // Early return if no results
    if (!Array.isArray(exportResultsSource) || exportResultsSource.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }

    // ENHANCEMENT: Build dynamic headers from metadata scores
    const allMetadataKeys = new Set();
    for (let i = 0; i < exportResultsSource.length; i++) {
        const result = exportResultsSource[i];
        result.m.forEach(match => {
            // CRITICAL FIX: Use mscores (compact format) with fallback
            const scores = match.metadata_scores || match.mscores;
            if (scores) {
                Object.keys(scores).forEach(key => allMetadataKeys.add(key));
            }
        });

        if (i > 0 && i % 500 === 0) {
            await yieldToUi();
        }
    }

    const metadataKeysArray = Array.from(allMetadataKeys).sort();

    // Build header row
    let headerRow = ['New Product', 'Category', 'SKU', 'Total Matches', 'Avg Similarity', 'Median Score', 'Best Score', 'Top Match Score'];

    // Add average metadata score headers
    metadataKeysArray.forEach(key => {
        headerRow.push(`Avg ${key.charAt(0).toUpperCase() + key.slice(1)} `);
    });

    // Add top match headers
    headerRow.push('Top Match Name', 'Top Match Overall Score');
    metadataKeysArray.forEach(key => {
        headerRow.push(`Top Match ${key} `);
    });

    // PERFORMANCE FIX #2: Use array.push() instead of string concatenation to avoid O(n²) complexity
    const csvRows = [headerRow.map(h => `"${h}"`).join(',')];

    for (let i = 0; i < exportResultsSource.length; i++) {
        const result = exportResultsSource[i];
        const product = result.p;
        const matches = result.m;
        const topMatch = matches[0];

        // PERFORMANCE: Use cached metadata statistics (avoids recalculation)
        const stats = getCachedMetadataStats(result);

        let row = [
            product.name || '',
            product.cat || 'Uncategorized',
            product.sku || '',
            matches.length,
            stats ? stats.overallAvg : 0,
            stats ? stats.medianScore : 0,
            stats ? stats.bestScore : 0,
            topMatch ? getScore(topMatch, 'similarity').toFixed(1) : 0
        ];

        // Add average metadata scores for this product
        metadataKeysArray.forEach(key => {
            const avgScore = stats && stats.metadataStats[key]
                ? stats.metadataStats[key].avg
                : '';
            row.push(avgScore);
        });

        // Add top match info
        if (topMatch) {
            row.push(topMatch.name || 'Unknown');
            row.push(getScore(topMatch, 'similarity').toFixed(1));

            // Add top match metadata scores
            // CRITICAL FIX: Use mscores (compact format) with fallback
            const topMatchScores = topMatch.metadata_scores || topMatch.mscores;
            metadataKeysArray.forEach(key => {
                const score = topMatchScores?.[key] || '';
                row.push(score ? score.toFixed(1) : '');
            });
        } else {
            row.push('No matches');
            row.push(0);
            metadataKeysArray.forEach(() => row.push(''));
        }

        const rowString = row.map(cell => {
            if (cell === null || cell === undefined) return '';
            const cellStr = String(cell);
            if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
                return `"${cellStr.replace(/"/g, '""')}"`;
            }
            return cellStr;
        }).join(',');

        csvRows.push(rowString);

        if (i > 0 && i % 500 === 0) {
            await yieldToUi();
        }
    }

    const filename = `match_results_${new Date().toISOString().slice(0, 10)}.csv`;
    if (window.pywebview) {
        try {
            const csv = csvRows.join('\n') + '\n';
            const result = await window.pywebview.api.save_file_auto(csv, filename);
            if (result) {
                showToast(`Results saved to Downloads folder: ${filename}`, 'success');
            } else {
                showToast('Export failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Export failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob(csvRows.map(row => `${row}\n`), { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        blobUrls.add(url);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('Results exported to CSV', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showToast('Export failed', 'error');
        } finally {
            setTimeout(() => {
                URL.revokeObjectURL(url);
                blobUrls.delete(url);
            }, 100);
        }
    }
}

async function resetApp() {
    if (confirm('Start over? This will clear all data and take you back to the upload step.')) {
        // Show loading message
        showToast('Resetting app...', 'info');

        // Clear UI state BEFORE cleanup to ensure visibility changes
        try {
            // Hide results and matching sections
            const resultsSection = document.getElementById('resultsSection');
            const matchingSection = document.getElementById('matchingSection');
            if (resultsSection) resultsSection.style.display = 'none';
            if (matchingSection) matchingSection.style.display = 'none';

            // Clear file info
            const historicalInfo = document.getElementById('historicalInfo');
            const newInfo = document.getElementById('newInfo');
            if (historicalInfo) {
                historicalInfo.innerHTML = '';
                historicalInfo.classList.remove('show');
            }
            if (newInfo) {
                newInfo.innerHTML = '';
                newInfo.classList.remove('show');
            }

            // Hide template download buttons
            const historicalTemplateBtn = document.getElementById('downloadHistoricalTemplateBtn');
            const newTemplateBtn = document.getElementById('downloadNewTemplateBtn');
            if (historicalTemplateBtn) historicalTemplateBtn.style.display = 'none';
            if (newTemplateBtn) newTemplateBtn.style.display = 'none';

            // Hide status messages
            const historicalStatus = document.getElementById('historicalStatus');
            const newStatus = document.getElementById('newStatus');
            if (historicalStatus) {
                historicalStatus.innerHTML = '';
                historicalStatus.classList.remove('show');
            }
            if (newStatus) {
                newStatus.innerHTML = '';
                newStatus.classList.remove('show');
            }

            // Reset all buttons to initial state
            const processBtn = document.getElementById('processHistoricalBtn');
            const processNewBtn = document.getElementById('processNewBtn');
            const resetBtn = document.getElementById('resetBtn');
            if (processBtn) {
                processBtn.disabled = true;
                processBtn.textContent = 'PROCESS';
            }
            if (processNewBtn) {
                processNewBtn.disabled = true;
                processNewBtn.textContent = 'PROCESS';
            }
            if (resetBtn) resetBtn.style.display = 'none';

            // Reset file input values
            const historicalInput = document.getElementById('historicalInput');
            const newInput = document.getElementById('newInput');
            const historicalCsvInput = document.getElementById('historicalCsvInput');
            const newCsvInput = document.getElementById('newCsvInput');
            if (historicalInput) historicalInput.value = '';
            if (newInput) newInput.value = '';
            if (historicalCsvInput) historicalCsvInput.value = '';
            if (newCsvInput) newCsvInput.value = '';

            // Reset file labels
            const historicalFileLabel = document.getElementById('historicalFileLabel');
            const newFileLabel = document.getElementById('newFileLabel');
            if (historicalFileLabel) historicalFileLabel.textContent = 'CSV optional - Use BUILD CSV for easy setup';
            if (newFileLabel) newFileLabel.textContent = 'CSV optional - Use BUILD CSV for easy setup';

        } catch (error) {
            console.error('Error clearing UI state:', error);
        }

        // Clean up memory
        cleanupMemory();

        // Clear saved state (webview only)
        await clearSavedState();

        // Small delay to ensure cleanup completes
        setTimeout(() => {
            showToast('Ready for new upload!', 'success');
            location.reload();
        }, 100);
    }
}

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
            // (Validation removed to support dynamic metadata schemas)

            // Use Web Worker for parallel CSV parsing (non-blocking)
            console.log('[CSV-PARSER] Starting Web Worker for CSV parsing');

            if (typeof (Worker) !== 'undefined') {
                // Web Workers supported - use parallel parsing
                const worker = new Worker('/static/csv-parser-worker.js');

                worker.onmessage = function (event) {
                    const result = event.data;

                    if (result.success) {
                        console.log(`[CSV-PARSER] ✓ Web Worker parsed ${result.lineCount} lines`);

                        // Save detected schema if available (for dynamic sliders)
                        if (result.detectedColumns && result.detectedColumns.length > 0) {
                            const columns = result.detectedColumns.map(col => ({
                                column_name: col,
                                display_name: col.charAt(0).toUpperCase() + col.slice(1),
                                data_type: (col === 'price' || col === 'performance') ? 'numeric' : 'string'
                            }));

                            console.log('[CSV-PARSER] Saving detected schema:', columns);
                            // Wait for schema save to ensure sliders can load it
                            saveMetadataSchema(columns).then(() => {
                                resolve(result.map);
                            });
                        } else {
                            resolve(result.map);
                        }
                    } else {
                        console.error('[CSV-PARSER] Web Worker error:', result.error);
                        showToast('CSV parsing error: ' + result.error, 'error');
                        resolve({});
                    }

                    worker.terminate();
                };

                worker.onerror = function (error) {
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

// PERFORMANCE OPTIMIZED: String replacement is 50-100x faster than DOM-based approach
const HTML_ESCAPE_MAP = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;'
};
const HTML_ESCAPE_REGEX = /[&<>"'\/]/g;

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(HTML_ESCAPE_REGEX, char => HTML_ESCAPE_MAP[char]);
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
                    // Add loading spinner overlay
                    img.classList.add('image-loading');

                    // Check if it's an API endpoint (needs blob URL for tracking)
                    if (src.startsWith('/api/products/')) {
                        // Create tracked blob URL to prevent memory leaks
                        createTrackedBlobUrl(src)
                            .then(blobUrl => {
                                img.src = blobUrl;
                                img.removeAttribute('data-src');
                                // Remove loading state when image loads - use addEventListener with {once:true} to prevent memory leak
                                img.addEventListener('load', () => img.classList.remove('image-loading'), { once: true });
                                img.addEventListener('error', () => img.classList.remove('image-loading'), { once: true });
                            })
                            .catch(error => {
                                console.error('Failed to load image:', error);
                                // Fallback to direct URL
                                img.src = src;
                                img.removeAttribute('data-src');
                                img.classList.remove('image-loading');
                            });
                    } else {
                        // For non-API images, load directly
                        img.src = src;
                        img.removeAttribute('data-src');
                        // Remove loading state when image loads - use addEventListener with {once:true} to prevent memory leak
                        img.addEventListener('load', () => img.classList.remove('image-loading'), { once: true });
                        img.addEventListener('error', () => img.classList.remove('image-loading'), { once: true });
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
runWhenDomReadyResults(() => {
    initLazyLoading();

    // MEMORY OPTIMIZATION: Periodically cleanup old blob URLs (prevents 10-30MB accumulation)
    // Track blob URL timestamps for age-based cleanup
    const blobUrlTimestamps = new Map();

    // Intercept blob URL creation to track timestamps
    const originalCreateObjectURL = URL.createObjectURL;
    URL.createObjectURL = function(blob) {
        const url = originalCreateObjectURL.call(URL, blob);
        blobUrlTimestamps.set(url, Date.now());
        blobUrls.add(url);
        return url;
    };

    // Cleanup interval: Revoke blob URLs older than 5 minutes
    blobUrlCleanupInterval = setInterval(() => {
        // PERFORMANCE FIX #7: Skip cleanup if no URLs to clean
        if (blobUrlTimestamps.size === 0) return;

        const now = Date.now();
        const fiveMinutes = 5 * 60 * 1000;
        const urlsToRevoke = []; // PERFORMANCE FIX #7: Batch deletions

        // PERFORMANCE FIX #7: Collect URLs to delete first, then delete (avoid modifying Map during iteration)
        for (const [url, timestamp] of blobUrlTimestamps) {
            if (now - timestamp > fiveMinutes) {
                urlsToRevoke.push(url);
            }
        }

        // PERFORMANCE FIX #7: Batch revoke and delete
        if (urlsToRevoke.length > 0) {
            urlsToRevoke.forEach(url => {
                URL.revokeObjectURL(url);
                blobUrls.delete(url);
                blobUrlTimestamps.delete(url);
            });
            console.log(`[BLOB-CLEANUP] Revoked ${urlsToRevoke.length} expired blob URLs`);
        }
    }, 60000); // Run every minute
});

// Retry Logic with Exponential Backoff
async function fetchWithRetry(url, options = {}, retryCount = 0) {
    const requestOptions = { ...(options || {}) };
    const timeoutMs = Number(requestOptions.timeoutMs || 0);
    delete requestOptions.timeoutMs;

    let timeoutId = null;
    let timeoutController = null;
    let didTimeout = false;

    // Optional per-request timeout (safe fallback: no timeout when not provided).
    if (timeoutMs > 0 && !requestOptions.signal) {
        timeoutController = new AbortController();
        requestOptions.signal = timeoutController.signal;
        timeoutId = setTimeout(() => {
            didTimeout = true;
            timeoutController.abort();
        }, timeoutMs);
    }

    try {
        const response = await fetch(url, requestOptions);

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
        // Preserve caller-triggered aborts (for explicit cancellation flows).
        if (error?.name === 'AbortError' && !didTimeout) {
            throw error;
        }

        // Network error - retry
        if (retryCount < RETRY_CONFIG.maxRetries) {
            const delay = Math.min(
                RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, retryCount),
                RETRY_CONFIG.maxDelay
            );

            const isTimeout = didTimeout || error?.name === 'AbortError';
            const label = isTimeout ? 'Request timed out' : 'Network error';
            showToast(`${label}. Retrying in ${delay / 1000} seconds... (Attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries})`, 'warning');

            await sleep(delay);
            return fetchWithRetry(url, options, retryCount + 1);
        }

        throw error;
    } finally {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

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

function showCsvHelp(type) {
    const modal = document.getElementById('csvHelpModal');
    modal.classList.add('show');
    // Re-initialize icons in modal only (scoped for performance)
    IconManager.reinit(50, modal);
}

function closeCsvHelp() {
    const modal = document.getElementById('csvHelpModal');
    modal.classList.remove('show');
}

async function downloadSampleCsv() {
    const csv = `filename,category,sku,name,price,price_history,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160;2024-03-15:200:1800:11.1:2400
product2.jpg,dinnerware,DW-002,White Plate Set,45.00,2024-01-15:45.00;2024-02-15:42.50;2024-03-15:44.00,2024-01-15:200:2000:10.0:9000;2024-02-15:220:2200:10.0:9900;2024-03-15:240:2400:10.0:10800
product3.jpg,textiles,TX-003,Cotton Napkins,15.99,15.99;16.50;15.75,100:800:12.5:1200;120:900:13.3:1440;110:850:12.9:1320
product4.jpg,placemats,PM-004,Red Placemat,32.00,,
product5.jpg,dinnerware,DW-005,Ceramic Bowl,22.50,2024-01-15:22.50;2024-02-15:23.00,80:600:13.3:960;90:650:13.8:1080`;

    const filename = 'sample_product_data.csv';

    if (window.pywebview) {
        try {
            const result = await window.pywebview.api.save_file_auto(csv, filename);
            if (result) {
                showToast(`Sample CSV saved to Downloads folder: ${filename}`, 'success');
            } else {
                showToast('Download failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Download failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);

        // MEMORY OPTIMIZATION: Track blob URL for cleanup (1-10MB per failure)
        blobUrls.add(url);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('Sample CSV downloaded! Open it in Excel or any text editor.', 'success');
        } catch (error) {
            console.error('Download failed:', error);
            showToast('Download failed', 'error');
        } finally {
            setTimeout(() => {
                URL.revokeObjectURL(url);
                blobUrls.delete(url);
            }, 100);
        }
    }
}

async function downloadExistingCsv(section) {
    try {
        showToast(`Downloading ${section} catalog CSV...`, 'info');

        // Call API to extract CSV from current database
        const response = await fetch(`/api/csv/extract?type=${section}`);

        if (!response.ok) {
            const error = await response.json();
            showToast(`Failed to download: ${error.message || 'Unknown error'}`, 'error');
            return;
        }

        // Get the CSV filename from response headers
        const contentDisposition = response.headers.get('content-disposition');
        let filename = `${section}-products.csv`;
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="?([^"]+)"?/);
            if (match) filename = match[1];
        }

        // Convert response to blob
        const blob = await response.blob();

        // Check if running in pywebview
        if (window.pywebview) {
            try {
                // For pywebview, read blob as text and save
                const text = await blob.text();
                const result = await window.pywebview.api.save_file_auto(text, filename);
                if (result) {
                    showToast(`${section} CSV saved! Review it and combine with your new data if needed.`, 'success');
                } else {
                    showToast('Download failed', 'error');
                }
            } catch (error) {
                console.error('Webview save failed:', error);
                showToast('Download failed - ' + error.message, 'error');
            }
        } else {
            // Browser fallback
            const url = URL.createObjectURL(blob);
            blobUrls.add(url);

            try {
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                showToast(`${section} CSV downloaded! Review it and combine with your new data if needed.`, 'success');
            } catch (error) {
                console.error('Download failed:', error);
                showToast('Download failed', 'error');
            } finally {
                setTimeout(() => {
                    URL.revokeObjectURL(url);
                    blobUrls.delete(url);
                }, 100);
            }
        }
    } catch (error) {
        console.error('Error downloading existing CSV:', error);
        showToast(`Error: ${error.message}`, 'error');
    }
}

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

    // PERFORMANCE FIX #10: Compute min/max iteratively to avoid spread operator stack overflow
    let max = -Infinity;
    let min = Infinity;
    prices.forEach(price => {
        if (price > max) max = price;
        if (price < min) min = price;
    });

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

// Advanced Features Implementation

// Toggle Advanced Settings
function toggleAdvancedSettings() {
    const panel = document.getElementById('advancedSettings');
    const btn = document.getElementById('advancedSettingsBtn');

    if (panel.style.display === 'none') {
        // Detect mode and show appropriate weight section
        detectAndShowWeightSection();
        panel.style.display = 'block';
        btn.textContent = 'Hide Advanced Settings';
    } else {
        panel.style.display = 'none';
        btn.textContent = 'Advanced Settings';
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

async function loadMetadataSchema() {
    try {
        const response = await fetch('/api/metadata-schema');
        if (!response.ok) {
            console.warn('Failed to load metadata schema:', response.statusText);
            return;
        }
        const data = await response.json();
        metadataSchema = data.schema || [];
        window.metadataSchema = metadataSchema; // Explicitly expose to window for other functions
        console.log('[METADATA] Loaded schema:', metadataSchema);

        // Initialize equal weights for all columns
        initializeEqualWeights();

        // Render sliders in both Mode 2 and Mode 3 containers
        renderDynamicWeightSliders('dynamicMetadataWeightsContainer', 'mode2');
        renderDynamicWeightSliders('dynamicHybridMetadataWeightsContainer', 'hybrid');
    } catch (error) {
        console.error('Error loading metadata schema:', error);
    }
}

/**
 * Save metadata schema to backend (after parsing CSV headers)
 * @param {Array} columns - Array of {column_name, data_type, display_name}
 */
async function saveMetadataSchema(columns) {
    try {
        const response = await fetch('/api/metadata-schema', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ columns: columns, clear_existing: true })
        });
        if (response.ok) {
            console.log('[METADATA] Schema saved successfully');
            // Reload the schema to populate sliders
            await loadMetadataSchema();
        }
    } catch (error) {
        console.error('Error saving metadata schema:', error);
    }
}

/**
 * Initialize equal weights for all columns in schema
 */
function initializeEqualWeights() {
    if (metadataSchema.length === 0) return;

    const equalWeight = Math.round(100 / metadataSchema.length);
    metadataWeights = {};
    metadataSchema.forEach((col, idx) => {
        // Last column gets remaining weight to ensure sum = 100
        if (idx === metadataSchema.length - 1) {
            metadataWeights[col.column_name] = 100 - (equalWeight * (metadataSchema.length - 1));
        } else {
            metadataWeights[col.column_name] = equalWeight;
        }
    });
    console.log('[METADATA] Initialized equal weights:', metadataWeights);
}

/**
 * Render dynamic weight sliders in a container
 * @param {string} containerId - DOM container ID
 * @param {string} prefix - Prefix for element IDs ('mode2' or 'hybrid')
 */
function renderDynamicWeightSliders(containerId, prefix) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (metadataSchema.length === 0) {
        container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #666; font-style: italic;">
                Upload a CSV to see available columns for weighting.
            </div>
        `;
        return;
    }

    let html = '<div style="display: flex; flex-direction: column; gap: 10px;">';

    metadataSchema.forEach(col => {
        const colName = col.column_name;
        const displayName = col.display_name || colName.toUpperCase();
        const dataType = col.data_type || 'string';
        const weight = metadataWeights[colName] || 0;
        const typeIcon = dataType === 'numeric' ? '#' : 'Aa';

        html += `
            <div style="display: flex; align-items: center; gap: 10px; padding: 8px; background: white; border: 1px solid #ddd;">
                <span style="width: 30px; text-align: center; font-size: 11px; color: #888; background: #f0f0f0; padding: 2px 4px; border-radius: 3px;"
                      title="${dataType === 'numeric' ? 'Numeric column' : 'Text column'}">${typeIcon}</span>
                <span style="flex: 1; font-weight: bold; font-size: 13px;">${displayName}</span>
                <input type="range" id="${prefix}_weight_${colName}"
                       min="0" max="100" value="${weight}"
                       style="width: 100px;"
                       oninput="updateDynamicWeight('${colName}', this.value, '${prefix}')">
                <span id="${prefix}_weight_val_${colName}" style="width: 40px; text-align: right; font-weight: bold;">${weight}%</span>
            </div>
        `;
    });

    html += '</div>';
    container.innerHTML = html;

    // Update totals
    updateWeightsTotal(prefix);
}

/**
 * Update a dynamic weight value
 */
function updateDynamicWeight(columnName, value, prefix) {
    metadataWeights[columnName] = parseInt(value);

    // Update display value
    const valSpan = document.getElementById(`${prefix}_weight_val_${columnName}`);
    if (valSpan) valSpan.textContent = value + '%';

    // Sync sliders between Mode 2 and Hybrid containers
    const otherPrefix = prefix === 'mode2' ? 'hybrid' : 'mode2';
    const otherSlider = document.getElementById(`${otherPrefix}_weight_${columnName}`);
    const otherValSpan = document.getElementById(`${otherPrefix}_weight_val_${columnName}`);
    if (otherSlider) otherSlider.value = value;
    if (otherValSpan) otherValSpan.textContent = value + '%';

    // Update totals
    updateWeightsTotal('mode2');
    updateWeightsTotal('hybrid');
}

/**
 * Update the total weight display
 */
function updateWeightsTotal(prefix) {
    const total = Object.values(metadataWeights).reduce((sum, w) => sum + w, 0);

    const totalSpan = prefix === 'mode2'
        ? document.getElementById('metadataWeightsTotal')
        : document.getElementById('hybridMetadataWeightsTotal');

    if (totalSpan) {
        totalSpan.textContent = total + '%';
        // Visual feedback for valid/invalid total
        totalSpan.style.color = (total === 100) ? '#4caf50' : '#f44336';
    }
}

/**
 * Equalize all metadata weights (Mode 2)
 */
function equalizeMetadataWeights() {
    initializeEqualWeights();
    renderDynamicWeightSliders('dynamicMetadataWeightsContainer', 'mode2');
    renderDynamicWeightSliders('dynamicHybridMetadataWeightsContainer', 'hybrid');
    console.log('[METADATA] Weights equalized');
}

/**
 * Equalize all metadata weights (Hybrid mode)
 */
function equalizeHybridMetadataWeights() {
    equalizeMetadataWeights(); // Same logic
}

/**
 * Get normalized metadata weights dict for API (sums to 1.0)
 */
function getNormalizedMetadataWeights() {
    const total = Object.values(metadataWeights).reduce((sum, w) => sum + w, 0);
    if (total === 0) return {};

    const normalized = {};
    Object.entries(metadataWeights).forEach(([col, weight]) => {
        normalized[col] = weight / total;
    });
    return normalized;
}

// Deprecated - metadata weights now handled dynamically
function updateMetadataWeights() {
    // Now handled by updateDynamicWeight() for flexible metadata columns
}

// Update Hybrid Weights - Single slider balances between visual and metadata
function updateHybridWeights() {
    const visualWeight = parseInt(document.getElementById('hybridBalanceSlider').value);
    const metadataWeight = 100 - visualWeight;

    document.getElementById('hybridVisualWeightValue').textContent = visualWeight;
    document.getElementById('hybridMetadataWeightValue').textContent = metadataWeight;
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


function updateHybridMetadataSubWeights() {
    // Dynamic metadata weights are managed via renderDynamicWeightSliders()
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
        // Mode 2 (Metadata) - reset to equal weights
        equalizeMetadataWeights();
    }

    if (hybridSection.style.display !== 'none') {
        // Reset Mode 3 (Hybrid) - 60% Visual, 40% Metadata
        document.getElementById('hybridBalanceSlider').value = 60;
        updateHybridWeights();

        // Reset visual sub-weights
        document.getElementById('hybridColorWeightSlider').value = 50;
        document.getElementById('hybridShapeWeightSlider').value = 30;
        document.getElementById('hybridTextureWeightSlider').value = 20;
        updateHybridVisualSubWeights();

        // Reset metadata sub-weights to equal
        equalizeHybridMetadataWeights();
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


    safeAddListener('colorWeightSlider', 'input', updateWeights);
    safeAddListener('shapeWeightSlider', 'input', updateWeights);
    safeAddListener('textureWeightSlider', 'input', updateWeights);
    safeAddListener('hybridBalanceSlider', 'input', updateHybridWeights);
    safeAddListener('hybridColorWeightSlider', 'input', updateHybridVisualSubWeights);
    safeAddListener('hybridShapeWeightSlider', 'input', updateHybridVisualSubWeights);
    safeAddListener('hybridTextureWeightSlider', 'input', updateHybridVisualSubWeights);
    safeAddListener('exportCsvBtn', 'click', exportResults);
    safeAddListener('exportWithImagesBtn', 'click', exportWithImages);
    safeAddListener('duplicateReportBtn', 'click', showDuplicateReport);
    safeAddListener('saveSessionBtn', 'click', saveSession);
    safeAddListener('loadSessionBtn', 'click', loadSession);
    safeAddListener('searchInput', 'input', applyFilters);
    safeAddListener('categoryFilter', 'change', applyFilters);
    safeAddListener('sortBySelect', 'change', applyFilters);
    safeAddListener('duplicatesOnlyCheckbox', 'change', applyFilters);
    safeAddListener('clearFiltersBtn', 'click', clearFilters);
}

// Export with Images
async function exportWithImages() {
    if (typeof JSZip === 'undefined') {
        showToast('JSZip library not loaded. Please refresh the page.', 'error');
        return;
    }

    const MAX_IMAGE_EXPORT_PRODUCTS = 2000;
    let exportResultsSource = matchResults;

    if (isSessionServerMode()) {
        try {
            showToast('Loading filtered results from server for image export...', 'info');
            const sessionData = await fetchAllServerSessionResults({
                actionLabel: 'Preparing image export',
                pageSize: 250,
                maxResults: MAX_IMAGE_EXPORT_PRODUCTS,
                showProgress: true
            });
            exportResultsSource = sessionData.results || [];
            if (sessionData.truncated) {
                showToast(
                    `Image export limited to first ${MAX_IMAGE_EXPORT_PRODUCTS.toLocaleString()} filtered products. Refine filters and export in parts for the full set.`,
                    'warning'
                );
            } else {
                showToast('Preparing export with images... This may take a few minutes.', 'info');
            }
        } catch (error) {
            console.error('Failed to load server results for image export:', error);
            showToast('Export failed - unable to load full filtered results', 'error');
            return;
        }
    } else {
        if (!Array.isArray(matchResults) || matchResults.length === 0) {
            showToast('No results to export', 'warning');
            return;
        }

        const chunkInfo = getChunkInfo();
        if (chunkInfo.totalResults > CHUNK_SIZE) {
            exportResultsSource = matchResults.slice(chunkInfo.startIdx, chunkInfo.endIdx);
            showToast(
                `Large dataset detected. Exporting current chunk only (${chunkInfo.startIdx.toLocaleString()}-${chunkInfo.endIdx.toLocaleString()}).`,
                'info'
            );
        } else {
            showToast('Preparing export with images... This may take a few minutes.', 'info');
        }

        if (exportResultsSource.length > MAX_IMAGE_EXPORT_PRODUCTS) {
            showToast(
                `Image export limited to first ${MAX_IMAGE_EXPORT_PRODUCTS.toLocaleString()} products. Use filters/chunks and export in parts.`,
                'warning'
            );
            exportResultsSource = exportResultsSource.slice(0, MAX_IMAGE_EXPORT_PRODUCTS);
        }
    }

    if (!Array.isArray(exportResultsSource) || exportResultsSource.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }

    try {
        const zip = new JSZip();
        const MAX_MATCHES_PER_PRODUCT = 5;  // Limit to top 5 matches to keep ZIP manageable
        const PRODUCT_BATCH_SIZE = 8;
        const MATCH_IMAGE_FETCH_CONCURRENCY = 3;
        let processedCount = 0;
        const totalProducts = exportResultsSource.length;
        const progressToastEvery = Math.max(20, Math.floor(totalProducts / 20));
        const imageBlobCache = new Map();

        const fetchImageBlobCached = async (productId) => {
            if (!productId) return null;
            if (imageBlobCache.has(productId)) {
                return imageBlobCache.get(productId);
            }

            const fetchPromise = (async () => {
                try {
                    const response = await fetch(`/api/products/${productId}/image`);
                    if (!response.ok) return null;

                    const blob = await response.blob();
                    const mimeType = (blob.type || '').toLowerCase();
                    const ext = mimeType.includes('/') ? mimeType.split('/')[1].split(';')[0] : 'jpg';
                    return { blob, ext: ext || 'jpg' };
                } catch (error) {
                    console.warn(`Failed to fetch image for product ${productId}:`, error);
                    return null;
                }
            })();

            imageBlobCache.set(productId, fetchPromise);
            return fetchPromise;
        };

        const mapWithConcurrency = async (items, concurrency, iterator) => {
            if (!Array.isArray(items) || items.length === 0) return [];
            const results = new Array(items.length);
            let nextIndex = 0;

            const workerCount = Math.min(Math.max(1, concurrency), items.length);
            const workers = Array.from({ length: workerCount }, async () => {
                while (true) {
                    const currentIndex = nextIndex++;
                    if (currentIndex >= items.length) return;
                    results[currentIndex] = await iterator(items[currentIndex], currentIndex);
                }
            });

            await Promise.all(workers);
            return results;
        };

        // Detect all metadata keys for comprehensive export
        const allMetadataKeys = new Set();
        exportResultsSource.forEach(result => {
            result.m.forEach(match => {
                // CRITICAL FIX: Use mscores (compact format) with fallback
                const scores = match.metadata_scores || match.mscores;
                if (scores) {
                    Object.keys(scores).forEach(key => allMetadataKeys.add(key));
                }
            });
        });
        const metadataKeysArray = Array.from(allMetadataKeys).sort();

        // Build CSV for results summary
        let csvRows = [];
        const csvHeader = ['New Product', 'Product ID', 'Category', 'SKU', 'Total Matches', 'Avg Similarity',
                          'Best Score', 'Top Match', 'Top Match Score', 'Image Path'];
        csvRows.push(csvHeader.map(h => `"${h}"`).join(','));
        const thresholdEl = document.getElementById('thresholdSlider');
        const thresholdValue = thresholdEl ? parseInt(thresholdEl.value, 10) : 0;

        // Prepare detailed JSON export data
        let exportData = {
            timestamp: new Date().toISOString(),
            mode: newMode,
            weights: similarityWeights,
            metadata_weights: newMode !== 'visual' ? metadataWeights : {},
            threshold: Number.isFinite(thresholdValue) ? thresholdValue : 0,
            metadata_fields: metadataKeysArray,
            results: []
        };

        for (let batchStart = 0; batchStart < exportResultsSource.length; batchStart += PRODUCT_BATCH_SIZE) {
            const batch = exportResultsSource.slice(batchStart, batchStart + PRODUCT_BATCH_SIZE);

            // RACE CONDITION FIX: Process batch in parallel, return data for deterministic ordering
            const batchResults = await Promise.all(batch.map(async (result) => {
                const product = result.p;
                const matches = result.m;
                const topMatches = matches.slice(0, MAX_MATCHES_PER_PRODUCT);
                const matchImagePaths = new Array(topMatches.length).fill('');

                processedCount++;
                if (processedCount === totalProducts || processedCount % progressToastEvery === 0) {
                    showToast(`Processing ${processedCount}/${totalProducts} products...`, 'info');
                }

                const metadataStats = getCachedMetadataStats(result);

                // Sanitize product name for folder naming
                const sanitizedName = (product.name || `product_${product.id}`).replace(/[<>:"/\\|?*]/g, '_');
                const productFolder = `products/${sanitizedName}_${product.id}`;
                const matchesFolder = `${productFolder}/matches`;
                let productImagePath = '';

                // Fetch and add new product image
                const productImgData = await fetchImageBlobCached(product.id);
                if (productImgData) {
                    productImagePath = `${productFolder}/new_product.${productImgData.ext}`;
                    zip.file(productImagePath, productImgData.blob);
                }

                // Fetch match images with bounded concurrency for lower memory/network spikes.
                const matchImageResults = await mapWithConcurrency(
                    topMatches,
                    MATCH_IMAGE_FETCH_CONCURRENCY,
                    async (match, i) => {
                        const matchImgData = await fetchImageBlobCached(match.mid);
                        if (!matchImgData) return null;

                        const matchName = (match.name || `match_${match.mid}`).replace(/[<>:"/\\|?*]/g, '_');
                        const similarity = getScore(match, 'similarity').toFixed(1);
                        const filename = `${matchesFolder}/${i + 1}_${matchName}_${similarity}pct.${matchImgData.ext}`;
                        matchImagePaths[i] = filename;

                        return {
                            index: i,
                            blob: matchImgData.blob,
                            filename: filename
                        };
                    }
                );

                // Add images to zip in deterministic order (prevents race conditions)
                matchImageResults.forEach(result => {
                    if (result) {
                        zip.file(result.filename, result.blob);
                    }
                });

                // Build CSV row (don't push yet - race condition)
                const topMatch = matches[0];
                const csvRow = [
                    product.name || '',
                    product.id,
                    product.cat || 'Uncategorized',
                    product.sku || '',
                    matches.length,
                    metadataStats ? metadataStats.overallAvg.toFixed(1) : 0,
                    metadataStats ? metadataStats.bestScore.toFixed(1) : 0,
                    topMatch ? topMatch.name : 'No matches',
                    topMatch ? getScore(topMatch, 'similarity').toFixed(1) : 0,
                    productImagePath || ''
                ];
                const csvRowString = csvRow.map(cell => {
                    const cellStr = String(cell);
                    if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
                        return `"${cellStr.replace(/"/g, '""')}"`;
                    }
                    return cellStr;
                }).join(',');

                // Build JSON export data (don't push yet - race condition)
                const productData = {
                    product: {
                        id: product.id,
                        name: product.name,
                        category: product.cat,
                        sku: product.sku,
                        image_path: productImagePath || ''
                    },
                    statistics: metadataStats ? {
                        total_matches: metadataStats.totalMatches,
                        avg_similarity: metadataStats.overallAvg,
                        median_score: metadataStats.medianScore,
                        best_score: metadataStats.bestScore,
                        worst_score: metadataStats.worstScore,
                        matches_above_threshold: metadataStats.matchesAboveThreshold,
                        metadata_averages: metadataStats.metadataStats
                    } : null,
                    top_matches: topMatches.map((m, idx) => ({
                        rank: idx + 1,
                        product_id: m.mid,
                        product_name: m.name,
                        similarity_score: getScore(m, 'similarity'),
                        color_score: getScore(m, 'color'),
                        shape_score: getScore(m, 'shape'),
                        texture_score: getScore(m, 'texture'),
                        // CRITICAL FIX: Use mscores (compact format) with fallback
                        metadata_scores: m.metadata_scores || m.mscores || {},
                        image_path: matchImagePaths[idx] || ''
                    }))
                };

                // Return data for deterministic ordering (prevents race condition)
                return { csvRowString, productData };
            }));

            // Add results to arrays in correct order (after parallel processing completes)
            batchResults.forEach(({ csvRowString, productData }) => {
                csvRows.push(csvRowString);
                exportData.results.push(productData);
            });

            // Yield between batches so UI stays responsive and GC has time to reclaim temps.
            if (batchStart + PRODUCT_BATCH_SIZE < exportResultsSource.length) {
                await yieldToUi();
            }
        }

        // Add CSV file to ZIP
        const csvContent = csvRows.join('\n');
        zip.file('results_summary.csv', csvContent);

        // Add detailed JSON file to ZIP
        const jsonContent = JSON.stringify(exportData, null, 2);
        zip.file('results_detailed.json', jsonContent);

        // Add README
        const readme = `# Match Results Export
Generated: ${new Date().toISOString()}
Mode: ${newMode}
Total Products: ${exportResultsSource.length}
Threshold: ${Number.isFinite(thresholdValue) ? thresholdValue : 0}%

## Files
- results_summary.csv: Quick summary of all matches
- results_detailed.json: Complete match data with scores and metadata
- products/: Folder containing each product and its top ${MAX_MATCHES_PER_PRODUCT} matches
  - Each product has its own folder with:
    - new_product.<ext>: The new product image
    - matches/: Folder with top matching products (ranked by similarity)

## Notes
- Only top ${MAX_MATCHES_PER_PRODUCT} matches per product are included to keep file size manageable
- Match images are named: rank_productname_similaritypct.<ext>
`;
        zip.file('README.txt', readme);

        // Generate ZIP
        showToast('Generating ZIP file...', 'info');
        let lastReportedDecile = -1;
        const zipBlob = await zip.generateAsync({
            type: 'blob',
            compression: 'DEFLATE',
            compressionOptions: { level: 6 },
            streamFiles: true
        }, (metadata) => {
            const decile = Math.floor(metadata.percent / 10);
            if (decile > lastReportedDecile && decile >= 0 && decile <= 10) {
                lastReportedDecile = decile;
                showToast(`Compressing ZIP: ${Math.min(100, decile * 10)}%`, 'info');
            }
        });

        // Download ZIP
        const filename = `match_results_with_images_${new Date().toISOString().slice(0, 10)}.zip`;

        // Check if running in pywebview
        if (window.pywebview) {
            try {
                // Convert blob to base64 for pywebview
                const reader = new FileReader();
                reader.onloadend = async function() {
                    try {
                        const result = await window.pywebview.api.save_file_auto(reader.result, filename);
                        if (result) {
                            showToast(`Export complete! ZIP saved to Downloads: ${filename}`, 'success');
                        } else {
                            showToast('Export failed', 'error');
                        }
                    } catch (error) {
                        console.error('Webview save failed:', error);
                        showToast('Export failed - ' + error.message, 'error');
                    }
                };
                reader.onerror = function() {
                    console.error('FileReader error:', reader.error);
                    showToast('Failed to read ZIP file: ' + (reader.error?.message || 'Unknown error'), 'error');
                };
                reader.readAsDataURL(zipBlob);
            } catch (error) {
                console.error('Webview export failed:', error);
                showToast('Export failed - ' + error.message, 'error');
            }
        } else {
            // Browser fallback
            const url = URL.createObjectURL(zipBlob);
            blobUrls.add(url);

            try {
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                showToast(`Export complete! Downloaded: ${filename}`, 'success');
            } catch (error) {
                console.error('Export failed:', error);
                showToast('Export failed', 'error');
            } finally {
                setTimeout(() => {
                    URL.revokeObjectURL(url);
                    blobUrls.delete(url);
                }, 100);
            }
        }
    } catch (error) {
        console.error('Export with images failed:', error);
        showToast('Failed to export with images: ' + error.message, 'error');
    }
}

async function showDuplicateReport() {
    let duplicateSource = matchResults;

    if (isSessionServerMode()) {
        try {
            showToast('Loading filtered results for duplicate analysis...', 'info');
            const sessionData = await fetchAllServerSessionResults({
                actionLabel: 'Preparing duplicate report',
                pageSize: 500,
                maxResults: 0,
                showProgress: true
            });
            duplicateSource = sessionData.results || [];
        } catch (error) {
            console.error('Failed to load server results for duplicate report:', error);
            showToast('Unable to load full filtered results for duplicate report', 'error');
            return;
        }
    }

    if (!Array.isArray(duplicateSource) || duplicateSource.length === 0) {
        showToast('No results to analyze', 'warning');
        return;
    }

    let duplicates = [];

    duplicateSource.forEach(result => {
        const product = result.p;  // Use compact format
        const highMatches = result.m.filter(m => getScore(m, 'similarity') > 90);

        if (highMatches.length > 0) {
            duplicates.push({
                product: product,
                matches: highMatches
            });
        }
    });

    const MAX_DUPLICATE_MODAL_ROWS = 500;
    if (duplicates.length > MAX_DUPLICATE_MODAL_ROWS) {
        duplicates = duplicates.slice(0, MAX_DUPLICATE_MODAL_ROWS);
        showToast(
            `Duplicate report preview limited to ${MAX_DUPLICATE_MODAL_ROWS.toLocaleString()} items. Use export for full data.`,
            'info'
        );
    }

    if (duplicates.length === 0) {
        showToast('No potential duplicates found (similarity > 90%)', 'info');
        return;
    }

    const modal = document.getElementById('detailModal');
    const modalBody = document.getElementById('modalBody');

    let html = `
        <div class="duplicate-report-modal">
            <div class="duplicate-report-header">
                <h2>Duplicate Detection Report</h2>
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
            const similarityScore = getScore(match, 'similarity');
            html += `
                <div class="duplicate-item" data-similarity="${similarityScore}">
                    <div class="duplicate-images">
                        <img src="/api/products/${product.id}/image" alt="${product.name}">
                        <img src="/api/products/${match.mid}/image" alt="${match.name}">
                    </div>
                    <div class="duplicate-info">
                        <h4>Potential Duplicate Detected</h4>
                        <div class="duplicate-score">${similarityScore.toFixed(1)}% Similar</div>
                        <div class="duplicate-details">
                            <p><strong>New Product:</strong> ${escapeHtml(product.name)} ${product.cat ? `(${product.cat})` : ''}</p>
                            <p><strong>Matched Product:</strong> ${escapeHtml(match.name || 'Unknown')}</p>
                            <p><strong>Recommendation:</strong> ${similarityScore > 95 ? 'Very likely duplicate - review carefully' : 'Possible duplicate - manual review recommended'}</p>
                        </div>
                    </div>
                </div>
            `;
        });
    });

    html += `
            </div>
            
            <div style="margin-top: 24px; text-align: center;">
                <button class="btn btn-primary" onclick="exportDuplicateReport()">Export Duplicate Report</button>
            </div>
        </div>
    `;

    modalBody.innerHTML = html;
    modal.classList.add('show');
}

// Export Duplicate Report
async function exportDuplicateReport() {
    let duplicateSource = matchResults;

    if (isSessionServerMode()) {
        try {
            showToast('Loading filtered results for duplicate export...', 'info');
            const sessionData = await fetchAllServerSessionResults({
                actionLabel: 'Preparing duplicate export',
                pageSize: 500,
                maxResults: 0,
                showProgress: true
            });
            duplicateSource = sessionData.results || [];
        } catch (error) {
            console.error('Failed to load server results for duplicate export:', error);
            showToast('Export failed - unable to load full filtered results', 'error');
            return;
        }
    }

    if (!Array.isArray(duplicateSource) || duplicateSource.length === 0) {
        showToast('No results to analyze', 'warning');
        return;
    }

    const duplicates = [];

    // Detect metadata keys for optional inclusion
    const allMetadataKeys = new Set();
    for (let i = 0; i < duplicateSource.length; i++) {
        const result = duplicateSource[i];
        result.m.forEach(match => {
            // CRITICAL FIX: Use mscores (compact format) with fallback
            const scores = match.metadata_scores || match.mscores;
            if (scores) {
                Object.keys(scores).forEach(key => allMetadataKeys.add(key));
            }
        });

        if (i > 0 && i % 500 === 0) {
            await yieldToUi();
        }
    }
    const metadataKeysArray = Array.from(allMetadataKeys).sort();
    const hasMetadataScores = metadataKeysArray.length > 0;

    for (let i = 0; i < duplicateSource.length; i++) {
        const result = duplicateSource[i];
        const product = result.p;  // Use compact format
        const highMatches = result.m.filter(m => getScore(m, 'similarity') > 90);

        if (highMatches.length > 0) {
            highMatches.forEach(match => {
                const similarityScore = getScore(match, 'similarity');
                const duplicateEntry = {
                    new_product: product.name,
                    new_category: product.cat || 'Uncategorized',
                    new_sku: product.sku || 'N/A',
                    matched_product: match.name || 'Unknown',
                    similarity_score: similarityScore.toFixed(1),
                    recommendation: similarityScore > 95 ? 'Very likely duplicate' : 'Possible duplicate'
                };

                // Add metadata scores if available
                // CRITICAL FIX: Use mscores (compact format) with fallback
                const matchScores = match.metadata_scores || match.mscores;
                if (hasMetadataScores && matchScores) {
                    metadataKeysArray.forEach(key => {
                        duplicateEntry[`${key}_score`] = matchScores[key]?.toFixed(1) || '';
                    });
                }

                duplicates.push(duplicateEntry);
            });
        }

        if (i > 0 && i % 400 === 0) {
            await yieldToUi();
        }
    }

    // Build header row
    let headerRow = ['New Product', 'New Category', 'New SKU', 'Matched Product', 'Similarity Score', 'Recommendation'];
    if (hasMetadataScores) {
        metadataKeysArray.forEach(key => {
            headerRow.push(`${key} Score`);
        });
    }

    const csvRows = [headerRow.map(h => `"${h}"`).join(',')];

    for (let i = 0; i < duplicates.length; i++) {
        const dup = duplicates[i];
        const row = headerRow.map(header => {
            const value = dup[header === 'Similarity Score' ? 'similarity_score' :
                header === 'Recommendation' ? 'recommendation' :
                    header === 'New Product' ? 'new_product' :
                        header === 'New Category' ? 'new_category' :
                            header === 'New SKU' ? 'new_sku' :
                                header === 'Matched Product' ? 'matched_product' :
                                    header];
            if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
                return `"${value.replace(/"/g, '""')}"`;
            }
            return typeof value === 'string' ? `"${value}"` : (value || '');
        });
        csvRows.push(row.join(','));

        if (i > 0 && i % 1000 === 0) {
            await yieldToUi();
        }
    }

    const filename = `duplicate_report_${new Date().toISOString().slice(0, 10)}.csv`;

    // Check if running in pywebview
    if (window.pywebview) {
        try {
            const csv = csvRows.join('\n') + '\n';
            const result = await window.pywebview.api.save_file_auto(csv, filename);
            if (result) {
                showToast(`Duplicate report saved to Downloads folder: ${filename}`, 'success');
            } else {
                showToast('Export failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Export failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob(csvRows.map(row => `${row}\n`), { type: 'text/csv' });
        const url = URL.createObjectURL(blob);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('Duplicate report exported to CSV', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showToast('Export failed', 'error');
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    }
}


async function saveSession() {
    // In server mode matchResults only holds the current page.
    // Fetch the full filtered set so the saved file contains everything.
    let allResults = matchResults;

    if (isSessionServerMode()) {
        try {
            showToast('Preparing session save — loading all results from server...', 'info');
            const sessionData = await fetchAllServerSessionResults({
                actionLabel: 'Saving session',
                pageSize: 500,
                maxResults: 0,
                showProgress: true
            });
            allResults = sessionData.results || [];
        } catch (error) {
            console.error('Failed to fetch full results for session save:', error);
            showToast('Save failed — could not load all results', 'error');
            return;
        }
    }

    if (!Array.isArray(allResults) || allResults.length === 0) {
        showToast('No session data to save', 'warning');
        return;
    }

    const sessionData = {
        version: '1.1',
        timestamp: new Date().toISOString(),
        weights: similarityWeights,
        threshold: parseInt(document.getElementById('thresholdSlider').value),
        limit: parseInt(document.getElementById('limitSelect').value),
        historicalProducts: historicalProducts,
        newProducts: newProducts,
        matchResults: allResults
    };

    const sessionContent = JSON.stringify(sessionData, null, 2);
    const filename = `matching_session_${new Date().toISOString().slice(0, 10)}.json`;


    if (window.pywebview) {
        try {
            const result = await window.pywebview.api.save_file_auto(sessionContent, filename);
            if (result) {
                showToast(`Session saved to Downloads folder: ${filename}`, 'success');
            } else {
                showToast('Save failed', 'error');
            }
        } catch (error) {
            console.error('Webview save failed:', error);
            showToast('Save failed - ' + error.message, 'error');
        }
    } else {
        // Browser fallback
        const blob = new Blob([sessionContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            showToast('Session saved successfully', 'success');
        } catch (error) {
            console.error('Save failed:', error);
            showToast('Save failed', 'error');
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    }
}


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

            // Restore state — loaded sessions are fully local, disable server mode
            serverResultsMode = false;
            currentMatchSessionId = null;
            resetServerResultsClientState();

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
            showResultsSectionWithCollapse();

            showToast(`Session loaded: ${matchResults.length} products with matches`, 'success');

            // Save to history
            saveToHistory('session_loaded', { timestamp: sessionData.timestamp });
        } catch (error) {
            showToast('Failed to load session: ' + error.message, 'error');
        }
    };

    input.click();
}


function applyFilters() {
    searchQuery = document.getElementById('searchInput').value.toLowerCase();
    filterCategory = document.getElementById('categoryFilter').value;
    filterDuplicatesOnly = document.getElementById('duplicatesOnlyCheckbox').checked;
    sortBy = document.getElementById('sortBySelect').value;

    // Update active filter badges
    updateActiveBadges();

    // Update main search status
    updateMainSearchStatus();

    // Re-render results with filters
    displayResults();

    // Save to history
    saveToHistory('filters_applied', { searchQuery, filterCategory, filterDuplicatesOnly, sortBy });
}

function updateMainSearchStatus() {
    const statusEl = document.getElementById('searchStatus');
    if (!searchQuery) {
        statusEl.innerHTML = '';
        return;
    }

    if (isSessionServerMode()) {
        const count = serverResultsFilteredCount || 0;
        statusEl.innerHTML = `<span class="search-count">${count} ${count === 1 ? 'product' : 'products'}</span>`;
        return;
    }

    // Count how many products match the search
    if (matchResults && matchResults.length > 0) {
        const filtered = matchResults.map(result => {
            const product = result.p;  // Use compact format
            const searchText = `${product.name || ''} ${product.sku || ''} ${product.cat || ''}`.toLowerCase();
            if (!searchText.includes(searchQuery)) {
                return null;
            }
            return result;
        }).filter(r => r !== null);

        const count = filtered.length;
        statusEl.innerHTML = `<span class="search-count">${count} ${count === 1 ? 'product' : 'products'}</span>`;
    }
}

// Update Active Filter Badges
function updateActiveBadges() {
    const badgesContainer = document.getElementById('activeBadges');
    const badges = [];

    // Search badge
    if (searchQuery) {
        badges.push({
            label: 'Search',
            value: searchQuery,
            clear: () => {
                document.getElementById('searchInput').value = '';
                applyFilters();
            }
        });
    }

    // Category badge
    if (filterCategory !== 'all' && filterCategory !== '') {
        badges.push({
            label: 'Category',
            value: filterCategory,
            clear: () => {
                document.getElementById('categoryFilter').value = 'all';
                applyFilters();
            }
        });
    }

    // Duplicates badge
    if (filterDuplicatesOnly) {
        badges.push({
            label: 'Duplicates Only',
            value: '',
            clear: () => {
                document.getElementById('duplicatesOnlyCheckbox').checked = false;
                applyFilters();
            }
        });
    }

    // Render badges
    if (badges.length === 0) {
        badgesContainer.innerHTML = '';
        return;
    }

    badgesContainer.innerHTML = badges.map(badge => `
        <div class="badge">
            <span>${badge.label}${badge.value ? ': <span class="badge-value">' + badge.value + '</span>' : ''}</span>
            <button class="badge-remove" onclick="event.stopPropagation(); this.parentElement.onclick(); this.parentElement.remove();">×</button>
        </div>
    `).join('');

    // Attach click handlers to badges
    badges.forEach((badge, index) => {
        const badgeElement = badgesContainer.children[index];
        badgeElement.onclick = badge.clear;
    });
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
    if (isSessionServerMode()) {
        populateCategoryFilterFromServerFacets();
        return;
    }

    const select = document.getElementById('categoryFilter');
    if (!select) return;

    const datasetToken = getResultsDatasetToken();
    const previousValue = select.value || filterCategory || 'all';

    if (categoryFilterDatasetToken === datasetToken && select.options.length > 0) {
        // Dataset unchanged: keep existing options and selection.
        if (previousValue && Array.from(select.options).some(opt => opt.value === previousValue)) {
            select.value = previousValue;
        } else {
            select.value = 'all';
        }
        return;
    }

    const categories = new Set();
    for (let i = 0; i < matchResults.length; i++) {
        const category = matchResults[i]?.p?.cat;
        if (category) categories.add(category);
    }

    select.innerHTML = '<option value="all">All Categories</option>';

    Array.from(categories).sort().forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    });

    if (previousValue && Array.from(select.options).some(opt => opt.value === previousValue)) {
        select.value = previousValue;
    } else {
        select.value = 'all';
    }

    categoryFilterDatasetToken = datasetToken;
}

// Populate dynamic sort options based on schema
function populateSortOptions() {
    const select = document.getElementById('sortBySelect');
    if (!select || !window.metadataSchema) return;

    // Check if options already exist to avoid duplicates
    const existingValues = new Set(Array.from(select.options).map(opt => opt.value));

    // Add SKU if not present
    if (!existingValues.has('sku')) {
        const option = document.createElement('option');
        option.value = 'sku';
        option.textContent = 'SKU';
        select.appendChild(option);
    }

    // Add dynamic columns
    window.metadataSchema.forEach(col => {
        const val = col.column_name;
        if (!existingValues.has(val) && val !== 'sku' && val !== 'name' && val !== 'category') {
            const option = document.createElement('option');
            option.value = val;
            // Add symbol for type
            const typeLabel = col.data_type === 'numeric' ? '#' : 'Aa';
            option.textContent = `${col.display_name} (${typeLabel})`;
            select.appendChild(option);
        }
    });
}


function filterAndSortResults(results) {
    // Populate sort options on first run
    if (!window.sortOptionsPopulated && window.metadataSchema) {
        populateSortOptions();
        window.sortOptionsPopulated = true;
    }

    // PERFORMANCE: Pre-calculate all filter conditions to avoid repeated lookups
    const activeMetadataFilters = window.metadataFilterCriteria || {};
    const metadataFilterKeys = Object.keys(activeMetadataFilters);
    const hasMetadataFilters = metadataFilterKeys.length > 0;
    const hasSearch = searchQuery && searchQuery.length > 0;
    const hasSearchResults = dynamicSearch && dynamicSearchResults && dynamicSearchResults.size > 0;
    const hasCategoryFilter = filterCategory !== 'all' && filterCategory !== '';
    const hasLimit = dynamicLimit > 0;
    const isDuplicatesFilter = filterDuplicatesOnly;
    const needsAverageCalc = sortBy === 'avg_similarity';

    // PERFORMANCE: Filter and transform in single pass
    let filtered = [];

    for (let i = 0; i < results.length; i++) {
        const result = results[i];
        let filteredMatches = result.m;


        if (dynamicThreshold > 30 || hasMetadataFilters || hasSearchResults) {
            filteredMatches = filteredMatches.filter(match => {
                // Check 1: Threshold (early exit if fails)
                if (dynamicThreshold > 30 && getScore(match, 'similarity') < dynamicThreshold) {
                    return false;
                }

                // Check 2: Metadata filters (early exit if fails)
                if (hasMetadataFilters) {
                    const values = match.mv || match.metadata_values || {};

                    for (let k = 0; k < metadataFilterKeys.length; k++) {
                        const field = metadataFilterKeys[k];
                        const criteria = activeMetadataFilters[field];
                        const val = values[field];

                        // Check multi-select (categorical) - NEW smart hybrid filter support
                        if (criteria.values && criteria.values.size > 0) {
                            if (!criteria.values.has(String(val))) {
                                return false; // Early exit if value not in selected set
                            }
                        }

                        // Check single equals (legacy categorical - deprecated but kept for compatibility)
                        if (criteria.equals !== undefined && val != criteria.equals) {
                            return false; // Early exit
                        }

                        // Check range min
                        if (criteria.min !== undefined) {
                            if (val === undefined || val === null || val === '') {
                                return false; // Missing value fails filter
                            }
                            const numVal = parseFloat(val);
                            if (isNaN(numVal) || numVal < criteria.min) {
                                return false; // Invalid or below min
                            }
                        }

                        // Check range max
                        if (criteria.max !== undefined) {
                            if (val === undefined || val === null || val === '') {
                                return false; // Missing value fails filter
                            }
                            const numVal = parseFloat(val);
                            if (isNaN(numVal) || numVal > criteria.max) {
                                return false; // Invalid or above max
                            }
                        }
                    }
                }

                // Check 3: Dynamic search (early exit if fails)
                if (hasSearchResults && !dynamicSearchResults.has(match.mid)) {
                    return false;
                }

                return true; // Passed all filters
            });

            if (filteredMatches.length === 0) continue; // Skip product if no matches
        }

        // Apply dynamic limit (only slice if needed, not every time)
        if (hasLimit && filteredMatches.length > dynamicLimit) {
            filteredMatches = filteredMatches.slice(0, dynamicLimit);
        }

        const product = result.p;

        // Check product filters before adding to filtered array
        // Duplicates filter
        if (isDuplicatesFilter) {
            let hasDuplicate = false;
            for (let j = 0; j < filteredMatches.length; j++) {
                if (getScore(filteredMatches[j], 'similarity') > 90) {
                    hasDuplicate = true;
                    break;
                }
            }
            if (!hasDuplicate) continue;
        }

        // Search filter (only build searchText if needed and not yet checked)
        if (hasSearch) {
            // PERFORMANCE: Build search text inline without creating unnecessary strings for all products
            const searchText = `${product.name || ''} ${product.sku || ''} ${product.cat || ''}`.toLowerCase();
            if (!searchText.includes(searchQuery)) {
                continue;
            }
        }

        // Category filter
        if (hasCategoryFilter && product.cat !== filterCategory) {
            continue;
        }

        // Add to filtered results
        const resultObj = {
            ...result,
            m: filteredMatches
        };

        // PERFORMANCE: Pre-calculate average for sorting to avoid O(n*m) recalculation in sort comparator
        if (needsAverageCalc) {
            let sum = 0;
            for (let j = 0; j < filteredMatches.length; j++) {
                sum += getScore(filteredMatches[j], 'similarity');
            }
            resultObj._avgSim = filteredMatches.length > 0 ? sum / filteredMatches.length : 0;
        }

        filtered.push(resultObj);
    }

    const schemaMap = new Map(window.metadataSchema?.map(c => [c.column_name, c]) || []);
    const sortSchemaCol = schemaMap.get(sortBy);
    const isMetadataSort = schemaMap.has(sortBy) || sortBy === 'sku';
    const isNumericSort = sortSchemaCol?.data_type === 'numeric';

    // Sort results - PERFORMANCE: Uses pre-calculated values where possible
    if (filtered.length > 1) {
        filtered.sort((a, b) => {
            // Priority: Check generic sorts first, then specific

            // 1. Check for Standard Sorts
            if (sortBy === 'similarity') {
                const aMatch = a.m.length > 0 ? a.m[0] : null;
                const bMatch = b.m.length > 0 ? b.m[0] : null;
                if (!aMatch && !bMatch) return 0;
                if (!aMatch) return 1;
                if (!bMatch) return -1;
                return getScore(bMatch, 'similarity') - getScore(aMatch, 'similarity');
            }

            if (sortBy === 'avg_similarity') {
                return b._avgSim - a._avgSim;
            }

            if (sortBy === 'category') {
                return (a.p.cat || '').localeCompare(b.p.cat || '');
            }

            if (sortBy === 'name') {
                return (a.p.name || '').localeCompare(b.p.name || '');
            }

            if (sortBy === 'match_count') {
                return b.m.length - a.m.length;
            }

            // 2. Dynamic Metadata Sort (using cached schema lookups)
            if (isMetadataSort) {
                const colName = sortBy;

                let valA = a.p[colName] !== undefined && a.p[colName] !== null ? a.p[colName] :
                           (a.p.meta && a.p.meta[colName] !== undefined && a.p.meta[colName] !== null ? a.p.meta[colName] : '');
                let valB = b.p[colName] !== undefined && b.p[colName] !== null ? b.p[colName] :
                           (b.p.meta && b.p.meta[colName] !== undefined && b.p.meta[colName] !== null ? b.p.meta[colName] : '');

                if (!window.debugSort) {
                    console.log(`[SORT] Sorting by ${colName}: "${valA}" vs "${valB}"`);
                    window.debugSort = true;
                }

                const isNumeric = isNumericSort ||
                    (!isNaN(parseFloat(valA)) && !isNaN(parseFloat(valB)) && valA !== '' && valB !== '');

                if (isNumeric) {
                    // Numeric sort (Descending default)
                    return parseFloat(valB) - parseFloat(valA);
                } else {
                    // String sort (Alphabetical Ascending)
                    return String(valA).localeCompare(String(valB));
                }
            }

            return 0;
        });

        // Clean up temporary values
        if (needsAverageCalc) {
            for (let i = 0; i < filtered.length; i++) {
                delete filtered[i]._avgSim;
            }
        }
    }

    return filtered;
}

// ENHANCEMENT: Initialize metadata filter criteria on window (single source of truth).
// All reads MUST go through window.metadataFilterCriteria to stay in sync with the
// handlers that write to it (updateMetadataFilterMulti, updateMetadataFilter, etc.).
if (!window.metadataFilterCriteria) window.metadataFilterCriteria = {};

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
}

// Initialize advanced features on page load
runWhenDomReadyResults(() => {
    initAdvancedFeatures();
});

// Toggle help text in CSV format modal
