
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

// Dynamic metadata schema and weights (loaded from backend after CSV upload)
let metadataSchema = [];  // Array of {column_name, data_type, display_name}
let metadataWeights = {}; // Dictionary of column_name -> weight (0-100)

// Pagination state
let currentPage = 1;
const RESULTS_PER_PAGE = 20; // Show 20 products per page
let serverResultsMode = false;
let currentMatchSessionId = null;
let currentMatchSessionSummary = null;
let currentMatchSessionTotal = 0;

// Undo/Redo state
let historyStack = [];
let historyIndex = -1;
const MAX_HISTORY = 50;
const PATH_SEPARATOR_REGEX = /[\\/]/;
const IMAGE_EXTENSION_REGEX = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;
const MAX_UPLOAD_FILES = 50000;
const LARGE_FOLDER_CHUNK_THRESHOLD = 5000;
const CATEGORY_COUNT_CHUNK_SIZE = 2000;
const MAX_FAILED_ITEM_DETAILS = 200;
const STREAM_UPLOAD_BATCH_SIZE = 100;
const STREAM_UPLOAD_BATCH_SIZE_DIRECT_MAX = 50;
const STREAM_UPLOAD_BATCH_SIZE_DIRECT_MIN = 15;
const DIRECT_UPLOAD_TARGET_BATCH_BYTES = 120 * 1024 * 1024; // 120MB target per HTTP upload
const UPLOAD_REQUEST_TIMEOUT_BASE_MS = 8 * 60 * 1000; // 8 min
const UPLOAD_REQUEST_TIMEOUT_MAX_MS = 30 * 60 * 1000; // 30 min hard cap
const UPLOAD_REQUEST_TIMEOUT_PER_MB_MS = 4000; // +4s per MB for slow links
const AUTO_FAST_FILE_THRESHOLD = 5000;
const AUTO_FAST_CPU_CORES_THRESHOLD = 4;
const CLIENT_DEBUG_LOGS = false;

function onDomReady(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback, { once: true });
        return;
    }
    setTimeout(callback, 0);
}
window.__catalogMatchOnDomReady = onDomReady;

function debugLog(...args) {
    if (CLIENT_DEBUG_LOGS) {
        console.log(...args);
    }
}

function debugWarn(...args) {
    if (CLIENT_DEBUG_LOGS) {
        console.warn(...args);
    }
}

// Excluded metadata keys for component display
const EXCLUDED_METADATA_KEYS = new Set(['sku', 'name', 'category', 'price', 'product_name', 'performance']);

function isImageLikeFile(file) {
    if (!file) return false;
    if (typeof file.type === 'string' && file.type.toLowerCase().startsWith('image/')) {
        return true;
    }
    return IMAGE_EXTENSION_REGEX.test(String(file.name || ''));
}

function showUploadLoadingState(infoElementId, message) {
    const info = document.getElementById(infoElementId);
    if (!info) return;
    info.innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <span class="btn-spinner" style="display: inline-block; margin-right: 8px;"></span>
            ${escapeHtml(message)}
        </div>
    `;
    info.classList.add('show');
}

async function countCategoriesWithYield(filesWithCategories) {
    const categoryCount = {};

    if (filesWithCategories.length <= LARGE_FOLDER_CHUNK_THRESHOLD) {
        filesWithCategories.forEach(({ category }) => {
            if (category) {
                categoryCount[category] = (categoryCount[category] || 0) + 1;
            }
        });
        return categoryCount;
    }

    for (let i = 0; i < filesWithCategories.length; i += CATEGORY_COUNT_CHUNK_SIZE) {
        const end = Math.min(i + CATEGORY_COUNT_CHUNK_SIZE, filesWithCategories.length);
        for (let j = i; j < end; j++) {
            const category = filesWithCategories[j]?.category;
            if (category) {
                categoryCount[category] = (categoryCount[category] || 0) + 1;
            }
        }

        if (i + CATEGORY_COUNT_CHUNK_SIZE < filesWithCategories.length) {
            await sleep(0);
        }
    }

    return categoryCount;
}

function getAutoProcessingProfile(totalFiles) {
    const cores = navigator.hardwareConcurrency || 0;
    const isSlowCpu = cores > 0 && cores <= AUTO_FAST_CPU_CORES_THRESHOLD;
    if (totalFiles >= AUTO_FAST_FILE_THRESHOLD || isSlowCpu) {
        return 'fast';
    }
    return 'auto';
}

function appendBatchUploadPayload(formData, filesWithCategories, categoryMap, isHistorical, totalOperationFiles, rebuildFaiss = true) {
    const useFilePaths = filesWithCategories.every(({ file }) =>
        file && typeof file.path === 'string' && file.path.trim().length > 0
    );

    const filePaths = [];
    const categories = [];
    const productNames = [];
    const skus = [];
    const metadataList = [];

    filesWithCategories.forEach(({ file, category }, idx) => {
        const metadata = categoryMap[file.name] || {};
        const finalCategory = metadata.category || category;

        categories.push(finalCategory || null);
        productNames.push(metadata.name || file.name || `image_${idx}`);
        skus.push(metadata.sku || null);

        const dynamicMeta = { ...metadata };
        delete dynamicMeta.category;
        delete dynamicMeta.sku;
        delete dynamicMeta.name;
        metadataList.push(JSON.stringify(dynamicMeta));

        if (useFilePaths) {
            filePaths.push(file.path);
        } else {
            formData.append('images', file, file.name || `image_${idx}`);
        }
    });

    if (useFilePaths) {
        formData.append('file_paths', JSON.stringify(filePaths));
    }

    formData.append('categories', JSON.stringify(categories));
    formData.append('product_names', JSON.stringify(productNames));
    formData.append('skus', JSON.stringify(skus));
    formData.append('metadata', JSON.stringify(metadataList));
    formData.append('is_historical', isHistorical ? 'true' : 'false');
    formData.append('operation_total_files', String(totalOperationFiles));
    formData.append('processing_profile', getAutoProcessingProfile(totalOperationFiles));
    formData.append('rebuild_faiss', rebuildFaiss ? 'true' : 'false');
    formData.append('skip_existing', 'true');

    return useFilePaths ? 'file_paths' : 'direct_upload';
}

function canUseFilePathTransport(filesWithCategories) {
    if (!Array.isArray(filesWithCategories) || filesWithCategories.length === 0) {
        return false;
    }

    return filesWithCategories.every(({ file }) =>
        file && typeof file.path === 'string' && file.path.trim().length > 0
    );
}

function estimateBatchPayloadBytes(filesWithCategories) {
    if (!Array.isArray(filesWithCategories) || filesWithCategories.length === 0) {
        return 0;
    }

    let total = 0;
    for (let i = 0; i < filesWithCategories.length; i++) {
        const size = Number(filesWithCategories[i]?.file?.size || 0);
        if (Number.isFinite(size) && size > 0) {
            total += size;
        }
    }
    return total;
}

function getAdaptiveUploadBatchSize(filesWithCategories) {
    if (!Array.isArray(filesWithCategories) || filesWithCategories.length === 0) {
        return STREAM_UPLOAD_BATCH_SIZE;
    }

    // File-path transport is local desktop fast-path; keep larger batch size.
    if (canUseFilePathTransport(filesWithCategories)) {
        return STREAM_UPLOAD_BATCH_SIZE;
    }

    // Direct HTTP uploads: size by payload budget for better reliability on slow links.
    const totalBytes = estimateBatchPayloadBytes(filesWithCategories);
    const avgBytes = totalBytes > 0 ? totalBytes / filesWithCategories.length : 0;

    if (avgBytes <= 0) {
        return STREAM_UPLOAD_BATCH_SIZE_DIRECT_MAX;
    }

    const byBytes = Math.max(
        STREAM_UPLOAD_BATCH_SIZE_DIRECT_MIN,
        Math.floor(DIRECT_UPLOAD_TARGET_BATCH_BYTES / avgBytes)
    );
    return Math.min(STREAM_UPLOAD_BATCH_SIZE_DIRECT_MAX, byBytes);
}

function getUploadRequestTimeoutMs(batchFiles, transportMode) {
    // File-path mode sends only metadata/path strings; keep baseline timeout.
    if (transportMode === 'file_paths') {
        return UPLOAD_REQUEST_TIMEOUT_BASE_MS;
    }

    const payloadBytes = estimateBatchPayloadBytes(batchFiles);
    const payloadMb = payloadBytes > 0 ? Math.ceil(payloadBytes / (1024 * 1024)) : 0;
    const dynamicTimeout = UPLOAD_REQUEST_TIMEOUT_BASE_MS + (payloadMb * UPLOAD_REQUEST_TIMEOUT_PER_MB_MS);

    return Math.min(UPLOAD_REQUEST_TIMEOUT_MAX_MS, Math.max(UPLOAD_REQUEST_TIMEOUT_BASE_MS, dynamicTimeout));
}

function recordFailedItem(failedItems, item) {
    if (failedItems.length < MAX_FAILED_ITEM_DETAILS) {
        failedItems.push(item);
    }
}

const IconManager = {
    // Track state
    debounceTimer: null,
    lastInitTime: 0,
    initCount: 0,

    /**
     * Initialize all Lucide icons on the page
     * Safe to call multiple times - Lucide skips already-initialized icons
     * @param {HTMLElement} [container] - Optional container to scope initialization
     */
    init(container = null) {
        if (typeof lucide === 'undefined') {
            console.warn('[IconManager] Lucide library not loaded');
            return false;
        }

        try {
            // Lucide's createIcons() intelligently skips already-converted icons
            // No memory leaks or duplicates from repeated calls
            if (container) {
                // Scoped initialization using 'root' parameter (more performant)
                // See: https://lucide.dev/guide/packages/lucide
                lucide.createIcons({
                    root: container,
                    attrs: { 'stroke-width': 2 }
                });
            } else {
                // Full page initialization
                lucide.createIcons();
            }

            this.lastInitTime = Date.now();
            this.initCount++;
            return true;
        } catch (e) {
            console.error('[IconManager] Failed to initialize icons:', e);
            return false;
        }
    },

    /**
     * Uses requestAnimationFrame for smoother, non-blocking updates
     * @param {number} delay - Debounce delay in ms (default: 50ms for better batching)
     * @param {HTMLElement} [container] - Optional container to scope re-init
     */
    reinit(delay = 50, container = null) {
        if (typeof lucide === 'undefined') {
            return;
        }

        // Clear any pending initialization
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Debounce to batch multiple rapid updates
        this.debounceTimer = setTimeout(() => {
            // Use requestAnimationFrame for non-blocking UI updates
            requestAnimationFrame(() => {
                this.init(container);
                this.debounceTimer = null;
            });
        }, delay);
    },

    /**
     * Immediate synchronous re-initialization (use sparingly)
     * Only use when icons must appear instantly (e.g., critical UI updates)
     * @param {HTMLElement} [container] - Optional container to scope re-init
     */
    reinitSync(container = null) {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
        this.init(container);
    },

    /**
     * Cleanup method - call before removing dynamic content
     * Prevents memory leaks by clearing pending timers
     */
    cleanup() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    },

    /**
     * Get initialization stats for debugging
     */
    getStats() {
        return {
            initCount: this.initCount,
            lastInitTime: this.lastInitTime,
            hasPendingInit: this.debounceTimer !== null
        };
    }
};

// Export for global access
window.IconManager = IconManager;

/**
 * Check if a value is numeric (for metadata comparison highlighting)
 * @param {*} val - Value to check
 * @returns {boolean} True if value is numeric
 */
function isNumericValue(val) {
    if (val === '-' || val === undefined || val === null) return false;
    const cleanVal = String(val).replace(/[$,]/g, '');
    const num = parseFloat(cleanVal);
    return !isNaN(num) && isFinite(num);
}

/**
 * Render metadata score bars with progress visualization
 * @param {Object} scores - Metadata scores object
 * @returns {string} HTML string for score bars
 */
function renderMetadataScoreBars(scores) {
    let html = '';
    for (const [key, score] of Object.entries(scores)) {
        if (EXCLUDED_METADATA_KEYS.has(key.toLowerCase())) continue;

        const escapedKey = escapeHtml(key.charAt(0).toUpperCase() + key.slice(1));
        const percentage = Number(score).toFixed(1);

        html += `
            <div style="margin-bottom: 4px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span style="font-size: 12px; color: #4a5568;">${escapedKey} Match</span>
                    <span style="font-size: 12px; font-weight: 600; color: #2d3748;">${percentage}%</span>
                </div>
                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden;">
                    <div style="width: ${percentage}%; height: 100%; background: #48bb78;"></div>
                </div>
            </div>
        `;
    }
    return html;
}

/**
 * Render full metadata comparison with optimized performance
 * @param {Array} keys - Array of metadata keys
 * @param {Object} newMeta - New product metadata
 * @param {Object} matchedMeta - Matched product metadata
 * @returns {string} HTML string for comparison table
 */
function renderMetadataComparison(keys, newMeta, matchedMeta) {
    let html = '';

    for (const key of keys) {
        const valNew = newMeta[key] ?? '-';
        const valMatch = matchedMeta[key] ?? '-';

        const strNew = String(valNew);
        const strMatch = String(valMatch);
        const isDiff = strNew !== strMatch && valNew !== '-' && valMatch !== '-';
        const numericField = isNumericValue(valNew) || isNumericValue(valMatch);

        // Pre-compute styles to avoid conditionals in template
        const bgColor = isDiff ? '#fff5f5' : 'white';
        const borderColor = isDiff ? '#feb2b2' : '#e2e8f0';
        const borderStyle = numericField ? 'border-left: 3px solid #4299e1;' : '';
        const fontSize = numericField ? '14px' : '13px';
        const fontWeight = numericField ? 'font-weight: 600;' : '';
        const labelColor = numericField ? '#2b6cb0' : '#4a5568';

        // Escape once and reuse
        const escapedKey = escapeHtml(String(key));
        const escapedNew = escapeHtml(strNew);
        const escapedMatch = escapeHtml(strMatch);

        html += `
            <div style="background: ${bgColor}; border: 1px solid ${borderColor}; border-radius: 6px; padding: 10px; ${borderStyle}">
                <div style="display: grid; grid-template-columns: 1fr 1.2fr 1.2fr; gap: 12px; align-items: center;">
                    <div style="font-weight: 600; color: ${labelColor}; font-size: 13px; text-transform: capitalize;">
                        ${numericField ? '<span style="display: inline-block; width: 6px; height: 6px; background: #4299e1; border-radius: 50%; margin-right: 6px;"></span>' : ''}
                        ${escapedKey}
                    </div>
                    <div style="padding: 6px 10px; background: #ebf8ff; border-radius: 4px; font-size: ${fontSize}; color: #2d3748; ${fontWeight}">${escapedNew}</div>
                    <div style="padding: 6px 10px; background: #fffaf0; border-radius: 4px; font-size: ${fontSize}; color: #2d3748; ${fontWeight}">${escapedMatch}</div>
                </div>
            </div>
        `;
    }

    return html;
}

const stringCache = new Map();
const MAX_CACHE_SIZE = 5000;  // MEMORY OPTIMIZATION: Prevent unbounded cache growth (1-5MB with 10K+ results)

/**
 * Intern a string to save memory on duplicates
 * Example: intern("Electronics") called 500 times = only 1 string in memory
 */
function intern(str) {
    if (!str || typeof str !== 'string') return str;
    if (!stringCache.has(str)) {
        // MEMORY OPTIMIZATION: Clear cache if it exceeds max size
        if (stringCache.size >= MAX_CACHE_SIZE) {
            stringCache.clear();
        }
        stringCache.set(str, str);
    }
    return stringCache.get(str);
}

/**
 * PERFORMANCE: Parse metadata JSON string with memoization
 * Reusable helper that caches parsed results to avoid re-parsing identical strings
 *
 * @param {string|object} metadata - Metadata JSON string or object
 * @param {string|number} id - Product/match ID for error logging
 * @returns {object} Parsed metadata object
 */
const metadataParseCache = new Map(); // Cache for parsed metadata (prevents re-parsing)
let metadataCacheSize = 0;
const MAX_METADATA_CACHE_SIZE = 1000; // Prevent unbounded cache growth

function parseMetadata(metadata, id) {
    if (!metadata) return {};
    if (typeof metadata === 'object') return metadata;

    // Check cache first (PERFORMANCE: ~10x faster for repeated strings)
    const cached = metadataParseCache.get(metadata);
    if (cached !== undefined) return cached;

    // Parse new string
    try {
        const parsed = JSON.parse(metadata);

        // Cache the result (with size limit to prevent memory leak)
        if (metadataCacheSize < MAX_METADATA_CACHE_SIZE) {
            metadataParseCache.set(metadata, parsed);
            metadataCacheSize++;
        }

        return parsed;
    } catch (e) {
        console.warn(`[METADATA] Failed to parse metadata for ID ${id}:`, e);
        return {};
    }
}

function clearMetadataCache() {
    metadataParseCache.clear();
    metadataCacheSize = 0;
}

function createCompactMatch(matchData) {
    // Use Float32Array for scores: 16 bytes vs 64 bytes for 4 numbers
    const scores = new Float32Array(4);
    scores[0] = matchData.similarity_score || 0;
    scores[1] = matchData.color_score || 0;
    scores[2] = matchData.shape_score || 0;
    scores[3] = matchData.texture_score || 0;

    const compact = {
        pid: matchData.product_id,
        mid: matchData.product_id || matchData.matched_product_id || matchData.id,
        s: scores,  // scores array
        cat: intern(matchData.category),  // Shared string, not duplicated
        sku: matchData.sku,
        name: matchData.product_name || matchData.name,
        fn: matchData.filename || matchData.image_path, // Capture filename or image path
        img: matchData.image_path
    };

    // Only include truthy optional properties
    if (matchData.is_potential_duplicate) {
        compact.dup = true;
    }

    // Include hybrid mode scores if present
    if (matchData.visual_score !== undefined) compact.vs = matchData.visual_score;
    if (matchData.metadata_score !== undefined) compact.ms = matchData.metadata_score;
    if (matchData.sku_score !== undefined) compact.skus = matchData.sku_score;
    if (matchData.name_score !== undefined) compact.ns = matchData.name_score;
    if (matchData.category_score !== undefined) compact.cs = matchData.category_score;
    if (matchData.price_score !== undefined) compact.ps = matchData.price_score;

    // Include full metadata values and scores dict (for dynamic metadata display in Mode 2/3)
    // CRITICAL FIX: Parse metadata_values JSON string for filtering
    // PERFORMANCE: Use cached parseMetadata helper instead of IIFE
    if (matchData.metadata_values) {
        compact.mv = parseMetadata(matchData.metadata_values, matchData.product_id);
    }
    if (matchData.metadata_scores) compact.mscores = matchData.metadata_scores;

    // PERFORMANCE: Don't freeze - allows V8 to optimize (10-20% faster object creation)
    return compact;
}

function createCompactProduct(productData) {
    const compact = {
        id: productData.id,
        name: productData.name || productData.filename || productData.product_name,
        cat: intern(productData.category),
        sku: productData.sku,
        hasF: productData.hasFeatures || false,
        img: productData.image_path,
        fn: productData.filename || (productData.image_path ? productData.image_path.split(/[\\/]/).pop() : ''),
        meta: parseMetadata(productData.metadata, productData.id)
    };

    // PERFORMANCE: Don't freeze - allows V8 to optimize (10-20% faster object creation)
    return compact;
}

/**
 * Get score from compact match object
 * @param {Object} match - Compact match object
 * @param {string} type - 'similarity', 'color', 'shape', or 'texture'
 */
function getScore(match, type) {
    const idx = { similarity: 0, color: 1, shape: 2, texture: 3 };
    return match.s[idx[type]];
}

function clearStringCache() {
    stringCache.clear();
    console.log('[MEMORY] String cache cleared');
}

let currentChunk = 0;
const CHUNK_SIZE = 10000;
let totalMatchCount = 0;  // Total count across all chunks

function getChunkInfo() {
    const startIdx = currentChunk * CHUNK_SIZE;
    const endIdx = Math.min(startIdx + CHUNK_SIZE, matchResults.length);
    return {
        chunkNumber: currentChunk + 1,
        startIdx: startIdx,
        endIdx: endIdx,
        hasMore: endIdx < matchResults.length,
        totalLoaded: endIdx,
        totalResults: totalMatchCount || matchResults.length
    };
}

function loadNextChunk() {
    const info = getChunkInfo();
    if (info.hasMore) {
        currentChunk++;
        displayResults(true);  // Reset to page 1 when changing chunks
        console.log(`[CHUNKING] Loaded chunk ${info.chunkNumber + 1}`);
    }
}

function loadPreviousChunk() {
    if (currentChunk > 0) {
        currentChunk--;
        displayResults(true);  // Reset to page 1 when changing chunks
        console.log(`[CHUNKING] Loaded chunk ${currentChunk + 1}`);
    }
}

function navigateToChunk(chunkIndex) {
    const maxChunk = Math.ceil(matchResults.length / CHUNK_SIZE) - 1;
    if (chunkIndex >= 0 && chunkIndex <= maxChunk) {
        currentChunk = chunkIndex;
        displayResults(true);  // Reset to page 1 when changing chunks
        console.log(`[CHUNKING] Navigated to chunk ${currentChunk + 1}`);
    }
}

function resetChunking() {
    currentChunk = 0;
    totalMatchCount = 0;
}

// Retry configuration
const RETRY_CONFIG = {
    maxRetries: 3,
    initialDelay: 1000, // 1 second
    maxDelay: 10000, // 10 seconds
    backoffMultiplier: 2
};

const eventListeners = {
    historical: [],
    new: [],
    matching: [],
    results: [],
    tooltips: []
};

const blobUrls = new Set();
let lazyLoadObserver = null;
const metadataStatsCache = new WeakMap();
let stateCheckInterval = null;
let catalogPollingInterval = null;
let catalogChannel = null;
let blobUrlCleanupInterval = null;
let lastAutoBackupTime = 0;
const AUTO_BACKUP_DEBOUNCE_MS = 5 * 60 * 1000; // 5 minute window for batch replace operations

function cleanupDynamicFilterDropdownListeners(rootContainer = null) {
    const container = rootContainer || document.getElementById('dynamicFiltersContainer');
    if (!container) return 0;

    let removedCount = 0;
    const dropdownContainers = container.querySelectorAll('.searchable-dropdown-container');
    dropdownContainers.forEach((dropdownContainer) => {
        if (dropdownContainer._closeDropdown) {
            document.removeEventListener('click', dropdownContainer._closeDropdown);
            delete dropdownContainer._closeDropdown;
            removedCount++;
        }
    });

    return removedCount;
}

function clearOperationData() {
    historicalProducts = [];
    newProducts = [];
    matchResults = [];
    currentPage = 1;
    clearStringCache();    // Free interned strings from memory optimization
    clearMetadataCache();  // PERFORMANCE: Free metadata parse cache (prevents unbounded growth)
    clearAllDebounces();   // MEMORY LEAK PREVENTION: Clear debounce timers
    resetChunking();       // Reset chunking state
    serverResultsMode = false;
    currentMatchSessionId = null;
    currentMatchSessionSummary = null;
    currentMatchSessionTotal = 0;
    if (typeof resetServerResultsClientState === 'function') {
        resetServerResultsClientState();
    }

    if (typeof dynamicSearchResults !== 'undefined' && dynamicSearchResults) {
        dynamicSearchResults.clear();
    }

    const dynamicFiltersContainer = document.getElementById('dynamicFiltersContainer');
    if (dynamicFiltersContainer) {
        // Remove document-level listeners attached by searchable dropdowns
        cleanupDynamicFilterDropdownListeners(dynamicFiltersContainer);
        // Remove the entire container
        dynamicFiltersContainer.remove();
    }

    // MEMORY OPTIMIZATION: Clear filter criteria and debug flags (prevents 10-50MB accumulation)
    if (window.metadataFilterCriteria) {
        window.metadataFilterCriteria = {};
    }
    if (window.debugSort !== undefined) {
        window.debugSort = false;
    }

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


const PROCESSING_BENCHMARKS = {
    'batch_match': {
        'visual': 0.3,      // Mode 1: Matching with FAISS (fast)
        'metadata': 0.02,   // Mode 2: Metadata only (very fast)
        'hybrid': 0.35      // Mode 3: Both modes
    },
    'upload': {
        'visual': 0.8,      // Upload + feature extraction
        'metadata': 0.05,   // CSV-only metadata
        'hybrid': 1.0       // Upload + extraction + both modes
    }
};

/**
 * Start a smooth time-based progress tracker
 * IMPORTANT: This is purely visual - backend processes independently
 *
 * @param {string} containerId - ID of container for progress UI
 * @param {string} operationType - 'batch_match' or 'upload'
 * @param {string} mode - 'visual', 'metadata', or 'hybrid'
 * @param {number} itemCount - Number of items being processed
 * @returns {object} Tracker object with stop() and complete() methods
 */
function startProgressEstimation(containerId, operationType, mode, itemCount) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Progress container ${containerId} not found`);
        return null;
    }

    // Calculate estimated duration based on benchmarks
    const timePerItem = PROCESSING_BENCHMARKS[operationType]?.[mode] || 0.5;
    const baseOverhead = 2; // seconds for setup/teardown
    const estimatedSeconds = (itemCount * timePerItem) + baseOverhead;

    console.log(`[PROGRESS] Starting estimation for ${itemCount} items (${operationType}/${mode})`);
    console.log(`[PROGRESS] Estimated duration: ${Math.round(estimatedSeconds)}s (${timePerItem}s per item + ${baseOverhead}s overhead)`);

    // Create progress UI
    container.innerHTML = `
        <div class="progress-estimation">
            <h4>Processing ${itemCount} items...</h4>
            <div class="progress-bar-modern">
                <div class="progress-fill-modern"></div>
                <span class="progress-percentage">0%</span>
            </div>
            <div class="progress-time">
                <span class="time-elapsed">Elapsed: 0s</span>
                <span class="time-remaining">Est. remaining: ${Math.round(estimatedSeconds)}s</span>
            </div>
        </div>
    `;
    container.classList.add('show');

    const startTime = Date.now();
    const progressFill = container.querySelector('.progress-fill-modern');
    const progressPercent = container.querySelector('.progress-percentage');
    const timeElapsed = container.querySelector('.time-elapsed');
    const timeRemaining = container.querySelector('.time-remaining');

    // Update progress every 100ms for smooth animation
    // NOTE: This runs independently - doesn't affect backend processing
    const intervalId = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        const percentage = Math.min(95, (elapsed / estimatedSeconds) * 100); // Cap at 95%
        const remaining = Math.max(0, estimatedSeconds - elapsed);

        progressFill.style.width = `${percentage}%`;
        progressPercent.textContent = `${Math.round(percentage)}%`;
        timeElapsed.textContent = `Elapsed: ${formatSeconds(elapsed)}`;
        timeRemaining.textContent = `Est. remaining: ${formatSeconds(remaining)}`;
    }, 100);

    // MEMORY OPTIMIZATION: Add timeout to prevent orphaned progress trackers (CPU/memory spike)
    // Auto-stop tracker after 1 hour to prevent indefinite interval running
    const timeoutId = setTimeout(() => {
        if (intervalId) {
            clearInterval(intervalId);
            console.warn(`[PROGRESS] Tracker auto-stopped after 1 hour timeout (orphaned tracker cleanup)`);
        }
    }, 3600000);  // 1 hour in milliseconds

    return {
        intervalId,
        timeoutId,
        startTime,
        stop: () => {
            clearInterval(intervalId);
            clearTimeout(timeoutId);
            console.log(`[PROGRESS] Tracker stopped`);
        },
        complete: (successMessage) => {
            clearInterval(intervalId);
            clearTimeout(timeoutId);
            const totalTime = (Date.now() - startTime) / 1000;
            progressFill.style.width = '100%';
            progressPercent.textContent = '100%';
            container.querySelector('h4').textContent = successMessage || 'Complete!';
            timeElapsed.textContent = `Completed in ${formatSeconds(totalTime)}`;
            timeRemaining.textContent = '';
            console.log(`[PROGRESS] Completed in ${formatSeconds(totalTime)} (estimated: ${Math.round(estimatedSeconds)}s)`);
        }
    };
}

/**
 * Format seconds into human-readable time
 */
function formatSeconds(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
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
    if (typeof catalogPollingInterval !== 'undefined' && catalogPollingInterval) {
        clearInterval(catalogPollingInterval);
        catalogPollingInterval = null;
        console.log('✓ Cleared catalog polling interval');
    }

    // Clear blob URL cleanup interval (Memory Leak Fix #1)
    if (typeof blobUrlCleanupInterval !== 'undefined' && blobUrlCleanupInterval) {
        clearInterval(blobUrlCleanupInterval);
        blobUrlCleanupInterval = null;
        console.log('✓ Cleared blob URL cleanup interval');
    }

    // Stop mobile results polling and flag checking
    if (typeof stopMatchResultsPolling === 'function') {
        stopMatchResultsPolling();
    }
    if (typeof mobileFlagCheckInterval !== 'undefined' && mobileFlagCheckInterval) {
        clearInterval(mobileFlagCheckInterval);
        mobileFlagCheckInterval = null;
    }

    // Close BroadcastChannel (Fix #10)
    if (typeof catalogChannel !== 'undefined' && catalogChannel) {
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

    // Clear search timeout (Fix #3)
    if (window.searchTimeout) {
        clearTimeout(window.searchTimeout);
        window.searchTimeout = null;
        console.log('✓ Cleared search timeout');
    }

    // Disconnect lazy load observer
    if (lazyLoadObserver) {
        lazyLoadObserver.disconnect();
        lazyLoadObserver = null;
    }

    // Clear state arrays
    matchResults = [];
    serverResultsMode = false;
    currentMatchSessionId = null;
    currentMatchSessionSummary = null;
    currentMatchSessionTotal = 0;
    historicalFiles = [];
    newFiles = [];
    categoryMap = {};
    historicalProducts = [];
    newProducts = [];

    // Clear CSV state
    historicalCsv = null;
    newCsv = null;

    // Clear dynamic search results cache (Memory Leak Fix #2)
    if (typeof dynamicSearchResults !== 'undefined' && dynamicSearchResults) {
        dynamicSearchResults.clear();
    }

    // Clear mode state
    historicalMode = 'visual';
    newMode = 'visual';
    historicalAdvancedMode = false;
    newAdvancedMode = false;

    // Clear metadata schema
    if (window.metadataSchema) {
        window.metadataSchema = null;
    }

    // Clear matching state
    if (window.matchingInProgress) {
        window.matchingInProgress = false;
    }

    console.log('✓ Memory cleanup complete - cleared all state arrays and globals');
}

async function clearSavedState() {
    /**
     * Clear any saved application state (webview-specific)
     * This is safe and non-breaking - only affects temporary saved state, not database
     */
    try {
        // Clear sessionStorage if available
        if (typeof sessionStorage !== 'undefined') {
            sessionStorage.clear();
        }

        // Clear localStorage if available
        if (typeof localStorage !== 'undefined') {
            localStorage.removeItem('catalogState');
            localStorage.removeItem('matchingState');
            localStorage.removeItem('appState');
        }

        // If pywebview API is available, optionally call its clear method
        if (window.pywebview && window.pywebview.api && window.pywebview.api.clear_saved_state) {
            try {
                await window.pywebview.api.clear_saved_state();
            } catch (e) {
                // API might not exist, that's OK
                console.debug('Pywebview clear_saved_state not available');
            }
        }

        console.log('✓ Saved state cleared (no database operations)');
    } catch (error) {
        console.warn('Warning: Could not clear saved state:', error);
        // Don't throw - this is not critical
    }
}

function updateCsvWarning(section) {

    const isHistorical = section === 'historical';
    const advancedMode = isHistorical ? historicalAdvancedMode : newAdvancedMode;
    const csvLoaded = isHistorical ? historicalCsv : newCsv;
    const filesLoaded = isHistorical ? historicalFiles.length > 0 : newFiles.length > 0;

    const warningDiv = document.getElementById(
        isHistorical ? 'historicalCsvWarning' : 'newCsvWarning'
    );

    if (!warningDiv) return;

    // Show warning if: advanced mode AND no CSV AND files uploaded
    if (advancedMode && !csvLoaded && filesLoaded) {
        warningDiv.style.display = 'block';
        // Re-initialize icons in warning div only (scoped for performance)
        IconManager.reinit(50, warningDiv);
    } else {
        warningDiv.style.display = 'none';
    }
}

function updateSectionCollapseToggle(sectionId, isCollapsed) {
    const section = document.getElementById(sectionId);
    if (!section) return;

    const toggleBtn = section.querySelector('.section-collapse-toggle');
    if (!toggleBtn) return;

    toggleBtn.textContent = isCollapsed ? 'EXPAND' : 'COLLAPSE';
    toggleBtn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
    toggleBtn.title = isCollapsed ? 'Expand section' : 'Collapse section';
}

function setSectionCollapsed(sectionId, shouldCollapse) {
    const section = document.getElementById(sectionId);
    if (!section) return;

    section.classList.toggle('is-collapsed', shouldCollapse);
    section.setAttribute('data-collapsed', shouldCollapse ? 'true' : 'false');
    updateSectionCollapseToggle(sectionId, shouldCollapse);
}

function toggleSectionCollapse(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    const isCollapsed = section.classList.contains('is-collapsed');
    setSectionCollapsed(sectionId, !isCollapsed);
}

function showNewSectionAfterHistoricalStep() {
    const newSection = document.getElementById('newSection');
    debugLog('[DEBUG] Attempting to show newSection:', newSection);

    if (!newSection) {
        console.error('[ERROR] newSection element not found in DOM!');
        return;
    }

    newSection.style.display = 'block';
    newSection.style.visibility = 'visible';
    setSectionCollapsed('historicalSection', true);
    setSectionCollapsed('newSection', false);
    debugLog('[DEBUG] newSection display set to block, current style:', newSection.style.display);

    // Re-apply mode settings to ensure UI is synced after section becomes visible
    setMode('new', newMode);

    setTimeout(() => {
        newSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

function showResultsSectionWithCollapse() {
    const resultsSection = document.getElementById('resultsSection');
    if (!resultsSection) return;

    resultsSection.style.display = 'block';
    setSectionCollapsed('newSection', true);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Initialize
onDomReady(() => {
    initHistoricalUpload();
    initNewUpload();
    initMatching();
    initResults();
    initTooltips();
    initGPUStatus();
    initCatalogOptions();
    setSectionCollapsed('historicalSection', false);
    setSectionCollapsed('newSection', false);

    // Check for crash recovery
    checkForCrashRecovery();
});

// Expose state variables to window for child window handlers (defined in index.html)
// These are set by handleCsvBuilderComplete when CSV Builder sends data
Object.defineProperty(window, 'historicalCsv', {
    get: () => historicalCsv,
    set: (value) => { historicalCsv = value; }
});
Object.defineProperty(window, 'newCsv', {
    get: () => newCsv,
    set: (value) => { newCsv = value; }
});
Object.defineProperty(window, 'historicalMode', {
    get: () => historicalMode
});
Object.defineProperty(window, 'newMode', {
    get: () => newMode
});

// Expose cleanup functions for child window handlers
window.removeWorkflowIndicators = removeWorkflowIndicators;
window.refreshCatalogInfo = typeof refreshCatalogInfo === 'function' ? refreshCatalogInfo : () => { };

// Cleanup on page unload (browser mode - prevents memory leaks)
// In webview mode, main window stays open so this rarely fires
window.addEventListener('beforeunload', () => {
    cleanupMemory();
});

// Also cleanup on pagehide for mobile browsers
window.addEventListener('pagehide', () => {
    cleanupMemory();
});

// Native folder selection helper for pywebview
async function selectFolderNative(handleFilesCallback) {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.select_folder) {
        try {
            const filesInfo = await window.pywebview.api.select_folder();
            if (filesInfo && filesInfo.length > 0) {
                // MEMORY OPTIMIZATION: Don't load images into memory
                // Instead, create file-like objects with paths for backend processing
                const files = filesInfo.map((info) => {
                    // Create a minimal file-like object with only necessary metadata
                    // The path will be used to read the file directly from disk on the backend
                    const file = {
                        name: info.name,
                        type: 'image/' + info.name.split('.').pop().toLowerCase(),
                        path: info.path,  // Absolute file path - backend reads directly from disk
                        size: info.size
                    };
                    // Add webkitRelativePath for category detection (needed by handleHistoricalFiles)
                    file.webkitRelativePath = info.relativePath;
                    return file;
                });
                Promise.resolve(handleFilesCallback(files)).catch((err) => {
                    console.error('Error handling native-selected files:', err);
                    showToast('Failed to process selected files', 'error');
                });
            }
        } catch (e) {
            console.error('Native folder selection error:', e);
            showToast('Error selecting folder: ' + e.message, 'error');
        }
        return true; // Handled natively
    }
    return false; // Fall back to HTML input
}

// Historical Catalog Upload
function initHistoricalUpload() {
    const dropZone = document.getElementById('historicalDropZone');
    const input = document.getElementById('historicalInput');
    const browseBtn = document.getElementById('historicalBrowseBtn');
    const csvInput = document.getElementById('historicalCsvInput');
    const processBtn = document.getElementById('processHistoricalBtn');

    // Use tracked listeners to prevent memory leaks
    addTrackedListener(browseBtn, 'click', async (e) => {
        e.stopPropagation();
        // Try native folder selection first (for pywebview)
        const handled = await selectFolderNative((files) => {
            showUploadLoadingState('historicalInfo', `Processing ${files.length.toLocaleString()} selected files...`);
            return handleHistoricalFiles(files);
        });
        if (!handled) {
            input.click();
        }
    }, 'historical');

    addTrackedListener(dropZone, 'click', async () => {
        const handled = await selectFolderNative((files) => {
            showUploadLoadingState('historicalInfo', `Processing ${files.length.toLocaleString()} selected files...`);
            return handleHistoricalFiles(files);
        });
        if (!handled) {
            input.click();
        }
    }, 'historical');

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
        const files = Array.from(e.dataTransfer.files);
        showUploadLoadingState('historicalInfo', `Processing ${files.length.toLocaleString()} dropped files...`);
        setTimeout(() => {
            void handleHistoricalFiles(files);
        }, 0);
    }, 'historical');

    addTrackedListener(input, 'change', (e) => {
        const files = Array.from(e.target.files);
        showUploadLoadingState('historicalInfo', `Processing ${files.length.toLocaleString()} selected files...`);
        setTimeout(() => {
            void handleHistoricalFiles(files);
        }, 0);
    }, 'historical');

    addTrackedListener(csvInput, 'change', async (e) => {
        if (e.target.files.length) {
            historicalCsv = e.target.files[0];
            showToast('CSV loaded for historical products', 'success');

            // IMMEDIATE DETECTION: Parse CSV immediately to populate weight sliders
            // This ensures sliders appear even if we don't click "Process" (e.g. for "use existing")
            try {
                const map = await parseCsv(historicalCsv);
                if (map && Object.keys(map).length > 0) {
                    await loadMetadataSchema();
                    console.log('[HIST-CSV] Metadata schema extracted and sliders populated');
                }
            } catch (err) {
                console.warn('[HIST-CSV] Immediate schema extraction failed:', err);
            }

            // Enable process button in advanced mode when CSV is uploaded
            if (historicalAdvancedMode) {
                processBtn.disabled = false;
            }

            // Update CSV warning when CSV is loaded
            updateCsvWarning('historical');
        }
    }, 'historical');

    // Don't set up the click handler here - it's handled by processHistoricalCatalogWithOptions
    // processBtn.addEventListener('click', processHistoricalCatalog);
}

async function handleHistoricalFiles(files) {
    const imageFiles = files.filter(isImageLikeFile);

    if (imageFiles.length === 0) {
        showToast('No image files found in folder', 'error');
        return;
    }

    if (imageFiles.length > MAX_UPLOAD_FILES) {
        showToast(`Too many files selected (${imageFiles.length.toLocaleString()}). Maximum supported is ${MAX_UPLOAD_FILES.toLocaleString()}.`, 'error');
        return;
    }

    // Extract categories from folder structure
    const filesWithCategories = imageFiles.map(file => {
        const category = extractCategoryFromPath(file.webkitRelativePath || file.name);
        return { file, category };
    });

    historicalFiles = filesWithCategories;

    // Count categories (yield for very large folders so UI stays responsive)
    const categoryCount = await countCategoriesWithYield(filesWithCategories);

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';

    const info = document.getElementById('historicalInfo');
    const displayLimit = 50;
    const hasMore = imageFiles.length > displayLimit;

    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearFolderUpload('historical')" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>${imageFiles.length} images loaded</h4>
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
    } else {
        // In simple mode, enable immediately
        document.getElementById('processHistoricalBtn').disabled = false;
    }

    // Update CSV warning
    updateCsvWarning('historical');

    showToast(`${imageFiles.length} historical images loaded from ${Object.keys(categoryCount).length || 0} categories`, 'success');
}

async function processHistoricalCatalog() {
    const statusDiv = document.getElementById('historicalStatus');
    const processBtn = document.getElementById('processHistoricalBtn');

    // Disable button immediately to prevent double-clicks
    if (processBtn.disabled) {
        console.warn('[GUARD] Historical catalog already processing, ignoring duplicate call');
        return;
    }

    statusDiv.classList.add('show');
    processBtn.disabled = true;
    showLoadingSpinner(processBtn, true);

    // Parse CSV if provided
    let categoryMap = {};
    if (historicalCsv) {
        try {
            categoryMap = await parseCsv(historicalCsv);
            // Load metadata schema for dynamic weight sliders
            await loadMetadataSchema();
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

    // Determine mode for progress tracker
    const mode = csvOnlyMode ? 'metadata' : historicalMode;

    // Start progress estimation (VISUAL ONLY - doesn't affect backend)
    const tracker = startProgressEstimation('historicalStatus', 'upload', mode, totalItems);

    // Load existing products from DB if using "add_to_existing" option
    const loadOption = getCatalogLoadOption();
    // MEMORY OPTIMIZATION: Load historical products with pagination
    let historicalProductsTotal = 0;  // Track total count
    let historicalProductsPage = 1;   // Track current page

    if (loadOption === 'add_to_existing') {
        try {
            console.log('[ADD_TO_EXISTING] Loading existing historical products from DB (first page)...');
            // Load first page only (50 products) to prevent memory bloat
            const response = await fetch('/api/catalog/products?type=historical&page=1&limit=50');
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
                historicalProductsTotal = data.total || historicalProducts.length;
                historicalProductsPage = 1;
                console.log(`[ADD_TO_EXISTING] Loaded ${historicalProducts.length} of ${historicalProductsTotal} existing products`);
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

    let successCount = 0;
    let failedCount = 0;
    const failedItems = [];
    let totalFeaturesExtracted = 0;
    let totalFeaturesFailed = 0;
    const uploadTransportsUsed = new Set();
    const extractionProfilesUsed = new Set();

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
        const progressText = statusDiv.querySelector('h4');

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
                    recordFailedItem(failedItems, { row: i + 1, fileName, reason: 'Missing SKU or Name' });
                    continue;
                }

                // Add to batch
                // Extract standard fields and separate extra metadata
                const { sku, name, category: _cat, price, performance, ...otherMetadata } = metadata;

                // Format price
                let numericPrice = null;
                if (price !== undefined && price !== null && price !== '') {
                    const parsed = parseFloat(String(price).replace(/[^0-9.-]+/g, ''));
                    if (!isNaN(parsed)) {
                        numericPrice = parsed;
                    }
                }

                // Format performance history (backend expects list for history)
                let performanceHistory = [];
                if (performance !== undefined && performance !== null && performance !== '') {
                    const parsed = parseFloat(String(performance).replace(/[^0-9.-]+/g, ''));
                    if (!isNaN(parsed)) {
                        performanceHistory = [parsed];
                    }
                }

                // Add to batch
                productsToCreate.push({
                    sku: sku,
                    product_name: name || fileName,
                    category: category,
                    price: numericPrice,
                    performance_history: performanceHistory,
                    metadata: otherMetadata,
                    is_historical: true
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
                        recordFailedItem(failedItems, { row: 'batch', fileName: 'all', reason: errorMsg });
                        console.error(`[BATCH-METADATA] Batch ${batchIdx + 1}/${totalBatches} failed:`, data);
                    }
                }

                console.log(`[BATCH-METADATA] ✓ Successfully created ${successCount} products in ${totalBatches} batches`);
            }
        } catch (error) {
            failedCount += csvOnlyItems.length;
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            recordFailedItem(failedItems, { row: 'batch', fileName: 'all', reason: error.message });
            console.error('[BATCH-METADATA] Batch creation error:', error);
        }
    }

    // Process image items in batch (Mode 1/3) - GPU batch processing
    if (imageItems.length > 0) {
        debugLog(`[BATCH-UPLOAD] Preparing to batch upload ${imageItems.length} images`);

        try {
            // OPTIMIZATION: Stream batch uploads every 100 images
            // This overlaps file I/O with network requests instead of waiting for all files to load
            const selectedUploadFiles = imageItems.map((idx) => historicalFiles[idx]);
            const uploadBatchSize = getAdaptiveUploadBatchSize(selectedUploadFiles);
            const totalBatches = Math.ceil(imageItems.length / uploadBatchSize);

            debugLog(`[BATCH-UPLOAD] Streaming ${imageItems.length} images in ${totalBatches} batch(es) of ${uploadBatchSize}`);

            // Process each batch
            for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                const batchStart = batchIdx * uploadBatchSize;
                const batchEnd = Math.min(batchStart + uploadBatchSize, imageItems.length);
                const batchItems = imageItems.slice(batchStart, batchEnd);

                debugLog(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches}: Preparing ${batchItems.length} images`);

                try {
                    let batchFormData = new FormData();
                    const batchFiles = batchItems.map((idx) => historicalFiles[idx]);
                    const transportMode = appendBatchUploadPayload(
                        batchFormData,
                        batchFiles,
                        categoryMap,
                        true,
                        imageItems.length,
                        batchIdx === totalBatches - 1
                    );
                    uploadTransportsUsed.add(transportMode);

                    debugLog(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches}: Sending ${batchItems.length} images`);
                    const requestTimeoutMs = getUploadRequestTimeoutMs(batchFiles, transportMode);

                    // Send this batch
                    const response = await fetchWithRetry('/api/products/batch-upload', {
                        method: 'POST',
                        body: batchFormData,
                        timeoutMs: requestTimeoutMs
                    });

                    // PERFORMANCE FIX: Removed debug logging to reduce network overhead during batch operations

                    let data;
                    try {
                        data = await response.json();
                    } catch (jsonError) {
                        throw new Error(`Failed to parse JSON response: ${jsonError.message}. Response status: ${response.status}`);
                    }

                    // MEMORY OPTIMIZATION: Clear FormData after request to prevent accumulation (10-50MB per batch)
                    batchFormData = null;

                    // PERFORMANCE FIX: Removed debug logging to reduce network overhead during batch operations

                    if (response.ok) {
                        // PERFORMANCE FIX: Removed debug logging to reduce network overhead during batch operations

                        debugLog(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches} response:`, {
                            total: data.total,
                            successful: data.successful,
                            failed: data.failed,
                            skipped: data.skipped
                        });

                        // SIMPLIFIED: Just trust the backend counts instead of complex index mapping
                        // The backend already calculated success/failure correctly

                        // Add backend's success count to our running total
                        const batchSuccessCount = data.successful || 0;
                        const batchFailedCount = (data.failed || 0) + (data.skipped || 0); // Skipped = failed for UI
                        totalFeaturesExtracted += data.features_extracted || 0;
                        totalFeaturesFailed += data.features_failed || 0;
                        if (data.processing_profile_used) {
                            extractionProfilesUsed.add(data.processing_profile_used);
                        }

                        successCount += batchSuccessCount;
                        failedCount += batchFailedCount;

                        // Process results to populate historicalProducts array for successful items
                        if (data.results) {
                            for (const result of data.results) {
                                if (result.status === 'success' && result.product_id) {
                                    // Find the corresponding file using the index
                                    const relativeIdx = result.index !== undefined ? result.index : 0;
                                    if (relativeIdx >= 0 && relativeIdx < batchItems.length) {
                                        const batchItemIdx = batchItems[relativeIdx];
                                        const fileObj = historicalFiles[batchItemIdx];
                                        const metadata = categoryMap[fileObj.file.name] || {};

                                        historicalProducts.push({
                                            id: result.product_id,
                                            filename: result.filename || fileObj.file.name,
                                            category: metadata.category || fileObj.category,
                                            sku: metadata.sku,
                                            name: metadata.name || fileObj.file.name,
                                            hasFeatures: true,
                                            hasPriceHistory: false
                                        });
                                    }
                                } else if (result.status === 'skipped' || result.status === 'failed') {
                                    // Collect failed items for detailed error display
                                    const relativeIdx = result.index !== undefined ? result.index : 0;
                                    if (relativeIdx >= 0 && relativeIdx < batchItems.length) {
                                        const batchItemIdx = batchItems[relativeIdx];
                                        const fileObj = historicalFiles[batchItemIdx];
                                        recordFailedItem(failedItems, {
                                            row: batchItemIdx + 1,
                                            fileName: result.filename || fileObj.file.name,
                                            reason: result.reason || result.error || 'Unknown error'
                                        });
                                    }
                                }
                            }
                        }
                    } else {
                        throw new Error(data.error || `Server returned ${response.status}`);
                    }
                } catch (error) {
                    // This batch failed - mark items as failed but CONTINUE to next batch
                    // PERFORMANCE FIX: Removed debug logging to reduce network overhead during batch operations
                    console.error(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches} failed:`, error);
                    for (const i of batchItems) {
                        const fileObj = historicalFiles[i];
                        failedCount++;
                        recordFailedItem(failedItems, { row: i + 1, fileName: fileObj.file.name, reason: error.message || 'Batch processing failed' });
                    }
                    // Continue to next batch
                }
            }
        } catch (error) {
            // Critical error in batch setup (very rare)
            console.error(`[BATCH-UPLOAD] Critical error:`, error);
            if (tracker) tracker.stop();
        }
    }

    // Complete progress tracker (backend finished, jump to 100%)
    if (tracker) {
        tracker.complete(`Successfully processed ${successCount} historical items!`);
    }

    const catalogOption = getCatalogLoadOption();
    const existingCount = catalogOption === 'add_to_existing' ? historicalProducts.filter(p => p.id).length - totalItems : 0;
    const newlyUploaded = historicalProducts.length - existingCount;
    const withoutMetadata = historicalProducts.filter(p => !p.category && !p.sku).length;

    let statusMsg = `<h4>Historical catalog processed</h4>`;
    statusMsg += `<p><strong>${successCount} successful</strong>, ${failedCount} failed</p>`;
    if (totalFeaturesExtracted > 0 || totalFeaturesFailed > 0) {
        statusMsg += `<p>Feature extraction: ${totalFeaturesExtracted} extracted, ${totalFeaturesFailed} failed</p>`;
    }
    if (totalFeaturesFailed > 0) {
        statusMsg += `<p style="color: #ed8936;">Warning: ${totalFeaturesFailed} images failed feature extraction</p>`;
    }
    if (extractionProfilesUsed.has('fast')) {
        statusMsg += `<p style="color: #2b6cb0;">Auto Fast Mode enabled for this upload (optimized CLIP preprocessing).</p>`;
    }
    if (uploadTransportsUsed.has('direct_upload')) {
        statusMsg += `<p style="color: #4a5568; font-size: 12px;">Browser file mode detected: using direct upload fallback (no native file paths available).</p>`;
    }

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
    } else if (failedCount > 0) {
        const detailsShown = Math.min(failedItems.length, MAX_FAILED_ITEM_DETAILS);
        const detailsNote = failedCount > detailsShown
            ? ` (showing first ${detailsShown})`
            : '';
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>${failedCount} items failed</strong>${detailsNote}</div>`;
        if (failedItems.length > 0) {
            debugLog('Failed item samples:', failedItems);
        }
    }

    statusDiv.innerHTML = statusMsg;

    showToast(`Historical catalog ready: ${successCount} products`, 'success');
    showLoadingSpinner(processBtn, false);

    // MEMORY OPTIMIZATION: Clear operation data to free 50-100MB
    clearOperationData();

    // Show next step
    showNewSectionAfterHistoricalStep();
}

function initNewUpload() {
    const dropZone = document.getElementById('newDropZone');
    const input = document.getElementById('newInput');
    const browseBtn = document.getElementById('newBrowseBtn');
    const csvInput = document.getElementById('newCsvInput');
    const processBtn = document.getElementById('processNewBtn');

    // Use tracked listeners to prevent memory leaks
    addTrackedListener(browseBtn, 'click', async (e) => {
        e.stopPropagation();
        // Try native folder selection first (for pywebview)
        const handled = await selectFolderNative((files) => {
            showUploadLoadingState('newInfo', `Processing ${files.length.toLocaleString()} selected files...`);
            return handleNewFiles(files);
        });
        if (!handled) {
            input.click();
        }
    }, 'new');

    addTrackedListener(dropZone, 'click', async () => {
        const handled = await selectFolderNative((files) => {
            showUploadLoadingState('newInfo', `Processing ${files.length.toLocaleString()} selected files...`);
            return handleNewFiles(files);
        });
        if (!handled) {
            input.click();
        }
    }, 'new');

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
        const files = Array.from(e.dataTransfer.files);
        showUploadLoadingState('newInfo', `Processing ${files.length.toLocaleString()} dropped files...`);
        setTimeout(() => {
            void handleNewFiles(files);
        }, 0);
    }, 'new');

    addTrackedListener(input, 'change', (e) => {
        const files = Array.from(e.target.files);
        showUploadLoadingState('newInfo', `Processing ${files.length.toLocaleString()} selected files...`);
        setTimeout(() => {
            void handleNewFiles(files);
        }, 0);
    }, 'new');

    addTrackedListener(csvInput, 'change', async (e) => {
        if (e.target.files.length) {
            newCsv = e.target.files[0];
            showToast('CSV loaded for new products', 'success');

            // IMMEDIATE DETECTION: Parse CSV immediately to populate weight sliders
            try {
                const map = await parseCsv(newCsv);
                if (map && Object.keys(map).length > 0) {
                    await loadMetadataSchema();
                    console.log('[NEW-CSV] Metadata schema extracted and sliders populated');
                }
            } catch (err) {
                console.warn('[NEW-CSV] Immediate schema extraction failed:', err);
            }

            // Enable process button in advanced mode when CSV is uploaded
            if (newAdvancedMode) {
                processBtn.disabled = false;
            }

            // Update CSV warning when CSV is loaded
            updateCsvWarning('new');
        }
    }, 'new');

}

async function handleNewFiles(files) {
    const imageFiles = files.filter(isImageLikeFile);

    if (imageFiles.length === 0) {
        showToast('No image files found in folder', 'error');
        return;
    }

    if (imageFiles.length > MAX_UPLOAD_FILES) {
        showToast(`Too many files selected (${imageFiles.length.toLocaleString()}). Maximum supported is ${MAX_UPLOAD_FILES.toLocaleString()}.`, 'error');
        return;
    }

    // Extract categories from folder structure
    const filesWithCategories = imageFiles.map(file => {
        const category = extractCategoryFromPath(file.webkitRelativePath || file.name);
        return { file, category };
    });

    newFiles = filesWithCategories;

    // Count categories (yield for very large folders so UI stays responsive)
    const categoryCount = await countCategoriesWithYield(filesWithCategories);

    const categorySummary = Object.keys(categoryCount).length > 0
        ? `<div style="margin-top: 10px;"><strong>Categories found:</strong> ${Object.entries(categoryCount).map(([cat, count]) => `${cat} (${count})`).join(', ')}</div>`
        : '<div style="margin-top: 10px; color: #ed8936;">No subfolders detected - all images will be uncategorized</div>';

    const info = document.getElementById('newInfo');
    const displayLimit = 50;
    const hasMore = imageFiles.length > displayLimit;

    info.innerHTML = `
        <button class="btn clear-btn" onclick="clearFolderUpload('new')" data-tooltip="Clear uploaded folder and start over">CLEAR</button>
        <h4>${imageFiles.length} images loaded</h4>
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
    } else {
        // In simple mode, enable immediately
        document.getElementById('processNewBtn').disabled = false;
    }

    // Update CSV warning
    updateCsvWarning('new');

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
            // Load metadata schema for dynamic weight sliders
            await loadMetadataSchema();
        } catch (error) {
            showToast('Failed to parse CSV file. Please check the format.', 'error');
            processBtn.disabled = false;
            showLoadingSpinner(processBtn, false);
            return;
        }
    }

    const hasImageFiles = newFiles && newFiles.length > 0;
    const hasCsvData = Object.keys(categoryMap).length > 0;
    const csvOnlyMode = !hasImageFiles && hasCsvData && newAdvancedMode;

    const itemsToProcess = hasImageFiles ? newFiles : (csvOnlyMode ? Object.keys(categoryMap) : []);
    const totalItems = itemsToProcess.length;

    // Determine mode for progress tracker
    const mode = csvOnlyMode ? 'metadata' : newMode;

    // Start progress estimation (VISUAL ONLY - doesn't affect backend)
    const tracker = startProgressEstimation('newStatus', 'upload', mode, totalItems);

    // Load existing products from DB if using "add_to_existing" option
    const newLoadOption = getNewCatalogLoadOption();
    let newProductsTotal = 0;
    if (newLoadOption === 'add_to_existing') {
        try {
            console.log('[ADD_TO_EXISTING] Loading existing new products from DB (first page)...');
            // Load first page only to avoid large in-memory arrays on huge catalogs.
            const response = await fetch('/api/catalog/products?type=new&page=1&limit=50');
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
                newProductsTotal = data.total || newProducts.length;
                console.log(`[ADD_TO_EXISTING] Loaded ${newProducts.length} of ${newProductsTotal} existing new products`);
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

    let successCount = 0;
    let failedCount = 0;
    const failedItems = [];
    let totalFeaturesExtracted = 0;
    let totalFeaturesFailed = 0;
    const uploadTransportsUsed = new Set();
    const extractionProfilesUsed = new Set();

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
                    recordFailedItem(failedItems, { row: i + 1, fileName, reason: 'Missing SKU or Name' });
                    continue;
                }

                // Prepare dynamic metadata
                const dynamicMeta = { ...metadata };
                delete dynamicMeta.sku;
                delete dynamicMeta.name; // This is product_name
                delete dynamicMeta.category;

                // Add to batch
                productsToCreate.push({
                    sku: metadata.sku,
                    product_name: metadata.name || fileName,
                    category: category,
                    is_historical: false,
                    ...dynamicMeta // Include all other dynamic fields
                });
                itemIndexMap.push({ i, fileName, metadata, category });
            }

            if (productsToCreate.length === 0) {
                console.warn('[BATCH-METADATA] No valid products to create');
            } else {
                // Step 2: Stream metadata creation in chunks to avoid massive request payloads.
                const STREAM_BATCH_SIZE = 100;
                const totalBatches = Math.ceil(productsToCreate.length / STREAM_BATCH_SIZE);
                const progressText = statusDiv.querySelector('h4');

                console.log(
                    `[BATCH-METADATA] Streaming ${productsToCreate.length} products in ${totalBatches} batch(es) of ${STREAM_BATCH_SIZE}`
                );

                for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                    const batchStart = batchIdx * STREAM_BATCH_SIZE;
                    const batchEnd = Math.min(batchStart + STREAM_BATCH_SIZE, productsToCreate.length);
                    const batchProducts = productsToCreate.slice(batchStart, batchEnd);

                    if (progressText) {
                        progressText.textContent =
                            `Creating metadata batch ${batchIdx + 1}/${totalBatches} (${batchProducts.length} products)...`;
                    }

                    const response = await fetchWithRetry('/api/products/metadata/batch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ products: batchProducts })
                    });

                    const data = await response.json();

                    if (response.ok && data.product_ids) {
                        successCount += data.product_ids.length;

                        for (let j = 0; j < data.product_ids.length; j++) {
                            const productId = data.product_ids[j];
                            const itemInfo = itemIndexMap[batchStart + j];

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
                    } else {
                        failedCount += batchProducts.length;
                        const errorMsg = getUserFriendlyError(data.error_code || 'BATCH_ERROR', data.error, data.suggestion);
                        recordFailedItem(failedItems, { row: 'batch', fileName: 'all', reason: errorMsg });
                        console.error(`[BATCH-METADATA] Batch ${batchIdx + 1}/${totalBatches} failed:`, data);
                    }
                }

                console.log(`[BATCH-METADATA] ✓ Successfully created ${successCount} products in ${totalBatches} batches`);
            }
        } catch (error) {
            failedCount += csvOnlyItems.length;
            const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
            recordFailedItem(failedItems, { row: 'batch', fileName: 'all', reason: error.message });
            console.error('[BATCH-METADATA] Batch creation error:', error);
        }
    }

    if (imageItems.length > 0) {
        debugLog(`[BATCH-UPLOAD] Preparing to batch upload ${imageItems.length} images`);

        try {
            const selectedUploadFiles = imageItems.map((idx) => newFiles[idx]);
            const uploadBatchSize = getAdaptiveUploadBatchSize(selectedUploadFiles);
            const totalBatches = Math.ceil(imageItems.length / uploadBatchSize);
            debugLog(`[BATCH-UPLOAD] Streaming ${imageItems.length} images in ${totalBatches} batch(es) of ${uploadBatchSize}`);

            for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                const batchStart = batchIdx * uploadBatchSize;
                const batchEnd = Math.min(batchStart + uploadBatchSize, imageItems.length);
                const batchItems = imageItems.slice(batchStart, batchEnd);

                try {
                    let batchFormData = new FormData();
                    const batchFiles = batchItems.map((idx) => newFiles[idx]);
                    const transportMode = appendBatchUploadPayload(
                        batchFormData,
                        batchFiles,
                        categoryMap,
                        false,
                        imageItems.length,
                        batchIdx === totalBatches - 1
                    );
                    uploadTransportsUsed.add(transportMode);

                    debugLog(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches}: Sending ${batchItems.length} images`);
                    const requestTimeoutMs = getUploadRequestTimeoutMs(batchFiles, transportMode);

                    const response = await fetchWithRetry('/api/products/batch-upload', {
                        method: 'POST',
                        body: batchFormData,
                        timeoutMs: requestTimeoutMs
                    });

                    const data = await response.json();
                    batchFormData = null;

                    if (response.ok) {
                        const batchSuccessCount = data.successful || 0;
                        const batchFailedCount = (data.failed || 0) + (data.skipped || 0);
                        totalFeaturesExtracted += data.features_extracted || 0;
                        totalFeaturesFailed += data.features_failed || 0;
                        if (data.processing_profile_used) {
                            extractionProfilesUsed.add(data.processing_profile_used);
                        }

                        successCount += batchSuccessCount;
                        failedCount += batchFailedCount;

                        if (data.results) {
                            for (const result of data.results) {
                                if (result.status === 'success' && result.product_id) {
                                    const relativeIdx = result.index !== undefined ? result.index : 0;
                                    if (relativeIdx >= 0 && relativeIdx < batchItems.length) {
                                        const itemIdx = batchItems[relativeIdx];
                                        const fileObj = newFiles[itemIdx];
                                        const metadata = categoryMap[fileObj.file.name] || {};

                                        newProducts.push({
                                            id: result.product_id,
                                            filename: result.filename || fileObj.file.name,
                                            category: metadata.category || fileObj.category,
                                            sku: metadata.sku,
                                            name: metadata.name || fileObj.file.name,
                                            hasFeatures: true,
                                            hasPriceHistory: false
                                        });
                                    }
                                } else if (result.status === 'skipped' || result.status === 'failed') {
                                    const relativeIdx = result.index !== undefined ? result.index : 0;
                                    if (relativeIdx >= 0 && relativeIdx < batchItems.length) {
                                        const itemIdx = batchItems[relativeIdx];
                                        const fileObj = newFiles[itemIdx];
                                        recordFailedItem(failedItems, {
                                            row: itemIdx + 1,
                                            fileName: result.filename || fileObj.file.name,
                                            reason: result.reason || result.error || 'Unknown error'
                                        });
                                    }
                                }
                            }
                        }
                    } else {
                        throw new Error(data.error || `Server returned ${response.status}`);
                    }
                } catch (error) {
                    console.error(`[BATCH-UPLOAD] Batch ${batchIdx + 1}/${totalBatches} failed:`, error);
                    for (const i of batchItems) {
                        const fileObj = newFiles[i];
                        failedCount++;
                        recordFailedItem(failedItems, { row: i + 1, fileName: fileObj.file.name, reason: error.message || 'Batch processing failed' });
                    }
                }
            }
        } catch (error) {
            console.error(`[BATCH-UPLOAD] Critical error:`, error);
            if (tracker) tracker.stop();
        }
    }

    // Complete progress tracker (backend finished, jump to 100%)
    if (tracker) {
        tracker.complete(`Successfully processed ${successCount} new products!`);
    }

    // Continue with the rest of the function (status display, etc.)
    const existingCount = newLoadOption === 'add_to_existing' ? newProducts.filter(p => p.id).length - totalItems : 0;
    const newlyUploaded = newProducts.length - existingCount;
    const withoutMetadata = newProducts.filter(p => !p.category && !p.sku).length;

    let statusMsg = `<h4>New products processed</h4>`;
    statusMsg += `<p><strong>${successCount} successful</strong>, ${failedCount} failed</p>`;
    if (totalFeaturesExtracted > 0 || totalFeaturesFailed > 0) {
        statusMsg += `<p>Feature extraction: ${totalFeaturesExtracted} extracted, ${totalFeaturesFailed} failed</p>`;
    }
    if (totalFeaturesFailed > 0) {
        statusMsg += `<p style="color: #ed8936;">Warning: ${totalFeaturesFailed} images failed feature extraction</p>`;
    }
    if (extractionProfilesUsed.has('fast')) {
        statusMsg += `<p style="color: #2b6cb0;">Auto Fast Mode enabled for this upload (optimized CLIP preprocessing).</p>`;
    }
    if (uploadTransportsUsed.has('direct_upload')) {
        statusMsg += `<p style="color: #4a5568; font-size: 12px;">Browser file mode detected: using direct upload fallback (no native file paths available).</p>`;
    }

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
    } else if (failedCount > 0) {
        const detailsShown = Math.min(failedItems.length, MAX_FAILED_ITEM_DETAILS);
        const detailsNote = failedCount > detailsShown
            ? ` (showing first ${detailsShown})`
            : '';
        statusMsg += `<div style="margin-top: 10px; color: #ed8936; font-size: 12px;"><strong>${failedCount} items failed</strong>${detailsNote}</div>`;
        if (failedItems.length > 0) {
            debugLog('Failed item samples:', failedItems);
        }
    }

    statusDiv.innerHTML = statusMsg;

    showToast(`New products ready: ${successCount} products`, 'success');
    showLoadingSpinner(processBtn, false);

    if (newLoadOption === 'replace') {
        newFiles = [];
        newCsv = null;
        categoryMap = {};
    }

    processBtn.disabled = false;

    // MEMORY OPTIMIZATION: Clear operation data to free 50-100MB
    clearOperationData();

    // Show matching section - force display
    const matchSection = document.getElementById('matchSection');
    debugLog('[DEBUG] Attempting to show matchSection:', matchSection);
    if (matchSection) {
        matchSection.style.display = 'block';
        matchSection.style.visibility = 'visible';
        debugLog('[DEBUG] matchSection display set to block, current style:', matchSection.style.display);
        setTimeout(() => {
            matchSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    } else {
        console.error('[ERROR] matchSection element not found in DOM!');
    }
}

function initMatching() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const matchBtn = document.getElementById('matchBtn');

    addTrackedListener(thresholdSlider, 'input', (e) => {
        thresholdValue.textContent = e.target.value;
    }, 'matching');

    addTrackedListener(matchBtn, 'click', startMatching, 'matching');
}

function appendBatchResults(batchResults) {
    if (!Array.isArray(batchResults) || batchResults.length === 0) {
        return 0;
    }

    let appended = 0;
    for (let i = 0; i < batchResults.length; i++) {
        const result = batchResults[i];

        // Use product_data provided directly in the batch result.
        let product = result.product_data;

        if (!product) {
            // Fallback: create minimal product object if details are missing.
            debugWarn(`[BATCH-MATCHING] Product ${result.product_id} data not found in response, using minimal object`);
            product = {
                id: result.product_id,
                filename: `Product ${result.product_id}`,
                name: `Product ${result.product_id}`,
                category: 'Unknown',
                sku: '',
                hasFeatures: false,
                metadata: {}
            };
        } else {
            product.hasFeatures = product.has_features || false;
            if (!product.name) product.name = product.product_name || product.filename || `Product ${product.id}`;
        }

        // Deduplicate matches to prevent duplicate entries.
        const rawMatches = result.matches || [];
        const seenMatchIds = new Set();
        const uniqueMatches = [];

        for (const m of rawMatches) {
            const mid = m.product_id || m.mid || m.id;
            if (mid && !seenMatchIds.has(mid)) {
                seenMatchIds.add(mid);
                uniqueMatches.push(m);
            }
        }

        const compactMatches = uniqueMatches.map(m => createCompactMatch(m));
        const compactProduct = createCompactProduct(product);

        const resultObj = {
            p: compactProduct,
            m: compactMatches,
            summary_stats: result.summary_stats
        };

        if (result.status !== 'success' && result.error) {
            resultObj.err = result.error;
        }

        matchResults.push(resultObj);
        appended++;
    }

    return appended;
}

async function loadBatchResultsFromSession(sessionId) {
    const pageSize = 250;
    let page = 1;
    let totalPages = 1;
    let totalResults = 0;
    let loaded = 0;

    while (page <= totalPages) {
        const url = `/api/products/match-results/session?session_id=${encodeURIComponent(sessionId)}&page=${page}&limit=${pageSize}`;
        const response = await fetchWithRetry(url, { method: 'GET' });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || `Failed to fetch session results (page ${page})`);
        }

        if (page === 1) {
            totalResults = data.total_results || 0;
            totalPages = data.total_pages || 1;
            totalMatchCount = totalResults || 0;
        }

        const pageResults = data.results || [];
        loaded += appendBatchResults(pageResults);

        if (page === totalPages || page % 5 === 0) {
            debugLog(`[BATCH-MATCHING] Session ${sessionId}: loaded ${loaded}/${totalResults || loaded} results`);
        }

        page++;
    }

    return { loaded, totalResults };
}

async function startMatching() {
    debugLog('[MATCHING] startMatching() called');
    const threshold = parseInt(document.getElementById('thresholdSlider').value);
    const limit = parseInt(document.getElementById('limitSelect').value);
    const progressDiv = document.getElementById('matchProgress');
    const matchBtn = document.getElementById('matchBtn');

    debugLog('[MATCHING] Threshold:', threshold, 'Limit:', limit);

    dynamicThreshold = threshold;
    dynamicLimit = limit;

    progressDiv.classList.add('show');
    matchBtn.disabled = true;
    showLoadingSpinner(matchBtn, true);
    debugLog('[MATCHING] UI updated, starting matching process');
    debugLog('[MATCHING] Using backend to query new products (memory efficient)');

    matchResults = [];
    resetChunking();  // Reset chunking for new match operation
    serverResultsMode = false;
    currentMatchSessionId = null;
    currentMatchSessionSummary = null;
    currentMatchSessionTotal = 0;
    if (typeof resetServerResultsClientState === 'function') {
        resetServerResultsClientState();
    }

    try {
        debugLog(`[BATCH-MATCHING] Starting batch matching (backend will query new products)`);

        let effectiveMode = newMode;

        // Determine mode for progress tracker
        let mode;
        if (effectiveMode === 'visual') mode = 'visual';
        else if (effectiveMode === 'metadata') mode = 'metadata';
        else mode = 'hybrid';

        // Start progress estimation (VISUAL ONLY - doesn't affect backend)
        const tracker = startProgressEstimation('matchProgress', 'batch_match', mode, newProducts.length);

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
            // Mode 3: Hybrid matching - read from single balance slider
            visualWeight = parseFloat(document.getElementById('hybridBalanceSlider').value) / 100;
            metadataWeight = 1.0 - visualWeight; // Always adds up to 100%
        }

        const batchPayload = {
            match_all_new: true,  // MEMORY OPTIMIZATION: Query IDs on backend
            threshold: threshold,
            limit: limit,
            match_against_all: false,
            visual_weight: visualWeight,
            metadata_weight: metadataWeight
        };

        // Add dynamic metadata weights if available (Mode 2 or Mode 3)
        if ((effectiveMode === 'metadata' || effectiveMode === 'hybrid') && Object.keys(metadataWeights).length > 0) {
            batchPayload.metadata_weights = getNormalizedMetadataWeights();
            debugLog(`[BATCH-MATCHING] Using dynamic metadata weights:`, batchPayload.metadata_weights);
        }

        debugLog(`[BATCH-MATCHING] Step 1: Prepare batch request`);
        debugLog(`[BATCH-MATCHING] Mode: Backend will query new product IDs`);
        debugLog(`[BATCH-MATCHING] Weights: visual=${visualWeight}, metadata=${metadataWeight}`);
        debugLog(`[BATCH-MATCHING] Threshold: ${threshold}, Limit: ${limit}`);
        debugLog(`[BATCH-MATCHING] Sending batch request with match_all_new=true (Effective Mode: ${effectiveMode})`);

        // Send batch request (backend processes independently)
        debugLog(`[BATCH-MATCHING] Step 2: Send POST request to /api/products/batch-match`);
        const response = await fetchWithRetry('/api/products/batch-match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(batchPayload)
        });

        const data = await response.json();
        debugLog(`[BATCH-MATCHING] Step 3: Received response - status: ${response.status}, ok: ${response.ok}`);
        debugLog(`[BATCH-MATCHING] Response data:`, data);

        if (!response.ok) {
            console.error(`[BATCH-MATCHING] Error response:`, data);
            if (tracker) tracker.stop();
            showToast('Batch matching failed: ' + (data.error || 'Unknown error'), 'error');
            showLoadingSpinner(matchBtn, false);
            matchBtn.disabled = false;
            return;
        }

        const sessionId = data.session_id;
        if (sessionId) {
            // Session-backed mode: fetch result pages on demand from backend.
            serverResultsMode = true;
            currentMatchSessionId = sessionId;
            currentMatchSessionSummary = data.summary || {};
            currentMatchSessionTotal = data.results_count || data.batch_size || 0;
            totalMatchCount = currentMatchSessionTotal;
            matchResults = [];

            if (typeof setServerResultsSessionState === 'function') {
                setServerResultsSessionState({
                    sessionId: currentMatchSessionId,
                    totalResults: currentMatchSessionTotal,
                    summary: currentMatchSessionSummary
                });
            }

            debugLog(`[BATCH-MATCHING] Step 4: Session mode enabled (${sessionId}), results will be loaded on demand`);
        } else {
            // Backward-compatible fallback if server still returns inline results.
            const batchResults = data.results || [];
            debugLog(`[BATCH-MATCHING] Step 4 fallback: Processing ${batchResults.length} inline results`);
            appendBatchResults(batchResults);
            currentMatchSessionTotal = matchResults.length;
            totalMatchCount = matchResults.length;
        }

        debugLog(`[BATCH-MATCHING] ✓ Complete! Processed ${currentMatchSessionTotal || matchResults.length} products`);

        // Complete progress tracker (backend finished, jump to 100%)
        if (tracker) {
            tracker.complete(`Successfully matched ${(currentMatchSessionTotal || matchResults.length)} products!`);
        }

    } catch (error) {
        console.error(`[BATCH-MATCHING] Error:`, error);
        if (tracker) tracker.stop();
        const errorMsg = getUserFriendlyError('NETWORK_ERROR', error.message);
        showToast('Batch matching failed: ' + errorMsg, 'error');
        showLoadingSpinner(matchBtn, false);
        matchBtn.disabled = false;
        return;
    }

    debugLog(`[BATCH-MATCHING] Matching complete. Total matchResults: ${matchResults.length}`);

    showToast('Matching complete!', 'success');
    showLoadingSpinner(matchBtn, false);

    // Show results
    displayResults();
    showResultsSectionWithCollapse();

    // Show save dialog after matching completes
    showSaveDialog('matching_complete');
}

function calculateProductMetadataStats(productResult) {
    // 1. Check if backend already provided summary stats (Optimized flow)
    if (productResult.summary_stats && Object.keys(productResult.summary_stats).length > 0) {
        // alert(JSON.stringify(productResult.summary_stats, null, 2)); // Debug
        const bs = productResult.summary_stats;
        const matches = productResult.m || [];

        // Fallback to _overall if _similarity is missing
        const mainStats = bs._similarity || bs._overall || { avg: 0, max: 0, min: 0 };
        const overall = bs._overall || { avg: 0, min: 0, max: 0, count: 0 };

        const metadataStats = {};
        const dynamicStats = {}; 

        Object.entries(bs).forEach(([key, stat]) => {
            // 1. BLACKLIST: Explicitly ignore these keys
            if (['_overall', '_similarity', 'match_count', 'color', 'shape', 'texture', 'visual'].includes(key)) return;

            // 2. Handle Explicit Types (New Backend)
            if (stat.type === 'numeric') {
                // Format numbers with proper decimals and commas
                const formatNum = (n) => {
                    if (n >= 1000) return n.toLocaleString('en-US', { maximumFractionDigits: 2 });
                    return n.toFixed(2);
                };

                dynamicStats[key] = {
                    type: 'numeric',
                    label: key.charAt(0).toUpperCase() + key.slice(1),
                    value: stat.avg,
                    subtext: `Avg: ${formatNum(stat.avg)} | Sum: ${formatNum(stat.sum)} | Min: ${formatNum(stat.min)} | Max: ${formatNum(stat.max)}`
                };
                // Also add to metadata stats dictionary for chart rendering
                metadataStats[key] = { avg: stat.avg, min: stat.min, max: stat.max, sum: stat.sum, count: stat.count };
            }
            else if (stat.type === 'categorical') {
                dynamicStats[key] = {
                    type: 'categorical',
                    label: key.charAt(0).toUpperCase() + key.slice(1),
                    value: stat.top_value || 'N/A',
                    subtext: `Top Value (${stat.distribution ? stat.distribution[stat.top_value] : 0})`
                };
            }
            else if (stat.type === 'similarity') {
                // Similarity scores for text fields (brand, description, etc.)
                metadataStats[key] = {
                    avg: stat.avg,
                    min: stat.min,
                    max: stat.max,
                    sum: undefined, // No sum for similarity scores
                    count: stat.count
                };
            }
            // 3. CATCH-ALL (Fixes Brand/Type/Description missing)
            // If it has no type, display it as a generic metadata field
            else {
                metadataStats[key] = {
                    avg: typeof stat.avg === 'number' ? stat.avg.toFixed(1) : stat.avg,
                    min: typeof stat.min === 'number' ? stat.min.toFixed(1) : stat.min,
                    max: typeof stat.max === 'number' ? stat.max.toFixed(1) : stat.max,
                    count: stat.count
                };
            }
        });

        // Calculate final scores for sorting/header
        const overallScores = matches.map(m => getScore(m, 'similarity'));
        const sortedScores = [...overallScores].sort((a, b) => a - b);

        return {
            totalMatches: matches.length,
            overallAvg: (mainStats.avg || 0).toFixed(1),
            bestScore: (mainStats.max || 0).toFixed(1),
            medianScore: sortedScores.length > 0 ? sortedScores[Math.floor(sortedScores.length / 2)].toFixed(1) : 0,
            worstScore: (overall.min || 0).toFixed(1),
            metadataStats,
            matchesAboveThreshold: matches.filter(m => getScore(m, 'similarity') >= (window.dynamicThreshold || 50)).length,
            dynamicStats, 
            _fromBackend: true
        };
    }

    // 2. FALLBACK (Client-side calculation)
    const matches = productResult.m || [];
    if (matches.length === 0) return null;

    const allMetadataKeys = new Set();
    matches.forEach(match => {
        const scores = match.metadata_scores || match.mscores;
        if (scores) {
            Object.keys(scores).forEach(key => {
                // Ensure fallback also excludes visual keys
                if (!['color', 'shape', 'texture', 'visual'].includes(key)) {
                    allMetadataKeys.add(key);
                }
            });
        }
    });


    const metadataStats = {};
    allMetadataKeys.forEach(key => {
        let sum = 0;
        let count = 0;
        let min = Infinity;
        let max = -Infinity;

        // Single loop through matches for this key
        matches.forEach(match => {
            const scores = match.metadata_scores || match.mscores || {};
            const val = scores[key];

            if (val !== undefined && val !== null && !isNaN(val)) {
                sum += val;
                count++;
                if (val < min) min = val;  // Track min iteratively
                if (val > max) max = val;  // Track max iteratively
            }
        });

        if (count > 0) {
            metadataStats[key] = {
                avg: (sum / count).toFixed(1),
                sum: sum.toFixed(1),
                min: min.toFixed(1),
                max: max.toFixed(1),
                count: count
            };
        }
    });

    const overallScores = matches.map(m => getScore(m, 'similarity'));
    const sortedScores = [...overallScores].sort((a, b) => a - b);

    // PERFORMANCE FIX #10: Use sorted array instead of Math.min/max spread operator
    return {
        totalMatches: matches.length,
        overallAvg: overallScores.length > 0 ? (overallScores.reduce((a, b) => a + b, 0) / overallScores.length).toFixed(1) : 0,
        medianScore: sortedScores.length > 0 ? sortedScores[Math.floor(sortedScores.length / 2)].toFixed(1) : 0,
        bestScore: sortedScores.length > 0 ? sortedScores[sortedScores.length - 1].toFixed(1) : 0,  // Last element = max
        worstScore: sortedScores.length > 0 ? sortedScores[0].toFixed(1) : 0,  // First element = min
        metadataStats,
        matchesAboveThreshold: matches.filter(m => getScore(m, 'similarity') >= (window.dynamicThreshold || 50)).length,
        dynamicStats: {}
    };
}
// Track selected metric for each product (per-product metric selection)
const productMetricSelections = {};

/**
 * Generate HTML for metadata statistics display
 * @param {Object} stats - The calculated statistics
 * @param {string} productId - The product ID for tracking metric selection
 * @param {string} selectedMetric - The currently selected metric ('avg', 'sum', 'min', 'max')
 */
function renderMetadataStats(stats, productId, selectedMetric = 'avg') {
    if (!stats) return '';

    // Store or retrieve the selected metric for this product
    if (productId) {
        if (selectedMetric && selectedMetric !== 'avg') {
            productMetricSelections[productId] = selectedMetric;
        } else if (!productMetricSelections[productId]) {
            productMetricSelections[productId] = 'avg';
        }
        selectedMetric = productMetricSelections[productId];
    }

    const html = `
        <div class="metadata-stats-container">
            <div class="stats-grid">
                <div class="stat-box">
                    <span class="stat-label">Total Matches</span>
                    <span class="stat-value">${stats.totalMatches}</span>
                </div>

                <div class="stat-box">
                    <span class="stat-label">Avg Similarity</span>
                    <span class="stat-value">${stats.overallAvg}%</span>
                </div>

                <div class="stat-box">
                    <span class="stat-label">Best Match</span>
                    <span class="stat-value">${stats.bestScore}%</span>
                </div>

                <div class="stat-box">
                    <span class="stat-label">Above Threshold</span>
                    <span class="stat-value">${stats.matchesAboveThreshold}</span>
                </div>
            </div>


            ${Object.keys(stats.metadataStats).length > 0 ? `
                <div class="metadata-breakdown">
                    <div class="metadata-breakdown-toolbar">
                        <h5 style="margin: 0;">Similarity Breakdown</h5>
                        ${productId ? `
                            <select class="metric-selector" onchange="updateProductMetric(${productId}, this.value)">
                                <option value="avg" ${selectedMetric === 'avg' ? 'selected' : ''}>Average</option>
                                <option value="sum" ${selectedMetric === 'sum' ? 'selected' : ''}>Sum</option>
                                <option value="min" ${selectedMetric === 'min' ? 'selected' : ''}>Minimum</option>
                                <option value="max" ${selectedMetric === 'max' ? 'selected' : ''}>Maximum</option>
                            </select>
                        ` : ''}
                    </div>
                    <div class="metadata-scores-grid">
                        ${Object.entries(stats.metadataStats).map(([key, data]) => {
                            // Check if this is a numeric field (has sum property indicating actual values)
                            const isNumeric = 'sum' in data && data.sum !== undefined;

                            const selectedValue = data[selectedMetric] || data.avg;

                            // Format display value
                            let displayValue;
                            if (isNumeric) {
                                // Format numeric values with commas
                                displayValue = typeof selectedValue === 'number'
                                    ? selectedValue.toLocaleString('en-US', { maximumFractionDigits: 2 })
                                    : selectedValue;
                            } else {
                                // Format percentage values
                                displayValue = typeof selectedValue === 'number' ? selectedValue.toFixed(1) : selectedValue;
                            }

                            // Calculate bar width
                            let barWidth = 0;
                            if (isNumeric) {
                                // For numeric values, normalize bar based on max value
                                if (data.max > 0) {
                                    barWidth = Math.min((parseFloat(selectedValue) / data.max) * 100, 100);
                                }
                            } else {
                                // For similarity percentages, use value directly (0-100)
                                if (selectedMetric === 'avg' || selectedMetric === 'min' || selectedMetric === 'max') {
                                    barWidth = Math.min(parseFloat(selectedValue), 100);
                                } else if (selectedMetric === 'sum') {
                                    const maxPossible = data.count * 100;
                                    barWidth = Math.min((parseFloat(selectedValue) / maxPossible) * 100, 100);
                                }
                            }

                            // Format range display
                            let rangeDisplay;
                            if (isNumeric) {
                                const formatNum = (n) => n.toLocaleString('en-US', { maximumFractionDigits: 2 });
                                rangeDisplay = `(${formatNum(data.min)} - ${formatNum(data.max)})`;
                            } else {
                                rangeDisplay = `(${data.min}% - ${data.max}%)`;
                            }

                            return `
                            <div class="metadata-score-item">
                                <span class="field-name">${key.charAt(0).toUpperCase() + key.slice(1)}</span>
                                <div class="score-details">
                                    <span class="score-value">${displayValue}${isNumeric ? '' : '%'}</span>
                                    <span class="score-range">${rangeDisplay}</span>
                                </div>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: ${barWidth}%"></div>
                                </div>
                            </div>
                        `;}).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;

    return html;
}

/**
 * Handle metric selection change for a specific product
 * @param {string} productId - The product ID
 * @param {string} metric - The selected metric ('avg', 'sum', 'min', 'max')
 */
function updateProductMetric(productId, metric) {
    productMetricSelections[productId] = metric;
    displayResults(false);  // Re-render without resetting pagination
}

function cleanupMetricSelections() {
    const currentProductIds = new Set();

    // Collect all currently displayed product IDs
    // CRITICAL FIX: Convert to string to match object keys (which are always strings)
    matchResults.forEach(result => {
        currentProductIds.add(String(result.p.id));
    });

    // Remove selections for products no longer in results
    Object.keys(productMetricSelections).forEach(productId => {
        if (!currentProductIds.has(productId)) {
            delete productMetricSelections[productId];
        }
    });
}

function getCachedMetadataStats(productResult) {
    if (!metadataStatsCache.has(productResult)) {
        metadataStatsCache.set(productResult, calculateProductMetadataStats(productResult));
    }
    return metadataStatsCache.get(productResult);
}

// Results
// Results/filter/history logic moved to /static/app.results.js

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
        label.textContent = input.files[0].name;
        label.classList.add('has-file');
    } else {
        label.textContent = 'Use BUILD CSV or see CSV FORMAT to create your file';
        label.classList.remove('has-file');
    }
}

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
        newMode = mode; // Sync the actual mode value
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
        historicalMode = mode; // Sync the actual mode value
    }

    // Update CSV warning displays
    updateCsvWarning('historical');
    updateCsvWarning('new');

    // Save state to localStorage
    saveMainAppState();
}


async function openCsvBuilderWithFiles(section) {
    const files = section === 'historical' ? historicalFiles : newFiles;

    // Prepare file data for CSV builder
    const fileData = files.map(({ file, category }) => ({
        filename: file.name,
        category: category || '',
        size: file.size,
        type: file.type
    }));

    // Use the openCsvBuilder function from index.html (handles webview vs browser)
    if (typeof openCsvBuilder === 'function') {
        // Generate window ID for staging
        const windowId = 'csv_builder_' + Math.random().toString(36).substr(2, 9);

        try {
            // Stage file data on server
            const response = await fetch('/api/csv-builder/stage', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    window_id: windowId,
                    file_data: fileData,
                    section: section
                })
            });

            if (!response.ok) {
                throw new Error('Failed to stage CSV builder data');
            }

            // Open CSV builder with window ID in query params
            openCsvBuilder(section, windowId);
        } catch (error) {
            console.error('Error staging CSV builder data:', error);
            showToast('Failed to open CSV builder', 'error');
        }
    } else {
        // Fallback - direct navigation (browser mode) - use sessionStorage
        sessionStorage.setItem('csvBuilderFiles', JSON.stringify(fileData));
        sessionStorage.setItem('csvBuilderSource', section);
        window.location.href = `/static/csv-builder.html?section=${section}`;
    }
}

// Deprecated - use openCsvBuilderWithFiles instead
function openIntegratedCsvBuilder(section) {
    openCsvBuilderWithFiles(section);
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

        // Update CSV warning
        updateCsvWarning('historical');

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

        // Update CSV warning
        updateCsvWarning('new');

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
            } else {
                // Initialize with default mode if no saved state
                setMode('historical', historicalMode);
            }

            if (state.newMode && ['visual', 'metadata', 'hybrid'].includes(state.newMode)) {
                setMode('new', state.newMode);
            } else {
                // Initialize with default mode if no saved state
                setMode('new', newMode);
            }
        } catch (e) {
            console.error('Failed to load main app state:', e);
            // On error, initialize with default modes
            setMode('historical', historicalMode);
            setMode('new', newMode);
        }
    } else {
        // No saved state - initialize with default modes to ensure UI is synced
        setMode('historical', historicalMode);
        setMode('new', newMode);
    }
}

// Call on page load
onDomReady(() => {
    loadMainAppState();
    setupMobileResultsListener();
});


// Mobile polling logic moved to /static/app.mobile.js
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
        statusIcon.innerHTML = '<i data-lucide="zap" style="width: 16px; height: 16px;"></i>';

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
        statusIcon.innerHTML = '<i data-lucide="alert-circle" style="width: 16px; height: 16px;"></i>';
        statusText.textContent = 'GPU Error';
        gpuStatusEl.setAttribute('data-tooltip', `GPU initialization failed: ${status.error}. Using CPU mode.`);
    } else {
        // CPU mode
        gpuStatusEl.classList.add('gpu-cpu');
        statusIcon.innerHTML = '<i data-lucide="cpu" style="width: 16px; height: 16px;"></i>';
        statusText.textContent = 'CPU Mode';

        let tooltip = 'Running on CPU - ';
        if (status.throughput) {
            tooltip += `${status.throughput} img/s. `;
        }
        tooltip += 'For faster processing, see GPU Setup Guide.';

        gpuStatusEl.setAttribute('data-tooltip', tooltip);
    }

    // Re-initialize icons in GPU status element only (scoped for performance)
    IconManager.reinit(50, gpuStatusEl);
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

// Catalog options/state/snapshot logic moved to /static/app.catalog.js
