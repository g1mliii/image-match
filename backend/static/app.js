
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
const AUTO_FAST_FILE_THRESHOLD = 5000;
const AUTO_FAST_CPU_CORES_THRESHOLD = 4;
const CLIENT_DEBUG_LOGS = false;

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

    return useFilePaths ? 'file_paths' : 'direct_upload';
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
    if (catalogPollingInterval) {
        clearInterval(catalogPollingInterval);
        catalogPollingInterval = null;
        console.log('✓ Cleared catalog polling interval');
    }

    // Clear blob URL cleanup interval (Memory Leak Fix #1)
    if (blobUrlCleanupInterval) {
        clearInterval(blobUrlCleanupInterval);
        blobUrlCleanupInterval = null;
        console.log('✓ Cleared blob URL cleanup interval');
    }

    // Stop mobile results polling and flag checking
    stopMatchResultsPolling();
    if (mobileFlagCheckInterval) {
        clearInterval(mobileFlagCheckInterval);
        mobileFlagCheckInterval = null;
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
document.addEventListener('DOMContentLoaded', () => {
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
            const totalBatches = Math.ceil(imageItems.length / STREAM_UPLOAD_BATCH_SIZE);

            debugLog(`[BATCH-UPLOAD] Streaming ${imageItems.length} images in ${totalBatches} batch(es) of ${STREAM_UPLOAD_BATCH_SIZE}`);

            // Process each batch
            for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                const batchStart = batchIdx * STREAM_UPLOAD_BATCH_SIZE;
                const batchEnd = Math.min(batchStart + STREAM_UPLOAD_BATCH_SIZE, imageItems.length);
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

                    // Send this batch
                    const response = await fetchWithRetry('/api/products/batch-upload', {
                        method: 'POST',
                        body: batchFormData
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
                    recordFailedItem(failedItems, { row: 'batch', fileName: 'all', reason: errorMsg });
                    console.error('[BATCH-METADATA] Batch creation failed:', data);
                }
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
            const totalBatches = Math.ceil(imageItems.length / STREAM_UPLOAD_BATCH_SIZE);
            debugLog(`[BATCH-UPLOAD] Streaming ${imageItems.length} images in ${totalBatches} batch(es) of ${STREAM_UPLOAD_BATCH_SIZE}`);

            for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                const batchStart = batchIdx * STREAM_UPLOAD_BATCH_SIZE;
                const batchEnd = Math.min(batchStart + STREAM_UPLOAD_BATCH_SIZE, imageItems.length);
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

                    const response = await fetchWithRetry('/api/products/batch-upload', {
                        method: 'POST',
                        body: batchFormData
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

        // Process batch results
        const batchResults = data.results || [];
        debugLog(`[BATCH-MATCHING] Step 4: Process results - Received ${batchResults.length} results from batch`);

        // Process each result
        for (let i = 0; i < batchResults.length; i++) {
            const result = batchResults[i];

            // Use product_data provided directly in the batch result (Optimized Flow)
            let product = result.product_data;

            if (!product) {
                // Fallback: create minimal product object if details not found (shouldn't happen with new enriched response)
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
                // Map backend keys to frontend expected keys if necessary
                product.hasFeatures = product.has_features || false;
                if (!product.name) product.name = product.product_name || product.filename || `Product ${product.id}`;
            }

            // Deduplicate matches to prevent "two product b's" issue
            const rawMatches = result.matches || [];
            const seenMatchIds = new Set();
            const uniqueMatches = [];

            for (const m of rawMatches) {
                // Handle various ID formats
                const mid = m.product_id || m.mid || m.id;
                if (mid && !seenMatchIds.has(mid)) {
                    seenMatchIds.add(mid);
                    uniqueMatches.push(m);
                }
            }
            const matches = uniqueMatches;

            debugLog(`[BATCH-MATCHING] Product ${product.id}: ${matches.length} matches found`);
            const compactMatches = matches.map(m => createCompactMatch(m));
            const compactProduct = createCompactProduct(product);

            const resultObj = {
                p: compactProduct,  // Compact product
                m: compactMatches,   // Compact matches
                summary_stats: result.summary_stats
            };

            // Only add error if present
            if (result.status !== 'success' && result.error) {
                resultObj.err = result.error;
            }

            matchResults.push(resultObj);
        }

        debugLog(`[BATCH-MATCHING] ✓ Complete! Processed ${matchResults.length} products`);

        // Complete progress tracker (backend finished, jump to 100%)
        if (tracker) {
            tracker.complete(`Successfully matched ${matchResults.length} products!`);
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

    // PERFORMANCE: Render only visible items (virtual scrolling concept)
    function renderItems(filteredValues) {
        // Clear existing items
        dropdownList.innerHTML = '';

        // PERFORMANCE: Use fragment for batch DOM updates
        const fragment = document.createDocumentFragment();
        const maxRender = Math.min(filteredValues.length, 100); // Limit initial render

        for (let i = 0; i < maxRender; i++) {
            const val = filteredValues[i];
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

            fragment.appendChild(item);
        }

        if (filteredValues.length > maxRender) {
            const moreInfo = document.createElement('div');
            moreInfo.style.padding = '8px 10px';
            moreInfo.style.fontSize = '10px';
            moreInfo.style.color = '#718096';
            moreInfo.style.textAlign = 'center';
            moreInfo.textContent = `Showing ${maxRender} of ${filteredValues.length}. Search to narrow down.`;
            fragment.appendChild(moreInfo);
        }

        dropdownList.appendChild(fragment);
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
        renderItems(values);
    });

    // PERFORMANCE: Debounced search
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        debounce(`dropdown-search-${key}`, () => {
            const filtered = searchTerm
                ? values.filter(v => v.toLowerCase().includes(searchTerm))
                : values;
            renderItems(filtered);
        }, 200);
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
    contentDiv.style.alignItems = 'center';
    contentDiv.style.width = '100%';

    // Loop schema cols
    let filterCount = 0;
    // PERFORMANCE FIX #4: Use DocumentFragment to batch DOM insertions and avoid multiple reflows
    const fragment = document.createDocumentFragment();

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
            inputs.innerHTML = `
                <input type="number" placeholder="Min" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'min')">
                <span style="color:#cbd5e0">-</span>
                <input type="number" placeholder="Max" class="input input-sm" style="width: 70px;" onchange="updateMetadataFilter('${key}', this.value, 'max')">
            `;
            wrapper.appendChild(inputs);
            fragment.appendChild(wrapper);  // PERFORMANCE FIX #4: Append to fragment instead of contentDiv
            filterCount++;
        } else {
            // SMART HYBRID FILTERS: Scan all matches to get unique values (no arbitrary limit)
            // PERFORMANCE: Use efficient Set for deduplication
            const uniqueVals = new Set();
            const maxScan = Math.min(matchResults.length, 500); // Limit scan for performance
            const maxUniqueValues = 100; // PERFORMANCE FIX #3: Early exit if we have enough unique values

            for (let i = 0; i < maxScan; i++) {
                const mList = matchResults[i].m;
                for (let j = 0; j < Math.min(mList.length, 10); j++) {
                    const m = mList[j];
                    const val = (m.mv && m.mv[key]) || (m.metadata_values && m.metadata_values[key]);
                    if (val) {
                        uniqueVals.add(String(val)); // Ensure string for consistency
                        // PERFORMANCE FIX #3: Break early if we've collected enough unique values
                        if (uniqueVals.size >= maxUniqueValues) break;
                    }
                }
                // Break outer loop too if we have enough values
                if (uniqueVals.size >= maxUniqueValues) break;
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

function displayResults(resetPage = true) {
    console.log('[DISPLAY] displayResults called');

    // MEMORY: Clean up selections for products no longer in results
    cleanupMetricSelections();

    // Populate dynamic sort options if schema is available
    populateDynamicSortOptions();

    // Remove existing dynamic filters to allow regeneration with new data
    const existingFilters = document.getElementById('dynamicFiltersContainer');
    if (existingFilters) {
        // Prevent leaking document-level dropdown listeners when filters are regenerated
        cleanupDynamicFilterDropdownListeners(existingFilters);
        existingFilters.remove();
    }

    // Check if dynamic filters need generation
    generateDynamicFilters();

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

        const response = await fetch(`/ api / catalog / products ? type = historical & page=${historicalProductsPage}& limit=50`);
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

async function updateDynamicSearch(value) {
    dynamicSearch = value.toLowerCase().trim();

    // Show spinner while typing
    const statusEl = document.getElementById('dynamicSearchStatus');

    // Clear cache if search is empty
    if (!dynamicSearch) {
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
        try {
            // Call backend search API
            const response = await fetch(`/ api / products / search ? q = ${encodeURIComponent(dynamicSearch)}& limit=1000`);
            const data = await response.json();

            if (data.success) {
                // Build a map of product IDs for fast lookup
                dynamicSearchResults.clear();
                data.results.forEach(product => {
                    dynamicSearchResults.set(product.id, product);
                });
                console.log(`[SEARCH] Found ${data.results.length} products matching "${dynamicSearch}"`);

                // Update status to show count
                const count = data.results.length;
                statusEl.innerHTML = `< span class="search-count" > ${count} ${count === 1 ? 'match' : 'matches'}</span > `;
            }
        } catch (error) {
            console.error('[SEARCH] Error:', error);
            statusEl.innerHTML = '<span style="color: #e53e3e; font-size: 0.75rem;">ERROR</span>';
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
                newData ? Promise.resolve({ ok: true, json: () => Promise.resolve(newData) }) : fetchWithRetry(`/ api / products / ${newProductId} `),
                matchData ? Promise.resolve({ ok: true, json: () => Promise.resolve(matchData) }) : fetchWithRetry(`/ api / products / ${matchedProductId} `)
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
    // Early return if no results
    if (matchResults.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }

    // ENHANCEMENT: Build dynamic headers from metadata scores
    const allMetadataKeys = new Set();
    matchResults.forEach(result => {
        result.m.forEach(match => {
            // CRITICAL FIX: Use mscores (compact format) with fallback
            const scores = match.metadata_scores || match.mscores;
            if (scores) {
                Object.keys(scores).forEach(key => allMetadataKeys.add(key));
            }
        });
    });

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

    matchResults.forEach(result => {
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
    });

    const csv = csvRows.join('\n') + '\n';
    const filename = `match_results_${new Date().toISOString().slice(0, 10)}.csv`;
    if (window.pywebview) {
        try {
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
        const blob = new Blob([csv], { type: 'text/csv' });
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
document.addEventListener('DOMContentLoaded', () => {
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
    if (matchResults.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }

    if (typeof JSZip === 'undefined') {
        showToast('JSZip library not loaded. Please refresh the page.', 'error');
        return;
    }

    showToast('Preparing export with images... This may take a few minutes.', 'info');

    try {
        const zip = new JSZip();
        const MAX_MATCHES_PER_PRODUCT = 5;  // Limit to top 5 matches to keep ZIP manageable
        let processedCount = 0;
        const totalProducts = matchResults.length;

        // Detect all metadata keys for comprehensive export
        const allMetadataKeys = new Set();
        matchResults.forEach(result => {
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

        // Prepare detailed JSON export data
        let exportData = {
            timestamp: new Date().toISOString(),
            mode: newMode,
            weights: similarityWeights,
            metadata_weights: newMode !== 'visual' ? metadataWeights : {},
            threshold: parseInt(document.getElementById('thresholdSlider').value),
            metadata_fields: metadataKeysArray,
            results: []
        };

        const BATCH_SIZE = 10;  // Process 10 products at a time
        for (let batchStart = 0; batchStart < matchResults.length; batchStart += BATCH_SIZE) {
            const batch = matchResults.slice(batchStart, batchStart + BATCH_SIZE);

            // RACE CONDITION FIX: Process batch in parallel, return data for deterministic ordering
            const batchResults = await Promise.all(batch.map(async (result) => {
                const product = result.p;
                const matches = result.m;
                const topMatches = matches.slice(0, MAX_MATCHES_PER_PRODUCT);

                processedCount++;
                if (processedCount % 5 === 0) {
                    showToast(`Processing ${processedCount}/${totalProducts} products...`, 'info');
                }

                const metadataStats = getCachedMetadataStats(result);

                // Sanitize product name for folder naming
                const sanitizedName = (product.name || `product_${product.id}`).replace(/[<>:"/\\|?*]/g, '_');
                const productFolder = `products/${sanitizedName}_${product.id}`;

                // Fetch and add new product image
                try {
                    const productImgResponse = await fetch(`/api/products/${product.id}/image`);
                    if (productImgResponse.ok) {
                        const productImgBlob = await productImgResponse.blob();
                        const productImgExt = productImgBlob.type.split('/')[1] || 'jpg';
                        zip.file(`${productFolder}/new_product.${productImgExt}`, productImgBlob);
                    }
                } catch (error) {
                    console.warn(`Failed to fetch image for product ${product.id}:`, error);
                }

                // PERFORMANCE OPTIMIZATION: Fetch match images in parallel (6x faster)
                const matchesFolder = `${productFolder}/matches`;

                // Fetch all match images concurrently
                const matchImageResults = await Promise.all(topMatches.map(async (match, i) => {
                    try {
                        const matchImgResponse = await fetch(`/api/products/${match.mid}/image`);
                        if (matchImgResponse.ok) {
                            const matchImgBlob = await matchImgResponse.blob();
                            const matchImgExt = matchImgBlob.type.split('/')[1] || 'jpg';
                            const matchName = (match.name || `match_${match.mid}`).replace(/[<>:"/\\|?*]/g, '_');
                            const similarity = getScore(match, 'similarity').toFixed(1);

                            // Return image data with index for deterministic ordering
                            return {
                                index: i,
                                blob: matchImgBlob,
                                filename: `${matchesFolder}/${i + 1}_${matchName}_${similarity}pct.${matchImgExt}`
                            };
                        }
                        return null;
                    } catch (error) {
                        console.warn(`Failed to fetch image for match ${match.mid}:`, error);
                        return null;
                    }
                }));

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
                    `${productFolder}/new_product.jpg`
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
                        image_path: `${productFolder}/new_product.jpg`
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
                        image_path: `${matchesFolder}/${idx + 1}_${(m.name || `match_${m.mid}`).replace(/[<>:"/\\|?*]/g, '_')}_${getScore(m, 'similarity').toFixed(1)}pct.jpg`
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

            // Small delay between batches to allow garbage collection
            if (batchStart + BATCH_SIZE < matchResults.length) {
                await new Promise(resolve => setTimeout(resolve, 50));
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
Total Products: ${matchResults.length}
Threshold: ${parseInt(document.getElementById('thresholdSlider').value)}%

## Files
- results_summary.csv: Quick summary of all matches
- results_detailed.json: Complete match data with scores and metadata
- products/: Folder containing each product and its top ${MAX_MATCHES_PER_PRODUCT} matches
  - Each product has its own folder with:
    - new_product.jpg: The new product image
    - matches/: Folder with top matching products (ranked by similarity)

## Notes
- Only top ${MAX_MATCHES_PER_PRODUCT} matches per product are included to keep file size manageable
- Match images are named: rank_productname_similaritypct.jpg
`;
        zip.file('README.txt', readme);

        // Generate ZIP
        showToast('Generating ZIP file...', 'info');
        const zipBlob = await zip.generateAsync({
            type: 'blob',
            compression: 'DEFLATE',
            compressionOptions: { level: 6 }
        }, (metadata) => {
            const percent = metadata.percent.toFixed(0);
            if (percent % 10 === 0) {
                showToast(`Compressing ZIP: ${percent}%`, 'info');
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

function showDuplicateReport() {
    if (matchResults.length === 0) {
        showToast('No results to analyze', 'warning');
        return;
    }

    const duplicates = [];

    matchResults.forEach(result => {
        const product = result.p;  // Use compact format
        const highMatches = result.m.filter(m => getScore(m, 'similarity') > 90);

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
    const duplicates = [];

    // Detect metadata keys for optional inclusion
    const allMetadataKeys = new Set();
    matchResults.forEach(result => {
        result.m.forEach(match => {
            // CRITICAL FIX: Use mscores (compact format) with fallback
            const scores = match.metadata_scores || match.mscores;
            if (scores) {
                Object.keys(scores).forEach(key => allMetadataKeys.add(key));
            }
        });
    });
    const metadataKeysArray = Array.from(allMetadataKeys).sort();
    const hasMetadataScores = metadataKeysArray.length > 0;

    matchResults.forEach(result => {
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
    });

    // Build header row
    let headerRow = ['New Product', 'New Category', 'New SKU', 'Matched Product', 'Similarity Score', 'Recommendation'];
    if (hasMetadataScores) {
        metadataKeysArray.forEach(key => {
            headerRow.push(`${key} Score`);
        });
    }

    const csvRows = [headerRow.map(h => `"${h}"`).join(',')];

    duplicates.forEach(dup => {
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
    });

    const csv = csvRows.join('\n') + '\n';

    const filename = `duplicate_report_${new Date().toISOString().slice(0, 10)}.csv`;

    // Check if running in pywebview
    if (window.pywebview) {
        try {
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
        const blob = new Blob([csv], { type: 'text/csv' });
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
    const categories = new Set();

    matchResults.forEach(result => {
        if (result.p.cat) {
            categories.add(result.p.cat);
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
    const metadataFilterKeys = Object.keys(metadataFilterCriteria || {});
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
                        const criteria = metadataFilterCriteria[field];
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

// ENHANCEMENT: Initialize metadata filter criteria
let metadataFilterCriteria = {};

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
document.addEventListener('DOMContentLoaded', () => {
    loadMainAppState();
    setupMobileResultsListener();
});


let matchResultsPollingInterval = null;
let mobileFlagCheckInterval = null;  // Check flag every 2 seconds

// Poll for mobile results flag via API
function setupMobileResultsListener() {
    if (mobileFlagCheckInterval) {
        return;
    }

    try {
        // Check mobile results flag every 2 seconds
        mobileFlagCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/mobile/check-flag');
                const data = await response.json();

                if (data.ready) {
                    debugLog('📱 [MOBILE] Flag set - results are ready');

                    // Stop flag checking
                    if (mobileFlagCheckInterval) {
                        clearInterval(mobileFlagCheckInterval);
                        mobileFlagCheckInterval = null;
                    }

                    // Start fetching results
                    startMatchResultsPolling();
                }
            } catch (error) {
                debugWarn('Error checking mobile flag:', error);
            }
        }, 2000);  // Check every 2 seconds

        debugLog('✓ Mobile results listener initialized (polling flag)');
    } catch (error) {
        console.warn('Failed to setup mobile listener:', error);
    }
}

function syncMobileResultsManual() {
    if (matchResultsPollingInterval) {
        showToast('Already checking mobile results...', 'info');
        return;
    }

    showToast('Checking mobile results...', 'info');
    startMatchResultsPolling();
}

function startMatchResultsPolling() {
    /**
     * Poll for stored mobile match results when triggered by mobile-upload window
     * Only polls 3 times (9 seconds total) to avoid unnecessary network requests
     * Results are displayed in the Results section
     */
    if (matchResultsPollingInterval) {
        clearInterval(matchResultsPollingInterval);
    }

    let pollAttempts = 0;
    const maxAttempts = 3;  // Try 3 times = 9 seconds max

    matchResultsPollingInterval = setInterval(async () => {
        pollAttempts++;

        try {
            const response = await fetch('/api/products/match-results');
            if (!response.ok) {
                if (pollAttempts >= maxAttempts) stopMatchResultsPolling();
                return;
            }

            const data = await response.json();
            const results = data.results || [];

            if (results.length > 0) {
                debugLog(`📱 [MOBILE] Received ${results.length} results`);

                // Add all results to display
                results.forEach(result => {
                    const compactProduct = createCompactProduct(result.product_data);
                    const compactMatches = result.matches.map(m => createCompactMatch(m));

                    matchResults.push({
                        p: compactProduct,
                        m: compactMatches
                    });
                });

                // Show results section
                showResultsSectionWithCollapse();

                displayResults(false);
                debugLog('✓ Mobile results displayed');

                // Stop polling - we got the results
                stopMatchResultsPolling();
            } else if (pollAttempts >= maxAttempts) {
                // No results after 9 seconds, give up
                debugWarn('📱 [MOBILE] No results found after 3 attempts');
                stopMatchResultsPolling();
            }
        } catch (error) {
            debugWarn('Error fetching mobile results:', error);
            if (pollAttempts >= maxAttempts) stopMatchResultsPolling();
        }
    }, 3000); // Try every 3 seconds

    debugLog('🔄 [MOBILE] Polling started (max 9 seconds)');
}

function stopMatchResultsPolling() {
    if (matchResultsPollingInterval) {
        clearInterval(matchResultsPollingInterval);
        matchResultsPollingInterval = null;
        debugLog('✓ [MOBILE] Polling stopped');

        // Clear the mobile flag so mobile can send new results
        fetch('/api/mobile/clear-flag', { method: 'POST' })
            .catch(error => debugWarn('Could not clear mobile flag:', error));

        // Resume lightweight flag listener for future mobile uploads.
        setupMobileResultsListener();
    }
}

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
                statsText += ` | <strong>${data.loaded_snapshot.name}</strong>`;
            }

            statsEl.innerHTML = statsText;

            // Check for large database warning
            if (data.database_size_mb && data.database_size_mb > 500) {
                showToast('Database is large (' + data.database_size_mb.toFixed(0) + ' MB). Consider cleaning up old products.', 'warning', 8000);
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

        // Initialize UI state based on selected option (fixes initial load bug)
        handleCatalogOptionChange();
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
    // Support both event-based calls and direct calls
    const option = e && e.target ? e.target.value : getCatalogLoadOption();
    const dropZone = document.getElementById('historicalDropZone');
    const processBtn = document.getElementById('processHistoricalBtn');
    const downloadDiv = document.getElementById('downloadExistingHistoricalDiv');

    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process, hide download
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING CATALOG';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // AUTO-LOAD CSV WHEN "USE EXISTING" IS SELECTED
        // This populates the CSV file label immediately so user can see it's loaded
        autoLoadCatalogCSV();
    } else if (option === 'replace') {
        // Replace catalog - show warning, hide download
        if (e && e.target && existingCatalogStats && existingCatalogStats.historical_products > 0) {
            // Only show confirmation dialog when user manually changes to replace (not on initial load)
            const confirmed = confirm(
                `WARNING: This will DELETE all ${existingCatalogStats.historical_products.toLocaleString()} existing historical products and create a NEW catalog!\n\n` +
                `A backup snapshot will be created automatically.\n\n` +
                `Are you sure you want to replace with a new catalog?`
            );
            if (!confirmed) {
                // Revert to use_existing
                document.querySelector('input[name="catalogLoadOption"][value="use_existing"]').checked = true;
                handleCatalogOptionChange();
                return;
            }
        }
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'REPLACE & PROCESS';
        if (downloadDiv) downloadDiv.style.display = 'none';
    } else {
        // Add to existing - show download button
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = historicalFiles.length === 0 && !historicalCsv;
        processBtn.textContent = 'ADD & PROCESS';
        if (downloadDiv) downloadDiv.style.display = 'block';
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
                `<p class="success">Loaded ${historicalProducts.length} products from existing catalog</p>`;

            // Show next section
            showNewSectionAfterHistoricalStep();

            // Load metadata schema to populate sliders if research was already done
            await loadMetadataSchema();

            // Auto-load CSV if available
            await autoLoadCatalogCSV();

        } catch (error) {
            console.error('Error loading existing catalog:', error);
            showToast('Failed to load existing catalog', 'error');
        }
        return;
    }

    if (option === 'replace') {
        // Create automatic backup snapshot before replacing (debounced to avoid duplicates in batch operations)
        const now = Date.now();
        if (now - lastAutoBackupTime > AUTO_BACKUP_DEBOUNCE_MS) {
            try {
                console.log('[REPLACE] Creating automatic backup snapshot...');
                showToast('Creating backup snapshot...', 'info');

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                const snapshotName = `auto-backup-before-replace-${timestamp}`;

                const snapshotResponse = await fetch('/api/catalogs/save-current', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: snapshotName,
                        description: 'Automatic backup created before batch replace operations',
                        tags: ['auto-backup', 'replace'],
                        skip_if_empty: true
                    })
                });
                const snapshotResult = await snapshotResponse.json().catch(() => ({}));

                if (snapshotResponse.ok) {
                    if (snapshotResult.skipped) {
                        console.log('[REPLACE] Skipping backup snapshot (catalog is empty)');
                        showToast('No existing catalog to back up', 'info');
                    } else {
                        console.log('[REPLACE] Backup snapshot created:', snapshotName);
                        showToast('Backup snapshot created', 'success');
                        // Wait a moment to ensure snapshot is complete
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                    lastAutoBackupTime = now;
                } else {
                    console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                    showToast('Warning: Could not create backup snapshot', 'warning');
                }
            } catch (error) {
                console.warn('[REPLACE] Error creating backup snapshot:', error);
                showToast('Warning: Could not create backup snapshot', 'warning');
                // Continue with replace even if snapshot fails
            }
        } else {
            console.log('[REPLACE] Skipping backup (within debounce window) - batch operation detected');
            showToast('Batch operation detected - using previous backup', 'info');
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

        // Initialize UI state based on selected option (fixes initial load bug)
        handleNewCatalogOptionChange();
    } catch (error) {
        console.error('Error checking existing new catalog:', error);
    }
}

function handleNewCatalogOptionChange() {
    const option = getNewCatalogLoadOption();
    const dropZone = document.getElementById('newDropZone');
    const processBtn = document.getElementById('processNewBtn');
    const downloadDiv = document.getElementById('downloadExistingNewDiv');

    if (option === 'use_existing') {
        // Using existing catalog - disable upload, enable process, hide download
        dropZone.style.opacity = '0.5';
        dropZone.style.pointerEvents = 'none';
        processBtn.disabled = false;
        processBtn.textContent = 'USE EXISTING NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // AUTO-LOAD CSV WHEN "USE EXISTING" IS SELECTED
        // This populates the CSV file label immediately so user can see it's loaded
        autoLoadCatalogCSV();
    } else if (option === 'replace') {
        // Replacing - enable upload, show warning, hide download
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'none';

        // Show warning
        if (existingCatalogStats && existingCatalogStats.new_products > 0) {
            const confirmed = confirm(
                `WARNING: This will DELETE all ${existingCatalogStats.new_products} existing new products and create a NEW catalog!\n\n` +
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
        // add_to_existing - enable upload, show download button
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
        processBtn.disabled = newFiles.length === 0 && !newCsv;
        processBtn.textContent = 'PROCESS NEW PRODUCTS';
        if (downloadDiv) downloadDiv.style.display = 'block';
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
                `<p class="success">Loaded ${newProducts.length} new products from existing catalog</p>`;

            // Show next section
            document.getElementById('matchSection').style.display = 'block';
            document.getElementById('matchSection').scrollIntoView({ behavior: 'smooth' });

            // Load metadata schema to populate sliders
            await loadMetadataSchema();

            // Auto-load CSV if available
            await autoLoadCatalogCSV();

        } catch (error) {
            console.error('Error loading existing new products:', error);
            showToast('Failed to load existing new products', 'error');
        }
        return;
    }

    if (option === 'replace') {
        // Create automatic backup snapshot before replacing (debounced to avoid duplicates in batch operations)
        const now = Date.now();
        if (now - lastAutoBackupTime > AUTO_BACKUP_DEBOUNCE_MS) {
            try {
                console.log('[REPLACE] Creating automatic backup snapshot...');
                showToast('Creating backup snapshot...', 'info');

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                const snapshotName = `auto-backup-before-replace-${timestamp}`;

                const snapshotResponse = await fetch('/api/catalogs/save-current', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: snapshotName,
                        description: 'Automatic backup created before batch replace operations',
                        tags: ['auto-backup', 'replace'],
                        skip_if_empty: true
                    })
                });
                const snapshotResult = await snapshotResponse.json().catch(() => ({}));

                if (snapshotResponse.ok) {
                    if (snapshotResult.skipped) {
                        console.log('[REPLACE] Skipping backup snapshot (catalog is empty)');
                        showToast('No existing catalog to back up', 'info');
                    } else {
                        console.log('[REPLACE] Backup snapshot created:', snapshotName);
                        showToast('Backup snapshot created', 'success');
                        // Wait a moment to ensure snapshot is complete
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                    lastAutoBackupTime = now;
                } else {
                    console.warn('[REPLACE] Failed to create backup snapshot, continuing anyway');
                    showToast('Warning: Could not create backup snapshot', 'warning');
                }
            } catch (error) {
                console.warn('[REPLACE] Error creating backup snapshot:', error);
                showToast('Warning: Could not create backup snapshot', 'warning');
                // Continue with replace even if snapshot fails
            }
        } else {
            console.log('[REPLACE] Skipping backup (within debounce window) - batch operation detected');
            showToast('Batch operation detected - using previous backup', 'info');
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
async function resetAppState(reason = 'Catalog data has changed') {
    console.log('Resetting app state:', reason);

    // Clear in-memory product data
    historicalProducts = [];
    newProducts = [];
    matchResults = [];
    historicalFiles = [];
    newFiles = [];
    historicalCsv = null;
    newCsv = null;

    // Clear saved state (webview only)
    await clearSavedState();

    // TRIGGER BACKEND CLEANUP: Wipe match results from DB to save space
    try {
        fetch('/api/cleanup-matches', { method: 'POST' })
            .then(res => res.json())
            .then(d => console.log('[CLEANUP] Transient matches wiped:', d))
            .catch(e => console.warn('[CLEANUP] Failed to wipe matches:', e));
    } catch (e) {
        console.warn('Backend cleanup error:', e);
    }

    // Reset UI to initial state
    const historicalSection = document.getElementById('historicalSection');
    const newSection = document.getElementById('newSection');
    const matchSection = document.getElementById('matchSection');
    const resultsSection = document.getElementById('resultsSection');

    if (newSection) newSection.style.display = 'none';
    if (matchSection) matchSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    setSectionCollapsed('historicalSection', false);
    setSectionCollapsed('newSection', false);

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
                text += ` | Loaded from: "${data.loaded_snapshot.name}"`;
            }

            summary.textContent = text;
            infoBar.style.display = 'block';

            // Also load metadata schema if a catalog exists
            loadMetadataSchema();
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
    // Use pywebview API to open in child window if available
    if (window.pywebview && window.pywebview.api) {
        console.log('[NAV] Opening Catalog Manager in child window (webviewer)...');
        try {
            window.pywebview.api.open_catalog_manager();
        } catch (e) {
            console.error('[NAV] Error opening catalog manager:', e);
            // Fallback to browser window
            window.open('/catalog-manager', '_blank');
        }
    } else {
        // Browser mode - open in new tab
        console.log('[NAV] Pywebview not available, opening in browser tab...');
        window.open('/catalog-manager', '_blank');
    }
}

// Listen for catalog changes from Catalog Manager (browser mode only)
// PyWebview mode uses window.handleChildWindowEvent in index.html
function initCatalogChangeListener() {
    // Listen via BroadcastChannel (browser mode only)
    try {
        catalogChannel = new BroadcastChannel('catalog_changes');
        catalogChannel.onmessage = (event) => {
            // In browser mode, call the same handler that pywebview uses
            if (typeof handleCatalogChanged === 'function') {
                handleCatalogChanged({ action: event.data.action, details: event.data.details });
            }
        };
    } catch (e) {
        // BroadcastChannel not supported, use polling
        if (catalogPollingInterval) clearInterval(catalogPollingInterval);
        catalogPollingInterval = setInterval(checkCatalogChangesInMainApp, 2000);
    }

    // Also check on visibility change (when user switches back to this tab)
    const catalogVisibilityHandler = () => {
        if (!document.hidden) {
            checkCatalogChangesInMainApp();

            // Restart polling if it was stopped and BroadcastChannel not available
            if (!catalogChannel && !catalogPollingInterval) {
                if (catalogPollingInterval) clearInterval(catalogPollingInterval);
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

// Check for catalog changes via sessionStorage (browser mode fallback)
function checkCatalogChangesInMainApp() {
    const changeData = sessionStorage.getItem('catalogManagerChange');
    if (changeData) {
        try {
            const change = JSON.parse(changeData);
            // Only process recent changes (within last 30 seconds)
            if (Date.now() - change.timestamp < 30000) {
                // Call the same handler that pywebview uses
                if (typeof handleCatalogChanged === 'function') {
                    handleCatalogChanged({ action: change.action, details: change.details });
                }
            }
        } catch (e) {
            console.error('Error parsing catalog change:', e);
        }
    }
}


let pendingSaveOperation = null;

/**
 * Show the save dialog to user for snapshot creation
 * @param {string} operation - The operation that triggered the save (e.g., 'comparison_complete', 'manual_save')
 * @param {string} defaultName - Default snapshot name to prefill
 */
function showSaveDialog(operation = 'snapshot', defaultName = null) {
    const dialog = document.getElementById('saveDialog');
    const nameInput = document.getElementById('snapshotNameInput');

    // Generate default name if not provided
    if (!defaultName) {
        const now = new Date();
        const date = now.toISOString().split('T')[0];
        const time = now.toTimeString().split(' ')[0].replace(/:/g, '-');
        defaultName = `${operation}-${date}_${time}`;
    }

    nameInput.value = defaultName;
    nameInput.focus();
    nameInput.select();

    // Store the operation for later use
    pendingSaveOperation = operation;

    // Show dialog
    dialog.classList.add('show');
}

/**
 * Close the save dialog without saving
 */
function closeSaveDialog() {
    const dialog = document.getElementById('saveDialog');
    dialog.classList.remove('show');
    pendingSaveOperation = null;
}

/**
 * Submit the save dialog
 */
async function submitSaveDialog() {
    const nameInput = document.getElementById('snapshotNameInput');
    const saveType = document.querySelector('input[name="saveType"]:checked').value;
    const snapshotName = nameInput.value.trim();

    if (!snapshotName) {
        showToast('Please enter a snapshot name', 'error');
        return;
    }

    if (saveType === 'skip') {
        closeSaveDialog();
        return;
    }

    try {
        const operation = pendingSaveOperation || 'snapshot';
        console.log(`[SAVE] Saving snapshot "${snapshotName}" as ${saveType}`);

        const response = await fetch('/api/catalogs/save-current', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: snapshotName,
                description: `Saved from ${operation}`,
                tags: [saveType, operation]
            })
        });

        const result = await response.json();

        if (response.ok && result.status === 'success') {
            const typeLabel = saveType === 'persistent' ? 'Persistent snapshot' : 'Session snapshot';
            console.log(`[SAVE] ✓ Snapshot saved successfully: ${snapshotName}`);
            showToast(`${typeLabel} saved: "${snapshotName}"`, 'success');
            closeSaveDialog();
        } else {
            console.error(`[SAVE] Error saving snapshot:`, result);
            showToast(`Error saving snapshot: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('[SAVE] Exception:', error);
        showToast('Error saving snapshot', 'error');
    }
}


let crashRecoveryData = null;

/**
 * Check for crash recovery on app startup
 */
async function checkForCrashRecovery() {
    try {
        const response = await fetch('/api/catalogs/check-crash-recovery');
        const result = await response.json();

        if (result.crash_detected && result.recovery_snapshot) {
            crashRecoveryData = result.recovery_snapshot;
            showCrashRecoveryDialog(result.recovery_snapshot);
        }
    } catch (error) {
        console.error('Error checking for crash recovery:', error);
    }
}

/**
 * Show crash recovery dialog to user
 */
function showCrashRecoveryDialog(snapshotInfo) {
    const dialog = document.getElementById('crashRecoveryDialog');
    const detailsDiv = document.getElementById('recoveryDetails');

    // Format snapshot info for display
    let html = `
        <div><strong>Name:</strong> ${escapeHtml(snapshotInfo.name || 'Unknown')}</div>
        <div><strong>Created:</strong> ${new Date(snapshotInfo.created_at).toLocaleString()}</div>
        <div><strong>Products:</strong> ${snapshotInfo.product_count || 'Unknown'}</div>
    `;

    if (snapshotInfo.created_by_operation) {
        html += `<div><strong>From:</strong> ${escapeHtml(snapshotInfo.created_by_operation)}</div>`;
    }

    // Show expiry warning since this is a session snapshot
    html += `<div style="margin-top: 10px; color: #c00;"><strong>WARNING:</strong> This recovery snapshot will expire in 1 hour if not restored</div>`;

    detailsDiv.innerHTML = html;
    dialog.classList.add('show');
}

/**
 * Close crash recovery dialog
 */
function closeCrashRecoveryDialog() {
    const dialog = document.getElementById('crashRecoveryDialog');
    dialog.classList.remove('show');
    crashRecoveryData = null;
}

/**
 * Discard the crash recovery snapshot
 */
async function discardCrashRecovery() {
    if (!crashRecoveryData) {
        closeCrashRecoveryDialog();
        return;
    }

    try {
        const snapshotName = encodeURIComponent(crashRecoveryData.id);
        const response = await fetch(`/api/catalogs/${snapshotName}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Recovery snapshot discarded', 'info');
        } else {
            const data = await response.json();
            showToast(`Error discarding snapshot: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error discarding recovery snapshot:', error);
        showToast('Error discarding recovery snapshot', 'error');
    }

    closeCrashRecoveryDialog();
}

/**
 * Restore from crash recovery snapshot
 */
async function restoreCrashRecovery() {
    if (!crashRecoveryData) {
        closeCrashRecoveryDialog();
        return;
    }

    try {
        // Load the snapshot
        const snapshotName = encodeURIComponent(crashRecoveryData.id);
        const response = await fetch(`/api/catalogs/load/${snapshotName}`, {
            method: 'POST'
        });

        const result = await response.json();

        if (response.ok) {
            showToast(`Recovered snapshot "${crashRecoveryData.name}" loaded`, 'success');

            // Refresh the UI
            await loadCatalogInfo();

            // Reload products if needed
            if (result.total_products && result.total_products > 0) {
                showToast(`${result.total_products} products restored`, 'info');
            }
        } else {
            showToast(`Error loading recovery snapshot: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error restoring recovery snapshot:', error);
        showToast('Error restoring recovery snapshot', 'error');
    }

    closeCrashRecoveryDialog();
}

async function autoLoadCatalogCSV() {
    try {
        // Load historical CSV
        const historicalResponse = await fetch('/api/catalogs/csv-content?section=historical');
        const historicalData = await historicalResponse.json();

        if (historicalData.has_csv) {
            const blob = new Blob([historicalData.csv_content], { type: 'text/csv' });
            const file = new File([blob], historicalData.filename || 'historical.csv');
            historicalCsv = file;

            // Update UI EXACTLY like manual upload does - just show the filename
            const historicalFileLabel = document.getElementById('historicalFileLabel');
            if (historicalFileLabel) {
                historicalFileLabel.textContent = historicalData.filename;
            }

            // Also update status section if it exists
            const historicalStatus = document.getElementById('historicalStatus');
            if (historicalStatus) {
                historicalStatus.innerHTML += `<p class="success">[✓] Historical CSV loaded: ${historicalData.filename} (${historicalData.row_count} rows)</p>`;
            }
        }

        // Load new CSV
        const newResponse = await fetch('/api/catalogs/csv-content?section=new');
        const newData = await newResponse.json();

        if (newData.has_csv) {
            const blob = new Blob([newData.csv_content], { type: 'text/csv' });
            const file = new File([blob], newData.filename || 'new.csv');
            newCsv = file;

            // Update UI EXACTLY like manual upload does - just show the filename
            const newFileLabel = document.getElementById('newFileLabel');
            if (newFileLabel) {
                newFileLabel.textContent = newData.filename;
            }

            // Also update status section if it exists
            const newStatus = document.getElementById('newStatus');
            if (newStatus) {
                newStatus.innerHTML += `<p class="success">[✓] New CSV loaded: ${newData.filename} (${newData.row_count} rows)</p>`;
            }
        }

        // Update CSV warnings after auto-load
        updateCsvWarning('historical');
        updateCsvWarning('new');
    } catch (error) {
        // Silently fail if no CSV available - this is normal if Mode 1 only
    }
}

async function openMobileModal() {
    const modal = document.getElementById('mobileModal');
    if (!modal) return;

    modal.style.display = 'block';

    // Load mobile connection info
    await loadMobileConnectionInfo();

    // Re-initialize icons in modal only (scoped for performance)
    IconManager.reinit(50, modal);
}

function closeMobileModal() {
    const modal = document.getElementById('mobileModal');
    if (!modal) return;
    modal.style.display = 'none';
}

async function loadMobileConnectionInfo() {
    try {
        // Local-only connection + PIN data for desktop modal
        const connectionResponse = await fetch('/api/mobile/connection-info');
        if (!connectionResponse.ok) throw new Error('Failed to fetch connection info');
        const connectionData = await connectionResponse.json();

        // Remote URL config metadata (source/editable)
        let remoteConfig = {
            remote_url: connectionData.remote_url || null,
            source: 'config',
            editable: true
        };
        let ngrokStatus = {
            ngrok_installed: false,
            has_token: false,
            token_source: null,
            tunnel_running: false,
            public_url: null
        };
        const [remoteResult, ngrokResult] = await Promise.allSettled([
            fetch('/api/mobile/remote-url'),
            fetch('/api/mobile/ngrok/status')
        ]);

        if (remoteResult.status === 'fulfilled') {
            try {
                if (remoteResult.value.ok) {
                    remoteConfig = await remoteResult.value.json();
                }
            } catch (_) {
                // Keep graceful fallback to connectionData.remote_url
            }
        }

        if (ngrokResult.status === 'fulfilled') {
            try {
                if (ngrokResult.value.ok) {
                    ngrokStatus = await ngrokResult.value.json();
                }
            } catch (_) {
                // Keep graceful fallback defaults
            }
        }

        const password = connectionData.password || '000000';
        const localUrl = connectionData.lan_url || connectionData.mobile_url || `http://${connectionData.primary_ip}:${connectionData.port}/mobile`;
        const remoteUrl = remoteConfig.remote_url || connectionData.remote_url || '';

        // Update UI with local network info + PIN
        document.getElementById('ipAddressInput').value = connectionData.primary_ip || 'localhost';
        document.getElementById('passwordInput').value = password;
        document.getElementById('localMobileUrl').textContent = localUrl;
        document.getElementById('mobilePassword').textContent = password;

        // Update remote URL controls and visibility
        const remoteInput = document.getElementById('remoteUrlInput');
        const remoteStatus = document.getElementById('remoteUrlStatus');
        const remoteRow = document.getElementById('remoteUrlRow');
        const remoteText = document.getElementById('remoteMobileUrl');
        const saveBtn = document.getElementById('saveRemoteUrlBtn');
        const clearBtn = document.getElementById('clearRemoteUrlBtn');
        const autoNgrokBtn = document.getElementById('autoNgrokRemoteUrlBtn');
        const ngrokTokenInput = document.getElementById('ngrokTokenInput');
        const saveNgrokTokenBtn = document.getElementById('saveNgrokTokenBtn');
        const clearNgrokTokenBtn = document.getElementById('clearNgrokTokenBtn');
        const ngrokTokenStatus = document.getElementById('ngrokTokenStatus');

        if (remoteInput) remoteInput.value = remoteUrl;
        if (remoteText) remoteText.textContent = remoteUrl || 'Not configured';
        if (remoteRow) remoteRow.style.display = remoteUrl ? 'list-item' : 'none';

        const editable = !!remoteConfig.editable;
        if (remoteInput) remoteInput.disabled = !editable;
        if (saveBtn) saveBtn.disabled = !editable;
        if (clearBtn) clearBtn.disabled = !editable;
        if (autoNgrokBtn) autoNgrokBtn.disabled = !editable;

        const tokenManagedByEnv = ngrokStatus.token_source === 'env';
        if (ngrokTokenInput) {
            ngrokTokenInput.disabled = tokenManagedByEnv;
            ngrokTokenInput.value = '';
        }
        if (saveNgrokTokenBtn) saveNgrokTokenBtn.disabled = tokenManagedByEnv;
        if (clearNgrokTokenBtn) clearNgrokTokenBtn.disabled = tokenManagedByEnv || !ngrokStatus.has_token;

        if (ngrokTokenStatus) {
            if (!ngrokStatus.ngrok_installed) {
                ngrokTokenStatus.textContent = 'ngrok binary not found. Install ngrok or bundle it in app package.';
            } else if (tokenManagedByEnv) {
                ngrokTokenStatus.textContent = 'Token managed by NGROK_AUTHTOKEN environment variable.';
            } else if (ngrokStatus.has_token) {
                if (ngrokStatus.public_url) {
                    ngrokTokenStatus.textContent = `Token configured. Tunnel live: ${ngrokStatus.public_url}`;
                } else {
                    ngrokTokenStatus.textContent = 'Token configured. Click AUTO NGROK to start or refresh tunnel URL.';
                }
            } else {
                ngrokTokenStatus.textContent = 'Token not configured. Paste token once, then click SETUP TOKEN.';
            }
        }

        if (remoteStatus) {
            if (!editable && remoteConfig.source === 'env') {
                remoteStatus.textContent = 'Managed by MOBILE_REMOTE_URL environment variable (read-only here).';
            } else if (remoteUrl) {
                remoteStatus.textContent = 'Remote URL configured. Use this secure URL outside office.';
            } else if (!ngrokStatus.has_token) {
                remoteStatus.textContent = 'Set ngrok token once, then click AUTO NGROK.';
            } else {
                remoteStatus.textContent = 'Click AUTO NGROK to start tunnel and save remote URL.';
            }
        }

        // Prefer secure remote URL for QR when available
        generateQRCode(remoteUrl || localUrl);

    } catch (error) {
        console.error('Error loading mobile connection info:', error);
        showToast('Failed to load mobile connection info', 'error');
    }
}

function generateQRCode(url) {
    const container = document.getElementById('qrCodeContainer');
    if (!container) return;

    // Clear previous QR code
    container.innerHTML = '';

    try {
        // Generate QR code using QRCode library
        new QRCode(container, {
            text: url,
            width: 200,
            height: 200,
            colorDark: '#000000',
            colorLight: '#ffffff',
            correctLevel: QRCode.CorrectLevel.H
        });
    } catch (error) {
        console.error('Error generating QR code:', error);
        container.innerHTML = '<p style="color: #999; font-size: 12px;">Failed to generate QR code</p>';
    }
}

function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const text = element.value || element.textContent;

    // Use clipboard API if available
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard!', 'success', 2000);
        }).catch(err => {
            console.error('Failed to copy:', err);
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
}

function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();

    try {
        document.execCommand('copy');
        showToast('Copied to clipboard!', 'success', 2000);
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showToast('Failed to copy', 'error');
    }

    document.body.removeChild(textarea);
}

async function generateNewPassword() {
    try {
        // Request new password from backend
        const response = await fetch('/api/mobile/password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'generate' })
        });

        if (!response.ok) throw new Error('Failed to generate password');

        const data = await response.json();
        const newPassword = data.password;

        // Update UI
        document.getElementById('passwordInput').value = newPassword;
        document.getElementById('mobilePassword').textContent = newPassword;

        showToast('New password generated! (Old PIN will stop working)', 'success', 3000);

    } catch (error) {
        console.error('Error generating password:', error);
        showToast('Failed to generate new password', 'error');
    }
}

async function saveRemoteUrl() {
    const input = document.getElementById('remoteUrlInput');
    if (!input) return;

    const remoteUrl = input.value.trim();
    if (!remoteUrl) {
        showToast('Enter an HTTPS remote URL or use CLEAR', 'error');
        return;
    }

    try {
        const response = await fetch('/api/mobile/remote-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ remote_url: remoteUrl })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to save remote URL');
        }

        showToast('Secure remote URL saved', 'success', 2500);
        await loadMobileConnectionInfo();
    } catch (error) {
        console.error('Error saving remote URL:', error);
        showToast(error.message || 'Failed to save remote URL', 'error');
    }
}

async function saveNgrokToken() {
    const input = document.getElementById('ngrokTokenInput');
    if (!input) return;

    const token = input.value.trim();
    if (!token) {
        showToast('Paste ngrok token first', 'error');
        return;
    }

    const saveBtn = document.getElementById('saveNgrokTokenBtn');
    if (saveBtn) saveBtn.disabled = true;

    try {
        const response = await fetch('/api/mobile/ngrok/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                authtoken: token,
                start_now: true
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || data.suggestion || 'Failed to save ngrok token');
        }

        input.value = '';
        if (data.public_url) {
            showToast('Token saved and tunnel started', 'success', 2500);
        } else if (data.start_error) {
            showToast(`Token saved. ${data.start_error}`, 'warning', 3500);
        } else {
            showToast('Token saved', 'success', 2500);
        }
        await loadMobileConnectionInfo();
    } catch (error) {
        console.error('Error saving ngrok token:', error);
        showToast(error.message || 'Failed to save ngrok token', 'error');
    } finally {
        const refreshBtn = document.getElementById('saveNgrokTokenBtn');
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

async function clearNgrokToken() {
    const clearBtn = document.getElementById('clearNgrokTokenBtn');
    if (clearBtn) clearBtn.disabled = true;

    try {
        const response = await fetch('/api/mobile/ngrok/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'clear' })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to clear ngrok token');
        }

        const input = document.getElementById('ngrokTokenInput');
        if (input) input.value = '';

        showToast('ngrok token cleared', 'success', 2200);
        await loadMobileConnectionInfo();
    } catch (error) {
        console.error('Error clearing ngrok token:', error);
        showToast(error.message || 'Failed to clear ngrok token', 'error');
    } finally {
        const refreshBtn = document.getElementById('clearNgrokTokenBtn');
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

async function autoDetectNgrokRemoteUrl() {
    const autoBtn = document.getElementById('autoNgrokRemoteUrlBtn');
    if (autoBtn) autoBtn.disabled = true;

    try {
        const response = await fetch('/api/mobile/remote-url/auto-ngrok', {
            method: 'POST'
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(
                data.error ||
                data.suggestion ||
                data.details?.hint ||
                'Failed to auto-detect ngrok URL'
            );
        }

        showToast('ngrok URL detected and saved', 'success', 2500);
        await loadMobileConnectionInfo();
    } catch (error) {
        console.error('Error auto-detecting ngrok URL:', error);
        showToast(error.message || 'Failed to auto-detect ngrok URL', 'error');
    } finally {
        const refreshAutoBtn = document.getElementById('autoNgrokRemoteUrlBtn');
        if (refreshAutoBtn) refreshAutoBtn.disabled = false;
    }
}

async function clearRemoteUrl() {
    try {
        const response = await fetch('/api/mobile/remote-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ remote_url: '' })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to clear remote URL');
        }

        showToast('Remote URL cleared', 'success', 2000);
        await loadMobileConnectionInfo();
    } catch (error) {
        console.error('Error clearing remote URL:', error);
        showToast(error.message || 'Failed to clear remote URL', 'error');
    }
}

function refreshMobileModal() {
    loadMobileConnectionInfo();
    showToast('Mobile connection info refreshed', 'success', 2000);
}


