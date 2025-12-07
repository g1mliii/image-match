/**
 * CSV Parser Web Worker
 * 
 * Runs CSV parsing in a separate thread to avoid blocking the UI.
 * Handles CSV line parsing, validation, and data extraction.
 */

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
                current += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    result.push(current);
    return result;
}

// Parse price history string
function parsePriceHistory(str) {
    if (!str) return null;
    
    const prices = [];
    const entries = str.split(';').map(e => e.trim()).filter(e => e);
    
    for (const entry of entries) {
        if (entry.includes(':')) {
            const [dateStr, priceStr] = entry.split(':').map(s => s.trim());
            const price = parseFloat(priceStr);
            if (!isNaN(price) && price >= 0) {
                prices.push({ date: dateStr, price: price });
            }
        } else {
            const price = parseFloat(entry);
            if (!isNaN(price) && price >= 0) {
                prices.push(price);
            }
        }
    }
    
    return prices.length > 0 ? prices : null;
}

// Parse performance history string
function parsePerformanceHistory(str) {
    if (!str) return null;
    
    const values = str.split(';').map(v => {
        const num = parseFloat(v.trim());
        return !isNaN(num) && num >= 0 ? num : null;
    }).filter(v => v !== null);
    
    return values.length > 0 ? values : null;
}

// Main worker message handler
self.onmessage = function(event) {
    const { csvText, hasHeader } = event.data;
    
    try {
        const lines = csvText.split('\n').filter(line => line.trim());
        const map = {};
        const errors = [];
        
        const dataLines = hasHeader ? lines.slice(1) : lines;
        
        // Process each line
        dataLines.forEach((line, index) => {
            try {
                const parts = parseCSVLine(line);
                
                if (parts.length >= 1) {
                    const filename = parts[0];
                    if (!filename) return;
                    
                    const category = parts[1] || null;
                    const sku = parts[2] || null;
                    const name = parts[3] || null;
                    
                    // Parse price history
                    let priceHistory = null;
                    const priceHistoryStr = parts[4] || parts[5] || null;
                    if (priceHistoryStr) {
                        priceHistory = parsePriceHistory(priceHistoryStr);
                    }
                    
                    // Parse performance history
                    let performanceHistory = null;
                    const perfHistoryStr = parts[6] || null;
                    if (perfHistoryStr) {
                        performanceHistory = parsePerformanceHistory(perfHistoryStr);
                    }
                    
                    map[filename] = {
                        category: category,
                        sku: sku,
                        name: name,
                        priceHistory: priceHistory,
                        performanceHistory: performanceHistory
                    };
                }
            } catch (e) {
                errors.push(`Line ${index + 1}: ${e.message}`);
            }
        });
        
        // Send results back to main thread
        self.postMessage({
            success: true,
            map: map,
            errors: errors,
            lineCount: dataLines.length
        });
    } catch (error) {
        self.postMessage({
            success: false,
            error: error.message
        });
    }
};
