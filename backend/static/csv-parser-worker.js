/**
 * CSV Parser Web Worker
 * 
 * Runs CSV parsing in a separate thread to avoid blocking the UI.
 * Handles CSV line parsing, validation, and data extraction.
 */

// Helper function to parse a CSV line properly handling quoted fields
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

// Main worker message handler
self.onmessage = function (event) {
    const { csvText, hasHeader } = event.data;

    try {
        const lines = csvText.split('\n').filter(line => line.trim());
        const map = {};
        const errors = [];

        // Default mapping if no header
        let colMap = {
            filename: 0,
            category: 1,
            sku: 2,
            name: 3
        };

        let startRow = 0;

        // Detect headers dynamically
        if (hasHeader && lines.length > 0) {
            try {
                const headerParts = parseCSVLine(lines[0]);
                const newMap = {};
                let foundAny = false;

                headerParts.forEach((h, i) => {
                    const norm = h.toLowerCase().trim().replace(/[\s_]/g, '');
                    const originalName = h.trim();

                    if (norm === 'filename' || norm === 'file') {
                        newMap.filename = i;
                        foundAny = true;
                    } else if (norm === 'category') {
                        newMap.category = i;
                        foundAny = true;
                    } else if (norm === 'sku' || norm === 'id') {
                        newMap.sku = i;
                        foundAny = true;
                    } else if (norm === 'name' || norm === 'productname' || norm === 'title') {
                        newMap.name = i;
                        foundAny = true;
                    } else if (norm === 'price' || norm === 'cost' || norm === 'msrp') {
                        newMap.price = i;
                        foundAny = true;
                    } else if (norm.includes('perform') || norm.includes('sales') || norm.includes('revenue')) {
                        newMap.performance = i;
                        foundAny = true;
                    } else {
                        // Dynamic column - capture it!
                        if (originalName) {
                            newMap[originalName] = i;
                            foundAny = true;
                        }
                    }
                });

                if (foundAny) {
                    colMap = { ...colMap, ...newMap }; // Overlay found columns
                    startRow = 1;
                }
            } catch (e) {
                // If header parsing fails, fall back to defaults
                console.warn('Header parsing failed, using defaults', e);
            }
        }

        const dataLines = lines.slice(startRow);

        // Process each line
        dataLines.forEach((line, index) => {
            try {
                const parts = parseCSVLine(line);

                if (parts.length >= 1) {
                    // Get filename using map or default to 0
                    const filenameIdx = colMap.filename !== undefined ? colMap.filename : 0;
                    const filename = parts[filenameIdx];

                    if (!filename) return;

                    // Extract all mapped data
                    const rowData = {};

                    // First, get system fields explicitly
                    rowData.category = (colMap.category !== undefined && parts[colMap.category]) ? parts[colMap.category] : null;
                    rowData.sku = (colMap.sku !== undefined && parts[colMap.sku]) ? parts[colMap.sku] : null;
                    rowData.name = (colMap.name !== undefined && parts[colMap.name]) ? parts[colMap.name] : null;
                    rowData.price = (colMap.price !== undefined && parts[colMap.price]) ? parts[colMap.price] : null;
                    rowData.performance = (colMap.performance !== undefined && parts[colMap.performance]) ? parts[colMap.performance] : null;

                    // Then get all dynamic columns
                    const systemKeys = ['filename', 'category', 'sku', 'name', 'price', 'performance'];
                    for (const [colName, colIdx] of Object.entries(colMap)) {
                        // Skip system internal keys if they are just aliases (we handled them above)
                        if (systemKeys.includes(colName)) continue;

                        if (colIdx !== undefined && parts[colIdx]) {
                            rowData[colName] = parts[colIdx];
                        }
                    }

                    map[filename] = rowData;
                }
            } catch (e) {
                errors.push(`Line ${index + 1 + startRow}: ${e.message}`);
            }
        });

        // Send results back to main thread
        self.postMessage({
            success: true,
            map: map,
            detectedColumns: Object.keys(colMap),
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
