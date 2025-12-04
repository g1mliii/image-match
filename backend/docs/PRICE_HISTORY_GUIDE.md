# Price History Feature Guide

## Overview

The Product Matching System now supports price history tracking! This allows you to:
- Track historical prices for products over time (up to 12 months)
- See price trends (up/down/stable) in match results
- View detailed price charts in product comparisons
- Export price statistics in CSV reports

## How to Add Price History

### Method 1: CSV Upload (Recommended)

Upload a CSV file along with your product images. The CSV can include price history in several flexible formats:

#### Format 1: With Dates (Most Accurate)
```csv
filename,category,sku,name,price_history
product1.jpg,placemats,PM-001,Blue Placemat,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75
```

#### Format 2: Prices Only (Auto-Generate Dates)
```csv
filename,category,sku,name,price_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99;31.50;28.75;32.00
```
Dates will be generated monthly going backwards from today.

#### Format 3: Single Current Price
```csv
filename,category,sku,name,price
product1.jpg,placemats,PM-001,Blue Placemat,29.99
```

### Method 2: API (For Developers)

Add price history via the REST API:

```bash
POST /api/products/{product_id}/price-history
Content-Type: application/json

{
  "prices": [
    {"date": "2024-01-15", "price": 29.99, "currency": "USD"},
    {"date": "2024-02-15", "price": 31.50, "currency": "USD"}
  ]
}
```

## CSV Format Details

### Required Fields
- **filename**: Must match your image file exactly (e.g., `product1.jpg`)

### Optional Fields
- **category**: Product category (auto-detected from folder structure if omitted)
- **sku**: Product SKU/identifier
- **name**: Product name
- **price**: Single current price
- **price_history**: Historical prices (see formats below)

### Price History Formats

#### With Dates
```
2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75
```
- Format: `date:price;date:price;date:price`
- Date format: `YYYY-MM-DD` (e.g., 2024-03-15)
- Separator: semicolon (`;`)
- Maximum: 12 price points

#### Without Dates
```
29.99;31.50;28.75;32.00
```
- Format: `price;price;price`
- Dates auto-generated monthly going backwards
- Separator: semicolon (`;`)
- Maximum: 12 price points

### Flexible Separators
The system accepts:
- Commas (`,`) between columns
- Tabs (`\t`) between columns
- Semicolons (`;`) between price entries

### Error Handling
The system gracefully handles:
- Missing price data (products without prices still work)
- Invalid date formats (skips invalid entries)
- Negative prices (rejected)
- Malformed CSV rows (skipped with warning)
- Mixed formats in same file (each row parsed independently)

## UI Features

### Match Results
- **Sparkline charts**: Quick visual of price trends
- **Current price**: Displayed prominently
- **Trend indicator**: ↑ (up), ↓ (down), → (stable)

### Detailed Comparison
- **Price statistics**: Min, max, average, current
- **Full price chart**: Interactive visualization
- **Trend analysis**: 5% threshold for up/down detection

### CSV Export
Exported CSV includes:
- Current price
- Average price
- Min price
- Max price
- Price trend

## Examples

### Example 1: Complete Data
```csv
filename,category,sku,name,price_history
blue_placemat.jpg,placemats,PM-001,Blue Placemat,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75
```

### Example 2: Minimal Data
```csv
filename,price
blue_placemat.jpg,29.99
```

### Example 3: Mixed Data Quality
```csv
filename,category,sku,name,price_history
product1.jpg,placemats,PM-001,Blue Placemat,2024-01-15:29.99;2024-02-15:31.50
product2.jpg,dinnerware,DW-002,White Plate,45.00
product3.jpg,,,Cotton Napkins,15.99;16.50;15.75
product4.jpg,placemats,PM-004,,
```

All rows will be processed successfully, with missing data handled gracefully.

## Performance

- **Batch operations**: Price history uploaded in bulk for efficiency
- **Indexed queries**: Fast retrieval even with thousands of products
- **Lazy loading**: Price data fetched only when needed
- **Caching**: Price statistics calculated once and cached

## Troubleshooting

### "Failed to parse price history"
- Check date format is YYYY-MM-DD
- Ensure prices are positive numbers
- Verify semicolons separate price entries
- Check for extra spaces or special characters

### "No price data displayed"
- Verify CSV was uploaded successfully
- Check filename matches image file exactly
- Look for validation warnings in browser console
- Ensure price column is in correct position (4th or 5th)

### "Price chart not showing"
- Ensure at least 2 price points exist
- Check dates are valid and in chronological order
- Verify prices are positive numbers

## Best Practices

1. **Use dates when available**: More accurate trend analysis
2. **Include 3-12 months**: Provides meaningful trends
3. **Keep prices current**: Update regularly for accuracy
4. **Use consistent currency**: All prices should be in same currency
5. **Validate before upload**: Check CSV in Excel/text editor first
6. **Start simple**: Begin with current prices, add history later
7. **Test with sample**: Use provided sample CSV to verify format

## Sample CSV

Download the sample CSV file: `sample_with_price_history.csv`

Or click "📖 Show CSV format examples" in the UI for interactive examples.

## API Reference

### Get Price History
```
GET /api/products/{product_id}/price-history?limit=12
```

### Add Price History
```
POST /api/products/{product_id}/price-history
{
  "prices": [
    {"date": "2024-01-15", "price": 29.99}
  ]
}
```

### Response Format
```json
{
  "status": "success",
  "product_id": 123,
  "price_history": [
    {"date": "2024-01-15", "price": 29.99, "currency": "USD"}
  ],
  "statistics": {
    "min": 28.75,
    "max": 32.00,
    "average": 30.50,
    "current": 30.25,
    "trend": "down",
    "data_points": 5
  }
}
```

## Support

For issues or questions:
1. Check browser console for detailed error messages
2. Verify CSV format matches examples
3. Test with sample CSV file
4. Review validation warnings in UI
