# Brutalist/Minimalist UI Version

A super minimalist, brutalist design alternative for the Catalog Match app.

## Design Philosophy

- **No gradients** - Pure black and white
- **No shadows** - Flat design
- **No rounded corners** - Sharp, geometric shapes
- **Monospace font** - Courier New throughout
- **Bold borders** - Thick, solid lines (3-4px)
- **High contrast** - Black on white, white on black
- **No animations** - Except essential transitions
- **Minimal spacing** - Efficient use of space
- **Uppercase labels** - Strong, direct typography

## Access

### Original Version (Gradient/Modern)
```
http://localhost:5000/
```

### Brutalist Version (Minimalist)
```
http://localhost:5000/brutalist
```

## Files

- `index-brutalist.html` - Stripped down HTML structure
- `styles-brutalist.css` - Brutalist styling (no gradients, shadows, or rounded corners)
- Uses the same `app.js` - No JavaScript changes needed

## Key Differences

| Feature | Original | Brutalist |
|---------|----------|-----------|
| Colors | Gradients (purple/blue) | Black & white only |
| Borders | 1-2px rounded | 3-4px sharp |
| Shadows | Multiple box-shadows | None |
| Font | Inter (sans-serif) | Courier New (monospace) |
| Buttons | Rounded, gradient | Square, solid |
| Spacing | Generous padding | Compact |
| Typography | Mixed case | UPPERCASE labels |
| Animations | Smooth transitions | Minimal |

## Benefits

1. **Faster loading** - Less CSS, no web fonts
2. **Better accessibility** - High contrast
3. **Clearer hierarchy** - Bold borders define sections
4. **Unique aesthetic** - Stands out from typical web apps
5. **Print-friendly** - Works great in black & white

## Customization

To adjust the brutalist theme:

1. **Change border thickness**: Search for `border:` in `styles-brutalist.css`
2. **Add color accents**: Replace `#000` with your accent color
3. **Adjust spacing**: Modify padding values
4. **Change font**: Replace `'Courier New', monospace` with another monospace font

## Notes

- The brutalist version uses the same backend API
- All functionality remains identical
- Session data is compatible between versions
- You can switch between versions at any time
