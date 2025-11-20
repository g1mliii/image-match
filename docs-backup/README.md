# CatalogMatch Marketing Website

This is the marketing website for CatalogMatch, hosted on GitHub Pages.

## Structure

```
docs/
├── index.html          # Landing page
├── pricing.html        # Pricing and plans
├── download.html       # Download page with installers
├── docs.html          # Documentation and user guide
├── contact.html       # Contact and support
├── styles.css         # Global styles (Material Design inspired)
├── scripts.js         # Interactive features and lazy loading
├── images/            # Images and screenshots
│   ├── favicon.png
│   ├── og-image.png
│   ├── screenshot-upload.png
│   └── screenshot-results.png
└── README.md          # This file
```

## Setup for GitHub Pages

1. Push this `docs` folder to your GitHub repository
2. Go to repository Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select branch: `main` and folder: `/docs`
5. Click Save
6. Your site will be available at: `https://yourusername.github.io/catalog-match/`

## Customization

### Update Links

Replace the following placeholders throughout the HTML files:

- `yourusername` → Your GitHub username
- `https://catalogmatch.lemonsqueezy.com/checkout` → Your actual LemonSqueezy checkout URL
- `support@catalogmatch.com` → Your actual support email
- `business@catalogmatch.com` → Your actual business email

### Add Images

Create the following images and place them in the `images/` folder:

1. **favicon.png** (32x32 or 64x64) - Browser tab icon with CatalogMatch logo
2. **og-image.png** (1200x630) - Social media preview image with CatalogMatch branding
3. **screenshot-upload.png** - Screenshot of the upload interface
4. **screenshot-results.png** - Screenshot of the results view

### Update Download Links

In `download.html`, update the download links to point to your actual GitHub releases:

```html
https://github.com/yourusername/product-matcher/releases/download/v1.0.0/ProductMatcher-Windows-1.0.0.exe
https://github.com/yourusername/product-matcher/releases/download/v1.0.0/ProductMatcher-macOS-1.0.0.dmg
```

### SEO Optimization

1. Update meta descriptions in each HTML file
2. Add your actual domain in Open Graph tags
3. Create a `sitemap.xml` file
4. Add `robots.txt` file
5. Submit to Google Search Console

### Analytics (Optional)

To add Google Analytics or Plausible:

1. Add tracking script to the `<head>` of each HTML file
2. Uncomment analytics code in `scripts.js`
3. Update the `trackEvent` function with your analytics provider

## Features

- **Pure HTML/CSS/JS** - No build process required
- **Material Design inspired** - Clean, modern aesthetic
- **Responsive** - Mobile-friendly design
- **Fast loading** - Optimized images and lazy loading
- **SEO optimized** - Meta tags, Open Graph, semantic HTML
- **Accessible** - ARIA labels, keyboard navigation
- **GPU accelerated animations** - Smooth hover effects

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## License

All website code is part of the CatalogMatch project.

## Contact

For website issues or suggestions, please open an issue on GitHub or contact support@catalogmatch.com.
