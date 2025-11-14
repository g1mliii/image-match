# Deployment Guide for Product Matcher Website

This guide will help you deploy the Product Matcher marketing website to GitHub Pages.

## Prerequisites

- GitHub account
- Git installed on your computer
- Product Matcher repository on GitHub

## Step 1: Prepare Your Repository

1. Ensure the `docs` folder is in your repository root
2. All HTML, CSS, and JS files should be in the `docs` folder

## Step 2: Update Placeholders

Before deploying, replace these placeholders throughout the HTML files:

### GitHub Username
Find and replace: `yourusername` → Your actual GitHub username

Files to update:
- All HTML files (index.html, pricing.html, download.html, docs.html, contact.html, 404.html)
- sitemap.xml
- _config.yml

### Email Addresses
- `support@productmatcher.com` → Your support email
- `business@productmatcher.com` → Your business email

### LemonSqueezy Checkout URL
- `https://productmatcher.lemonsqueezy.com/checkout` → Your actual checkout URL

### Download Links
In `download.html`, update:
```html
https://github.com/YOURUSERNAME/product-matcher/releases/download/v1.0.0/ProductMatcher-Windows-1.0.0.exe
https://github.com/YOURUSERNAME/product-matcher/releases/download/v1.0.0/ProductMatcher-macOS-1.0.0.dmg
```

## Step 3: Add Images

Create and add these images to `docs/images/`:

1. **favicon.png** (32x32 or 64x64 px)
2. **og-image.png** (1200x630 px)
3. **screenshot-upload.png** (1200x800 px)
4. **screenshot-results.png** (1200x800 px)

### Temporary Placeholders

Until you have real screenshots, you can use placeholder services:

```html
<!-- In index.html, replace data-src with: -->
<img src="https://placehold.co/1200x800/0077be/ffffff?text=Upload+Interface" alt="Upload interface">
<img src="https://placehold.co/1200x800/0077be/ffffff?text=Results+View" alt="Results view">
```

## Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Scroll down to **Pages** (left sidebar)
4. Under **Source**, select:
   - Branch: `main` (or your default branch)
   - Folder: `/docs`
5. Click **Save**
6. Wait 1-2 minutes for deployment

Your site will be available at:
```
https://YOURUSERNAME.github.io/product-matcher/
```

## Step 5: Configure Custom Domain (Optional)

If you have a custom domain:

1. In GitHub Pages settings, enter your domain (e.g., `productmatcher.com`)
2. Create a `CNAME` file in the `docs` folder:
   ```
   productmatcher.com
   ```
3. Configure DNS records with your domain provider:
   - Add a CNAME record pointing to `YOURUSERNAME.github.io`
   - Or add A records pointing to GitHub's IPs:
     - 185.199.108.153
     - 185.199.109.153
     - 185.199.110.153
     - 185.199.111.153

## Step 6: Set Up LemonSqueezy

1. Create a LemonSqueezy account at https://lemonsqueezy.com
2. Create a new product:
   - Name: "Product Matcher Pro"
   - Price: $49 (one-time)
   - Enable "Generate unique license keys"
3. Get your checkout URL
4. Update `pricing.html` with the checkout URL

## Step 7: Test Your Site

Visit your GitHub Pages URL and test:

- [ ] All pages load correctly
- [ ] Navigation works
- [ ] Images display (or placeholders show)
- [ ] Download links work (after creating releases)
- [ ] Contact form opens mailto link
- [ ] Responsive design on mobile
- [ ] All internal links work
- [ ] External links open in new tabs

## Step 8: SEO Optimization

### Submit to Google Search Console

1. Go to https://search.google.com/search-console
2. Add your property (your GitHub Pages URL)
3. Verify ownership (use HTML tag method)
4. Submit your sitemap: `https://YOURUSERNAME.github.io/product-matcher/sitemap.xml`

### Update Meta Tags

In each HTML file, update:
- `<meta property="og:url">` with your actual URL
- `<meta property="og:image">` with your actual image URL
- `<meta property="twitter:url">` with your actual URL

## Step 9: Analytics (Optional)

### Google Analytics

1. Create a Google Analytics account
2. Get your tracking ID (G-XXXXXXXXXX)
3. Add to the `<head>` of each HTML file:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### Plausible Analytics (Privacy-Friendly Alternative)

1. Create a Plausible account
2. Add to the `<head>` of each HTML file:

```html
<script defer data-domain="yourdomain.com" src="https://plausible.io/js/script.js"></script>
```

## Step 10: Create GitHub Releases

For the download page to work, create releases:

1. Go to your repository → Releases → Create a new release
2. Tag version: `v1.0.0`
3. Release title: `Product Matcher v1.0.0`
4. Upload your built executables:
   - `ProductMatcher-Windows-1.0.0.exe`
   - `ProductMatcher-macOS-1.0.0.dmg`
5. Publish release

## Maintenance

### Updating Content

1. Edit HTML files in the `docs` folder
2. Commit and push changes
3. GitHub Pages will automatically rebuild (takes 1-2 minutes)

### Updating Version Numbers

When releasing a new version:
1. Update version in `download.html`
2. Update changelog in `download.html`
3. Create new GitHub release
4. Update download links

## Troubleshooting

### Site Not Loading
- Check GitHub Pages settings are correct
- Ensure `docs` folder is in the root of your repository
- Wait a few minutes after enabling GitHub Pages

### Images Not Showing
- Check image paths are correct (relative to docs folder)
- Ensure images are committed to the repository
- Check file extensions match (case-sensitive on Linux)

### 404 Errors
- Ensure all links use relative paths
- Check file names match exactly (case-sensitive)
- Verify `404.html` exists in docs folder

### Custom Domain Not Working
- Verify DNS records are correct
- Wait 24-48 hours for DNS propagation
- Check CNAME file exists and contains correct domain

## Support

If you encounter issues:
- Check GitHub Pages documentation: https://docs.github.com/pages
- Open an issue on the repository
- Contact support@productmatcher.com

## Checklist

Before going live:

- [ ] All placeholders replaced
- [ ] Images added (or placeholders working)
- [ ] LemonSqueezy checkout URL updated
- [ ] GitHub releases created
- [ ] Download links tested
- [ ] All pages tested on desktop and mobile
- [ ] SEO meta tags updated
- [ ] Sitemap submitted to Google
- [ ] Analytics configured (optional)
- [ ] Custom domain configured (optional)
- [ ] Contact form tested
- [ ] All external links verified

Congratulations! Your Product Matcher website is now live! 🎉
