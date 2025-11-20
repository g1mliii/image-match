# Security Documentation

This document outlines the security measures implemented in the Product Matcher marketing website.

## XSS (Cross-Site Scripting) Protection

### Static Content Security
- All HTML pages are static with no user-generated content rendering
- No `innerHTML` usage - only `textContent` for dynamic content
- All form inputs are sanitized before processing
- URL validation prevents `javascript:` protocol injection

### JavaScript Security Measures

1. **Input Sanitization**
   ```javascript
   function sanitizeInput(input) {
     const div = document.createElement('div');
     div.textContent = input;
     return div.innerHTML;
   }
   ```

2. **URL Validation**
   ```javascript
   function isValidUrl(url) {
     try {
       const parsed = new URL(url, window.location.origin);
       return ['http:', 'https:', 'mailto:'].includes(parsed.protocol);
     } catch {
       return false;
     }
   }
   ```

3. **Contact Form Protection**
   - All inputs sanitized using `textContent`
   - Email validation with regex
   - Required field validation
   - `encodeURIComponent` for mailto links

4. **Safe DOM Manipulation**
   - Use `textContent` instead of `innerHTML`
   - Explicit button types (`type="button"`)
   - No `eval()` or `Function()` constructors
   - No dynamic script injection

## Content Security Policy (CSP)

### CSP via Meta Tags (Currently Implemented)

GitHub Pages doesn't allow custom HTTP headers, so we use meta tags for CSP:

```html
<meta http-equiv="Content-Security-Policy" content="
  default-src 'self';
  script-src 'self' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self';
  connect-src 'self';
  base-uri 'self';
  form-action 'self' mailto:;
">
```

**Note:** The following CSP directives are **NOT supported** in meta tags and require HTTP headers:
- `frame-ancestors` (clickjacking protection)
- `report-uri` / `report-to` (violation reporting)
- `sandbox` (iframe sandboxing)

### Full CSP Headers (For Custom Domain with CDN)

If using a custom domain with Cloudflare, Netlify, or similar:

```
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://plausible.io; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' https://www.google-analytics.com; frame-ancestors 'none'; base-uri 'self'; form-action 'self' mailto:;
```

### CSP Directives Explained

- `default-src 'self'` - Only load resources from same origin
- `script-src` - Allow scripts from self and analytics services
- `style-src 'self' 'unsafe-inline'` - Allow inline styles (for dynamic styling)
- `img-src 'self' data: https:` - Allow images from self, data URIs, and HTTPS
- `frame-ancestors 'none'` - Prevent clickjacking
- `base-uri 'self'` - Restrict base tag to same origin
- `form-action 'self' mailto:` - Only allow form submissions to self or mailto

## Additional Security Headers

### For Production Deployment

If using a custom domain with a CDN or reverse proxy, add these headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

### Header Explanations

- **X-Content-Type-Options**: Prevents MIME type sniffing (HTTP header only)
- **X-Frame-Options**: Prevents clickjacking attacks (HTTP header only)
- **X-XSS-Protection**: Browser XSS filter - legacy support (HTTP header only)
- **Referrer-Policy**: Controls referrer information (can use meta tag)
- **Permissions-Policy**: Restricts browser features (HTTP header only)
- **Strict-Transport-Security**: Forces HTTPS connections (HTTP header only)

**Important:** Most security headers require HTTP headers and cannot be set via meta tags. GitHub Pages provides HTTPS automatically but doesn't allow custom headers.

## GitHub Pages Security

### Built-in Protections

GitHub Pages provides:
- Automatic HTTPS via Let's Encrypt
- DDoS protection via Fastly CDN
- No server-side code execution
- Static file serving only
- Basic security headers (HSTS, X-Content-Type-Options)

### Limitations

- Cannot set custom HTTP headers (most security headers unavailable)
- CSP via meta tags only (some directives not supported)
- No server-side validation
- No rate limiting on forms
- No clickjacking protection (X-Frame-Options requires HTTP header)

### Workaround for Full Security Headers

To get full security header support:
1. Use a custom domain
2. Put Cloudflare in front (free plan includes security headers)
3. Or migrate to Netlify/Vercel (both support custom headers)

## Form Security

### Contact Form

The contact form uses:
1. Client-side validation (email format, required fields)
2. Input sanitization before processing
3. `mailto:` fallback (no server-side processing)
4. `encodeURIComponent` for URL encoding

### Recommendations for Production

For a production contact form, consider:
1. Backend API with rate limiting
2. CAPTCHA (reCAPTCHA, hCaptcha)
3. Server-side validation
4. Email service (SendGrid, Mailgun)
5. CSRF token protection

## Third-Party Integrations

### LemonSqueezy (Payment Processing)
- External checkout page (not embedded)
- No payment data handled on our site
- PCI DSS compliant (handled by LemonSqueezy)

### Analytics (Optional)
- Google Analytics or Plausible
- No PII collection
- Cookie consent banner recommended (GDPR)

## Dependency Security

### No External Dependencies

The website uses:
- Pure HTML/CSS/JavaScript
- No npm packages
- No build process
- No third-party libraries

This eliminates:
- Supply chain attacks
- Vulnerable dependencies
- Build-time injection risks

## Best Practices Implemented

### 1. Minimal Attack Surface
- Static site with no backend
- No user authentication
- No database
- No file uploads (except via desktop app)

### 2. Secure Defaults
- HTTPS enforced by GitHub Pages
- No cookies used
- No localStorage for sensitive data
- No session management

### 3. Input Validation
- Email format validation
- URL protocol validation
- Required field checks
- Length limits on form fields

### 4. Output Encoding
- `textContent` for text insertion
- `encodeURIComponent` for URLs
- HTML entity encoding for display

### 5. Safe JavaScript
- No `eval()` or `Function()`
- No `innerHTML` with user input
- No dynamic script loading
- Strict mode enabled

## GPU Acceleration Security

GPU acceleration is implemented safely:
- Uses CSS transforms only (`translateZ(0)`)
- No WebGL or Canvas manipulation
- No shader code execution
- Pure CSS animations

## Monitoring and Incident Response

### Recommended Monitoring

1. **GitHub Security Alerts**
   - Enable Dependabot (if using dependencies)
   - Monitor repository security tab

2. **Analytics Monitoring**
   - Watch for unusual traffic patterns
   - Monitor for spam form submissions

3. **User Reports**
   - Contact form for security reports
   - GitHub Issues for public disclosure

### Incident Response

If a security issue is discovered:
1. Assess severity and impact
2. Create a fix in a private branch
3. Test thoroughly
4. Deploy fix immediately
5. Document in changelog
6. Notify users if necessary

## Security Checklist

Before deploying:

- [ ] All user inputs sanitized
- [ ] No `innerHTML` with user data
- [ ] URL validation implemented
- [ ] Form validation in place
- [ ] HTTPS enforced
- [ ] CSP meta tags added (optional)
- [ ] External links use `rel="noopener"`
- [ ] No sensitive data in client-side code
- [ ] No API keys in JavaScript
- [ ] Analytics configured properly
- [ ] Contact form tested
- [ ] All links validated
- [ ] 404 page configured

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** open a public GitHub issue
2. Email: security@productmatcher.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work on a fix immediately.

## Regular Security Audits

Recommended schedule:
- **Monthly**: Review analytics for anomalies
- **Quarterly**: Check for new security best practices
- **Annually**: Full security audit
- **On updates**: Security review of all changes

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [MDN Web Security](https://developer.mozilla.org/en-US/docs/Web/Security)
- [GitHub Pages Security](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https)
- [Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)

## License

This security documentation is part of the Product Matcher project.

Last Updated: November 13, 2024
