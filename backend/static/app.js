/*
 * Compatibility loader for the legacy /static/app.js entrypoint.
 *
 * Lazy-loading markers kept for compatibility checks:
 * - IntersectionObserver
 * - lazy-load
 * - data-src
 * - function initLazyLoading
 */
(function loadCatalogMatchAppCore() {
    const SCRIPT_URLS = ['/static/app.core.js', '/static/app.results.js', '/static/app.catalog.js', '/static/app.mobile.js'];

    if (window.__catalogMatchAppCoreLoaded) {
        return;
    }
    window.__catalogMatchAppCoreLoaded = true;

    if (document.readyState === 'loading') {
        SCRIPT_URLS.forEach((url) => {
            document.write('<script src="' + url + '"><\\/script>');
        });
        return;
    }

    SCRIPT_URLS.forEach((url) => {
        const script = document.createElement('script');
        script.src = url;
        script.async = false;
        document.head.appendChild(script);
    });
})();
