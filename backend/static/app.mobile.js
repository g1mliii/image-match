/*
 * Mobile integration module extracted from app.core.js.
 * Keeps mobile upload polling and modal/connectivity actions isolated.
 */

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
                if (!response.ok) return;
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


