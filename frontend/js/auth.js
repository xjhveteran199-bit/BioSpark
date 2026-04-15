/**
 * Auth module — full-page login/register gate.
 * Users must authenticate before accessing the main app.
 */

const Auth = (() => {
    const TOKEN_KEY = 'biospark_token';
    const USER_KEY = 'biospark_user';

    // --- Token storage ---

    function getToken() {
        return localStorage.getItem(TOKEN_KEY);
    }

    function getUser() {
        try {
            const raw = localStorage.getItem(USER_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch {
            return null;
        }
    }

    function _saveSession(token, user) {
        localStorage.setItem(TOKEN_KEY, token);
        localStorage.setItem(USER_KEY, JSON.stringify(user));
    }

    function _clearSession() {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
    }

    function isLoggedIn() {
        return !!getToken();
    }

    function authHeaders() {
        const token = getToken();
        if (!token) return {};
        return { 'Authorization': `Bearer ${token}` };
    }

    // --- Tab switching ---

    function switchTab(tab) {
        document.querySelectorAll('.auth-gate-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });
        document.getElementById('auth-login-form').classList.toggle('hidden', tab !== 'login');
        document.getElementById('auth-register-form').classList.toggle('hidden', tab !== 'register');
        // Clear errors
        document.getElementById('login-error').classList.add('hidden');
        document.getElementById('register-error').classList.add('hidden');
    }

    // --- Form handlers ---

    async function handleLogin(e) {
        e.preventDefault();
        const errorEl = document.getElementById('login-error');
        const btn = e.target.querySelector('.auth-submit-btn');
        errorEl.classList.add('hidden');
        btn.disabled = true;

        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;

        try {
            const resp = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Login failed');
            _saveSession(data.access_token, data.user);
            _enterApp();
        } catch (err) {
            errorEl.textContent = err.message;
            errorEl.classList.remove('hidden');
        } finally {
            btn.disabled = false;
        }
    }

    async function handleRegister(e) {
        e.preventDefault();
        const errorEl = document.getElementById('register-error');
        const btn = e.target.querySelector('.auth-submit-btn');
        errorEl.classList.add('hidden');
        btn.disabled = true;

        const email = document.getElementById('register-email').value.trim();
        const username = document.getElementById('register-username').value.trim();
        const password = document.getElementById('register-password').value;

        try {
            const resp = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, username, password }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Registration failed');
            _saveSession(data.access_token, data.user);
            _enterApp();
        } catch (err) {
            errorEl.textContent = err.message;
            errorEl.classList.remove('hidden');
        } finally {
            btn.disabled = false;
        }
    }

    function logout() {
        _clearSession();
        _showGate();
    }

    // --- View control ---

    function _enterApp() {
        if (window.LoginFX) LoginFX.destroy();
        document.getElementById('auth-gate').classList.add('hidden');
        document.getElementById('app-wrapper').classList.remove('hidden');
        _updateHeader();
    }

    function _showGate() {
        document.getElementById('auth-gate').classList.remove('hidden');
        document.getElementById('app-wrapper').classList.add('hidden');
        // Reset forms
        document.getElementById('auth-login-form').reset();
        document.getElementById('auth-register-form').reset();
        document.getElementById('login-error').classList.add('hidden');
        document.getElementById('register-error').classList.add('hidden');
        switchTab('login');
        // Start login animations
        setTimeout(() => { if (window.LoginFX) LoginFX.init(); }, 50);
    }

    function _updateHeader() {
        const user = getUser();
        const authArea = document.getElementById('auth-area');
        if (!authArea || !user) return;

        authArea.innerHTML = `
            <span class="auth-user-name">${_escHtml(user.username)}</span>
            <button class="auth-btn auth-btn-logout" onclick="Auth.logout()"
                data-en="Logout" data-zh="退出">Logout</button>
        `;

        if (window.App && App.lang) App.applyLang();
    }

    // --- Session check on page load ---

    async function checkSession() {
        const token = getToken();
        if (!token) {
            _showGate();
            return;
        }

        try {
            const resp = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (resp.ok) {
                const user = await resp.json();
                localStorage.setItem(USER_KEY, JSON.stringify(user));
                _enterApp();
            } else {
                _clearSession();
                _showGate();
            }
        } catch {
            // Network error — still try to enter if token exists
            _enterApp();
        }
    }

    function _escHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    return {
        getToken, getUser, isLoggedIn, authHeaders,
        switchTab, handleLogin, handleRegister, logout,
        checkSession, _updateHeader,
    };
})();

window.Auth = Auth;
