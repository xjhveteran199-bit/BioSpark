/**
 * Auth module — handles registration, login, logout, and token management.
 * Stores JWT token in localStorage. Exposes Auth global.
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

    // --- Auth headers helper ---

    function authHeaders() {
        const token = getToken();
        if (!token) return {};
        return { 'Authorization': `Bearer ${token}` };
    }

    // --- API calls ---

    async function register(email, username, password) {
        const resp = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, username, password }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Registration failed');
        _saveSession(data.access_token, data.user);
        _updateUI();
        return data.user;
    }

    async function login(username, password) {
        const resp = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Login failed');
        _saveSession(data.access_token, data.user);
        _updateUI();
        return data.user;
    }

    function logout() {
        _clearSession();
        _updateUI();
    }

    // --- Verify saved token on page load ---

    async function checkSession() {
        const token = getToken();
        if (!token) { _updateUI(); return; }

        try {
            const resp = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (resp.ok) {
                const user = await resp.json();
                localStorage.setItem(USER_KEY, JSON.stringify(user));
                _updateUI();
            } else {
                _clearSession();
                _updateUI();
            }
        } catch {
            // Network error — keep session, user might be offline
            _updateUI();
        }
    }

    // --- UI Updates ---

    function _updateUI() {
        const loggedIn = isLoggedIn();
        const user = getUser();

        // Auth area in header
        const authArea = document.getElementById('auth-area');
        if (!authArea) return;

        if (loggedIn && user) {
            authArea.innerHTML = `
                <span class="auth-user-name">${_escHtml(user.username)}</span>
                <button class="auth-btn auth-btn-logout" onclick="Auth.logout()"
                    data-en="Logout" data-zh="退出">Logout</button>
            `;
        } else {
            authArea.innerHTML = `
                <button class="auth-btn auth-btn-login" onclick="Auth.showModal('login')"
                    data-en="Login" data-zh="登录">Login</button>
                <button class="auth-btn auth-btn-register" onclick="Auth.showModal('register')"
                    data-en="Register" data-zh="注册">Register</button>
            `;
        }

        // Re-apply language if App is loaded
        if (window.App && App.lang) App.applyLang();
    }

    // --- Modal ---

    function showModal(mode) {
        const isZh = window.App && App.lang === 'zh';
        const isLogin = mode === 'login';

        const title = isLogin
            ? (isZh ? '登录' : 'Login')
            : (isZh ? '注册' : 'Create Account');

        const switchText = isLogin
            ? (isZh ? '没有账号？<a href="#" onclick="Auth.showModal(\'register\'); return false;">注册</a>' : 'No account? <a href="#" onclick="Auth.showModal(\'register\'); return false;">Register</a>')
            : (isZh ? '已有账号？<a href="#" onclick="Auth.showModal(\'login\'); return false;">登录</a>' : 'Already have an account? <a href="#" onclick="Auth.showModal(\'login\'); return false;">Login</a>');

        const emailField = isLogin ? '' : `
            <label class="auth-field">
                <span>${isZh ? '邮箱' : 'Email'}</span>
                <input type="email" id="auth-email" placeholder="${isZh ? '你的邮箱' : 'your@email.com'}" required>
            </label>`;

        const overlay = document.createElement('div');
        overlay.id = 'auth-overlay';
        overlay.className = 'auth-overlay';
        overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };

        overlay.innerHTML = `
            <div class="auth-modal">
                <button class="auth-modal-close" onclick="Auth.closeModal()">&times;</button>
                <h2>${title}</h2>
                <form id="auth-form" onsubmit="Auth.handleSubmit(event, '${mode}')">
                    ${emailField}
                    <label class="auth-field">
                        <span>${isZh ? '用户名' : 'Username'}</span>
                        <input type="text" id="auth-username" placeholder="${isZh ? '用户名' : 'username'}" required minlength="2">
                    </label>
                    <label class="auth-field">
                        <span>${isZh ? '密码' : 'Password'}</span>
                        <input type="password" id="auth-password" placeholder="${isZh ? '密码（至少6位）' : 'password (min 6 chars)'}" required minlength="6">
                    </label>
                    <div id="auth-error" class="auth-error hidden"></div>
                    <button type="submit" class="btn primary auth-submit-btn">${title}</button>
                </form>
                <p class="auth-switch">${switchText}</p>
            </div>
        `;

        // Remove existing overlay if present
        closeModal();
        document.body.appendChild(overlay);

        // Focus first input
        setTimeout(() => {
            const first = overlay.querySelector('input');
            if (first) first.focus();
        }, 100);
    }

    function closeModal() {
        const existing = document.getElementById('auth-overlay');
        if (existing) existing.remove();
    }

    async function handleSubmit(e, mode) {
        e.preventDefault();
        const errorEl = document.getElementById('auth-error');
        const submitBtn = document.querySelector('.auth-submit-btn');
        errorEl.classList.add('hidden');
        submitBtn.disabled = true;

        const username = document.getElementById('auth-username').value.trim();
        const password = document.getElementById('auth-password').value;

        try {
            if (mode === 'register') {
                const email = document.getElementById('auth-email').value.trim();
                await register(email, username, password);
            } else {
                await login(username, password);
            }
            closeModal();
        } catch (err) {
            errorEl.textContent = err.message;
            errorEl.classList.remove('hidden');
            submitBtn.disabled = false;
        }
    }

    // --- Helper ---

    function _escHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    return {
        getToken, getUser, isLoggedIn, authHeaders,
        register, login, logout, checkSession,
        showModal, closeModal, handleSubmit,
        _updateUI,
    };
})();

window.Auth = Auth;
