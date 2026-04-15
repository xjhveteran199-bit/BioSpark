/**
 * Login Animations — Sci-fi Neural Gateway experience
 * Canvas particle network + bio-signal waveforms + HUD effects
 */

const LoginFX = (() => {
    let canvas, ctx, particles, mouse, animId, waveOffset = 0;
    let hueShift = 0;
    const CONFIG = {
        particleCount: 90,
        connectionDist: 150,
        mouseRadius: 200,
        baseSpeed: 0.3,
        colors: {
            cyan: [34, 211, 238],
            indigo: [99, 102, 241],
            pink: [244, 114, 182],
            purple: [139, 92, 246],
        }
    };

    // ── Particle Network ──────────────────────────────────

    class Particle {
        constructor(w, h) {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.vx = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.vy = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.r = Math.random() * 2 + 1;
            const keys = Object.keys(CONFIG.colors);
            this.color = CONFIG.colors[keys[Math.floor(Math.random() * keys.length)]];
            this.pulse = Math.random() * Math.PI * 2;
        }

        update(w, h, mx, my) {
            this.pulse += 0.02;

            // Mouse attraction
            if (mx !== null && my !== null) {
                const dx = mx - this.x;
                const dy = my - this.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONFIG.mouseRadius) {
                    const force = (CONFIG.mouseRadius - dist) / CONFIG.mouseRadius * 0.02;
                    this.vx += dx * force * 0.01;
                    this.vy += dy * force * 0.01;
                }
            }

            // Damping
            this.vx *= 0.99;
            this.vy *= 0.99;

            this.x += this.vx;
            this.y += this.vy;

            // Wrap
            if (this.x < 0) this.x = w;
            if (this.x > w) this.x = 0;
            if (this.y < 0) this.y = h;
            if (this.y > h) this.y = 0;
        }

        draw(ctx) {
            const glow = 0.6 + Math.sin(this.pulse) * 0.4;
            const [r, g, b] = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * (1 + Math.sin(this.pulse) * 0.3), 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r},${g},${b},${glow})`;
            ctx.fill();
            // Outer glow
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * 4, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r},${g},${b},${glow * 0.08})`;
            ctx.fill();
        }
    }

    function drawConnections(ctx) {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONFIG.connectionDist) {
                    const alpha = (1 - dist / CONFIG.connectionDist) * 0.25;
                    const [r, g, b] = particles[i].color;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }
    }

    // ── Bio-signal Waveforms ──────────────────────────────

    function drawBioWaveforms(ctx, w, h) {
        waveOffset += 0.015;

        // ECG-style waveform (top area)
        drawECGWave(ctx, w, h * 0.18, waveOffset, 'rgba(34,211,238,0.12)', 1.5);
        drawECGWave(ctx, w, h * 0.22, waveOffset + 1.5, 'rgba(99,102,241,0.08)', 1);

        // EEG-style waveform (bottom area)
        drawEEGWave(ctx, w, h * 0.78, waveOffset, 'rgba(244,114,182,0.10)', 1.2);
        drawEEGWave(ctx, w, h * 0.82, waveOffset + 2, 'rgba(139,92,246,0.07)', 0.8);
    }

    function drawECGWave(ctx, w, y, offset, color, lineW) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        const segLen = 120;
        for (let x = 0; x < w; x += 1) {
            const t = ((x + offset * 80) % segLen) / segLen;
            let val = 0;
            // P wave
            if (t > 0.1 && t < 0.2) val = Math.sin((t - 0.1) / 0.1 * Math.PI) * 8;
            // QRS complex
            else if (t > 0.3 && t < 0.33) val = -12 * ((t - 0.3) / 0.03);
            else if (t > 0.33 && t < 0.38) val = -12 + 60 * ((t - 0.33) / 0.05);
            else if (t > 0.38 && t < 0.42) val = 48 - 56 * ((t - 0.38) / 0.04);
            // T wave
            else if (t > 0.55 && t < 0.7) val = Math.sin((t - 0.55) / 0.15 * Math.PI) * 12;

            const yy = y + val * (0.8 + Math.sin(offset + x * 0.001) * 0.2);
            if (x === 0) ctx.moveTo(x, yy);
            else ctx.lineTo(x, yy);
        }
        ctx.stroke();
    }

    function drawEEGWave(ctx, w, y, offset, color, lineW) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        for (let x = 0; x < w; x += 1) {
            const val = Math.sin(x * 0.02 + offset * 3) * 10
                + Math.sin(x * 0.05 + offset * 5) * 5
                + Math.sin(x * 0.1 + offset * 8) * 3
                + Math.sin(x * 0.003 + offset) * 15;
            if (x === 0) ctx.moveTo(x, y + val);
            else ctx.lineTo(x, y + val);
        }
        ctx.stroke();
    }

    // ── Mouse-tracking Radial Glow ──────────────────────

    function drawMouseGlow(ctx) {
        if (mouse.x === null) return;
        const grad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 250);
        grad.addColorStop(0, 'rgba(99,102,241,0.06)');
        grad.addColorStop(0.5, 'rgba(34,211,238,0.02)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(mouse.x - 250, mouse.y - 250, 500, 500);
    }

    // ── Grid overlay ─────────────────────────────────────

    function drawGrid(ctx, w, h) {
        ctx.strokeStyle = 'rgba(99,102,241,0.035)';
        ctx.lineWidth = 0.5;
        const size = 50;
        for (let x = 0; x < w; x += size) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += size) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }

    // ── Scanning ring around logo ────────────────────────

    function drawScanRing(ctx, cx, cy) {
        hueShift += 0.005;
        for (let i = 0; i < 3; i++) {
            const r = 70 + i * 22;
            const a = 0.15 - i * 0.04;
            const rot = hueShift * (1 + i * 0.5);
            ctx.beginPath();
            ctx.arc(cx, cy, r, rot, rot + Math.PI * 1.2);
            ctx.strokeStyle = i === 0
                ? `rgba(34,211,238,${a})`
                : i === 1
                    ? `rgba(99,102,241,${a})`
                    : `rgba(244,114,182,${a})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        // Center glow
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 55);
        grad.addColorStop(0, 'rgba(99,102,241,0.08)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 55, 0, Math.PI * 2);
        ctx.fill();
    }

    // ── Main loop ────────────────────────────────────────

    function animate() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        drawGrid(ctx, w, h);
        drawBioWaveforms(ctx, w, h);
        drawMouseGlow(ctx);

        // Scan ring centered on logo
        const logoEl = document.querySelector('.auth-gate-logo');
        if (logoEl) {
            const rect = logoEl.getBoundingClientRect();
            const scale = window.devicePixelRatio || 1;
            drawScanRing(ctx, (rect.left + rect.width / 2) * scale, (rect.top + rect.height / 2) * scale);
        }

        particles.forEach(p => {
            p.update(w, h, mouse.x, mouse.y);
            p.draw(ctx);
        });
        drawConnections(ctx);

        animId = requestAnimationFrame(animate);
    }

    // ── Typing effect ────────────────────────────────────

    function typeText(el, text, speed = 40) {
        el.textContent = '';
        el.style.borderRight = '2px solid var(--neon-green)';
        let i = 0;
        const timer = setInterval(() => {
            el.textContent += text[i];
            i++;
            if (i >= text.length) {
                clearInterval(timer);
                // Blink cursor then remove
                setTimeout(() => { el.style.borderRight = 'none'; }, 2000);
            }
        }, speed);
    }

    // ── Floating HUD data panels ─────────────────────────

    function createHUDPanels() {
        const container = document.getElementById('auth-gate');
        if (!container) return;

        const panels = [
            { x: '2%', y: '8%', lines: ['SYS.NEURAL_LINK', '> STATUS: ONLINE', '> LATENCY: 2.3ms'] },
            { x: '80%', y: '8%', lines: ['BIO.SIGNAL_PROC', '> ECG: ■■■■■■□□', '> SAMPLING: 360Hz'] },
            { x: '2%', y: '85%', lines: ['SEC.AUTH_MODULE', '> AES-256-GCM', '> JWT VALID'] },
            { x: '80%', y: '85%', lines: ['AI.MODEL_STATUS', '> CNN: READY', '> PARAMS: 44,293'] },
        ];

        panels.forEach((p, i) => {
            const el = document.createElement('div');
            el.className = 'hud-panel';
            el.style.left = p.x;
            el.style.top = p.y;
            el.style.animationDelay = `${i * 0.3}s`;
            el.innerHTML = p.lines.map((l, j) =>
                j === 0
                    ? `<div class="hud-panel-title">${l}</div>`
                    : `<div class="hud-panel-line">${l}</div>`
            ).join('');
            container.appendChild(el);
        });
    }

    // ── Hexagon decorations ─────────────────────────────

    function createHexDecorations() {
        const container = document.getElementById('auth-gate');
        if (!container) return;

        const count = 6;
        for (let i = 0; i < count; i++) {
            const hex = document.createElement('div');
            hex.className = 'hex-decoration';
            hex.style.left = `${Math.random() * 90 + 5}%`;
            hex.style.top = `${Math.random() * 90 + 5}%`;
            hex.style.animationDelay = `${Math.random() * 5}s`;
            hex.style.animationDuration = `${6 + Math.random() * 4}s`;
            container.appendChild(hex);
        }
    }

    // ── Staggered form entrance ──────────────────────────

    function staggerFormEntrance() {
        const items = document.querySelectorAll('.auth-gate-container > *');
        items.forEach((el, i) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(25px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            setTimeout(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 300 + i * 150);
        });
    }

    // ── Init / Destroy ───────────────────────────────────

    function init() {
        const gate = document.getElementById('auth-gate');
        if (!gate || gate.classList.contains('hidden')) return;

        // Canvas setup
        canvas = document.getElementById('auth-canvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');
        const scale = window.devicePixelRatio || 1;

        function resize() {
            canvas.width = window.innerWidth * scale;
            canvas.height = window.innerHeight * scale;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
        }
        resize();
        window.addEventListener('resize', resize);

        // Mouse tracking
        mouse = { x: null, y: null };
        window.addEventListener('mousemove', e => {
            mouse.x = e.clientX * scale;
            mouse.y = e.clientY * scale;
        });
        window.addEventListener('mouseleave', () => {
            mouse.x = null;
            mouse.y = null;
        });

        // Create particles
        particles = [];
        for (let i = 0; i < CONFIG.particleCount; i++) {
            particles.push(new Particle(canvas.width, canvas.height));
        }

        // Create HUD & decorations
        createHUDPanels();
        createHexDecorations();

        // Typing effect on subtitle
        const subtitle = document.querySelector('.auth-gate-subtitle');
        if (subtitle) {
            const lang = (window.App && App.lang) || 'en';
            const text = lang === 'zh'
                ? subtitle.getAttribute('data-zh') || subtitle.textContent
                : subtitle.getAttribute('data-en') || subtitle.textContent;
            setTimeout(() => typeText(subtitle, text, 35), 800);
        }

        // Stagger entrance
        staggerFormEntrance();

        // Start animation
        animate();
    }

    function destroy() {
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        // Remove HUD panels and hex decorations
        document.querySelectorAll('.hud-panel, .hex-decoration').forEach(el => el.remove());
    }

    return { init, destroy };
})();

window.LoginFX = LoginFX;
