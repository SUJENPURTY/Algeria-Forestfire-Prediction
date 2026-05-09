document.addEventListener('DOMContentLoaded', () => {
    initStarfield();
    initNavbar();
    initReveal();
    initForm();
    initBars();
});

function initStarfield() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let stars = [];

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function makeStar() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.4,
            a: Math.random() * 0.5 + 0.1,
            v: Math.random() * 0.1 + 0.02,
            c: Math.random() > 0.85 ? '249,115,22' : '255,255,255'
        };
    }

    function init() {
        const n = Math.min(Math.floor((canvas.width * canvas.height) / 14000), 110);
        stars = Array.from({ length: n }, makeStar);
    }

    function drawStar(s) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${s.c}, ${s.a})`;
        ctx.fill();
    }

    function drawLine(s1, s2) {
        const dx = s1.x - s2.x;
        const dy = s1.y - s2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 140) {
            ctx.beginPath();
            ctx.moveTo(s1.x, s1.y);
            ctx.lineTo(s2.x, s2.y);
            ctx.strokeStyle = `rgba(249,115,22,${0.05 * (1 - dist / 140)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
    }

    function tick() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (const s of stars) {
            drawStar(s);
            s.y += s.v;
            if (s.y > canvas.height) {
                s.y = 0;
                s.x = Math.random() * canvas.width;
            }
        }
        for (let i = 0; i < stars.length; i++) {
            for (let j = i + 1; j < stars.length; j++) {
                drawLine(stars[i], stars[j]);
            }
        }
        requestAnimationFrame(tick);
    }

    resize();
    init();
    tick();

    window.addEventListener('resize', () => {
        resize();
        init();
    });
}

function initNavbar() {
    const nav = document.querySelector('.navbar');
    if (!nav) return;
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 20);
    });
}

function initReveal() {
    const items = document.querySelectorAll('.feature-card, .reveal');
    if (!items.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                setTimeout(() => entry.target.classList.add('visible'), i * 80);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });

    items.forEach(el => observer.observe(el));
}

function initForm() {
    const form = document.getElementById('predict-form');
    const overlay = document.getElementById('loading-overlay');
    if (!form) return;

    form.querySelectorAll('.field-input').forEach(input => {
        input.addEventListener('blur', () => checkField(input));
        input.addEventListener('input', () => {
            const group = input.closest('.field-group');
            if (group.classList.contains('has-err')) checkField(input);
        });
    });

    form.addEventListener('submit', () => {
        let ok = true;
        form.querySelectorAll('.field-input').forEach(inp => {
            if (!checkField(inp)) ok = false;
        });
        if (ok && overlay) overlay.classList.add('active');
    });
}

function checkField(input) {
    const group = input.closest('.field-group');
    const val = input.value.trim();

    group.classList.remove('has-err');
    input.classList.remove('err');

    if (input.dataset.req === '1' && !val) {
        group.classList.add('has-err');
        input.classList.add('err');
        return false;
    }
    if (val && input.dataset.num === '1') {
        if (isNaN(parseFloat(val))) {
            group.classList.add('has-err');
            input.classList.add('err');
            return false;
        }
    }
    return true;
}

function initBars() {
    const bars = document.querySelectorAll('.bar');
    if (!bars.length) return;
    const heights = [62, 38, 75, 50, 88, 42, 68, 35];
    bars.forEach((bar, i) => {
        bar.style.height = '0px';
        setTimeout(() => {
            bar.style.height = heights[i] + '%';
        }, i * 70 + 600);
    });
}
