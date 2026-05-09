document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavbar();
    initReveal();
    initForm();
});

function initParticles() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let stars = [];
    let frameId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createStar() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.4 + 0.3,
            opacity: Math.random() * 0.5 + 0.1,
            speed: Math.random() * 0.15 + 0.02,
            color: Math.random() > 0.8 ? '249, 115, 22' : '255, 255, 255'
        };
    }

    function initStars() {
        const n = Math.min(Math.floor((canvas.width * canvas.height) / 12000), 100);
        stars = Array.from({ length: n }, createStar);
    }

    function drawStar(s) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${s.color}, ${s.opacity})`;
        ctx.fill();
    }

    function drawLink(s1, s2) {
        const dx = s1.x - s2.x;
        const dy = s1.y - s2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
            ctx.beginPath();
            ctx.moveTo(s1.x, s1.y);
            ctx.lineTo(s2.x, s2.y);
            ctx.strokeStyle = `rgba(249, 115, 22, ${0.06 * (1 - dist / 130)})`;
            ctx.lineWidth = 0.6;
            ctx.stroke();
        }
    }

    function tick() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < stars.length; i++) {
            drawStar(stars[i]);
            for (let j = i + 1; j < stars.length; j++) {
                drawLink(stars[i], stars[j]);
            }
            stars[i].y += stars[i].speed;
            if (stars[i].y > canvas.height) {
                stars[i].y = 0;
                stars[i].x = Math.random() * canvas.width;
            }
        }
        frameId = requestAnimationFrame(tick);
    }

    resize();
    initStars();
    tick();

    window.addEventListener('resize', () => {
        resize();
        initStars();
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
    const items = document.querySelectorAll('.feature-card, .reveal-item');
    if (!items.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                setTimeout(() => entry.target.classList.add('visible'), i * 80);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

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
            if (group.classList.contains('has-error')) checkField(input);
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

    group.classList.remove('has-error');
    input.classList.remove('field-error');

    if (input.dataset.req === '1' && !val) {
        group.classList.add('has-error');
        input.classList.add('field-error');
        return false;
    }
    if (val && input.dataset.num === '1') {
        if (isNaN(parseFloat(val))) {
            group.classList.add('has-error');
            input.classList.add('field-error');
            return false;
        }
    }
    return true;
}

function animateBars() {
    const bars = document.querySelectorAll('.chart-bar');
    if (!bars.length) return;
    const heights = [62, 38, 75, 50, 88, 42, 68, 35, 80];
    bars.forEach((bar, i) => {
        bar.style.height = '0px';
        setTimeout(() => {
            bar.style.height = heights[i] + '%';
        }, i * 80);
    });
}

if (document.querySelector('.chart-bar')) {
    animateBars();
}
