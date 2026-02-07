// ======================================
// Main JavaScript for SafeCross Website
// ======================================

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add scroll animation effects
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe sections for fade-in animations
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });

    // Parallax effect for hero section
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero');
        if (hero) {
            const heroVisual = hero.querySelector('.hero-visual');
            if (heroVisual) {
                heroVisual.style.transform = `translate(-50%, -50%) translateY(${scrolled * 0.5}px)`;
            }
        }
    });

    // Add hover effect to pipeline steps
    const pipelineSteps = document.querySelectorAll('.pipeline-step');
    pipelineSteps.forEach((step, index) => {
        step.style.transitionDelay = `${index * 0.1}s`;
    });

    // Code block syntax highlighting (simple version)
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        let html = block.innerHTML;

        // Simple syntax highlighting
        html = html.replace(/\b(def|class|import|from|if|else|elif|for|while|return|True|False|None)\b/g, '<span style="color: #ff79c6">$1</span>');
        html = html.replace(/\b(cv2|np|numpy|detections|net)\b/g, '<span style="color: #8be9fd">$1</span>');
        html = html.replace(/#.*/g, '<span style="color: #6272a4">$&</span>');
        html = html.replace(/"([^"]*)"/g, '<span style="color: #f1fa8c">"$1"</span>');
        html = html.replace(/'([^']*)'/g, '<span style="color: #f1fa8c">\'$1\'</span>');
        html = html.replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #bd93f9">$1</span>');

        block.innerHTML = html;
    });

    // Add loading animation to tech badges
    const techBadges = document.querySelectorAll('.tech-badge');
    techBadges.forEach((badge, index) => {
        badge.style.animation = `fadeIn 0.5s ease ${index * 0.1}s both`;
    });

    // Dynamic year in footer
    const currentYear = new Date().getFullYear();
    const yearElements = document.querySelectorAll('.footer-bottom p');
    yearElements.forEach(el => {
        if (el.textContent.includes('2025')) {
            el.textContent = el.textContent.replace('2025', currentYear);
        }
    });

    // Add intersection observer for pipeline steps
    const pipelineObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0) scale(1)';
                }, index * 100);
            }
        });
    }, { threshold: 0.1 });

    pipelineSteps.forEach(step => {
        step.style.opacity = '0';
        step.style.transform = 'translateY(20px) scale(0.95)';
        step.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        pipelineObserver.observe(step);
    });

    // Add fade-in animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(style);

    console.log('SafeCross website loaded successfully! 🚀');
});
