// ======================================
// SafeCross Interactive Simulation
// ======================================

class Vehicle {
    constructor(type, x, y, speed, lane) {
        this.type = type;
        this.x = x;
        this.y = y;
        this.speed = speed;
        this.lane = lane;
        this.width = this.getWidth();
        this.height = this.getHeight();
        this.confidence = 0.7 + Math.random() * 0.27; // 0.70 - 0.97
        this.displacement = 0;
        this.moving = true;
        this.color = this.getColor();
    }

    getWidth() {
        const sizes = { car: 80, bus: 120, truck: 100, motorbike: 50 };
        return sizes[this.type] || 80;
    }

    getHeight() {
        const sizes = { car: 40, bus: 50, truck: 45, motorbike: 30 };
        return sizes[this.type] || 40;
    }

    getColor() {
        const colors = {
            car: '#3498db',
            bus: '#e74c3c',
            truck: '#f39c12',
            motorbike: '#9b59b6'
        };
        return colors[this.type] || '#3498db';
    }

    update(deltaTime, speed) {
        if (this.moving) {
            this.x += this.speed * deltaTime * speed;
            this.displacement = Math.abs(this.speed * deltaTime * speed * 10);
        }
    }

    draw(ctx, showBoxes) {
        // Draw vehicle body
        ctx.fillStyle = this.color;
        ctx.fillRect(this.x, this.y, this.width, this.height);

        // Draw vehicle details based on type
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        if (this.type === 'car') {
            // Windows
            ctx.fillRect(this.x + 10, this.y + 5, 25, 15);
            ctx.fillRect(this.x + 45, this.y + 5, 25, 15);
        } else if (this.type === 'bus') {
            // Multiple windows
            for (let i = 0; i < 4; i++) {
                ctx.fillRect(this.x + 10 + i * 25, this.y + 10, 20, 20);
            }
        } else if (this.type === 'truck') {
            // Cab
            ctx.fillRect(this.x + 10, this.y + 5, 30, 25);
        } else if (this.type === 'motorbike') {
            // Rider
            ctx.fillRect(this.x + 15, this.y + 3, 20, 15);
        }

        // Draw wheels
        ctx.fillStyle = '#2c3e50';
        const wheelY = this.y + this.height - 5;
        if (this.type === 'bus' || this.type === 'truck') {
            ctx.fillRect(this.x + 15, wheelY, 10, 8);
            ctx.fillRect(this.x + this.width / 2, wheelY, 10, 8);
            ctx.fillRect(this.x + this.width - 25, wheelY, 10, 8);
        } else if (this.type === 'motorbike') {
            ctx.beginPath();
            ctx.arc(this.x + 15, wheelY, 5, 0, Math.PI * 2);
            ctx.arc(this.x + 35, wheelY, 5, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.fillRect(this.x + 10, wheelY, 10, 8);
            ctx.fillRect(this.x + this.width - 20, wheelY, 10, 8);
        }

        // Draw bounding box
        if (showBoxes) {
            const boxColor = this.moving ? '#00ff88' : '#3366ff';
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(this.x - 5, this.y - 5, this.width + 10, this.height + 10);

            // Draw label
            ctx.fillStyle = boxColor;
            ctx.font = '12px Space Grotesk, sans-serif';
            const label = `${this.type} (${this.confidence.toFixed(2)})`;
            ctx.fillText(label, this.x, this.y - 20);

            // Draw displacement
            ctx.fillStyle = '#a0a0a0';
            ctx.font = '10px Space Grotesk, sans-serif';
            ctx.fillText(`Disp: ${this.displacement.toFixed(2)}`, this.x, this.y + this.height + 15);
        }
    }

    isOffScreen(canvasWidth) {
        return this.x > canvasWidth + 50;
    }

    getBoundingBox() {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height,
            centerY: this.y + this.height / 2
        };
    }
}

class SafeCrossSimulation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        this.vehicles = [];
        this.pedestrian = { x: 50, y: 0 };
        this.running = true;
        this.speed = 1.0;
        this.showCriticalZone = true;
        this.showBoundingBoxes = true;
        
        this.setupCanvas();

        // Safety decision
        this.safetyHistory = [];
        this.historyWindowSize = 50; // 2 seconds at ~25 FPS
        this.safetyThreshold = 0.7;
        this.currentDecision = true; // true = safe, false = unsafe

        // Animation
        this.lastTime = 0;
        this.spawnTimer = 0;
        this.spawnInterval = 2000; // ms

        // Initialize with some vehicles
        this.addVehicle();
        this.addVehicle();

        // Start animation
        requestAnimationFrame((time) => this.animate(time));
    }

    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        // Update pedestrian position
        this.pedestrian.y = this.canvas.height / 2;

        // Handle resize
        window.addEventListener('resize', () => {
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.pedestrian.y = this.canvas.height / 2;
        });
    }

    addVehicle() {
        const types = ['car', 'bus', 'truck', 'motorbike'];
        const type = types[Math.floor(Math.random() * types.length)];
        const lanes = [0.3, 0.4, 0.5, 0.6]; // Different y positions
        const lane = lanes[Math.floor(Math.random() * lanes.length)];
        const y = this.canvas.height * lane;
        const speed = 50 + Math.random() * 100; // pixels per second
        const vehicle = new Vehicle(type, -100, y, speed, lane);
        this.vehicles.push(vehicle);
    }

    clearRoad() {
        this.vehicles = [];
    }

    togglePause() {
        this.running = !this.running;
        return this.running;
    }

    setSpeed(speed) {
        this.speed = speed;
    }

    toggleCriticalZone() {
        this.showCriticalZone = !this.showCriticalZone;
        return this.showCriticalZone;
    }

    toggleBoundingBoxes() {
        this.showBoundingBoxes = !this.showBoundingBoxes;
        return this.showBoundingBoxes;
    }

    drawScene() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        // Draw sidewalk (top)
        ctx.fillStyle = '#555555';
        ctx.fillRect(0, 0, width, height * 0.2);

        // Draw road
        ctx.fillStyle = '#333333';
        ctx.fillRect(0, height * 0.2, width, height * 0.6);

        // Draw lane markings
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([20, 10]);
        for (let i = 1; i < 3; i++) {
            const y = height * 0.2 + (height * 0.6 * i / 3);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Draw sidewalk (bottom)
        ctx.fillStyle = '#555555';
        ctx.fillRect(0, height * 0.8, width, height * 0.2);

        // Draw critical zone overlay
        if (this.showCriticalZone) {
            ctx.fillStyle = 'rgba(255, 51, 102, 0.15)';
            const criticalZoneStart = height * 0.4; // Lower 60% of frame
            ctx.fillRect(0, criticalZoneStart, width, height - criticalZoneStart);

            // Draw critical zone label
            ctx.fillStyle = 'rgba(255, 51, 102, 0.5)';
            ctx.font = '12px Space Grotesk, sans-serif';
            ctx.fillText('Critical Zone (lower 60%)', 10, criticalZoneStart + 15);
        }

        // Draw pedestrian
        ctx.fillStyle = '#00ff88';
        ctx.beginPath();
        ctx.arc(this.pedestrian.x, this.pedestrian.y, 15, 0, Math.PI * 2);
        ctx.fill();

        // Draw pedestrian body
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(this.pedestrian.x, this.pedestrian.y + 15);
        ctx.lineTo(this.pedestrian.x, this.pedestrian.y + 40);
        ctx.stroke();

        // Draw arms
        ctx.beginPath();
        ctx.moveTo(this.pedestrian.x - 10, this.pedestrian.y + 25);
        ctx.lineTo(this.pedestrian.x + 10, this.pedestrian.y + 25);
        ctx.stroke();

        // Draw legs
        ctx.beginPath();
        ctx.moveTo(this.pedestrian.x, this.pedestrian.y + 40);
        ctx.lineTo(this.pedestrian.x - 8, this.pedestrian.y + 55);
        ctx.moveTo(this.pedestrian.x, this.pedestrian.y + 40);
        ctx.lineTo(this.pedestrian.x + 8, this.pedestrian.y + 55);
        ctx.stroke();

        // Label
        ctx.fillStyle = '#00ff88';
        ctx.font = '12px Space Grotesk, sans-serif';
        ctx.fillText('Pedestrian', this.pedestrian.x - 25, this.pedestrian.y - 25);
    }

    updateSafetyDecision() {
        // Check if any moving vehicle is in critical zone
        const criticalZoneY = this.canvas.height * 0.4;
        let unsafe = false;

        for (const vehicle of this.vehicles) {
            const bbox = vehicle.getBoundingBox();
            const vehicleTypes = ['bus', 'car', 'truck', 'motorbike'];

            if (vehicle.moving &&
                vehicleTypes.includes(vehicle.type) &&
                bbox.centerY > criticalZoneY) {
                unsafe = true;
                break;
            }
        }

        // Add to history
        this.safetyHistory.push(!unsafe);
        if (this.safetyHistory.length > this.historyWindowSize) {
            this.safetyHistory.shift();
        }

        // Calculate smoothed decision
        if (this.safetyHistory.length > 0) {
            const safeCount = this.safetyHistory.filter(v => v).length;
            const ratio = safeCount / this.safetyHistory.length;
            this.currentDecision = ratio >= this.safetyThreshold;
        } else {
            this.currentDecision = !unsafe;
        }

        return this.currentDecision;
    }

    updateUI() {
        // Update safety banner
        const banner = document.getElementById('safetyBanner');
        if (this.currentDecision) {
            banner.textContent = '✅ SAFE TO CROSS';
            banner.className = 'safety-banner safe';
        } else {
            banner.textContent = '🚫 DO NOT CROSS';
            banner.className = 'safety-banner unsafe';
        }

        // Update vibration motors
        const motorLeft = document.getElementById('motorLeft');
        const motorRight = document.getElementById('motorRight');

        if (!this.currentDecision) {
            motorLeft.classList.add('active');
            motorRight.classList.add('active');
        } else {
            motorLeft.classList.remove('active');
            motorRight.classList.remove('active');
        }

        // Update safety history bar
        const historyBar = document.getElementById('safetyHistoryBar');
        historyBar.innerHTML = '';
        this.safetyHistory.forEach((safe) => {
            const segment = document.createElement('div');
            segment.className = `history-segment ${safe ? 'safe' : 'unsafe'}`;
            historyBar.appendChild(segment);
        });

        // Fill remaining segments
        const remaining = this.historyWindowSize - this.safetyHistory.length;
        for (let i = 0; i < remaining; i++) {
            const segment = document.createElement('div');
            segment.className = 'history-segment';
            segment.style.background = '#0a0a0a';
            historyBar.appendChild(segment);
        }
    }

    animate(currentTime) {
        const deltaTime = (currentTime - this.lastTime) / 1000; // Convert to seconds
        this.lastTime = currentTime;

        if (this.running) {
            // Auto-spawn vehicles
            this.spawnTimer += deltaTime * 1000;
            if (this.spawnTimer > this.spawnInterval) {
                this.addVehicle();
                this.spawnTimer = 0;
            }

            // Update vehicles
            this.vehicles = this.vehicles.filter(vehicle => {
                vehicle.update(deltaTime, this.speed);
                return !vehicle.isOffScreen(this.canvas.width);
            });

            // Update safety decision
            this.updateSafetyDecision();
            this.updateUI();
        }

        // Draw scene
        this.drawScene();

        // Draw vehicles
        this.vehicles.forEach(vehicle => {
            vehicle.draw(this.ctx, this.showBoundingBoxes);
        });

        requestAnimationFrame((time) => this.animate(time));
    }
}

// Initialize simulation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const simulation = new SafeCrossSimulation('simulationCanvas');

    // Control panel event listeners
    document.getElementById('playPauseBtn').addEventListener('click', (e) => {
        const running = simulation.togglePause();
        e.target.textContent = running ? '⏸ Pause' : '▶ Play';
    });

    document.getElementById('addVehicleBtn').addEventListener('click', () => {
        simulation.addVehicle();
    });

    document.getElementById('clearRoadBtn').addEventListener('click', () => {
        simulation.clearRoad();
    });

    document.getElementById('speedSlider').addEventListener('input', (e) => {
        const speed = parseFloat(e.target.value);
        simulation.setSpeed(speed);
        document.getElementById('speedValue').textContent = speed.toFixed(1);
    });

    document.getElementById('showCriticalZone').addEventListener('change', (e) => {
        simulation.toggleCriticalZone();
    });

    document.getElementById('showBoundingBoxes').addEventListener('change', (e) => {
        simulation.toggleBoundingBoxes();
    });
});
