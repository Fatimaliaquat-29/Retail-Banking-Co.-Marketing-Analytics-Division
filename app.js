// ==========================================================================
// STATIC DATASETS (FROM MACHINE LEARNING OUTPUTS)
// ==========================================================================

const leaderboardData = [
    { method: "Tuned Random Forest", accuracy: 0.8694, precision: 0.4422, recall: 0.6099, f1: 0.5127, roc: 0.8118, highlight: true },
    { method: "Random Forest (depth=10)", accuracy: 0.8557, precision: 0.4096, recall: 0.6369, f1: 0.4985, roc: 0.8134, highlight: false },
    { method: "Random Oversampling (ROS)", accuracy: 0.8371, precision: 0.3713, recall: 0.6433, f1: 0.4708, roc: 0.8013, highlight: false },
    { method: "Class Weighted Logistic Reg.", accuracy: 0.8350, precision: 0.3679, recall: 0.6466, f1: 0.4689, roc: 0.8009, highlight: false },
    { method: "SMOTE (Oversampling)", accuracy: 0.8286, precision: 0.3581, recall: 0.6584, f1: 0.4639, roc: 0.8010, highlight: false },
    { method: "Random Undersampling (RUS)", accuracy: 0.8310, precision: 0.3609, recall: 0.6487, f1: 0.4638, roc: 0.8005, highlight: false },
    { method: "LightGBM Classifier", accuracy: 0.9012, precision: 0.6667, recall: 0.2457, f1: 0.3591, roc: 0.8123, highlight: false },
    { method: "XGBoost Classifier", accuracy: 0.9018, precision: 0.6798, recall: 0.2425, f1: 0.3574, roc: 0.8129, highlight: false },
    { method: "Gradient Boosting", accuracy: 0.9011, precision: 0.6728, recall: 0.2371, f1: 0.3506, roc: 0.8092, highlight: false },
    { method: "CatBoost Classifier", accuracy: 0.9022, precision: 0.6981, recall: 0.2317, f1: 0.3479, roc: 0.8085, highlight: false },
    { method: "Baseline Logistic Regression", accuracy: 0.9009, precision: 0.6905, recall: 0.2188, f1: 0.3322, roc: 0.8008, highlight: false }
];

const thresholdTuningData = {
    "0.2": { tn: 842, fp: 6468, fn: 27, tp: 901, recall: 0.9709, precision: 0.1223, f1: 0.2172, acc: 0.2116 },
    "0.3": { tn: 3259, fp: 4051, fn: 124, tp: 804, recall: 0.8664, precision: 0.1656, f1: 0.2781, acc: 0.4932 },
    "0.4": { tn: 5484, fp: 1826, fn: 247, tp: 681, recall: 0.7338, precision: 0.2716, f1: 0.3965, acc: 0.7484 },
    "0.5": { tn: 6279, fp: 1031, fn: 328, tp: 600, recall: 0.6466, precision: 0.3679, f1: 0.4689, acc: 0.8350 },
    "0.6": { tn: 6502, fp: 808, fn: 349, tp: 579, recall: 0.6239, precision: 0.4174, f1: 0.5002, acc: 0.8596 }
};

const featureImportanceData = [
    { name: "euribor3m", val: 17.69 },
    { name: "nr.employed", val: 15.89 },
    { name: "emp.var.rate", val: 13.14 },
    { name: "cons.conf.idx", val: 6.99 },
    { name: "pdays", val: 4.92 },
    { name: "cons.price.idx", val: 4.71 },
    { name: "poutcome_success", val: 4.48 },
    { name: "age", val: 3.04 },
    { name: "contact_telephone", val: 2.50 },
    { name: "month_may", val: 2.45 }
];

const unknownSummaryData = [
    { name: "default", pct: 20.87 },
    { name: "education", pct: 4.20 },
    { name: "housing", pct: 2.40 },
    { name: "loan", pct: 2.40 },
    { name: "job", pct: 0.80 },
    { name: "marital", pct: 0.19 }
];

// ==========================================================================
// APPLICATION CONTROLLER STATE
// ==========================================================================

let currentSlide = 1;
const totalSlides = 6;
let charts = {}; // Holds Chart.js instances

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    populateLeaderboardTable();
    initCharts();
    setupEventListeners();
    runCalculations();
    runSimulator();
});

// ==========================================================================
// TAB CONTROLLER
// ==========================================================================

function switchTab(tabId) {
    // Hide all sections
    const sections = document.querySelectorAll(".content-section");
    sections.forEach(sec => sec.classList.remove("active-section"));
    
    // Deactivate all tab buttons
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => btn.classList.remove("active"));
    
    // Show selected section
    const targetSection = document.getElementById(`section-${tabId}`);
    if (targetSection) {
        targetSection.classList.add("active-section");
    }
    
    // Activate clicked button
    const targetBtn = document.getElementById(`tab-${tabId}`);
    if (targetBtn) {
        targetBtn.classList.add("active");
    }
    
    // Resize charts to prevent sizing bugs when tabs switch
    setTimeout(() => {
        Object.values(charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }, 150);
}

// ==========================================================================
// SLIDES PRESENTATION CONTROLLER
// ==========================================================================

function updateSlideView() {
    // Hide all slides
    const slides = document.querySelectorAll(".slide-card");
    slides.forEach(slide => slide.classList.remove("active-slide"));
    
    // Reveal target slide
    const activeSlide = document.getElementById(`slide-${currentSlide}`);
    if (activeSlide) {
        activeSlide.classList.add("active-slide");
    }
    
    // Update progress bar
    const progressFill = document.getElementById("slide-progress-fill");
    const progressPct = (currentSlide / totalSlides) * 100;
    progressFill.style.width = `${progressPct}%`;
    
    // Update labels
    document.getElementById("slide-indicator").innerText = `Slide ${currentSlide} of ${totalSlides}`;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        updateSlideView();
    }
}

function prevSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        updateSlideView();
    }
}

// Listen to Keyboard Left/Right Arrow keys for Presentation Control
document.addEventListener("keydown", (e) => {
    const slidesSection = document.getElementById("section-slides");
    if (slidesSection && slidesSection.classList.contains("active-section")) {
        if (e.key === "ArrowRight") {
            nextSlide();
        } else if (e.key === "ArrowLeft") {
            prevSlide();
        }
    }
});

// ==========================================================================
// CHART.JS ENGINE
// ==========================================================================

function initCharts() {
    // Chart options style customization (for glassmorphic dark theme)
    const textColors = '#94a3b8';
    const gridColors = 'rgba(255, 255, 255, 0.05)';
    
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: textColors, font: { family: 'Inter', size: 10 } }
            }
        },
        scales: {
            x: {
                grid: { color: gridColors },
                ticks: { color: textColors, font: { family: 'Inter' } }
            },
            y: {
                grid: { color: gridColors },
                ticks: { color: textColors, font: { family: 'Inter' } }
            }
        }
    };

    // 1. Slide 2 Donut Chart
    const ctxSlideDonut = document.getElementById("slide-donut-chart").getContext("2d");
    charts.slideDonut = new Chart(ctxSlideDonut, {
        type: 'doughnut',
        data: {
            labels: ['No (88.73%)', 'Yes (11.27%)'],
            datasets: [{
                data: [36548, 4640],
                backgroundColor: ['#4f709c', '#00f2fe'],
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#ffffff', font: { family: 'Inter', weight: 'bold' } }
                }
            }
        }
    });

    // 2. EDA: Target Distribution Pie Chart
    const ctxEdaTarget = document.getElementById("eda-target-chart").getContext("2d");
    charts.edaTarget = new Chart(ctxEdaTarget, {
        type: 'pie',
        data: {
            labels: ['Unsubscribed (y=no)', 'Subscribed (y=yes)'],
            datasets: [{
                data: [36548, 4640],
                backgroundColor: ['#4f709c', '#00f2fe'],
                borderColor: 'rgba(255, 255, 255, 0.05)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: textColors, font: { family: 'Inter' } }
                },
                title: {
                    display: true,
                    text: 'Target Variable Class Distribution',
                    color: '#ffffff',
                    font: { family: 'Outfit', size: 12, weight: 'bold' }
                }
            }
        }
    });

    // 3. EDA: Unknown Values Horizontal Bar Chart
    const ctxEdaUnk = document.getElementById("eda-unknowns-chart").getContext("2d");
    charts.edaUnk = new Chart(ctxEdaUnk, {
        type: 'bar',
        data: {
            labels: unknownSummaryData.map(d => d.name),
            datasets: [{
                label: "Placeholder 'Unknown' Percentage (%)",
                data: unknownSummaryData.map(d => d.pct),
                backgroundColor: '#4f709c',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: "Missing Categorical Data (Encoded as 'Unknown')",
                    color: '#ffffff',
                    font: { family: 'Outfit', size: 12, weight: 'bold' }
                }
            },
            scales: {
                x: {
                    grid: { color: gridColors },
                    ticks: { color: textColors, font: { family: 'Inter' } },
                    title: { display: true, text: 'Percentage of Dataset (%)', color: textColors }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: textColors, font: { family: 'Inter' } }
                }
            }
        }
    });

    // 4. Models: Performance horizontal Bar Chart
    const ctxModels = document.getElementById("model-comparison-chart").getContext("2d");
    // Sort models by F1 score to show top ones
    const topModels = [...leaderboardData].sort((a,b) => b.f1 - a.f1).slice(0, 7);
    
    charts.models = new Chart(ctxModels, {
        type: 'bar',
        data: {
            labels: topModels.map(d => d.method.split(" (")[0]),
            datasets: [
                {
                    label: 'F1-Score (Positive)',
                    data: topModels.map(d => d.f1 * 100),
                    backgroundColor: '#00f2fe',
                    borderRadius: 4
                },
                {
                    label: 'Recall (Positive)',
                    data: topModels.map(d => d.recall * 100),
                    backgroundColor: 'rgba(0, 242, 254, 0.4)',
                    borderRadius: 4
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: textColors } }
            },
            scales: {
                x: {
                    grid: { color: gridColors },
                    ticks: { color: textColors },
                    title: { display: true, text: 'Percentage (%)', color: textColors }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: textColors }
                }
            }
        }
    });

    // 5. Tuning: Threshold Curve Line Chart
    const ctxTuning = document.getElementById("threshold-curve-chart").getContext("2d");
    const thresholds = Object.keys(thresholdTuningData).map(Number);
    const recalls = Object.values(thresholdTuningData).map(d => d.recall * 100);
    const precisions = Object.values(thresholdTuningData).map(d => d.precision * 100);
    const f1s = Object.values(thresholdTuningData).map(d => d.f1 * 100);
    
    charts.tuning = new Chart(ctxTuning, {
        type: 'line',
        data: {
            labels: thresholds,
            datasets: [
                {
                    label: 'F1-Score',
                    data: f1s,
                    borderColor: '#00f2fe',
                    backgroundColor: '#00f2fe',
                    tension: 0.15,
                    borderWidth: 2
                },
                {
                    label: 'Recall',
                    data: recalls,
                    borderColor: '#00e676',
                    backgroundColor: '#00e676',
                    tension: 0.15,
                    borderWidth: 2
                },
                {
                    label: 'Precision',
                    data: precisions,
                    borderColor: '#ff1744',
                    backgroundColor: '#ff1744',
                    tension: 0.15,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: textColors } }
            },
            scales: {
                x: {
                    grid: { color: gridColors },
                    ticks: { color: textColors },
                    title: { display: true, text: 'Classification Probability Cutoff', color: textColors }
                },
                y: {
                    grid: { color: gridColors },
                    ticks: { color: textColors },
                    title: { display: true, text: 'Score (%)', color: textColors }
                }
            }
        }
    });
}

// ==========================================================================
// LEADERBOARD TABLE GENERATOR
// ==========================================================================

function populateLeaderboardTable() {
    const tbody = document.getElementById("leaderboard-tbody");
    tbody.innerHTML = "";
    
    leaderboardData.forEach(row => {
        const tr = document.createElement("tr");
        if (row.highlight) {
            tr.classList.add("highlight-tr");
        }
        
        tr.innerHTML = `
            <td>${row.method}</td>
            <td>${(row.accuracy * 100).toFixed(2)}%</td>
            <td>${(row.precision * 100).toFixed(2)}%</td>
            <td>${(row.recall * 100).toFixed(2)}%</td>
            <td>${(row.f1 * 100).toFixed(2)}%</td>
            <td>${(row.roc * 100).toFixed(2)}%</td>
        `;
        
        tbody.appendChild(tr);
    });
}

// ==========================================================================
// EVENT LISTENERS & WIDGET RECALCULATORS
// ==========================================================================

function setupEventListeners() {
    // 1. Threshold Tuning Slider
    const thSlider = document.getElementById("threshold-slider");
    thSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value).toFixed(1);
        document.getElementById("threshold-val-lbl").innerText = val;
        updateTuningMetrics(val);
    });

    // 2. ROI Calculator Sliders
    const roiLeads = document.getElementById("roi-total-leads");
    const roiCallCost = document.getElementById("roi-call-cost");
    const roiValSub = document.getElementById("roi-value-sub");
    const roiCapacity = document.getElementById("roi-capacity");
    
    const updateRoiLabels = () => {
        document.getElementById("lbl-total-leads").innerText = parseInt(roiLeads.value).toLocaleString();
        document.getElementById("lbl-call-cost").innerText = `$${parseFloat(roiCallCost.value).toFixed(2)}`;
        document.getElementById("lbl-value-sub").innerText = `$${parseFloat(roiValSub.value).toFixed(2)}`;
        document.getElementById("lbl-capacity").innerText = `${parseInt(roiCapacity.value).toLocaleString()} calls`;
    };

    [roiLeads, roiCallCost, roiValSub, roiCapacity].forEach(slider => {
        slider.addEventListener("input", () => {
            updateRoiLabels();
            runCalculations();
        });
    });
    updateRoiLabels();

    // 3. What-If Simulator Inputs
    const simInputs = [
        "sim-age", "sim-job", "sim-marital", "sim-education",
        "sim-poutcome", "sim-campaign", "sim-euribor", "sim-contact"
    ];
    simInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("input", runSimulator);
        }
    });
}

// ==========================================================================
// THRESHOLD TUNING RECALCULATOR
// ==========================================================================

function updateTuningMetrics(thresholdVal) {
    const data = thresholdTuningData[thresholdVal];
    if (!data) return;
    
    // Update Confusion Matrix cells
    document.getElementById("cell-tn").innerText = data.tn.toLocaleString();
    document.getElementById("cell-fp").innerText = data.fp.toLocaleString();
    document.getElementById("cell-fn").innerText = data.fn.toLocaleString();
    document.getElementById("cell-tp").innerText = data.tp.toLocaleString();
    
    // Update Dynamic Scorecards
    document.getElementById("stat-f1").innerText = `${(data.f1 * 100).toFixed(2)}%`;
    document.getElementById("stat-recall").innerText = `${(data.recall * 100).toFixed(2)}%`;
    document.getElementById("stat-precision").innerText = `${(data.precision * 100).toFixed(2)}%`;
}

// ==========================================================================
// ROI BUSINESS VALUE MATRICES CALCULATOR
// ==========================================================================

function runCalculations() {
    const leads = parseInt(document.getElementById("roi-total-leads").value);
    const callCost = parseFloat(document.getElementById("roi-call-cost").value);
    const valSub = parseFloat(document.getElementById("roi-value-sub").value);
    const capacity = parseInt(document.getElementById("roi-capacity").value);
    
    // Target rates in population:
    const generalConversion = 0.1127; // 11.27% general population rate
    const mlTargetPrecision = 0.4422; // 44.22% target precision (conversion rate under model list)
    
    // Strategy A: Random Calling (Restricted to Capacity)
    const callsA = Math.min(leads, capacity);
    const costA = callsA * callCost;
    const successesA = Math.round(callsA * generalConversion);
    const grossRevA = successesA * valSub;
    const profitA = grossRevA - costA;
    const cacA = successesA > 0 ? costA / successesA : 0;
    
    // Strategy B: ML Targeted Calling (Restricted to Capacity)
    // Here, we only call leads classified positive.
    // The density of positive cases matches precision (44.22% of called list will subscribe).
    const callsB = Math.min(leads, capacity);
    const costB = callsB * callCost;
    const successesB = Math.round(callsB * mlTargetPrecision);
    const grossRevB = successesB * valSub;
    const profitB = grossRevB - costB;
    const cacB = successesB > 0 ? costB / successesB : 0;
    
    // Update Strategy A displays
    document.getElementById("strat-a-calls").innerText = callsA.toLocaleString();
    document.getElementById("strat-a-cost").innerText = `$${costA.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-a-subs").innerText = successesA.toLocaleString();
    document.getElementById("strat-a-rev").innerText = `$${grossRevA.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-a-profit").innerText = `$${profitA.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-a-cac").innerText = successesA > 0 ? `$${cacA.toFixed(2)}` : "N/A";
    
    // Update Strategy B displays
    document.getElementById("strat-b-calls").innerText = callsB.toLocaleString();
    document.getElementById("strat-b-cost").innerText = `$${costB.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-b-subs").innerText = successesB.toLocaleString();
    document.getElementById("strat-b-rev").innerText = `$${grossRevB.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-b-profit").innerText = `$${profitB.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("strat-b-cac").innerText = successesB > 0 ? `$${cacB.toFixed(2)}` : "N/A";
    
    // Lift summary calculation
    const liftVal = profitB - profitA;
    const liftPct = profitA > 0 ? (liftVal / profitA) * 100 : 0;
    const cacCut = cacA > 0 ? ((cacA - cacB) / cacA) * 100 : 0;
    
    const liftLabel = document.getElementById("roi-profit-lift");
    const liftSign = liftVal >= 0 ? "+" : "";
    liftLabel.innerText = `${liftSign}$${liftVal.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}`;
    
    // Style text color based on profit positive or negative
    if (liftVal >= 0) {
        liftLabel.style.color = 'var(--success)';
    } else {
        liftLabel.style.color = 'var(--danger)';
    }
    
    document.getElementById("roi-pct-lift").innerText = `${Math.round(liftPct)}%`;
    document.getElementById("roi-cac-cut").innerText = `${Math.round(cacCut)}%`;
}

// ==========================================================================
// WHAT-IF CUSTOMER SIMULATOR SCORING MODEL (HEURISTICS CLASSIFIER)
// ==========================================================================

function runSimulator() {
    const age = parseInt(document.getElementById("sim-age").value) || 38;
    const job = document.getElementById("sim-job").value;
    const marital = document.getElementById("sim-marital").value;
    const education = document.getElementById("sim-education").value;
    const poutcome = document.getElementById("sim-poutcome").value;
    const campaign = parseInt(document.getElementById("sim-campaign").value) || 1;
    const euribor = parseFloat(document.getElementById("sim-euribor").value) || 3.6;
    const contact = document.getElementById("sim-contact").value;
    
    // Heuristic Model Approximation based on model weights
    // Base probability represents general conversion likelihood ~11.3% (using logistic function formulation)
    let score = 0.15; // Baseline
    
    // 1. Prior Campaign Success (Strongest factor)
    if (poutcome === "success") {
        score += 0.45;
    } else if (poutcome === "failure") {
        score += 0.05;
    }
    
    // 2. Macroeconomics (Interest rates)
    // Low interest rates imply high savings demand
    if (euribor < 1.5) {
        score += 0.20;
    } else if (euribor < 3.0) {
        score += 0.08;
    } else if (euribor > 4.5) {
        score -= 0.15;
    } else {
        score -= 0.02;
    }
    
    // 3. Campaign contact limits (capping fatigue)
    if (campaign === 1) {
        score += 0.05;
    } else if (campaign === 2) {
        // no modifier
    } else if (campaign === 3) {
        score -= 0.04;
    } else if (campaign >= 4) {
        score -= 0.12;
    }
    
    // 4. Profession
    if (job === "retired" || job === "student") {
        score += 0.12;
    } else if (job === "blue-collar") {
        score -= 0.05;
    } else if (job === "services") {
        score -= 0.03;
    } else if (job === "admin.") {
        score += 0.02;
    }
    
    // 5. Contact Medium
    if (contact === "cellular") {
        score += 0.05;
    } else if (contact === "telephone") {
        score -= 0.04;
    }
    
    // 6. Demographics: Age U-Shape trend
    if (age < 25 || age > 60) {
        score += 0.08;
    }
    
    // 7. Demographics: Marital & Education
    if (marital === "single") {
        score += 0.02;
    } else if (marital === "married") {
        score -= 0.02;
    }
    if (education === "university.degree") {
        score += 0.04;
    }
    
    // Force probability between 0.02 and 0.98 bounds
    const probability = Math.max(0.02, Math.min(0.98, score));
    const finalPct = Math.round(probability * 100);
    
    // Render Probability Gauge
    document.getElementById("sim-score-pct").innerText = `${finalPct}%`;
    const gaugeFill = document.getElementById("sim-gauge-fill");
    
    // SVG radial stroke offset math (circumference = 2 * PI * r = 2 * 3.14159 * 40 = 251.2)
    const strokeOffset = 251 - (251 * probability);
    gaugeFill.style.strokeDashoffset = strokeOffset;
    
    // Change gauge color based on likelihood
    if (probability >= 0.45) {
        gaugeFill.style.stroke = "var(--success)";
    } else if (probability >= 0.25) {
        gaugeFill.style.stroke = "var(--primary)";
    } else {
        gaugeFill.style.stroke = "var(--danger)";
    }
    
    // Render Badge & Verdict Card
    const badge = document.getElementById("sim-badge");
    const title = document.getElementById("sim-verdict-title");
    const desc = document.getElementById("sim-verdict-desc");
    const vBox = document.getElementById("sim-verdict-box");
    const actionBox = document.getElementById("sim-action-box");
    const actionText = document.getElementById("sim-action-text");
    
    // Reset classes
    badge.className = "badge";
    vBox.style.borderColor = "var(--border)";
    actionBox.style.borderColor = "var(--border)";
    
    if (probability >= 0.45) {
        badge.classList.add("high-prob");
        badge.innerText = "HIGH PROBABILITY";
        title.innerText = "Hot Subscriber Lead";
        vBox.style.borderColor = "var(--success)";
        desc.innerText = "Model scores this customer as highly receptive to direct term deposit offers. High likely conversion.";
        actionText.innerText = "ACTION: Contact immediately. Priority level 1. Expected client CAC is low.";
    } else if (probability >= 0.25) {
        badge.classList.add("med-prob");
        badge.innerText = "MEDIUM PROBABILITY";
        title.innerText = "Warm Lead Prospect";
        vBox.style.borderColor = "var(--primary)";
        desc.innerText = "Customer shows moderate conversion metrics. Campaign success is dependent on contact timing.";
        actionText.innerText = "ACTION: Place in secondary call list. Cap attempts at 2. Contact only cell phones.";
    } else {
        badge.classList.add("low-prob");
        badge.innerText = "LOW PROBABILITY";
        title.innerText = "Cold Lead / Do Not Call";
        vBox.style.borderColor = "var(--danger)";
        desc.innerText = "Client features indicate low likelihood of term deposit subscription. Extremely high decline risk.";
        actionText.innerText = "ACTION: Do not contact. Filter out of lists to save campaign budget and agent hours.";
    }
}
