document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    console.log('Initializing application...');
    await checkAndLoadData();
}

async function checkAndLoadData() {
    try {
        // Check API health
        const healthResponse = await fetch('/api/health');
        const healthData = await healthResponse.json();
        console.log('Health check:', healthData);

        if (!healthData.data_loaded) {
            showError('Warning: Data not properly loaded');
            return;
        }

        // Load branches
        await loadBranches();
        
        // Setup other components
        setupFormHandling();
        setupProbabilitySlider();
        
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize application');
    }
}

async function loadBranches() {
    try {
        const branchSelect = document.getElementById('preferred_branch');
        if (!branchSelect) {
            throw new Error('Branch select element not found');
        }

        branchSelect.disabled = true;
        branchSelect.innerHTML = '<option value="">Loading branches...</option>';

        const response = await fetch('/api/branches');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Branch data:', data);

        branchSelect.innerHTML = '<option value="">Select Branch</option>';
        
        if (data.branches && Array.isArray(data.branches)) {
            data.branches.forEach(branch => {
                if (branch && branch !== "All") {
                    const option = document.createElement('option');
                    option.value = branch;
                    option.textContent = branch
                        .split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                        .join(' ');
                    branchSelect.appendChild(option);
                }
            });
        }

        branchSelect.disabled = false;

    } catch (error) {
        console.error('Error loading branches:', error);
        showError('Failed to load branches');
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
    `;
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 5000);
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    if (!validateForm(this)) {
        return;
    }

    showLoading();
    
    try {
        const formData = {
            jee_rank: parseInt(document.getElementById('jee_rank').value),
            category: document.getElementById('category').value,
            college_type: document.getElementById('college_type').value,
            preferred_branch: document.getElementById('preferred_branch').value,
            round_no: parseInt(document.getElementById('round_no').value),
            min_probability: parseFloat(document.getElementById('min_prob').value)
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
        hideLoading();

    } catch (error) {
        console.error('Prediction error:', error);
        hideLoading();
        showError('Failed to get predictions. Please try again.');
    }
}

function validateForm(form) {
    // JEE Rank validation
    const jeeRank = form.querySelector('#jee_rank').value;
    if (!jeeRank || isNaN(jeeRank) || jeeRank < 1) {
        showError('Please enter a valid JEE rank');
        return false;
    }

    // Category validation
    const category = form.querySelector('#category').value;
    if (!category) {
        showError('Please select a category');
        return false;
    }

    // College Type validation
    const collegeType = form.querySelector('#college_type').value;
    if (!collegeType) {
        showError('Please select a college type');
        return false;
    }

    // Branch validation
    const branch = form.querySelector('#preferred_branch').value;
    if (!branch) {
        showError('Please select a branch');
        return false;
    }

    // Round validation
    const round = form.querySelector('#round_no').value;
    if (!round) {
        showError('Please select a counseling round');
        return false;
    }

    return true;
}

// Probability Slider
function setupProbabilitySlider() {
    const slider = document.getElementById('min_prob');
    const valueDisplay = document.getElementById('min_prob_value');
    
    if (slider && valueDisplay) {
        slider.addEventListener('input', function() {
            valueDisplay.textContent = this.value + '%';
        });
    }
}

// UI Helpers
function showLoading() {
    // Add loading indicator logic
    const resultsSection = document.querySelector('.results-section');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    // You can add a loading spinner here
}

function hideLoading() {
    // Remove loading indicator logic
}

function showError(message) {
    alert(message); // You can replace this with a better error display
}

function displayResults(data) {
    const resultsSection = document.querySelector('.results-section');
    if (!resultsSection) return;

    resultsSection.style.display = 'block';

    if (!data.predictions || data.predictions.length === 0) {
        resultsSection.innerHTML = '<p>No colleges found matching your criteria.</p>';
        return;
    }

    // Create and populate the results table
    const table = document.createElement('table');
    table.className = 'results-table';

    // Add table headers
    table.innerHTML = `
        <thead>
            <tr>
                <th>Preference</th>
                <th>Institute</th>
                <th>Branch</th>
                <th>Category</th>
                <th>Opening Rank</th>
                <th>Closing Rank</th>
                <th>Probability</th>
            </tr>
        </thead>
        <tbody>
            ${data.predictions.map(pred => `
                <tr>
                    <td>${pred.Preference}</td>
                    <td>${pred.Institute}</td>
                    <td>${pred.Branch}</td>
                    <td>${pred.Category}</td>
                    <td>${pred.Opening_Rank}</td>
                    <td>${pred.Closing_Rank}</td>
                    <td>${pred['Admission Probability (%)']}%</td>
                </tr>
            `).join('')}
        </tbody>
    `;

    // Clear previous results and add new table
    resultsSection.innerHTML = '';
    resultsSection.appendChild(table);

    // If there's plot data, display it
    if (data.plot_data) {
        const plotDiv = document.createElement('div');
        plotDiv.id = 'probability-plot';
        resultsSection.appendChild(plotDiv);
        Plotly.newPlot('probability-plot', data.plot_data);
    }
}
