document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupFormHandling();
    setupTableSorting();
    setupFilteringAndSearch();
    setupCollegeTypeToggle();
    setupProbabilitySlider();
    setupResponsiveHandling();
}

// Form Handling
function setupFormHandling() {
    const searchForm = document.getElementById('prediction-form');
    if (searchForm) {
        searchForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!validateForm(this)) {
                return;
            }

            showLoading();
            
            try {
                const formData = new FormData(this);
                const response = await fetch('/api/p
