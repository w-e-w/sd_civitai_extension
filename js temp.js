async function fetchUrlDictionary() {
    const apiUrl = '/civitai/get-mhtml_dict';
    try {
        const response = await fetch(apiUrl);
        if (!response.ok) throw new Error('Failed to fetch the URL dictionary');
        return await response.json();
    } catch (error) {
        console.error('Error fetching URL dictionary:', error);
        return {};
    }
}

async function processButtons() {
    const urlDictionary = await fetchUrlDictionary();

    document.querySelectorAll('.copy-path-button.card-button').forEach(button => {
        const filepath = button.getAttribute('data-clipboard-text');
        
        if (!filepath) return;

        const filepathWithoutExt = filepath.replace(/\.[^/.]+$/, '');

        if (urlDictionary[filepathWithoutExt]) {
            const filename = urlDictionary[filepathWithoutExt];
            const url = `./civitai/get-mhtml?filename=${encodeURIComponent(filename)}`;

            const newDiv = document.createElement('div');
            newDiv.classList.add('open-civ-page', 'card-button');

            newDiv.onclick = function (event) {
                event.stopPropagation();
                window.open(url, '_blank');
            };

            button.parentNode.insertBefore(newDiv, button);
        }
    });
}

processButtons();
// add button to every card-button with data-clipboard-text
