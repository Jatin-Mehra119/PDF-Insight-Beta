// DOM Elements
const uploadBox = document.getElementById('upload-box');
const pdfUpload = document.getElementById('pdf-upload');
const welcomeScreen = document.getElementById('welcome-screen');
const chatContainer = document.getElementById('chat-container');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const searchToggle = document.getElementById('search-toggle');
const modelSelect = document.getElementById('model-select');
const sessionInfo = document.getElementById('session-info');
const currentFileName = document.getElementById('current-file-name');
const clearHistoryBtn = document.getElementById('clear-history');
const removePdfBtn = document.getElementById('remove-pdf');
const newChatBtn = document.getElementById('new-chat');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const getStartedBtn = document.getElementById('get-started-btn');
const contextContent = document.getElementById('context-content');
const contextSidebar = document.getElementById('context-sidebar');
const toggleContextBtn = document.getElementById('toggle-context');
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.querySelector('.sidebar');

// App state
let currentSessionId = null;
let lastContextData = null;
let isMobile = window.innerWidth <= 768;

// Event listeners
uploadBox.addEventListener('click', () => pdfUpload.click());
pdfUpload.addEventListener('change', handleFileUpload);
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
chatInput.addEventListener('input', () => {
    sendButton.disabled = chatInput.value.trim() === '';
});
clearHistoryBtn.addEventListener('click', clearChatHistory);
removePdfBtn.addEventListener('click', removePdf);
newChatBtn.addEventListener('click', resetApp);
getStartedBtn.addEventListener('click', () => {
    uploadBox.click();
});
toggleContextBtn.addEventListener('click', toggleContextSidebar);

// Mobile menu event listeners
if (menuToggle) {
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('show');
    });
}

// Handle window resize
window.addEventListener('resize', () => {
    const wasMobile = isMobile;
    isMobile = window.innerWidth <= 768;
    
    // If we're transitioning between mobile/desktop
    if (wasMobile !== isMobile) {
        updateMobileUI();
    }
});

// Initialize the app
initializeApp();

// Functions
function initializeApp() {
    updateMobileUI();
    
    // Check if the page was refreshed
    const pageWasRefreshed = (
        window.performance && 
        window.performance.navigation && 
        window.performance.navigation.type === 1
    ) || document.referrer === document.location.href;
    
    // If page was refreshed, clear any saved session
    if (pageWasRefreshed) {
        localStorage.removeItem('pdf_insight_session');
        return;
    }
    
    // Check if we have a session in localStorage
    const savedSession = localStorage.getItem('pdf_insight_session');
    if (savedSession) {
        try {
            const session = JSON.parse(savedSession);
            currentSessionId = session.id;
            currentFileName.textContent = session.fileName;
            
            // Show chat interface
            welcomeScreen.classList.add('hidden');
            chatContainer.classList.remove('hidden');
            sessionInfo.classList.remove('hidden');
            
            // Load chat history
            fetchChatHistory();
        } catch (e) {
            console.error('Failed to load saved session:', e);
            localStorage.removeItem('pdf_insight_session');
        }
    }
}

// Update UI based on mobile/desktop view
function updateMobileUI() {
    if (isMobile) {
        if (menuToggle) menuToggle.classList.remove('hidden');
        contextSidebar.classList.add('collapsed');
    } else {
        if (menuToggle) menuToggle.classList.add('hidden');
        sidebar.classList.remove('show');
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    try {
        // Show loading overlay
        showLoading('Processing Document...');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_name', modelSelect.value);
        
        const response = await fetch('/upload-pdf', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentSessionId = data.session_id;
            currentFileName.textContent = file.name;
            
            // Save session to localStorage
            localStorage.setItem('pdf_insight_session', JSON.stringify({
                id: currentSessionId,
                fileName: file.name
            }));
            
            // Show chat interface
            welcomeScreen.classList.add('hidden');
            chatContainer.classList.remove('hidden');
            sessionInfo.classList.remove('hidden');
            
            // Reset chat history view
            chatMessages.innerHTML = `
                <div class="system-message">
                    <p>Upload successful! You can now ask questions about "${file.name}".</p>
                </div>
            `;
            
            // Enable input
            chatInput.disabled = false;
            chatInput.placeholder = 'Ask a question about the document...';
            
            // Close sidebar on mobile after uploading
            if (isMobile) {
                sidebar.classList.remove('show');
            }
        } else {
            // Enhanced error display
            let errorDetails = '';
            if (data.detail) {
                errorDetails = data.detail;
            }
            if (data.type) {
                errorDetails = `${data.type}: ${errorDetails}`;
            }
            
            showError('Error: ' + (errorDetails || 'Failed to process document'));
            
            // Add a more detailed error in the chat area if we're already in chat mode
            if (!welcomeScreen.classList.contains('hidden')) {
                const errorMessageDiv = document.createElement('div');
                errorMessageDiv.className = 'system-message error';
                errorMessageDiv.innerHTML = `
                    <p><strong>Error Processing Document</strong></p>
                    <p>${errorDetails}</p>
                    <p>Please make sure you have set up all required API keys in the .env file.</p>
                `;
                chatMessages.appendChild(errorMessageDiv);
            }
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        showError('Failed to upload document. Please try again.');
    } finally {
        hideLoading();
    }
}

async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query || !currentSessionId) return;
    
    // Disable input and show typing indicator
    chatInput.disabled = true;
    sendButton.disabled = true;
    
    // Add user message to chat
    const userMessageElement = createMessageElement('user', query);
    chatMessages.appendChild(userMessageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Clear input
    chatInput.value = '';
    
    try {
        // Show loading state
        const typingIndicator = createTypingIndicator();
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                query: query,
                use_search: searchToggle.checked,
                model_name: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        chatMessages.removeChild(typingIndicator);
        
        if (data.status === 'success') {
            // Add assistant message
            const assistantMessageElement = createMessageElement('assistant', data.answer);
            chatMessages.appendChild(assistantMessageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Apply syntax highlighting to code blocks
            applyCodeHighlighting();
            
            // Update context sidebar
            updateContextSidebar(data.context_used);
            lastContextData = data.context_used;
        } else {
            showError('Failed to get response: ' + data.detail);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        showError('Failed to get response. Please try again.');
        
        // Remove typing indicator if it exists
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            chatMessages.removeChild(indicator);
        }
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        chatInput.focus();
    }
}

async function fetchChatHistory() {
    if (!currentSessionId) return;
    
    try {
        showLoading('Loading chat history...');
        
        const response = await fetch('/chat-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' && data.history.length > 0) {
            // Clear chat messages and add system message
            chatMessages.innerHTML = `
                <div class="system-message">
                    <p>Continuing your conversation about "${currentFileName.textContent}".</p>
                </div>
            `;
            
            // Add all messages from history
            data.history.forEach(item => {
                const userMessage = createMessageElement('user', item.user);
                const assistantMessage = createMessageElement('assistant', item.assistant);
                chatMessages.appendChild(userMessage);
                chatMessages.appendChild(assistantMessage);
            });
            
            // Apply syntax highlighting to code blocks
            applyCodeHighlighting();
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    } catch (error) {
        console.error('Error fetching chat history:', error);
        showError('Failed to load chat history.');
    } finally {
        hideLoading();
    }
}

async function clearChatHistory() {
    if (!currentSessionId) return;
    
    try {
        showLoading('Clearing chat history...');
        
        const response = await fetch('/clear-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Reset chat messages
            chatMessages.innerHTML = `
                <div class="system-message">
                    <p>Chat history cleared. You can continue asking questions about "${currentFileName.textContent}".</p>
                </div>
            `;
            
            // Reset context
            contextContent.innerHTML = '<p class="no-context">No context available yet. Ask a question first.</p>';
            lastContextData = null;
        } else {
            showError('Failed to clear chat history: ' + data.detail);
        }
    } catch (error) {
        console.error('Error clearing chat history:', error);
        showError('Failed to clear chat history.');
    } finally {
        hideLoading();
    }
}

async function removePdf() {
    if (!currentSessionId) return;
    
    try {
        showLoading('Removing PDF from the system...');
        
        const response = await fetch('/remove-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Reset the app
            resetApp();
            showError('PDF file has been removed from the system.');
        } else {
            showError('Failed to remove PDF: ' + data.detail);
        }
    } catch (error) {
        console.error('Error removing PDF:', error);
        showError('Failed to remove PDF. Please try again.');
    } finally {
        hideLoading();
    }
}

function resetApp() {
    // Clear current session
    currentSessionId = null;
    lastContextData = null;
    localStorage.removeItem('pdf_insight_session');
    
    // Reset UI
    welcomeScreen.classList.remove('hidden');
    chatContainer.classList.add('hidden');
    sessionInfo.classList.add('hidden');
    
    // Reset file input
    pdfUpload.value = '';
    
    // Reset chat messages
    chatMessages.innerHTML = `
        <div class="system-message">
            <p>Upload successful! You can now ask questions about the document.</p>
        </div>
    `;
    
    // Reset context sidebar
    contextContent.innerHTML = '<p class="no-context">No context available yet. Ask a question first.</p>';
}

function createMessageElement(type, content) {
    const div = document.createElement('div');
    div.className = `message ${type}-message`;
    
    // For user messages, just escape HTML
    if (type === 'user') {
        div.innerHTML = `
            <div class="message-content">${escapeHTML(content)}</div>
            <div class="message-timestamp">${formatTimestamp(new Date())}</div>
        `;
    } 
    // For assistant messages, render with Markdown
    else {
        // Configure marked.js options
        marked.setOptions({
            breaks: true, // Add <br> on single line breaks
            gfm: true,    // GitHub Flavored Markdown
            sanitize: false // Allow HTML in the input
        });
        
        // Process the content with marked
        const renderedContent = marked.parse(content);
        
        div.innerHTML = `
            <div class="message-content">${renderedContent}</div>
            <div class="message-timestamp">${formatTimestamp(new Date())}</div>
        `;
    }
    
    return div;
}

function createTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant-message typing-indicator';
    div.innerHTML = `
        <div class="typing-animation">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;
    return div;
}

function updateContextSidebar(contextData) {
    if (!contextData || contextData.length === 0) {
        contextContent.innerHTML = '<p class="no-context">No context available for this response.</p>';
        return;
    }
    
    contextContent.innerHTML = '';
    
    contextData.forEach((item, index) => {
        const contextItem = document.createElement('div');
        contextItem.className = 'context-item';
        
        const score = Math.round((1 - item.score) * 100); // Convert distance to similarity score
        
        contextItem.innerHTML = `
            <span class="context-score">Relevance: ${score}%</span>
            <div class="context-text">${truncateText(item.text, 300)}</div>
        `;
        
        contextContent.appendChild(contextItem);
    });
}

function toggleContextSidebar() {
    contextSidebar.classList.toggle('collapsed');
    
    // Update icon
    const icon = toggleContextBtn.querySelector('i');
    if (contextSidebar.classList.contains('collapsed')) {
        icon.className = 'fas fa-angle-left';
    } else {
        icon.className = 'fas fa-angle-right';
    }
}

function applyCodeHighlighting() {
    // Find all code blocks in the assistant messages
    document.querySelectorAll('.assistant-message pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

function showLoading(message) {
    loadingText.textContent = message || 'Loading...';
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function showError(message) {
    // Add error message to chat
    const errorDiv = document.createElement('div');
    errorDiv.className = 'system-message error';
    errorDiv.innerHTML = `<p>${message}</p>`;
    
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Also show an alert for critical errors
    if (message.includes('API_KEY') || message.includes('environment variables')) {
        alert('Configuration Error: ' + message);
    }
    
    // Remove after 10 seconds
    setTimeout(() => {
        if (errorDiv.parentNode === chatMessages) {
            chatMessages.removeChild(errorDiv);
        }
    }, 10000);
}

// Helper functions
function formatTimestamp(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}