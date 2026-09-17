let firstRecordAdded = false;
let currentChat = [];
// —— 新增：记录当前快照的 key（时间戳） ——
let recordTs = null;


// 自动滚动到底部
function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// 收起边栏按钮
document.getElementById('toggle-sidebar').addEventListener('click', () => {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
});

document.getElementById('image-upload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // 清除欢迎消息
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // 读取并显示用户上传的图片
    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = function () {
            // 调整尺寸
            const targetWidth = 300, targetHeight = 200;
            const canvas = document.createElement('canvas');
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
            const resizedDataUrl = canvas.toDataURL('image/jpeg');

            // 构造用户消息 HTML，加入模型
            const time = new Date().toLocaleTimeString();
            const userImgHtml = `
                <div class="message-header">
                  <span class="message-time">${time}</span>
                  <span class="message-label">您上传的图片</span>
                </div>
                <img src="${resizedDataUrl}" class="chat-image single-image">
            `;
            // 推入 currentChat 并渲染
            currentChat.push({role: 'user', html: userImgHtml});
            updateHistoryRecord();
            const userMsg = document.createElement('div');
            userMsg.className = 'message user-message';
            userMsg.innerHTML = userImgHtml;
            document.getElementById('chat-area').appendChild(userMsg);

            // 显示 loading
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'message ai-message loading';
            loadingMsg.innerHTML = '<div class="loader"></div><p>正在分析图片，请稍候...</p>';
            document.getElementById('chat-area').appendChild(loadingMsg);
            scrollToBottom();

            // 发送到后端
            const formData = new FormData();
            formData.append('image', file);
            formData.append('model', document.getElementById('model-select').value);

            fetch('/predict', {method: 'POST', body: formData})
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    loadingMsg.remove();
                    if (data.error) throw new Error(data.error);

                    // 构造分析结果 HTML
                    const formatMetrics = (metrics) => {
                        let html = '<div class="metrics-grid">';
                        for (const [k, v] of Object.entries(metrics)) {
                            if (typeof v === 'object') {
                                html += `<div class="metric-category"><strong>${k.replace(/_/g, ' ')}:</strong>${formatMetrics(v)}</div>`;
                            } else {
                                html += `<div class="metric-item"><span class="metric-key">${k.replace(/_/g, ' ')}:</span><span class="metric-value">${v}</span></div>`;
                            }
                        }
                        return html + '</div>';
                    };
                    const analysisHtml = `
                    <div class="message-header">
                      <span class="message-time">${new Date().toLocaleTimeString()}</span>
                      <span class="message-label">分析结果</span>
                    </div>
                    <div class="image-grid">
                      <div class="image-column"><p class="image-label">原图</p><img src="${data.original_url || resizedDataUrl}" class="grid-image"></div>
                      <div class="image-column"><p class="image-label">分割结果</p><img src="${data.segmentation_url}" class="grid-image"></div>
                      <div class="image-column"><p class="image-label">叠加效果</p><img src="${data.overlay_url}" class="grid-image"></div>
                    </div>
                    <div class="analysis-section"><h3>分割指标</h3>${formatMetrics(data.segmentation_metrics)}</div>
                    <div class="analysis-section"><h3>性能指标</h3>${formatMetrics(data.performance_metrics)}</div>
                    <div class="analysis-section"><h3>模型信息</h3>${formatMetrics(data.model_info)}</div>
                `;
                    // 推入 currentChat 并渲染
                    currentChat.push({role: 'ai', html: analysisHtml});
                    updateHistoryRecord();
                    const aiMsg = document.createElement('div');
                    aiMsg.className = 'message ai-message';
                    aiMsg.innerHTML = analysisHtml;
                    document.getElementById('chat-area').appendChild(aiMsg);
                    scrollToBottom();

                    // 仅首次添加历史记录
                    if (!firstRecordAdded) {
                        const ts = new Date().toLocaleString().replace(/\//g, '-').replace(/上午|下午/, '');
                        addHistoryRecord(ts);
                        firstRecordAdded = true;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingMsg.remove();
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'message ai-message error';
                    errorMsg.innerHTML = `
                    <div class="error-content">
                      <p>处理失败: ${error.message}</p>
                      <button class="retry-btn">重试</button>
                    </div>
                `;
                    document.getElementById('chat-area').appendChild(errorMsg);
                    scrollToBottom();
                    errorMsg.querySelector('.retry-btn').addEventListener('click', () => {
                        errorMsg.remove();
                        this.click();
                    });
                })
                .finally(() => {
                    e.target.value = '';
                });
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});


// 发送按钮事件
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;

    // 构造并展示用户消息
    const time = new Date().toLocaleTimeString();
    const userHtml = `
      <div class="message-header">
        <span class="message-time">${time}</span>
      </div>
      <p>${text}</p>`;
    currentChat.push({role: 'user', html: userHtml});
    updateHistoryRecord();    // 保存用户这条
    const userDiv = document.createElement('div');
    userDiv.className = 'message user-message';
    userDiv.innerHTML = userHtml;
    document.getElementById('chat-area').appendChild(userDiv);
    scrollToBottom();
    input.value = '';

    // 显示 AI 正在输入
    const aiTyping = document.createElement('div');
    aiTyping.className = 'message ai-message typing';
    aiTyping.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    document.getElementById('chat-area').appendChild(aiTyping);
    scrollToBottom();

    // 发送到后端并处理响应
    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: text,
            model: document.getElementById('model-select').value,
        })
    })
        .then(response => response.json())
        .then(data => {
            // 移除 “正在输入” 指示器
            aiTyping.remove();

            // 构造并展示 AI 回复
            const aiHtml = `
            <div class="message-header">
              <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <p>${data.reply}</p>`;
            currentChat.push({role: 'ai', html: aiHtml});
            updateHistoryRecord();  // 保存 AI 回复
            const aiDiv = document.createElement('div');
            aiDiv.className = 'message ai-message';
            aiDiv.innerHTML = aiHtml;
            document.getElementById('chat-area').appendChild(aiDiv);
            scrollToBottom();

            // —— 在这里首次保存历史快照 ——
            if (!firstRecordAdded) {
                const ts1 = new Date().toLocaleString()
                    .replace(/\//g, '-')
                    .replace(/上午|下午/, '');
                addHistoryRecord(ts1);
                firstRecordAdded = true;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            aiTyping.classList.remove('typing');
            aiTyping.classList.add('error');
            aiTyping.innerHTML = '<p>请求失败，请稍后再试</p>';
        });
}

// 页面初始化：加载侧栏已有历史
// 修改页面初始化代码
document.addEventListener('DOMContentLoaded', () => {
    const historyList = document.getElementById('history-list');

    // 清空现有列表（避免重复）
    historyList.innerHTML = '';

    // 加载并创建带有按钮的历史记录项
    Object.keys(localStorage)
        .filter(key => key.startsWith('chat-'))
        .sort()
        .forEach(key => {
            const ts = key.replace('chat-', '');
            createHistoryItem(ts); // 使用我们的工厂函数
        });
});


function renderChatArea() {
    const chatArea = document.getElementById('chat-area');
    chatArea.innerHTML = '';
    currentChat.forEach(msg => {
        const wrapper = document.createElement('div');
        wrapper.className = `message ${msg.role}-message`;
        wrapper.innerHTML = msg.html;   // 直接使用保存好的那一条消息的 innerHTML
        chatArea.appendChild(wrapper);
    });
    scrollToBottom();
}

// 恢复某条历史

// 在侧栏和 localStorage 中添加新历史
// 修改 addHistoryRecord 函数
// 修改 addHistoryRecord 函数
// 修改 addHistoryRecord 函数
// 修改 addHistoryRecord 函数
function addHistoryRecord(ts) {
    // 确保时间戳唯一，避免覆盖
    let uniqueTs = ts;
    let counter = 1;
    while (localStorage.getItem(`chat-${uniqueTs}`)) {
        uniqueTs = `${ts} (${counter++})`;
    }
    ts = uniqueTs;

    recordTs = ts;
    localStorage.setItem('chat-' + ts, JSON.stringify(currentChat));

    // 查找是否已有相同时间戳的DOM元素
    const historyList = document.getElementById('history-list');
    let existingItem = Array.from(historyList.children).find(item =>
        item.dataset.timestamp === ts
    );

    if (!existingItem) {
        // 创建新的历史记录项
        existingItem = createHistoryItem(ts);
        historyList.appendChild(existingItem);
    }

    return existingItem;
}

// 新增：创建历史记录项的函数
// 修改 createHistoryItem 函数（确保可以被初始化代码调用）
function createHistoryItem(ts) {
    const historyList = document.getElementById('history-list');

    // 检查是否已存在
    const existingItem = Array.from(historyList.children).find(item =>
        item.dataset.timestamp === ts
    );
    if (existingItem) return existingItem;

    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = ts;
    item.dataset.timestamp = ts;

    // 创建更多操作按钮
    const moreBtn = document.createElement('button');
    moreBtn.className = 'more-btn';
    moreBtn.innerHTML = '...';
    moreBtn.style.display = 'none';

    item.appendChild(moreBtn);
    historyList.appendChild(item);

    // 鼠标事件
    item.addEventListener('mouseenter', () => {
        moreBtn.style.display = 'inline-block';
    });
    item.addEventListener('mouseleave', () => {
        moreBtn.style.display = 'none';
    });

    // 点击事件
    item.addEventListener('click', (e) => {
        if (!e.target.closest('.more-btn')) {
            restoreChat(ts);
        }
    });

    // 更多按钮事件
    moreBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showHistoryItemMenu(moreBtn, item);
    });

    return item;
}

// 修改 showHistoryItemMenu 函数
function showHistoryItemMenu(button, item) {
    const ts = item.dataset.timestamp;

    // 移除现有的菜单
    const existingMenu = document.querySelector('.history-item-menu');
    if (existingMenu) existingMenu.remove();

    // 创建菜单
    const menu = document.createElement('div');
    menu.className = 'history-item-menu';
    menu.innerHTML = `
        <div class="menu-item rename-item">重命名</div>
        <div class="menu-item delete-item">删除</div>
    `;

    // 定位菜单
    const rect = button.getBoundingClientRect();
    menu.style.position = 'absolute';
    menu.style.top = `${rect.bottom + window.scrollY}px`;
    menu.style.left = `${rect.left + window.scrollX - 80}px`;
    menu.style.zIndex = '1000';

    document.body.appendChild(menu);

    // 重命名功能
    menu.querySelector('.rename-item').addEventListener('click', (e) => {
        e.stopPropagation();
        const oldTs = item.dataset.timestamp;
        let newName = prompt('请输入新的会话名称:', item.textContent);

        if (newName) {
            newName = newName.trim();
            if (newName === '') return;

            // 检查名称是否已存在
            let uniqueName = newName;
            let counter = 1;
            while (localStorage.getItem(`chat-${uniqueName}`)) {
                if (uniqueName === oldTs) break; // 允许改回原名
                uniqueName = `${newName} (${counter++})`;
            }
            newName = uniqueName;

            // 更新本地存储
            const chatData = localStorage.getItem(`chat-${oldTs}`);
            if (chatData) {
                localStorage.setItem(`chat-${newName}`, chatData);
                localStorage.removeItem(`chat-${oldTs}`);

                // 更新当前记录key
                if (recordTs === oldTs) {
                    recordTs = newName;
                }

                // 完全重建历史项以确保按钮存在
                const parent = item.parentNode;
                const newItem = createHistoryItem(newName);
                parent.replaceChild(newItem, item);

                // 如果正在查看被重命名的记录，重新加载
                if (recordTs === newName) {
                    restoreChat(newName);
                }
            }
        }
        menu.remove();
    });

    // 删除功能
    menu.querySelector('.delete-item').addEventListener('click', (e) => {
        e.stopPropagation();
        if (confirm('确定要删除此会话记录吗？')) {
            // 从本地存储删除
            localStorage.removeItem(`chat-${ts}`);

            // 从UI删除
            item.remove();

            // 如果是当前会话，重置
            if (recordTs === ts) {
                document.getElementById('new-chat').click();
            }
        }
        menu.remove();
    });

    // 点击其他地方关闭菜单
    const closeMenu = (e) => {
        if (!menu.contains(e.target)) {
            menu.remove();
            document.removeEventListener('click', closeMenu);
        }
    };

    setTimeout(() => {
        document.addEventListener('click', closeMenu);
    }, 100);
}

// 修改 restoreChat 函数
function restoreChat(ts) {
    // 尝试从本地存储获取数据
    const json = localStorage.getItem('chat-' + ts);

    if (!json) {
        // 如果找不到，尝试查找是否有重命名的记录
        const allKeys = Object.keys(localStorage);
        const matchingKey = allKeys.find(key =>
            key.startsWith('chat-') && localStorage.getItem(key) === JSON.stringify(currentChat)
        );

        if (matchingKey) {
            // 如果找到匹配的记录，使用它
            currentChat = JSON.parse(localStorage.getItem(matchingKey));
            recordTs = matchingKey.replace('chat-', '');
            renderChatArea();
            firstRecordAdded = true;
            return;
        }

        alert('找不到此聊天记录，可能已被删除或重命名');
        return;
    }

    currentChat = JSON.parse(json);
    renderChatArea();
    firstRecordAdded = true;
    recordTs = ts;
}

// 重命名历史记录项

function updateHistoryRecord() {
    if (recordTs) {
        localStorage.setItem('chat-' + recordTs, JSON.stringify(currentChat));
    }
}

document.getElementById('new-chat').addEventListener('click', () => {
    // 重置模型与 UI
    currentChat = [];
    firstRecordAdded = false;
    // **关键：清除旧的 recordTs，避免后续覆盖**
    recordTs = null;
    document.getElementById('chat-area').innerHTML = `
    <div class="message welcome-message">
      <div class="welcome-content">
        <h3>我是MedSeg，很高兴为您服务！</h3>
        <p>请上传皮肤病变图片，我将为您分析分割结果</p>
      </div>
    </div>`;
});

document.getElementById('clear-history').addEventListener('click', () => {
    if (!confirm('确定要清空所有历史记录吗？')) return;
    Object.keys(localStorage).filter(k => k.startsWith('chat-')).forEach(k => localStorage.removeItem(k));
    document.getElementById('history-list').innerHTML = '';
    // 同时也重置当前会话
    document.getElementById('new-chat').click();
});

// 添加搜索按钮事件监听
document.getElementById('search-btn').addEventListener('click', () => {
    toggleSearchPanel();
});

// 创建搜索面板
function toggleSearchPanel() {
    let searchPanel = document.querySelector('.search-panel');

    if (!searchPanel) {
        // 创建搜索面板
        searchPanel = document.createElement('div');
        searchPanel.className = 'search-panel';
        searchPanel.innerHTML = `
            <div class="search-header">
                <input type="text" id="search-input" placeholder="搜索历史记录...">
                <button id="close-search" class="icon-btn">×</button>
            </div>
            <div class="search-results"></div>
        `;

        document.querySelector('.sidebar').appendChild(searchPanel);

        // 搜索输入事件
        document.getElementById('search-input').addEventListener('input', (e) => {
            performSearch(e.target.value);
        });

        // 关闭按钮事件
        document.getElementById('close-search').addEventListener('click', () => {
            searchPanel.remove();
        });
    } else {
        searchPanel.remove();
    }
}

// 执行搜索
// 修改 performSearch 函数
function performSearch(query) {
    if (!query.trim()) {
        document.querySelector('.search-results').innerHTML = '';
        return;
    }

    const resultsContainer = document.querySelector('.search-results');
    resultsContainer.innerHTML = '<div class="search-loading">搜索中...</div>';

    setTimeout(() => {
        const results = [];
        const searchLower = query.toLowerCase();

        // 搜索所有历史记录
        Object.keys(localStorage).forEach(key => {
            if (key.startsWith('chat-')) {
                try {
                    const timestamp = key.replace('chat-', '');
                    const chatData = JSON.parse(localStorage.getItem(key));

                    // 1. 首先检查记录名称是否匹配
                    const nameMatch = timestamp.toLowerCase().includes(searchLower);

                    // 2. 检查对话内容是否匹配
                    let contentMatch = false;
                    let previewText = '';
                    const chatText = chatData.map(msg =>
                        msg.html.replace(/<[^>]*>/g, ' ') // 移除HTML标签
                    ).join(' ');

                    if (chatText.toLowerCase().includes(searchLower)) {
                        contentMatch = true;
                        previewText = getTextPreview(chatText, searchLower);
                    }

                    // 如果名称或内容匹配，则加入结果
                    if (nameMatch || contentMatch) {
                        results.push({
                            key,
                            timestamp,
                            preview: nameMatch ?
                                `名称匹配: ${highlightMatch(timestamp, searchLower)}` :
                                previewText,
                            matchType: nameMatch ? 'name' : 'content'
                        });
                    }
                } catch (e) {
                    console.error('解析聊天记录失败:', e);
                }
            }
        });

        // 按匹配类型排序（名称匹配优先）
        results.sort((a, b) => {
            if (a.matchType === 'name' && b.matchType !== 'name') return -1;
            if (a.matchType !== 'name' && b.matchType === 'name') return 1;
            return 0;
        });

        displaySearchResults(results);
    }, 300);
}

// 高亮匹配的名称
function highlightMatch(text, query) {
    const regex = new RegExp(`(${escapeRegExp(query)})`, 'gi');
    return text.replace(regex, '<span class="highlight">$1</span>');
}

// 转义正则特殊字符
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function displaySearchResults(results) {
    const resultsContainer = document.querySelector('.search-results');

    if (results.length === 0) {
        resultsContainer.innerHTML = '<div class="no-results">未找到匹配的对话</div>';
        return;
    }

    let html = '';
    results.forEach(result => {
        html += `
            <div class="search-result-item" data-key="${result.key}">
                <div class="result-title">${result.timestamp}</div>
                <div class="result-preview">${result.preview}</div>
                <div class="result-match-type">
                    ${result.matchType === 'name' ? '名称匹配' : '内容匹配'}
                </div>
            </div>
        `;
    });

    resultsContainer.innerHTML = html;

    // 添加点击事件
    document.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', () => {
            const key = item.dataset.key;
            const ts = key.replace('chat-', '');
            restoreChat(ts);
            document.querySelector('.search-panel')?.remove();
        });
    });
}


// 获取文本预览
function getTextPreview(text, query) {
    const index = text.toLowerCase().indexOf(query.toLowerCase());
    if (index === -1) return text.substring(0, 100) + '...';

    const start = Math.max(0, index - 20);
    const end = Math.min(text.length, index + query.length + 80);
    let preview = text.substring(start, end);

    if (start > 0) preview = '...' + preview;
    if (end < text.length) preview += '...';

    // 高亮匹配词
    const regex = new RegExp(`(${query})`, 'gi');
    return preview.replace(regex, '<span class="highlight">$1</span>');
}

document.addEventListener('DOMContentLoaded', function () {
    const settingsBtn = document.getElementById('settings-btn');
    const closeBtn = document.getElementById('close-settings');
    const settingsPanel = document.getElementById('settings-panel');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // 显示面板
    settingsBtn.addEventListener('click', () => {
        settingsPanel.classList.add('active');
    });

    // 关闭面板
    closeBtn.addEventListener('click', () => {
        settingsPanel.classList.remove('active');
    });

    // 标签切换功能
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // 移除所有active类
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // 添加active类到当前按钮和对应内容
            button.classList.add('active');
            const tabId = button.dataset.tab + '-tab';
            document.getElementById(tabId).classList.add('active');

            // 根据标签切换面板大小
            if (button.dataset.tab === 'model' || button.dataset.tab === 'dataset') {
                settingsPanel.classList.add('expanded');
            } else {
                settingsPanel.classList.remove('expanded');
            }
        });
    });

});

document.addEventListener('DOMContentLoaded', function () {
    const themeButtons = document.querySelectorAll('.theme-btn');

    // 初始化主题 - 默认浅色
    function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);

        // 设置对应按钮的active状态
        themeButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.theme === savedTheme) {
                btn.classList.add('active');
            }
        });
    }

    // 主题切换功能
    themeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const theme = button.dataset.theme;

            // 移除所有按钮的active状态
            themeButtons.forEach(btn => btn.classList.remove('active'));

            // 设置当前按钮为active
            button.classList.add('active');

            // 应用主题
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
        });
    });

    // 初始化
    initTheme();
});

document.addEventListener('DOMContentLoaded', () => {
    const langButtons = document.querySelectorAll('.language-btn');

    const translations = {
        zh: {
            "title": "医学图像分割系统",
            "new-chat-text": "开启新对话",
            "clear-history-text": "清空记录",
            "welcome-title": "我是MedSeg，很高兴为您服务！",
            "welcome-msg": "请上传皮肤病变图片，我将为您分析分割结果",
            "user-input-placeholder": "输入问题或拖放图片进行分析...",
            "disclaimer": "AI生成内容仅供参考",
            "aboutset":"通用设置",
            "aboutmodel":"模型相关",
            "aboutdata":"数据相关",
            "theme_set":"主题设置",
            "language_set":"语言设置"
        },
        en: {
            "title": "Medical Image Segmentation System",
            "new-chat-text": "Start New Chat",
            "clear-history-text": "Clear History",
            "welcome-title": "I'm MedSeg, glad to assist you!",
            "welcome-msg": "Please upload a skin lesion image, and I will analyze the segmentation results for you.",
            "user-input-placeholder": "Enter your question or drop an image to analyze...",
            "disclaimer": "AI-generated content is for reference only",
            "aboutset": "General Settings",
            "aboutmodel": "Model Settings",
            "aboutdata": "Dataset Settings",
            "theme_set": "Theme Settings",
            "language_set": "Language Settings"
        }
    };

    function applyLanguage(lang) {
        const t = translations[lang];
        for (const id in t) {
            if (id === "user-input-placeholder") {
                document.getElementById("user-input").placeholder = t[id];
            } else {
                const el = document.getElementById(id);
                if (el) el.textContent = t[id];
            }
        }
        // 保存语言偏好
        localStorage.setItem('language', lang);

        // 设置按钮高亮
        langButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.lang === lang) {
                btn.classList.add('active');
            }
        });
    }

    // 初始化语言
    const savedLang = localStorage.getItem('language') || 'zh';
    applyLanguage(savedLang);

    langButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            applyLanguage(btn.dataset.lang);
        });
    });
});
