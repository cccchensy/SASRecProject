/**
 * SASRec 番剧推荐系统 - 前端逻辑
 * 纯 JavaScript，无框架依赖
 */

// ========== 全局状态 ==========
let sessionId = null;
let currentCandidate = null;
let likedCount = 0;
let isLoading = false;

// ========== DOM 元素引用 ==========
const $ = (id) => document.getElementById(id);

// ========== 初始化 ==========
document.addEventListener("DOMContentLoaded", () => {
    startSession();
});

// ========== API 调用 ==========

async function startSession() {
    setLoading(true);
    hideError();
    hideResults();

    try {
        const res = await fetch("/api/session/start", { method: "POST" });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "启动会话失败");
        }
        const data = await res.json();

        sessionId = data.session_id;
        likedCount = data.current_count;
        currentCandidate = data.candidate;

        renderCandidate();
        renderProgress();
        showCandidateSection();
    } catch (e) {
        showError("启动推荐会话失败: " + e.message);
    } finally {
        setLoading(false);
    }
}

async function submitFeedback(feedback) {
    if (isLoading || !sessionId || !currentCandidate) return;

    setLoading(true);
    hideError();

    try {
        const res = await fetch("/api/session/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                item_id: currentCandidate.item_id,
                feedback: feedback,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            // 会话不存在则自动重启
            if (res.status === 404) {
                showError("会话已过期，正在重新开始...");
                setTimeout(() => startSession(), 1500);
                return;
            }
            throw new Error(err.detail || "提交反馈失败");
        }

        const data = await res.json();

        likedCount = data.current_count;
        currentCandidate = data.candidate;

        renderProgress();
        renderLikedList(data.liked_items);
        renderCandidate();
    } catch (e) {
        showError("提交反馈失败: " + e.message);
    } finally {
        setLoading(false);
    }
}

async function generateRecommendations() {
    if (isLoading || !sessionId) return;

    setLoading(true);
    hideError();

    try {
        const res = await fetch("/api/session/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });

        if (!res.ok) {
            const err = await res.json();
            if (res.status === 404) {
                showError("会话已过期，请重新开始");
                return;
            }
            throw new Error(err.detail || "生成推荐失败");
        }

        const data = await res.json();
        renderResults(data);
        hideCandidateSection();
        showResults();
    } catch (e) {
        showError("生成推荐失败: " + e.message);
    } finally {
        setLoading(false);
    }
}

function restartSession() {
    sessionId = null;
    currentCandidate = null;
    likedCount = 0;
    startSession();
}

// ========== 渲染函数 ==========

function renderCandidate() {
    if (!currentCandidate) {
        $("candidate-name").textContent = "所有候选番剧已浏览完毕";
        $("candidate-tag").textContent = "";
        $("candidate-tag").className = "tag";
        $("btn-like").disabled = true;
        $("btn-dislike").disabled = true;
        return;
    }

    $("candidate-name").textContent = "《" + currentCandidate.name + "》";
    $("candidate-tag").textContent = currentCandidate.tag;

    // 根据 tag 设置标签颜色
    const tagEl = $("candidate-tag");
    tagEl.className = "tag";
    if (currentCandidate.tag === "热门初始") {
        tagEl.classList.add("tag-hot");
    } else if (currentCandidate.tag === "随机探索") {
        tagEl.classList.add("tag-explore");
    } else if (currentCandidate.tag === "智能关联") {
        tagEl.classList.add("tag-smart");
    }

    $("btn-like").disabled = false;
    $("btn-dislike").disabled = false;
}

function renderProgress() {
    $("liked-count").textContent = likedCount;
    const pct = Math.min((likedCount / 5) * 100, 100);
    $("progress-bar").style.width = pct + "%";
}

function renderLikedList(items) {
    const list = $("liked-list");
    list.innerHTML = "";
    if (!items || items.length === 0) {
        $("liked-section").classList.add("hidden");
        return;
    }
    $("liked-section").classList.remove("hidden");
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item.name;
        list.appendChild(li);
    });
}

function renderResults(data) {
    // Top-10
    const top10List = $("top10-list");
    top10List.innerHTML = "";
    data.top10.forEach((item) => {
        const li = document.createElement("li");
        li.innerHTML =
            '<span class="rank-badge">' +
            item.rank +
            '</span><span>《' +
            escapeHtml(item.name) +
            "》</span>";
        top10List.appendChild(li);
    });

    // I2I 关联推荐
    const i2iContainer = $("i2i-container");
    i2iContainer.innerHTML = "";
    data.related.forEach((group) => {
        const div = document.createElement("div");
        div.className = "i2i-group";

        let html =
            '<p class="i2i-source">因为为您推荐了: 《' +
            escapeHtml(group.source.name) +
            "》</p>";
        html += '<ul class="i2i-items">';
        group.items.forEach((item) => {
            html += "<li>《" + escapeHtml(item.name) + "》</li>";
        });
        html += "</ul>";

        div.innerHTML = html;
        i2iContainer.appendChild(div);
    });

    // 更新已喜欢列表
    renderLikedList(data.liked_items);

    // 更新进度
    likedCount = data.liked_count;
    renderProgress();
}

// ========== UI 切换 ==========

function showCandidateSection() {
    $("candidate-section").classList.remove("hidden");
    $("progress-section").classList.remove("hidden");
    $("action-section").classList.remove("hidden");
}

function hideCandidateSection() {
    $("candidate-section").classList.add("hidden");
    $("action-section").classList.add("hidden");
}

function showResults() {
    $("result-section").classList.remove("hidden");
}

function hideResults() {
    $("result-section").classList.add("hidden");
}

function setLoading(state) {
    isLoading = state;
    $("loading-overlay").classList.toggle("hidden", !state);

    // 禁用/启用所有按钮
    const buttons = document.querySelectorAll(".btn");
    buttons.forEach((btn) => {
        if (state) {
            btn.dataset.wasDisabled = btn.disabled;
            btn.disabled = true;
        } else {
            // 恢复原来的 disabled 状态
            if (btn.dataset.wasDisabled === "true") {
                btn.disabled = true;
            } else {
                btn.disabled = false;
            }
        }
    });
}

function showError(msg) {
    const box = $("error-box");
    box.textContent = msg;
    box.classList.remove("hidden");
}

function hideError() {
    $("error-box").classList.add("hidden");
}

// ========== 工具函数 ==========

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
