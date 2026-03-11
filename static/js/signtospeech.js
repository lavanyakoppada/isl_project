async function startCamera() {
    const stopBtn = document.getElementById('vedio-stop');
    const video = document.getElementById('video');
    const placeholder = document.getElementById('video-placeholder');
    const text = document.querySelector('.text');
    try {
        await fetch('/api/record/start/', { method: 'POST' });
        if (stopBtn) stopBtn.disabled = false;
        if (video) {
            video.src = "/api/record/stream/";
            video.style.display = "block";
        }
        if (placeholder) {
            placeholder.style.display = "none";
            placeholder.style.visibility = "hidden";
        }
        if (text) text.textContent = "Recording started. Use Stop to finish.";
    } catch (e) {
        alert("Failed to start recording: " + e.message);
    }
}

async function stopCamera() {
    const stopBtn = document.getElementById('vedio-stop');
    const text = document.querySelector('.text');
    try {
        const res = await fetch('/api/record/stop/', { method: 'POST' });
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || 'Failed to stop');
        }
        if (stopBtn) stopBtn.disabled = true;
        const video = document.getElementById('video');
        const placeholder = document.getElementById('video-placeholder');
        if (video) {
            video.src = "";
            video.style.display = "none";
        }
        if (placeholder) {
            placeholder.style.display = "flex";
            placeholder.style.visibility = "visible";
        }
        if (text) text.textContent = "Processing signs...";
        const result = await fetchSentenceWithRetry(12, 800);
        if (text) {
            const sentence = result && typeof result.sentence === 'string' ? result.sentence.trim() : "";
            const gloss = result && typeof result.gloss === 'string' ? result.gloss.trim() : "";
            const validSentence = sentence && sentence.toLowerCase() !== "no signs detected.";
            text.textContent = validSentence ? sentence : (gloss || "No signs detected.");
        }
    } catch (e) {
        alert("Failed to stop recording: " + e.message);
    }
}

async function fetchSentenceWithRetry(maxAttempts = 10, delayMs = 800) {
    await new Promise(res => setTimeout(res, 400));
    for (let i = 0; i < maxAttempts; i++) {
        try {
            const r = await fetch('/api/gloss/sentence/?t=' + Date.now(), { cache: 'no-store' });
            const j = await r.json();
            const sentence = j && typeof j.sentence === 'string' ? j.sentence.trim() : "";
            const gloss = j && typeof j.gloss === 'string' ? j.gloss.trim() : "";
            const sentenceOk = sentence && sentence.toLowerCase() !== "no signs detected.";
            const glossOk = gloss && gloss.length > 0;
            if (sentenceOk || glossOk) return j;
        } catch (e) {
        }
        await new Promise(res => setTimeout(res, delayMs));
    }
    try {
        const r = await fetch('/api/gloss/sentence/?t=' + Date.now(), { cache: 'no-store' });
        const j = await r.json();
        return j || { sentence: "", gloss: "" };
    } catch {
        return { sentence: "", gloss: "" };
    }
}

// Play Audio
const playBtn = document.getElementById('read');
if (playBtn) {
    playBtn.addEventListener('click', async () => {
        const textEl = document.querySelector('.text');
        const msg = textEl ? (textEl.textContent || "").trim() : "";
        if (!msg) { alert("No text to speak"); return; }
        try {
            if ('speechSynthesis' in window) {
                const u = new SpeechSynthesisUtterance(msg);
                u.rate = 1.0;
                u.pitch = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
            } else {
                await fetch('/api/tts/speak/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: msg })
                });
            }
        } catch (e) {
            alert("Failed to play audio: " + e.message);
        }
    });
}