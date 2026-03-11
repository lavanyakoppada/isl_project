const startBtn = document.getElementById("startBtn");
const generateBtn = document.getElementById("generateBtn");
const spokenText = document.getElementById("spokenText");
const video = document.getElementById("video");
const currentWordDisplay = document.getElementById("currentWord");
const textInput = document.getElementById("textInput");

let lastSpeech = "";
let gloss = [];
let wordIndex = 0;
let letterIndex = 0;
let spellingMode = false;
let currentWord = "";

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  spokenText.innerText = "Speech recognition not supported.";
  startBtn.disabled = true;
} else {

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = false;

  startBtn.addEventListener("click", () => {
    recognition.start();
    spokenText.innerText = "Listening...";
  });

  recognition.onresult = function(event) {
    lastSpeech = event.results[0][0].transcript;
    spokenText.innerText = "Recognized Text: " + lastSpeech;
    generateBtn.disabled = false;
  };

  recognition.onerror = function(event) {
    spokenText.innerText = "Error: " + event.error;
  };
}

// Update lastSpeech when typing
textInput.addEventListener("input", () => {
  lastSpeech = textInput.value;
  generateBtn.disabled = lastSpeech.trim() === "";
});

generateBtn.addEventListener("click", async () => {

  const response = await fetch("/convert_speech/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: lastSpeech })
  });

  const data = await response.json();

  gloss = data.gloss;
  wordIndex = 0;
  playWord();
});

function playWord() {
  if (wordIndex >= gloss.length) {
    currentWordDisplay.innerText = "";
    video.pause();
    return;
  }

  currentWord = gloss[wordIndex];
  currentWordDisplay.innerText = currentWord;

  video.src = `/media/signs/${currentWord}.mp4`;
  video.load();
  video.play();
}

function playLetter() {
  if (letterIndex >= currentWord.length) {
    spellingMode = false;
    letterIndex = 0;
    wordIndex++;
    playWord();
    return;
  }

  const letter = currentWord[letterIndex];
  currentWordDisplay.innerText = letter;

  video.src = `/media/signs/${letter}.mp4`;
  video.load();
  video.play();

  letterIndex++;
}

video.addEventListener("ended", () => {
  if (spellingMode) {
    playLetter();
  } else {
    wordIndex++;
    playWord();
  }
});

video.addEventListener("error", () => {
  spellingMode = true;
  letterIndex = 0;
  playLetter();
});