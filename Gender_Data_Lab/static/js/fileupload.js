// Toggle the form sections based on selected source type (Internal or External)
function toggleFormSections() {
  const selectedType = document.getElementById("sourceType").value;
  document.getElementById("hidden_source_type").value = selectedType;

  document.getElementById("internalForm").style.display = (selectedType === "internal") ? "block" : "none";
  document.getElementById("externalForm").style.display = (selectedType === "external") ? "block" : "none";
}

// General function to show error messages
function showMessage(message, type = "alert") {
  const messageContainer = document.getElementById("messages");
  messageContainer.innerHTML = `<div class="${type}">${message}</div>`;
}

// Simulate progress and submit the form after upload
function simulateUploadProgress(file, callback) {
  const loading = document.getElementById("loading");
  const progress = document.getElementById("progress");
  const progressText = document.getElementById("progressText");

  const fileSizeMB = file.size / (1024 * 1024);
  const maxTime = Math.min(fileSizeMB * 100, 5000); // Max 5 seconds
  const intervalTime = maxTime / 100;

  loading.style.display = "block";
  progress.style.width = "0%";
  progressText.textContent = "Uploading...";

  let width = 0;
  const interval = setInterval(() => {
    if (width >= 100) {
      clearInterval(interval);
      progressText.textContent = "Upload complete!";
      setTimeout(callback, 300); // Callback to submit
    } else {
      width += 1;
      progress.style.width = width + "%";
    }
  }, intervalTime);
}

// Upload button that decides which input to use and submits
function submitToServer() {
  const sourceType = document.getElementById("sourceType").value;
  const form = document.getElementById("uploadForm");

  if (!sourceType) {
    showMessage("Please select a file source before uploading.");
    return;
  }

  let fileInput;
  if (sourceType === "internal") {
    fileInput = document.getElementById("dataFile_internal");
  } else if (sourceType === "external") {
    fileInput = document.getElementById("dataFile_external");
  }

  if (!fileInput || !fileInput.files.length) {
    showMessage("Please select a file to upload.");
    return;
  }

  const file = fileInput.files[0];
  simulateUploadProgress(file, () => form.submit());
}

// Optional testing button (e.g., for development)
function simulateUpload() {
  submitToServer(); // You can separate logic here if you want different behavior
}
