document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("video-input");
  const file = fileInput.files[0];
  if (!file) return;

  const loading = document.getElementById("loading");
  const result = document.getElementById("result");
  const resultMessage = document.getElementById("result-message");
  const confidence = document.getElementById("confidence");

  loading.classList.remove("hidden");
  result.classList.add("hidden");

  const formData = new FormData();
  formData.append("video", file);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      resultMessage.textContent = `Error: ${data.error}`;
      confidence.textContent = "";
    } else {
      resultMessage.textContent = data.message;
      confidence.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
      result.className = data.is_deepfake ? "deepfake" : "genuine";
    }
  } catch (error) {
    resultMessage.textContent = "An error occurred during analysis";
    confidence.textContent = "";
  } finally {
    loading.classList.add("hidden");
    result.classList.remove("hidden");
  }
});

document.addEventListener("DOMContentLoaded", function () {
  const videoInput = document.getElementById("video-input");
  const uploadSection = document.querySelector(".upload-section");

  // Create video preview container
  const videoPreviewContainer = document.createElement("div");
  videoPreviewContainer.id = "video-preview-container";
  videoPreviewContainer.className = "hidden";

  // Create video element
  const videoPreview = document.createElement("video");
  videoPreview.id = "video-preview";
  videoPreview.controls = true;
  videoPreview.preload = "metadata";

  // Add video element to container
  videoPreviewContainer.appendChild(videoPreview);

  // Insert after the upload section
  uploadSection.parentNode.insertBefore(
    videoPreviewContainer,
    uploadSection.nextSibling
  );

  // Handle file selection for video preview
  videoInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      // Create object URL for the video file
      const videoURL = URL.createObjectURL(file);

      // Set the video source and show the preview
      videoPreview.src = videoURL;
      videoPreviewContainer.classList.remove("hidden");

      // Clean up the object URL when no longer needed
      videoPreview.onload = function () {
        URL.revokeObjectURL(videoURL);
      };
    } else {
      // Hide preview if no file selected
      videoPreviewContainer.classList.add("hidden");
    }
  });
});
