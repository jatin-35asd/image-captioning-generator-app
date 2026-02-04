function previewImage(event) {
  const img = document.getElementById("preview");
  img.src = URL.createObjectURL(event.target.files[0]);
  img.classList.remove("hidden");
}
