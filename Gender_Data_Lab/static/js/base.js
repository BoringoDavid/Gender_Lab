// working on provide feedback button of top navigation 
document.addEventListener("DOMContentLoaded", function () {
    const feedbackBtn = document.getElementById("feedbackBtn");
  
    if (feedbackBtn) {
      feedbackBtn.addEventListener("click", function () {
        window.open(
          "https://surveys.paris21.org/limesurvey/index.php/796189?lang=en",
          "_blank"
        );
      });
    }
  });
  // end of navigation

  
  