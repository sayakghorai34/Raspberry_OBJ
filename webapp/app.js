const express = require("express");
const multer = require("multer");
const axios = require("axios");
const path = require("path");
const fs = require("fs");
const app = express();
const PORT = 3000;

// Python server configuration
//const PYTHON_SERVER = "http://localhost:8000"; // Update with the Python server IP/port
const PYTHON_SERVER = process.env.SERVER_URL || "http://python-server:8000";

// Static folder for CSS and client-side assets
app.use(express.static(path.join(__dirname, "public")));

// Middleware for parsing form data
const upload = multer({ dest: "uploads/" });

// View engine setup
app.set("view engine", "ejs");

// Routes
// Home Page
app.get('/', (req, res) => {
    res.render('index', { message: null }); // Ensure 'message' is passed
  });
  

// Stream Route
app.get("/stream", (req, res) => {
  res.redirect(`${PYTHON_SERVER}/stream`); // Redirect to the Python server's camera stream
});

// Upload Image
app.post("/upload/image", upload.single("image"), async (req, res) => {
  try {
    const filePath = req.file.path;
    const response = await axios.post(`${PYTHON_SERVER}/upload/image`, 
      fs.readFileSync(filePath), 
      { headers: { "Content-Type": "image/jpeg" } }
    );

    const { output_image, image_data, class_counts } = response.data;

    res.render("index", {
      message: "Image processed successfully!",
      result: output_image, 
      imgData: image_data, 
      videoData: null, // Include videoData and set to null
      classCounts: class_counts 
    });
  } catch (error) {
    console.error(error);
    res.render("index", { 
      message: "Failed to process image.", 
      result: null, 
      imgData: null, 
      videoData: null, 
      classCounts: null 
    });
  } finally {
    fs.unlinkSync(req.file.path); // Clean up the uploaded file
  }
});


app.post("/upload/video", upload.single("video"), async (req, res) => {
  try {
    const videoData = fs.readFileSync(req.file.path);

    const response = await axios.post(`${PYTHON_SERVER}/upload/video`, videoData, {
      headers: { "Content-Type": "video/mp4" },
      responseType: "arraybuffer", // Receive binary data
    });

    const videoBase64 = Buffer.from(response.data, "binary").toString("base64");
    const videoSrc = `data:video/mp4;base64,${videoBase64}`;

    res.render("index", {
      message: "Video processed successfully!",
      videoData: videoSrc,
      imgData: null, // Include imgData and set to null
      classCounts: null // Include classCounts and set to null
    });
  } catch (error) {
    console.error(error);
    res.render("index", { 
      message: "Failed to process video.", 
      videoData: null, 
      imgData: null, 
      classCounts: null 
    });
  } finally {
    fs.unlinkSync(req.file.path); // Clean up the uploaded file
  }
});



// Start Server
app.listen(PORT, () => {
  console.log(`Frontend running at http://localhost:${PORT}`);
});
