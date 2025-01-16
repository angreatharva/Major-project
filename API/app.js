const express = require("express");
const cors = require("cors");
const multer = require("multer");
const userController = require("./controller/user.controller");
const UserModel = require("./model/user.model");
const DrModel = require("./model/doctor.model");
const doctorController = require("./controller/doctor.controller");

const app = express();

// Increase the JSON payload limit
app.use(express.json({ limit: "50mb" }));
// If you're also using URL-encoded data, increase that limit too
app.use(express.urlencoded({ limit: "50mb", extended: true }));

app.use(
  cors({
    origin: "http://192.168.0.104",
  })
);

// Setup multer for handling image uploads
const storage = multer.memoryStorage();

const upload = multer({
  storage: storage,
  limits: { fileSize: 20 * 1024 * 1024 }, // Limit to 20MB
}).single("image");

app.use(upload);

app.post("/registerUser", userController.registerUser);
app.get("/users", userController.getRegisteredUsers);
app.post("/registerDoctor", doctorController.registerDoctor);

app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    // Check if the user is a patient
    const user = await UserModel.findOne({ email });
    if (user && (await user.comparePassword(password))) {
      return res.status(200).json({
        success: true,
        role: "patient",
        userId: user._id,
        userName: user.userName,
        message: "Login successful as a patient",
      });
    }

    // Check if the user is a doctor
    const doctor = await DrModel.findOne({ email });
    if (doctor && (await doctor.comparePassword(password))) {
      return res.status(200).json({
        success: true,
        role: "doctor",
        userId: doctor._id,
        userName: doctor.doctorName,
        message: "Login successful as a doctor",
      });
    }

    // If no match is found
    return res.status(401).json({
      success: false,
      message: "Invalid email or password",
    });
  } catch (err) {
    console.error("Error during login:", err);
    res.status(500).json({
      success: false,
      message: "An error occurred while logging in",
    });
  }
});

module.exports = app;
