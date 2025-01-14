const doctorService = require("../services/doctor.services");

const registerDoctor = async (req, res) => {
  const { image } = req.body; // Get the image as base64 string

  if (!image) {
    return res.status(400).json({
      success: false,
      message: "No image uploaded.",
    });
  }

  try {
    // Decode the base64 string to buffer
    const imageBuffer = Buffer.from(image, "base64");

    // Prepare doctor data from request body and image buffer
    const doctorData = {
      doctorName: req.body.doctorName,
      phone: req.body.phone,
      age: req.body.age,
      gender: req.body.gender,
      email: req.body.email,
      qualification: req.body.qualification,
      specialization: req.body.specialization,
      licenseNumber: req.body.licenseNumber,
      password: req.body.password,
      image: imageBuffer, // Use the decoded buffer
    };

    const savedDoctor = await doctorService.registerDoctor(doctorData);

    res.status(201).json({
      success: true,
      data: savedDoctor,
      message: "Doctor registered successfully!",
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Error registering doctor: " + error.message,
    });
  }
};

module.exports = {
  registerDoctor,
};
