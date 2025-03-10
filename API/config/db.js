const mongoose = require("mongoose");
require("dotenv").config();

const dbUri = process.env.MONGODB_URI;

mongoose
  .connect(dbUri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => {
    console.log("MongoDB Connected");
  })
  .catch((error) => {
    console.error("Connection Error:", error);
  });

module.exports = mongoose;
