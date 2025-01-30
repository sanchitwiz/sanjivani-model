const express = require('express');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const app = express();

app.get('/send-image', async (req, res) => {
    const imagePath = 'C:\\Users\\sanid\\Downloads\\vitty\\download.jpg'; // Your specific file path

    try {
        // Create form data and append the image file
        const formData = new FormData();
        formData.append('file', fs.createReadStream(imagePath));

        // Send the form data to Flask
        const flaskResponse = await axios.post('http://localhost:5000/predict', formData, {
            headers: {
                ...formData.getHeaders(), // Properly set headers for multipart/form-data
            },
        });

        // Log the response from Flask
        console.log('Flask response:', flaskResponse.data);

        // Send the Flask response back to the client
        res.send(flaskResponse.data);
    } catch (error) {
        console.error('Error processing image:', error.message);
        res.status(500).send('Error processing image.');
    }
});

// Start the Node.js server
app.listen(3000, () => {
    console.log('Server is running on port 3000');
});