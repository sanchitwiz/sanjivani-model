const express = require('express');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const app = express();

app.get('/send-image', async (req, res) => {
    const imagePath = "./download.jpg"; 

    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(imagePath));

        const flaskResponse = await axios.post('http://localhost:5000/predict', formData, {
            headers: {
                ...formData.getHeaders(),
            },
        });

        console.log('Flask response:', flaskResponse.data);

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