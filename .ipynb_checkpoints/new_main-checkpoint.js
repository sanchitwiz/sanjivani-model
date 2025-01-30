const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const app = express();

app.get('/send-image', async (req, res) => {
    const imageUrl = req.query.image_url;

    if (!imageUrl) {
        return res.status(400).send('No image URL provided');
    }

    try {

        const imageResponse = await axios({
            url: imageUrl,
            responseType: 'stream', 
        });

        const formData = new FormData();
        formData.append('file', imageResponse.data, 'image.jpg'); 

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