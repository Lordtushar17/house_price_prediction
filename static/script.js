let predictedPrice = 0;  // Variable to store predicted price in dollars
let converted = false;  // Flag to check the conversion state

async function predictPrice() {
    const size = document.getElementById('size').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input_data: [parseInt(size)] }),
    });
    const data = await response.json();
    predictedPrice = data.prediction[0];
    document.getElementById('result').innerText = `Predicted Price: $${predictedPrice}`;
    converted = false;  // Reset conversion state
    document.getElementById('convert-button').innerText = 'Convert to Rupees';
}

function convertCurrency() {
    const conversionRate = 74.5;  // Example conversion rate (1 USD = 74.5 INR)
    if (!converted) {
        const priceInRupees = predictedPrice * conversionRate;
        document.getElementById('result').innerText = `Predicted Price: ₹${priceInRupees.toFixed(2)}`;
        document.getElementById('convert-button').innerText = 'Convert to Dollars';
        converted = true;
    } else {
        document.getElementById('result').innerText = `Predicted Price: $${predictedPrice}`;
        document.getElementById('convert-button').innerText = 'Convert to Rupees';
        converted = false;
    }
}
