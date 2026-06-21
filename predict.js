const url = 'http://127.0.0.1:8000/predict';

const payload = {
  age: 25,
  resting_blood_pressure: 180,
  cholesterol: 410.5,
  max_heart_rate: 250,
  exercise_induced_angina: 0,
  num_major_vessels: 0,
  chest_pain_type: "non_anginal"
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log('Success:', data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
  