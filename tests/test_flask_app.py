import unittest
import sys
import os

# Add the project root to path so we can import from flask_app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure app for testing
        app.testing = True
        cls.client = app.test_client()

    def test_home_page(self):
        """Test if the home page loads successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Check for the title we used in index.html
        self.assertIn(b'Movie Sentiment Analysis', response.data)

    def test_predict_route(self):
        """Test the prediction logic via the web interface"""
        # Simulate a form submission
        response = self.client.post('/predict', data=dict(text="I loved this movie, it was fantastic!"))
        
        self.assertEqual(response.status_code, 200)
        
        # Check if our template logic (Positive/Negative) is present
        # Note: In your HTML, you output 'Positive' or 'Negative' inside the H2 tags
        is_positive = b'Positive' in response.data
        is_negative = b'Negative' in response.data
        
        self.assertTrue(is_positive or is_negative, "Response should contain prediction result")

if __name__ == '__main__':
    unittest.main()