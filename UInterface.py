# Import the Flask library
from flask import Flask, redirect

# Create a Flask web application
app = Flask(__name__)

# Define a route for the root URL ('/')
@app.route('/')
def index():
    # Redirect users to a specific URL (in this case, an HTML page hosted on Amazon S3)
    return redirect('https://movie-recommendation-system.s3.amazonaws.com/index.html')

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
