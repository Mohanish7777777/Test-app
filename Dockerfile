# Use an official Python image as a base
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir flask pycryptodome numpy

# Expose the port Flask runs on
EXPOSE 8000

# Define the command to run the application
CMD ["python", "app.py"]
