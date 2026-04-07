# Use Python 3.9 as the base image
FROM python:3.10

# Set the working directory
WORKDIR /code

# Copy all files from your repo into the container
COPY . .

# Install the dependencies listed in your requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set up a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Move to the app directory and set ownership
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Run your main file
# We use quotes because of the spaces in "app file .py"
CMD ["python", "inference.py"]
