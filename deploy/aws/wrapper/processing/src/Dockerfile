FROM mambaorg/micromamba:debian12-slim

USER root

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

# Copy dependencies files
COPY preprocess-env.yml /tmp/env.yml
COPY requirements.txt /tmp/extra-requirements.txt

# Create a new environment with the dependencies of the source code
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean -a -y
# Install extra dependencies for the AWS wrapper 
RUN pip install --no-cache-dir -r /tmp/extra-requirements.txt


COPY app.py /app/
COPY training_trigger.py /app/

# At runtime, we'll mount the full code from S3
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "app.py"]