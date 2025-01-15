FROM ottomator/base-python:latest

WORKDIR /app

# Switch to root user
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch to the 'appuser' user
USER appuser

# Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install dependencies
# RUN pip install fastapi
# RUN pip install uvicorn
# RUN pip install pydantic
# RUN pip install supabase
# RUN pip install python-dotenv
# RUN pip install asyncpg
# RUN pip install pandas


# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application
COPY . .

# Run the application
CMD ["uvicorn", "Watson:app", "--host", "0.0.0.0", "--port", "8001"]