# FROM ollama/ollama

# # Copy the model weights into the container
# COPY ollama_files/models /root/.ollama/models

# # Expose the necessary port
# EXPOSE 11434

# # Set the default command to run when starting the container
# CMD ["ollama", "serve"]


# First stage: Use a full Linux distribution for copying files
FROM ubuntu:20.04 AS builder

# Install necessary tools
RUN apt-get update && apt-get install -y wget

# Create the directory for model weights
RUN mkdir -p /root/.ollama/models

# Copy the model weights into the container
COPY ollama_files/models /root/.ollama/models

# Second stage: Use the official ollama image
FROM ollama/ollama

# Copy the model weights from the builder stage
COPY --from=builder /root/.ollama/models /root/.ollama/models

# Expose the necessary port
EXPOSE 11434

# Set the default command to run when starting the container
CMD ["serve"]
