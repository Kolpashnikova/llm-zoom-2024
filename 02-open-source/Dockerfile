FROM ollama/ollama

# Copy the model weights into the container
COPY ollama_files/models /root/.ollama/models

# Expose the necessary port
EXPOSE 11434

# Set the default command to run when starting the container
CMD ["ollama", "serve"]