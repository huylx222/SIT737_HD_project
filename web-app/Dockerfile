FROM node:16-slim

# Set working directory
WORKDIR /app

# Copy package.json files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app source
COPY . .

# Create uploads directory if it doesn't exist
RUN mkdir -p ./uploads

# Expose port
EXPOSE 3000

# Start the application
CMD ["node", "app.js"]