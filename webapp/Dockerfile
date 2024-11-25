# Use Node.js official image
FROM node:18

# Copy package.json and install dependencies
COPY . /webapp
WORKDIR /webapp
RUN npm install
#RUN npm i express 
#RUN npm i multer 
#RUN npm i axios
#RUN npm i ejs
# Copy the entire project


# Expose the port the app runs on
EXPOSE 3000
