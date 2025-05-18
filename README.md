1. Git clone the code
   
   git clone https://github.com/huylx222/SIT737_HD_project
   cd SIT737_HD_project

3. Download the model
   
   mkdir -p api_server/models
   curl -L -o "api_server/models/${MODEL_FILENAME}" https://github.com/huylx222/SIT753_HD/releases/download/v1.0.0/resnet50_epoch007_acc199.290_bpcer_0.019_apcer_0.014.pth

5. Build the images
   
   docker build -t web:newest ./web-app
   docker build -t api:newest ./api_server

7. Docker compose
   
   docker compose-up

9. Push the image to gcp and run Kubernetes
    
   docker tag web:newest {GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/test_project-api:${IMAGE_TAG}
   docker tag api:newest ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/test_project-web:${IMAGE_TAG}
   
   docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/test_project-api:${IMAGE_TAG}
   docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/test_project-web:${IMAGE_TAG}

   kubectl apply -f api-deployment.yaml
   kubectl apply -f api-service.yaml
   kubectl apply -f web-deployment.yaml
   kubectl apply -f web-service.yaml
