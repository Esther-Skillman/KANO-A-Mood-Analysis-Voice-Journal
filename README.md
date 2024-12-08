# KANO-A-Mood-Analysis-Voice-Journal

### Project Title
KANO: A Mood Analysis Voice Journal

### Vision
KANO ('Keep A New Outlook')  will be a mood analysis voice journal that will use AI to discern the emotions of a person’s voice entry. Enabling them to reflect on the emotions they convey, providing a deeper and unbiased insight into their overall well-being.

### Supervisor
Vassilis Cutsuridis

### Project Features

| **Feature**                  | **MVP (Docker)**                                     | **Full-Scale (Google Cloud)**                        |
|------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Back-End API**             | Flask/FastAPI in a Docker container (local/VPS).    | Cloud Run (serverless, scalable API).               |
| **AI Model Hosting**         | Model runs in the same Docker container as API.     | Cloud Run for separate model hosting.  |
| **File Storage**             | Local storage or temporary container storage.       | Google Cloud Storage (scalable and reliable).       |
| **Front-End**                | React hosted locally or in another Docker container.| React hosted on Firebase Hosting.                   |
| **Scalability**              | Limited by hardware.                                | Fully scalable with Google Cloud.                   |
| **Cost**                     | Free (local) or ~$5/month (VPS).                    | Free under Google Cloud’s free tier; scales with use.|
