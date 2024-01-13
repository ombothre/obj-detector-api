# Object Detection and Analysis with FastAPI

This FastAPI application facilitates the processing and analysis of videos, extracting object information and providing predictions based on user-defined criteria. The application utilizes the YOLO (You Only Look Once) object detection model and integrates with a PostgreSQL database to store video metadata and detection results.

## Prerequisites

Before running the application, ensure you have the required dependencies installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the FastAPI application, run the following command:

```bash
uvicorn obj_detector:app --reload
```



# Routes

## Home

- **Route:** `/`
- **Method:** GET
- **Description:** Simple welcome message to verify the application is running.

## Process Video

- **Route:** `/process-video/`
- **Method:** POST
- **Description:** Upload a video file for processing and analysis.

  **Request Parameters:**
  - `name`: Name of the video.
  - `video`: Video file to be uploaded.

  **Response:**
  Successful response includes the uploaded video's name and filename.

  ![image](https://github.com/ombothre/obj-detector-api/assets/92716010/d98ba3b2-f8fa-4136-8fad-463e921eb9eb)


## Predict Objects

- **Route:** `/predict/{name}/`
- **Method:** POST
- **Description:** Obtain predictions based on specified object frequencies for a particular video.

  **Request Parameters:**
  - `name`: Name of the video for which predictions are requested.
  - `objects`: List of objects with frequencies to predict.

  **Response:**
  Predicted timestamps for specified objects based on the given criteria.

  ![image](https://github.com/ombothre/obj-detector-api/assets/92716010/361b2fa7-fef2-411c-98b2-7f68dcac0024)


# Database

The application uses a PostgreSQL database to store video metadata and detection results. The database schema includes two tables:

1. **videos:** Stores video metadata.
2. **detections:** Stores object detection results linked to the corresponding video.

# Usage

1. Upload a video for processing using the `/process-video/` route.
2. Use the `/predict/{name}/` route to obtain predictions based on specified object frequencies.

**Note:** Update database connection details in the code to match your PostgreSQL configuration.
