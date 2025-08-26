---
title: AuraScanAI API
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
---

# AuraScanAI - Vehicle Damage Assessment API

This repository contains the complete backend service for the AuraScanAI project, a sophisticated AI-powered system for analyzing vehicle damage from images. The API is built with Flask and serves a custom-trained, multi-task Vision Transformer (ViT) model capable of identifying damage areas and assessing their severity.

The live API is deployed as a Docker container on Hugging Face Spaces.

---

## 🚀 Key Features

*   **AI-Powered Analysis:** Leverages a state-of-the-art Vision Transformer (ViT) model fine-tuned on over 15,000 images of vehicle damage.
*   **Multi-Task Learning:** The model simultaneously predicts:
    1.  The location of the primary damage area (bounding box).
    2.  The overall severity of the damage (`minor`, `moderate`, `severe`).
*   **Business Logic Engine:** Includes a post-processing layer to translate AI outputs into actionable business insights, including a realistic estimated repair cost range.
*   **Scalable Architecture:** Built with a professional, singleton pattern to ensure the large AI model is loaded only once, providing fast and efficient inference.
*   **Containerized & Deployable:** Fully containerized with Docker and configured for seamless deployment on cloud platforms like Hugging Face Spaces.

---

## 🛠️ Technology Stack

*   **AI Framework:** PyTorch
*   **Vision Model Library:** `timm` (PyTorch Image Models)
*   **API Framework:** Flask
*   **WSGI Server:** Gunicorn
*   **Containerization:** Docker
*   **Cloud Deployment:** Hugging Face Spaces

---

## ⚙️ API Endpoints

The server provides two main endpoints:

### 1. Health Check

A simple endpoint to verify that the server is running and responsive.

*   **Endpoint:** `/ping`
*   **Method:** `GET`
*   **Success Response (200):**
    ```json
    {
      "message": "Server is alive!",
      "status": "ok"
    }
    ```

### 2. Damage Analysis

The core endpoint for analyzing an image.

*   **Endpoint:** `/analyze`
*   **Method:** `POST`
*   **Request Body:** `multipart/form-data` with a single field:
    *   `file`: The vehicle image file (`.jpg`, `.png`, etc.).
*   **Success Response (200):** A detailed JSON object containing the full analysis.
    ```json
    {
      "success": true,
      "totalDamages": 1,
      "overallSeverity": "severe",
      "confidence": "0.66",
      "costRange": {
        "min": 800,
        "max": 2500
      },
      "damages": [
        {
          "id": "dmg-1",
          "type": "Primary Damage Area",
          "location": "Detected by AI",
          "severity": "severe",
          "estimatedCost": { "min": 800, "max": 2500 },
          "coordinates": [
            [ 423.06, 364.49, 1229.91, 859.14 ]
          ]
        }
      ]
    }
    ```
*   **Error Response (4xx/5xx):**
    ```json
    {
      "success": false,
      "error": "Descriptive error message."
    }
    ```

---

## 📜 MVP Approach & Future Roadmap

This project serves as a powerful Proof of Concept (MVP), demonstrating a complete end-to-end pipeline for AI-powered vehicle damage assessment.

**Current Capability (MVP):**
The current AI model is an **Image Assessment Model**, designed to identify the single most prominent damage area in an image. It provides a holistic analysis, including an overall severity classification, estimated repair cost, and a bounding box for the primary damage region. This successfully proves the core technology is viable.

**Future Roadmap:**
The next phase of this project will involve evolving the AI core into a full **Multi-Object Detector** (e.g., using a DETR or YOLO architecture). This will enable the system to:
-   Identify and draw bounding boxes for multiple, distinct damages in a single image.
-   Provide a detailed breakdown and cost estimate for each individual damage in the Damage Ledger.