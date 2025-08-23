---
title: AuraScanAI
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
---

# AuraScanAI - Vehicle Damage Assessment API

This is the backend API for the AuraScanAI project. It uses a Vision Transformer model
to detect vehicle damage and assess its severity from an image.

## Endpoints

-   `/ping`: A simple GET endpoint to check if the server is alive.
-   `/analyze`: A POST endpoint that accepts a single image file (`multipart/form-data`) and returns a JSON object with the analysis.