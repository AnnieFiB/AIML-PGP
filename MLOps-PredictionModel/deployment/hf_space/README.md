
---
title: Wellness Tourism Package Prediction
emoji: 🌍
sdk: docker
app_port: 7860
---

# 🌍 Wellness Tourism Package Prediction

A Docker-based Hugging Face Space providing:

- Streamlit prediction interface
- FastAPI prediction service
- Model loaded from Hugging Face Model Hub

## Architecture

```text
Streamlit UI
     ↓
FastAPI
     ↓
Preprocessing Pipeline
     ↓
Prediction Model
