# Insurance Premium Prediction API

A Machine Learning-powered REST API built with **FastAPI** that predicts an individual's insurance premium category based on their personal, lifestyle, financial, and demographic information.

The model classifies users into three insurance premium categories:

- **Low**
- **Medium**
- **High**

The API uses **Pydantic** for request/response validation and **Docker** for containerization.

---

## 🚀 Project Overview

Insurance premium prediction is a Machine Learning classification problem where customer information is analysed to estimate their potential insurance premium category.

The model considers factors such as:

- Age
- Weight
- Height
- Income
- Smoking status
- City
- Occupation

Based on these inputs, the trained Machine Learning model predicts whether the user's insurance premium category is:

```text
Low / Medium / High
