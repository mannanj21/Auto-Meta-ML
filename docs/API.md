# API Documentation

## Base URL
- Development: `http://localhost:8000`
- Production: `https://your-backend.railway.app`

## Endpoints

### GET /
Health check
```json
{
  "message": "Meta-Learning Algorithm Recommender API",
  "version": "1.0.0"
}
```

### POST /api/predict
Upload dataset and get recommendations

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "algorithm": "Random Forest",
      "confidence": 0.85,
      "rank": 1,
      "explanation": "Robust ensemble method...",
      "characteristics": {
        "speed": "Medium",
        "interpretability": "Medium"
      }
    }
  ],
  "visualization_url": "/api/visualization/viz_abc123.png",
  "dataset_info": {
    "filename": "data.csv",
    "rows": 1000,
    "columns": 20
  }
}
```

### GET /api/algorithms
List supported algorithms

**Response:**
```json
{
  "algorithms": [
    "Naive Bayes",
    "Random Forest",
    "Decision Tree (J48/C4.5)",
    ...
  ]
}
```

## Error Responses

```json
{
  "detail": "Error message here"
}
```

Status codes:
- 400: Bad Request (invalid file)
- 404: Not Found
- 500: Internal Server Error
