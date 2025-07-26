from fastapi import FastAPI, Request
import json
import uvicorn
from datetime import datetime

app = FastAPI(title="MLOps Alert Webhook Receiver")

@app.post("/webhook")
async def receive_alert(request: Request):
    """Receive and log alerts from Grafana"""
    try:
        payload = await request.json()
        
        # Log the alert
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*50}")
        print(f"ALERT RECEIVED AT: {timestamp}")
        print(f"{'='*50}")
        
        # Extract alert information
        if "alerts" in payload:
            for alert in payload["alerts"]:
                print(f"Alert Name: {alert.get('labels', {}).get('alertname', 'Unknown')}")
                print(f"Severity: {alert.get('labels', {}).get('severity', 'Unknown')}")
                print(f"Component: {alert.get('labels', {}).get('component', 'Unknown')}")
                print(f"Status: {alert.get('status', 'Unknown')}")
                print(f"Description: {alert.get('annotations', {}).get('description', 'No description')}")
                print(f"Value: {alert.get('annotations', {}).get('value', 'No value')}")
                print("-" * 30)
        
        # Log full payload for debugging
        print(f"Full payload: {json.dumps(payload, indent=2)}")
        print(f"{'='*50}\n")
        
        return {"status": "received", "timestamp": timestamp}
        
    except Exception as e:
        print(f"Error processing alert: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("Starting MLOps Alert Webhook Receiver on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
