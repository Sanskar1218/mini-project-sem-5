# Stock Market Forecasting API

AI-powered stock market forecasting using ARIMA, LSTM, and Meta-Learning with Angel One API integration.

## 🔒 Security Features

- ✅ Credentials stored in `.env` file (never committed to git)
- ✅ Manual TOTP entry for Angel One API
- ✅ Environment variable validation
- ✅ CORS protection for frontend integration

## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd stock-forecasting-api
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Credentials

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or use any text editor
```

**Edit `.env` file:**
```
ANGEL_API_KEY=mBLsZ8Kv
ANGEL_CLIENT_ID=S57846669
ANGEL_PASSWORD=1812
```

⚠️ **IMPORTANT**: Never commit `.env` to GitHub! It's already in `.gitignore`.

### 5. Run the API

```bash
python stock_forecast_api.py
```

API will start at: `http://localhost:5000`

## 📡 API Endpoints

### 1. Forecast Stock Price

**POST** `/api/forecast`

```json
{
  "totp": "123456",
  "symbol": "RELIANCE-EQ",
  "use_angelone": true,
  "forecast_days": 7
}
```

**Response:**
```json
{
  "success": true,
  "data_source": "angel_one",
  "live_quote": {
    "ltp": 1364.50,
    "change": 15.30,
    "change_pct": 1.13
  },
  "results": {
    "metrics": {
      "arima": {"rmse": 45.23, "mae": 38.15, "accuracy": 97.25},
      "lstm": {"rmse": 42.10, "mae": 35.80, "accuracy": 97.68},
      "meta": {"rmse": 39.87, "mae": 33.42, "accuracy": 97.89}
    },
    "forecast": {
      "dates": ["2025-10-03", "2025-10-04", ...],
      "arima": [1370.25, 1375.80, ...],
      "lstm": [1368.40, 1373.20, ...],
      "meta": [1369.15, 1374.35, ...]
    }
  }
}
```

### 2. Get Available Stocks

**GET** `/api/stocks`

```json
{
  "success": true,
  "stocks": [
    {"symbol": "RELIANCE-EQ", "name": "Reliance Industries"},
    {"symbol": "TCS-EQ", "name": "Tata Consultancy Services"}
  ]
}
```

### 3. Health Check

**GET** `/health`

## 🌐 Frontend Integration (Lovable AI)

### Example React/Next.js Code:

```javascript
// Frontend code to call your API
const forecast = async () => {
  const totp = prompt("Enter your 6-digit TOTP code:");
  
  const response = await fetch('http://localhost:5000/api/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      totp: totp,
      symbol: 'RELIANCE-EQ',
      use_angelone: true,
      forecast_days: 7
    })
  });
  
  const data = await response.json();
  
  if (data.success) {
    console.log('Forecast:', data.results.forecast);
    console.log('Accuracy:', data.results.metrics);
  }
};
```

## 🔐 Getting Angel One TOTP

## 📱 Simple Local Frontend

We've added a minimal static frontend in the `frontend/` folder to quickly test the API.

Open the file directly in a browser:

1. Start the backend (see Run the API section).
2. Open `frontend/index.html` in your browser (File > Open or drag it into a browser window).

Serve the frontend (recommended if you hit CORS issues) from the project root:

```powershell
# from project root
& ".venv\Scripts\python.exe" -m http.server 8000
# then open http://127.0.0.1:8000/frontend/index.html
```

The page includes buttons to call `/health`, list `/api/stocks`, and submit a `/api/forecast` request (it will prompt for the Angel One TOTP when enabled).

1. Go to [Angel One SmartAPI](https://smartapi.angelbroking.com/)
2. Click **"Enable TOTP"**
3. Scan QR code with Google Authenticator
4. Enter the 6-digit code when API requests it

## 📊 Available Stocks

- RELIANCE-EQ (Reliance Industries)
- TCS-EQ (Tata Consultancy Services)
- INFY-EQ (Infosys)
- HDFCBANK-EQ (HDFC Bank)
- ICICIBANK-EQ (ICICI Bank)
- SBIN-EQ (State Bank of India)
- TATAMOTORS-EQ (Tata Motors)
- WIPRO-EQ (Wipro)
- ITC-EQ (ITC)
- BHARTIARTL-EQ (Bharti Airtel)

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY stock_forecast_api.py .

CMD ["python", "stock_forecast_api.py"]
```

```bash
docker build -t stock-forecast-api .
docker run -p 5000:5000 --env-file .env stock-forecast-api
```

## ☁️ Deploy to Cloud

### Heroku

```bash
heroku create your-app-name
heroku config:set ANGEL_API_KEY=mBLsZ8Kv
heroku config:set ANGEL_CLIENT_ID=S57846669
heroku config:set ANGEL_PASSWORD=1812
git push heroku main
```

### Railway.app

1. Push to GitHub (`.env` won't be pushed due to `.gitignore`)
2. Import repo to Railway
3. Add environment variables in Railway dashboard
4. Deploy automatically

### Render

1. Connect GitHub repo
2. Add environment variables in Render dashboard
3. Deploy

## 🛡️ Security Best Practices

1. ✅ **Never commit `.env`** - Already in `.gitignore`
2. ✅ **Use environment variables** - Credentials loaded from `.env`
3. ✅ **TOTP authentication** - User enters TOTP manually each time
4. ✅ **HTTPS in production** - Use SSL certificates
5. ✅ **Rate limiting** - Add rate limiting for production
6. ✅ **API keys** - Consider adding API key authentication for frontend

## 📝 License

MIT

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. **Never commit credentials**
4. Submit pull request

## ❓ Support

## Automatic TOTP generation

If you want the server to generate TOTP codes automatically (so you don't need to paste them every time), add your base32 TOTP secret to the `.env` file as `ANGEL_TOTP_SECRET`.

Example `.env` entry:
```
ANGEL_TOTP_SECRET=C2DA3JAWEKZW4REKDVFBEHAPQA
```

When `ANGEL_TOTP_SECRET` is present the backend will create a TOTP code automatically whenever a request sets `use_angelone: true` but omits `totp` in the POST body. Keep this secret secure and never commit `.env` to version control.

Issues? Open a GitHub issue or contact support.