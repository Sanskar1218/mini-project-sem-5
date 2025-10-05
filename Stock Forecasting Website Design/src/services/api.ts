import { z } from 'zod';

// API Response schemas for type safety
const StockSchema = z.object({
  symbol: z.string(),
  name: z.string()
});

const ForecastMetricsSchema = z.object({
  rmse: z.number().nullable(),
  mae: z.number().nullable(),
  accuracy: z.number().nullable(),
  error: z.string().optional()
});

const ForecastDataSchema = z.object({
  dates: z.array(z.string()),
  arima: z.array(z.number()),
  lstm: z.array(z.number()),
  meta: z.array(z.number())
});

const ForecastResultsSchema = z.object({
  metrics: z.object({
    arima: ForecastMetricsSchema,
    lstm: ForecastMetricsSchema,
    meta: ForecastMetricsSchema
  }),
  forecast: ForecastDataSchema
});

const LiveQuoteSchema = z.object({
  ltp: z.number(),
  open: z.number(),
  high: z.number(),
  low: z.number(),
  volume: z.number(),
  change: z.number(),
  change_pct: z.number()
});

const ForecastResponseSchema = z.object({
  success: z.boolean(),
  data_source: z.string(),
  live_quote: LiveQuoteSchema.nullable(),
  results: ForecastResultsSchema,
  latest_close: z.number(),
  data_range: z.object({
    start: z.string(),
    end: z.string(),
    records: z.number()
  })
});

const StocksResponseSchema = z.object({
  success: z.boolean(),
  stocks: z.array(StockSchema)
});

const HealthResponseSchema = z.object({
  status: z.string()
});

// Types derived from schemas
export type Stock = z.infer<typeof StockSchema>;
export type ForecastMetrics = z.infer<typeof ForecastMetricsSchema>;
export type ForecastData = z.infer<typeof ForecastDataSchema>;
export type ForecastResults = z.infer<typeof ForecastResultsSchema>;
export type LiveQuote = z.infer<typeof LiveQuoteSchema>;
export type ForecastResponse = z.infer<typeof ForecastResponseSchema>;
export type StocksResponse = z.infer<typeof StocksResponseSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export interface ForecastRequestParams {
  symbol: string;
  totp?: string;
  use_angelone?: boolean;
  forecast_days?: number;
}

// API service class
export class StockForecastAPI {
  private baseURL: string;

  constructor(baseURL: string = 'http://127.0.0.1:5000') {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    responseSchema: z.ZodSchema<T>
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    try {
      return responseSchema.parse(data);
    } catch (error) {
      console.error('API response validation failed:', error);
      throw new Error('Invalid API response format');
    }
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request('/health', {}, HealthResponseSchema);
  }

  async getStocks(): Promise<StocksResponse> {
    return this.request('/api/stocks', {}, StocksResponseSchema);
  }

  async getForecast(params: ForecastRequestParams): Promise<ForecastResponse> {
    const payload: any = {
      symbol: params.symbol,
      forecast_days: params.forecast_days || 7,
    };

    if (params.totp) {
      payload.totp = params.totp;
    }

    if (params.use_angelone !== undefined) {
      payload.use_angelone = params.use_angelone;
    }

    return this.request('/api/forecast', {
      method: 'POST',
      body: JSON.stringify(payload),
    }, ForecastResponseSchema);
  }

  async getMockForecast(params: ForecastRequestParams): Promise<ForecastResponse> {
    const payload: any = {
      symbol: params.symbol,
      forecast_days: params.forecast_days || 7,
    };

    if (params.totp) {
      payload.totp = params.totp;
    }

    return this.request('/api/forecast-mock', {
      method: 'POST',
      body: JSON.stringify(payload),
    }, ForecastResponseSchema);
  }
}

// Export singleton instance
export const stockForecastAPI = new StockForecastAPI();

// Hook for React components to use the API service
export function useApiService() {
  const getStockOptions = () => {
    return [
      { value: 'RELIANCE-EQ', label: 'Reliance Industries' },
      { value: 'TCS-EQ', label: 'Tata Consultancy Services' },
      { value: 'INFY-EQ', label: 'Infosys' },
      { value: 'HDFCBANK-EQ', label: 'HDFC Bank' },
      { value: 'ICICIBANK-EQ', label: 'ICICI Bank' },
      { value: 'SBIN-EQ', label: 'State Bank of India' },
      { value: 'TATAMOTORS-EQ', label: 'Tata Motors' },
      { value: 'WIPRO-EQ', label: 'Wipro' },
      { value: 'ITC-EQ', label: 'ITC' },
      { value: 'BHARTIARTL-EQ', label: 'Bharti Airtel' }
    ];
  };

  return {
    getStockOptions,
    api: stockForecastAPI
  };
}
