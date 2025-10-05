import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { DashboardSidebar } from "./DashboardSidebar";
import { ForecastForm, ForecastParams } from "./ForecastForm";
import { MetaForecastResults } from "./MetaForecastResults";
import { Search, Bell, Clock, CheckCircle, AlertCircle, Activity, Sparkles, TrendingUp } from "lucide-react";
import { stockForecastAPI, ForecastResponse } from "../services/api";
import { toast } from "sonner";

export function Dashboard() {
  // CRITICAL: This proves the updated file is loaded
  alert('DASHBOARD LOADED - UPDATED VERSION');
  console.log('=== DASHBOARD COMPONENT LOADED - USING UPDATED VERSION ===');
  console.log('API Base URL:', 'http://127.0.0.1:5000');
  console.log('Timestamp:', new Date().toISOString());
  
  const [isLoading, setIsLoading] = useState(false);
  const [forecastResults, setForecastResults] = useState<{
    symbol: string;
    days: number;
    results: any;
  } | null>(null);
  const [activityLog, setActivityLog] = useState<Array<{
    id: string;
    timestamp: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
  }>>([
    {
      id: '1',
      timestamp: new Date().toISOString(),
      message: 'Meta ensemble dashboard initialized successfully',
      type: 'success'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      message: 'ARIMA + LSTM models loaded and ready',
      type: 'info'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      message: 'NSE/BSE market data connection established',
      type: 'info'
    }
  ]);

  const handleForecastSubmit = async (params: ForecastParams) => {
    console.log('=== FORECAST SUBMIT CALLED ===');
    console.log('Params:', params);
    console.log('API Base URL:', 'http://127.0.0.1:5000');
    
    setIsLoading(true);

    // Add to activity log
    const newLogEntry = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      message: `Starting Meta ensemble forecast for ${params.symbol} (${params.days} days)`,
      type: 'info' as const
    };
    setActivityLog(prev => [newLogEntry, ...prev]);

    try {
      let response: ForecastResponse;

      console.log(`=== Calling ${params.useMock ? 'MOCK' : 'REAL'} API endpoint: /api/${params.useMock ? 'forecast-mock' : 'forecast'} ===`);

      if (params.useMock) {
        // Use mock data for testing
        response = await stockForecastAPI.getMockForecast({
          symbol: params.symbol,
          forecast_days: params.days,
          totp: params.totp,
          use_angelone: params.useAngelOne
        });
      } else {
        // Use real API
        response = await stockForecastAPI.getForecast({
          symbol: params.symbol,
          forecast_days: params.days,
          totp: params.totp,
          use_angelone: params.useAngelOne
        });
      }

      console.log(`=== ${params.useMock ? 'Mock' : 'Real'} API Response:`, response);

      if (!response.success) {
        throw new Error('Forecast request failed');
      }

      // Transform API response to match expected format for MetaForecastResults
      const apiResults = {
        meta: {
          predictions: response.results.forecast.meta.map((value, index) => ({
            date: response.results.forecast.dates[index],
            value: value,
            confidence_upper: value * 1.05,
            confidence_lower: value * 0.95
          })),
          historical: [],
          metrics: {
            mse: response.results.metrics.meta.rmse || 0,
            mae: response.results.metrics.meta.mae || 0,
            r2: response.results.metrics.meta.accuracy ? response.results.metrics.meta.accuracy / 100 : 0.85,
            confidence: response.results.metrics.meta.accuracy ? response.results.metrics.meta.accuracy / 100 : 0.85
          }
        },
        arima: {
          predictions: response.results.forecast.arima.map((value, index) => ({
            date: response.results.forecast.dates[index],
            value: value,
            confidence_upper: value * 1.05,
            confidence_lower: value * 0.95
          })),
          historical: [],
          metrics: {
            mse: response.results.metrics.arima.rmse || 0,
            mae: response.results.metrics.arima.mae || 0,
            r2: response.results.metrics.arima.accuracy ? response.results.metrics.arima.accuracy / 100 : 0.80,
            confidence: response.results.metrics.arima.accuracy ? response.results.metrics.arima.accuracy / 100 : 0.80
          }
        },
        lstm: {
          predictions: response.results.forecast.lstm.map((value, index) => ({
            date: response.results.forecast.dates[index],
            value: value,
            confidence_upper: value * 1.05,
            confidence_lower: value * 0.95
          })),
          historical: [],
          metrics: {
            mse: response.results.metrics.lstm.rmse || 0,
            mae: response.results.metrics.lstm.mae || 0,
            r2: response.results.metrics.lstm.accuracy ? response.results.metrics.lstm.accuracy / 100 : 0.82,
            confidence: response.results.metrics.lstm.accuracy ? response.results.metrics.lstm.accuracy / 100 : 0.82
          }
        }
      };

      setForecastResults({
        symbol: params.symbol,
        days: params.days,
        results: apiResults
      });

      // Add success log
      const successLog = {
        id: (Date.now() + 1).toString(),
        timestamp: new Date().toISOString(),
        message: `Meta ensemble forecast completed for ${params.symbol} with ${response.data_source} data source`,
        type: 'success' as const
      };
      setActivityLog(prev => [successLog, ...prev]);

      // Show success toast
      toast.success(`Forecast completed using ${response.data_source} data`);

    } catch (error) {
      console.error('Forecast error:', error);

      // Add error log
      const errorLog = {
        id: (Date.now() + 1).toString(),
        timestamp: new Date().toISOString(),
        message: `Forecast failed for ${params.symbol}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error' as const
      };
      setActivityLog(prev => [errorLog, ...prev]);

      // Show error toast
      toast.error(`Forecast failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen bg-background flex">
      <DashboardSidebar />

      <div className="flex-1 flex flex-col">
        {/* Top Navigation */}
        <div className="glass-panel-subtle border-b border-glass-border px-6 py-4 backdrop-blur-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                <Input
                  placeholder="Search Indian stocks (NSE/BSE)..."
                  className="pl-12 w-80 glass-panel-subtle border-glass-border hover:glass-panel transition-all duration-200"
                />
              </div>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
                <span>Market Open</span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="glass-panel-subtle hover:glass-panel">
                <Bell className="w-4 h-4" />
              </Button>
              <div className="w-9 h-9 glass-panel rounded-xl flex items-center justify-center">
                <span className="text-muted-foreground text-sm font-medium">AI</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden flex">
          {/* Main Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="space-y-6">
              {/* Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="glass-panel rounded-xl p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-accent/20 to-accent/10 rounded-xl flex items-center justify-center">
                      <CheckCircle className="w-5 h-5 text-accent" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Meta Ensemble</h3>
                      <p className="text-xs text-muted-foreground">System Status</p>
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-accent">Operational</div>
                  <p className="text-sm text-muted-foreground">All AI models active</p>
                </div>

                <div className="glass-panel rounded-xl p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl flex items-center justify-center">
                      <Clock className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Last Forecast</h3>
                      <p className="text-xs text-muted-foreground">Recent Activity</p>
                    </div>
                  </div>
                  <div className="text-2xl font-bold">3m ago</div>
                  <p className="text-sm text-muted-foreground">RELIANCE.NS prediction</p>
                </div>

                <div className="glass-panel rounded-xl p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-chart-4/20 to-chart-4/10 rounded-xl flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-chart-4" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Market Coverage</h3>
                      <p className="text-xs text-muted-foreground">Indian Equities</p>
                    </div>
                  </div>
                  <div className="text-2xl font-bold">NSE + BSE</div>
                  <p className="text-sm text-muted-foreground">Live data available</p>
                </div>
              </div>

              {/* Forecast Form */}
              <ForecastForm onSubmit={handleForecastSubmit} isLoading={isLoading} />

              {/* Forecast Results */}
              {forecastResults && (
                <>
                  <MetaForecastResults
                    symbol={forecastResults.symbol}
                    days={forecastResults.days}
                    results={forecastResults.results}
                  />

                  {/* Market Alerts */}
                  <div className="glass-panel rounded-xl p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="w-10 h-10 bg-gradient-to-br from-chart-4/20 to-chart-4/10 rounded-xl flex items-center justify-center">
                        <TrendingUp className="w-5 h-5 text-chart-4" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Market Alerts</h3>
                        <p className="text-sm text-muted-foreground">Real-time forecast notifications</p>
                      </div>
                    </div>
                    <div className="text-center py-8 space-y-3">
                      <div className="w-12 h-12 glass-panel-subtle rounded-xl flex items-center justify-center mx-auto">
                        <Bell className="w-6 h-6 text-muted-foreground" />
                      </div>
                      <h4 className="font-medium">No active alerts</h4>
                      <p className="text-sm text-muted-foreground max-w-md mx-auto">
                        Forecast notifications and market alerts will appear here when available
                      </p>
                    </div>
                  </div>

                  {/* Session Stats */}
                  <div className="glass-panel rounded-xl p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="w-10 h-10 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl flex items-center justify-center">
                        <Activity className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Session Statistics</h3>
                        <p className="text-sm text-muted-foreground">Current session performance metrics</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-primary mb-2">1</div>
                        <p className="text-sm text-muted-foreground">Total Forecasts</p>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-accent mb-2">
                          {(forecastResults.results.meta.metrics.confidence * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-muted-foreground">Average Confidence</p>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-chart-4 mb-2">Meta</div>
                        <p className="text-sm text-muted-foreground">Models Used</p>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Empty State */}
              {!forecastResults && !isLoading && (
                <div className="glass-panel-elevated rounded-2xl p-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="w-16 h-16 glass-panel rounded-2xl flex items-center justify-center mb-6">
                      <Sparkles className="w-8 h-8 text-primary" />
                    </div>
                    <h3 className="text-2xl font-semibold mb-4">Ready for Meta Forecasting</h3>
                    <p className="text-muted-foreground text-lg max-w-2xl leading-relaxed mb-8">
                      Select an Indian stock symbol from NSE or BSE, configure your parameters, and generate
                      ensemble AI predictions combining ARIMA and LSTM models.
                    </p>
                    <div className="flex items-center space-x-6 text-sm text-muted-foreground">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-primary rounded-full"></div>
                        <span>ARIMA Ready</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-accent rounded-full"></div>
                        <span>LSTM Ready</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-chart-4 rounded-full"></div>
                        <span>Meta Ensemble Ready</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Activity Log */}
              <div className="glass-panel rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="w-8 h-8 bg-gradient-to-br from-primary/20 to-primary/10 rounded-lg flex items-center justify-center">
                    <Activity className="w-4 h-4 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Activity Console</h3>
                    <p className="text-sm text-muted-foreground">System events and forecast logs</p>
                  </div>
                </div>
                <ScrollArea className="h-72">
                  <div className="space-y-4">
                    {activityLog.map((log, index) => (
                      <div key={log.id}>
                        <div className="flex items-start space-x-3">
                          <div className="flex-shrink-0 mt-1">
                            {log.type === 'success' && <CheckCircle className="w-4 h-4 text-accent" />}
                            {log.type === 'info' && <Activity className="w-4 h-4 text-primary" />}
                            {log.type === 'warning' && <AlertCircle className="w-4 h-4 text-chart-4" />}
                            {log.type === 'error' && <AlertCircle className="w-4 h-4 text-destructive" />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm leading-relaxed">{log.message}</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              {new Date(log.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        {index < activityLog.length - 1 && (
                          <div className="h-px bg-glass-border my-4"></div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}