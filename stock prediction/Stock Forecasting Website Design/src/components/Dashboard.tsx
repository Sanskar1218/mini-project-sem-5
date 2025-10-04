import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";
import { DashboardSidebar } from "./DashboardSidebar";
import { ForecastForm, ForecastParams } from "./ForecastForm";
import { MetaForecastResults } from "./MetaForecastResults";
import { Search, Bell, QrCode, Clock, Shield, CheckCircle, AlertCircle, Activity, Sparkles, TrendingUp, Brain } from "lucide-react";

export function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [forecastResults, setForecastResults] = useState<any>(null);
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
    setIsLoading(true);
    
    // Add to activity log
    const newLogEntry = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      message: `Starting Meta ensemble forecast for ${params.symbol} (${params.days} days)`,
      type: 'info' as const
    };
    setActivityLog(prev => [newLogEntry, ...prev]);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Generate mock forecast data for Indian stocks
    const generateMockData = (basePrice: number, days: number, volatility: number = 0.05) => {
      const historical = Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (30 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        value: basePrice + (Math.random() - 0.5) * basePrice * volatility
      }));

      const predictions = Array.from({ length: days }, (_, i) => {
        const trend = 1 + (Math.random() - 0.3) * 0.015; // Indian market tendency
        const price = basePrice * Math.pow(trend, i + 1) + (Math.random() - 0.5) * basePrice * volatility;
        return {
          date: new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          value: price,
          confidence_upper: price * 1.08,
          confidence_lower: price * 0.92
        };
      });

      return {
        predictions,
        historical,
        metrics: {
          mse: Math.random() * 0.005 + 0.002,
          mae: Math.random() * 0.01 + 0.005,
          r2: 0.82 + Math.random() * 0.15,
          confidence: 0.75 + Math.random() * 0.2
        }
      };
    };

    // Indian stock base prices
    const getIndianStockPrice = (symbol: string) => {
      const prices: { [key: string]: number } = {
        'RELIANCE': 2487,
        'TCS': 4156,
        'INFY': 1832,
        'HDFC': 1654,
        'ICICI': 1278,
        'SBI': 825,
        'LT': 3456,
        'WIPRO': 567,
        'MARUTI': 12450,
        'HCLTECH': 1789
      };
      return prices[symbol] || 2000;
    };

    const basePrice = getIndianStockPrice(params.symbol);

    const mockResults = {
      meta: generateMockData(basePrice, params.days, 0.04), // Meta has lower volatility
      arima: generateMockData(basePrice, params.days, 0.06),
      lstm: generateMockData(basePrice * 1.015, params.days, 0.05)
    };

    setForecastResults({
      symbol: params.symbol,
      days: params.days,
      results: mockResults
    });

    // Add success log
    const successLog = {
      id: (Date.now() + 1).toString(),
      timestamp: new Date().toISOString(),
      message: `Meta ensemble forecast completed for ${params.symbol} with ${(mockResults.meta.metrics.confidence! * 100).toFixed(1)}% confidence`,
      type: 'success' as const
    };
    setActivityLog(prev => [successLog, ...prev]);

    if (params.useAngelOne && !params.totp) {
      const totpLog = {
        id: (Date.now() + 2).toString(),
        timestamp: new Date().toISOString(),
        message: 'Angel One TOTP generated automatically (value redacted for security)',
        type: 'info' as const
      };
      setActivityLog(prev => [totpLog, ...prev]);
    }

    setIsLoading(false);
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
                <MetaForecastResults
                  symbol={forecastResults.symbol}
                  days={forecastResults.days}
                  results={forecastResults.results}
                />
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

          {/* Right Sidebar */}
          <div className="w-80 border-l border-glass-border glass-panel-subtle p-6 space-y-6 backdrop-blur-xl">
            {/* Meta Ensemble Tips */}
            <div className="glass-panel rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Brain className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-semibold">Meta Ensemble Tips</h3>
              </div>
              <div className="space-y-4 text-sm">
                <div className="flex items-start space-x-3">
                  <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                  <span className="leading-relaxed">Meta combines ARIMA stability with LSTM pattern recognition</span>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-1.5 h-1.5 bg-accent rounded-full mt-2 flex-shrink-0"></div>
                  <span className="leading-relaxed">Higher confidence scores indicate stronger prediction consensus</span>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-1.5 h-1.5 bg-chart-4 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="leading-relaxed">Mock mode uses realistic Indian market volatility patterns</span>
                </div>
              </div>
            </div>

            {/* Angel One TOTP Setup */}
            <div className="glass-panel rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Shield className="w-4 h-4 text-accent" />
                <h3 className="text-sm font-semibold">Angel One TOTP</h3>
              </div>
              <div className="space-y-4">
                <div className="glass-panel-subtle rounded-xl p-6 flex items-center justify-center">
                  <QrCode className="w-20 h-20 text-muted-foreground" />
                </div>
                <div className="text-center space-y-3">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Scan QR code with Google Authenticator for live NSE/BSE data access
                  </p>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full text-xs glass-panel-subtle hover:glass-panel border-glass-border"
                  >
                    Show Secret Key
                  </Button>
                </div>
              </div>
            </div>

            {/* Market Alerts */}
            <div className="glass-panel rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <TrendingUp className="w-4 h-4 text-chart-4" />
                <h3 className="text-sm font-semibold">Market Alerts</h3>
              </div>
              <div className="text-center py-6 space-y-2">
                <div className="w-8 h-8 glass-panel-subtle rounded-lg flex items-center justify-center mx-auto">
                  <Bell className="w-4 h-4 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">No active alerts</p>
                <p className="text-xs text-muted-foreground">
                  Forecast notifications will appear here
                </p>
              </div>
            </div>

            {/* Performance Stats */}
            <div className="glass-panel rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Activity className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-semibold">Session Stats</h3>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-muted-foreground">Forecasts</span>
                  <span className="text-sm font-medium">0</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-muted-foreground">Avg Confidence</span>
                  <span className="text-sm font-medium">—</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-muted-foreground">Models Used</span>
                  <span className="text-sm font-medium">Meta</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}