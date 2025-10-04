import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import { Download, TrendingUp, TrendingDown, Activity, Sparkles, Brain, BarChart3, Share, Eye } from "lucide-react";

interface MetaForecastResultsProps {
  symbol: string;
  days: number;
  results: ForecastData;
}

interface ForecastData {
  meta: ModelResult;
  arima: ModelResult;
  lstm: ModelResult;
}

interface ModelResult {
  predictions: Array<{
    date: string;
    value: number;
    confidence_upper?: number;
    confidence_lower?: number;
  }>;
  metrics: {
    mse: number;
    mae: number;
    r2: number;
    confidence?: number;
  };
  historical: Array<{
    date: string;
    value: number;
  }>;
}

export function MetaForecastResults({ symbol, days, results }: MetaForecastResultsProps) {
  const [activeTab, setActiveTab] = useState("meta");

  const formatChartData = (historical: ModelResult['historical'], predictions: ModelResult['predictions']) => {
    const historicalData = historical.slice(-30).map(item => ({
      date: item.date,
      historical: item.value,
      forecast: null,
      upper: null,
      lower: null
    }));

    const forecastData = predictions.map(item => ({
      date: item.date,
      historical: null,
      forecast: item.value,
      upper: item.confidence_upper,
      lower: item.confidence_lower
    }));

    return [...historicalData, ...forecastData];
  };

  const metaResult = results.meta;
  const chartData = formatChartData(metaResult.historical, metaResult.predictions);
  const lastPrice = metaResult.historical[metaResult.historical.length - 1]?.value || 0;
  const lastForecast = metaResult.predictions[metaResult.predictions.length - 1]?.value || 0;
  const totalChange = ((lastForecast - lastPrice) / lastPrice) * 100;
  const currentPrice = lastPrice;
  const targetPrice = lastForecast;

  const getSymbolCurrency = (symbol: string) => {
    return "₹"; // All Indian stocks use INR
  };

  const currency = getSymbolCurrency(symbol);

  return (
    <div className="space-y-8">
      {/* Meta Prediction - Prominent Display */}
      <div className="glass-panel-elevated rounded-2xl p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="space-y-2">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-accent/10 rounded-xl flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Meta Ensemble Prediction</h2>
                <p className="text-muted-foreground">
                  {symbol} • {days}-day forecast combining ARIMA + LSTM
                </p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Badge 
              variant={totalChange >= 0 ? "default" : "destructive"} 
              className="glass-panel text-lg px-4 py-2"
            >
              {totalChange >= 0 ? <TrendingUp className="w-4 h-4 mr-2" /> : <TrendingDown className="w-4 h-4 mr-2" />}
              {totalChange.toFixed(2)}%
            </Badge>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" className="glass-panel-subtle hover:glass-panel border-glass-border">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm" className="glass-panel-subtle hover:glass-panel border-glass-border">
                <Share className="w-4 h-4 mr-2" />
                Share
              </Button>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          <div className="glass-panel rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-foreground">{currency}{currentPrice.toFixed(2)}</div>
            <p className="text-sm text-muted-foreground">Current Price</p>
          </div>
          <div className="glass-panel rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-primary">{currency}{targetPrice.toFixed(2)}</div>
            <p className="text-sm text-muted-foreground">Target Price</p>
          </div>
          <div className="glass-panel rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-accent">{(metaResult.metrics.confidence || 0.85 * 100).toFixed(1)}%</div>
            <p className="text-sm text-muted-foreground">Confidence</p>
          </div>
          <div className="glass-panel rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-chart-4">{metaResult.metrics.r2.toFixed(3)}</div>
            <p className="text-sm text-muted-foreground">R-Squared</p>
          </div>
        </div>

        {/* Main Chart */}
        <div className="h-96 glass-panel rounded-xl p-6">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="confidenceBand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
                axisLine={false}
                tickLine={false}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `${currency}${value.toFixed(0)}`}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  color: 'var(--foreground)'
                }}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: any, name: string) => [
                  value ? `${currency}${value.toFixed(2)}` : null,
                  name === 'historical' ? 'Historical' : 
                  name === 'forecast' ? 'Meta Forecast' :
                  name === 'upper' ? 'Upper Bound' : 'Lower Bound'
                ]}
              />
              
              {/* Confidence Band */}
              {metaResult.predictions[0]?.confidence_upper && (
                <Area
                  type="monotone"
                  dataKey="upper"
                  stackId="1"
                  stroke="none"
                  fill="url(#confidenceBand)"
                  connectNulls={false}
                />
              )}
              {metaResult.predictions[0]?.confidence_lower && (
                <Area
                  type="monotone"
                  dataKey="lower"
                  stackId="1"
                  stroke="none"
                  fill="rgba(6, 7, 10, 1)"
                  connectNulls={false}
                />
              )}
              
              {/* Historical Data */}
              <Line
                type="monotone"
                dataKey="historical"
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
              />
              
              {/* Meta Forecast */}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="hsl(var(--primary))"
                strokeWidth={3}
                dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 5 }}
                connectNulls={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Supporting Model Results */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* ARIMA Results */}
        <div className="glass-panel rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-chart-3/20 to-chart-3/10 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-chart-3" />
              </div>
              <div>
                <h3 className="font-semibold">ARIMA Model</h3>
                <p className="text-xs text-muted-foreground">Time Series Analysis</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-xs">
              <Eye className="w-3 h-3 mr-1" />
              Details
            </Button>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-lg font-bold">{results.arima.metrics.mse.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">MSE</p>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{results.arima.metrics.mae.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">MAE</p>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{results.arima.metrics.r2.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">R²</p>
            </div>
          </div>

          {/* Mini Chart */}
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={formatChartData(results.arima.historical.slice(-10), results.arima.predictions)}>
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="hsl(var(--chart-3))"
                  strokeWidth={2}
                  strokeDasharray="3 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* LSTM Results */}
        <div className="glass-panel rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-accent/20 to-accent/10 rounded-lg flex items-center justify-center">
                <Brain className="w-4 h-4 text-accent" />
              </div>
              <div>
                <h3 className="font-semibold">LSTM Model</h3>
                <p className="text-xs text-muted-foreground">Neural Network</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-xs">
              <Eye className="w-3 h-3 mr-1" />
              Details
            </Button>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-lg font-bold">{results.lstm.metrics.mse.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">MSE</p>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{results.lstm.metrics.mae.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">MAE</p>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{results.lstm.metrics.r2.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">R²</p>
            </div>
          </div>

          {/* Mini Chart */}
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={formatChartData(results.lstm.historical.slice(-10), results.lstm.predictions)}>
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="hsl(var(--accent))"
                  strokeWidth={2}
                  strokeDasharray="3 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Forecast Table */}
      <div className="glass-panel rounded-xl p-6">
        <h3 className="font-semibold mb-6">Meta Prediction Timeline</h3>
        <div className="glass-panel-subtle rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-glass-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Date</TableHead>
                <TableHead className="text-muted-foreground">Meta Price</TableHead>
                <TableHead className="text-muted-foreground">Change</TableHead>
                <TableHead className="text-muted-foreground">Lower Bound</TableHead>
                <TableHead className="text-muted-foreground">Upper Bound</TableHead>
                <TableHead className="text-muted-foreground">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {metaResult.predictions.map((prediction, index) => {
                const prevPrice = index === 0 ? lastPrice : metaResult.predictions[index - 1].value;
                const change = ((prediction.value - prevPrice) / prevPrice) * 100;
                
                return (
                  <TableRow key={prediction.date} className="border-glass-border hover:glass-panel-subtle">
                    <TableCell className="font-medium">
                      {new Date(prediction.date).toLocaleDateString()}
                    </TableCell>
                    <TableCell className="font-semibold">
                      {currency}{prediction.value.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={change >= 0 ? "default" : "destructive"} className="text-xs glass-panel border-0">
                        {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {prediction.confidence_lower ? `${currency}${prediction.confidence_lower.toFixed(2)}` : '—'}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {prediction.confidence_upper ? `${currency}${prediction.confidence_upper.toFixed(2)}` : '—'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                            style={{ width: `${(metaResult.metrics.confidence || 0.85) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {((metaResult.metrics.confidence || 0.85) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}