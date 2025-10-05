import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, AreaChart } from "recharts";
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
    rmse: number;
    mae: number;
    r2: number;
    accuracy_pct: number;
  };
  historical: Array<{
    date: string;
    value: number;
  }>;
}

export function MetaForecastResults({ symbol, days, results }: MetaForecastResultsProps) {
  // Guard clause to prevent rendering with incomplete data during loading
  if (!results || !results.meta || !results.arima || !results.lstm) {
    return null;
  }

  const formatChartData = (historicalData: ModelResult['historical'], predictions: ModelResult['predictions']) => {
    if (!historicalData || historicalData.length === 0) return [];

    const historicalPoints: Array<{ date: string; historical: number; forecast: number | null }> = historicalData.slice(-30).map(item => ({
      date: item.date,
      historical: item.value,
      forecast: null,
    }));

    const forecastPoints = predictions.map(item => ({
      date: item.date,
      historical: null,
      forecast: item.value,
    }));

    const lastHistoricalPoint = historicalPoints[historicalPoints.length - 1];
    if (lastHistoricalPoint && predictions.length > 0) {
      // To connect the lines, the last historical point should also have a 'forecast' value equal to its own 'historical' value.
      // This creates the anchor point from which the forecast line will begin.
      lastHistoricalPoint.forecast = lastHistoricalPoint.historical;
    }

    return [...historicalPoints, ...forecastPoints];
  };

  const metaResult = results.meta;
  const chartData = formatChartData(metaResult.historical, metaResult.predictions);
  const lastPrice = metaResult.historical.length > 0 ? metaResult.historical[metaResult.historical.length - 1].value : 0;
  const lastForecast = metaResult.predictions.length > 0 ? metaResult.predictions[metaResult.predictions.length - 1].value : 0;
  const totalChange = lastPrice > 0 ? ((lastForecast - lastPrice) / lastPrice) * 100 : 0;
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
            <div className="text-3xl font-bold text-accent">{metaResult.metrics.accuracy_pct.toFixed(1)}%</div>
            <p className="text-sm text-muted-foreground">Accuracy</p>
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
                <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#FF6B6B" stopOpacity={0.5}/>
                  <stop offset="30%" stopColor="#FF8E53" stopOpacity={0.4}/>
                  <stop offset="70%" stopColor="#FFD93D" stopOpacity={0.3}/>
                  <stop offset="100%" stopColor="#6BCF7F" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4ECDC4" stopOpacity={0.3}/>
                  <stop offset="50%" stopColor="#45B7D1" stopOpacity={0.2}/>
                  <stop offset="100%" stopColor="#96CEB4" stopOpacity={0.05}/>
                </linearGradient>
                <linearGradient id="forecastStroke" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#FF6B6B"/>
                  <stop offset="50%" stopColor="#FF8E53"/>
                  <stop offset="100%" stopColor="#FFD93D"/>
                </linearGradient>
                <linearGradient id="historicalStroke" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#4ECDC4"/>
                  <stop offset="100%" stopColor="#45B7D1"/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.15)" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: '#A6B0BF' }}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
                axisLine={false}
                tickLine={false}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: '#A6B0BF' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `${currency}${value.toFixed(0)}` }
                domain={['dataMin - (dataMax - dataMin) * 0.1', 'dataMax + (dataMax - dataMin) * 0.1']}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  backdropFilter: 'blur(24px)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: '12px',
                  color: '#E6EEF8',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                }}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: any, name: string) => [
                  value ? `${currency}${value.toFixed(2)}`  : null,
                  name === 'historical' ? '📈 Historical Price' : 
                  name === 'forecast' ? '🚀 Meta Forecast' : name
                ]}
              />
              
              {/* Historical Data Area */}
              <Area
                type="monotone"
                dataKey="historical"
                stroke="url(#historicalStroke)"
                strokeWidth={3}
                fill="url(#historicalGradient)"
                fillOpacity={0.5}
                dot={false}
                connectNulls={true}
              />
              
              {/* Meta Forecast Area */}
              <Area
                type="monotone"
                dataKey="forecast"
                stroke="url(#forecastStroke)"
                strokeWidth={5}
                fill="url(#forecastGradient)"
                fillOpacity={0.7}
                dot={{
                  fill: "#FF6B6B", 
                  strokeWidth: 3, 
                  r: 8, 
                  stroke: "#FFD93D",
                }}
                connectNulls={true}
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
              <div className="text-lg font-bold">{results.arima.metrics.rmse.toFixed(2)}</div>
              <p className="text-xs text-muted-foreground">RMSE</p>
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

          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={formatChartData(results.arima.historical.slice(-10), results.arima.predictions)}>
                <defs>
                  <linearGradient id="arimaForecastGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8B5CF6" stopOpacity={0.5}/>
                    <stop offset="50%" stopColor="#A78BFA" stopOpacity={0.3}/>
                    <stop offset="100%" stopColor="#C4B5FD" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="arimaHistoricalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4ECDC4" stopOpacity={0.2}/>
                    <stop offset="100%" stopColor="#45B7D1" stopOpacity={0.05}/>
                  </linearGradient>
                  <linearGradient id="arimaForecastStroke" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#8B5CF6"/>
                    <stop offset="100%" stopColor="#A78BFA"/>
                  </linearGradient>
                  <linearGradient id="arimaHistoricalStroke" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#4ECDC4"/>
                    <stop offset="100%" stopColor="#45B7D1"/>
                  </linearGradient>
                </defs>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(139, 92, 246, 0.2)',
                    borderRadius: '8px',
                    color: '#E6EEF8'
                  }}
                  formatter={(value: any, name: string) => [
                    value ? `${currency}${value.toFixed(2)}`  : null,
                    name === 'historical' ? '📈 Historical' : '🔮 ARIMA'
                  ]}
                  labelFormatter={() => ''}
                />
                <Area
                  type="monotone"
                  dataKey="historical"
                  stroke="url(#arimaHistoricalStroke)"
                  strokeWidth={2}
                  fill="url(#arimaHistoricalGradient)"
                  fillOpacity={0.4}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="forecast"
                  stroke="url(#arimaForecastStroke)"
                  strokeWidth={3}
                  strokeDasharray="4 2"
                  fill="url(#arimaForecastGradient)"
                  fillOpacity={0.6}
                  dot={{ fill: "#8B5CF6", strokeWidth: 2, r: 4 }}
                />
              </AreaChart>
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
              <div className="text-lg font-bold">{results.lstm.metrics.rmse.toFixed(2)}</div>
              <p className="text-xs text-muted-foreground">RMSE</p>
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

          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={formatChartData(results.lstm.historical.slice(-10), results.lstm.predictions)}>
                <defs>
                  <linearGradient id="lstmForecastGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.5}/>
                    <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="lstmHistoricalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4ECDC4" stopOpacity={0.2}/>
                    <stop offset="100%" stopColor="#45B7D1" stopOpacity={0.05}/>
                  </linearGradient>
                  <linearGradient id="lstmForecastStroke" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#3B82F6"/>
                    <stop offset="100%" stopColor="#60A5FA"/>
                  </linearGradient>
                  <linearGradient id="lstmHistoricalStroke" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#4ECDC4"/>
                    <stop offset="100%" stopColor="#45B7D1"/>
                  </linearGradient>
                </defs>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(59, 130, 246, 0.2)',
                    borderRadius: '8px',
                    color: '#E6EEF8'
                  }}
                  formatter={(value: any, name: string) => [
                    value ? `${currency}${value.toFixed(2)}`  : null,
                    name === 'historical' ? '📈 Historical' : '🧠 LSTM'
                  ]}
                  labelFormatter={() => ''}
                />
                <Area
                  type="monotone"
                  dataKey="historical"
                  stroke="url(#lstmHistoricalStroke)"
                  strokeWidth={2}
                  fill="url(#lstmHistoricalGradient)"
                  fillOpacity={0.4}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="forecast"
                  stroke="url(#lstmForecastStroke)"
                  strokeWidth={3}
                  strokeDasharray="4 2"
                  fill="url(#lstmForecastGradient)"
                  fillOpacity={0.6}
                  dot={{ fill: "#3B82F6", strokeWidth: 2, r: 4 }}
                />
              </AreaChart>
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
                <TableHead className="text-muted-foreground">Forecast Price</TableHead>
                <TableHead className="text-muted-foreground">Change</TableHead>
                <TableHead className="text-muted-foreground">Accuracy</TableHead>
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
                    <TableCell className="font-semibold text-primary">
                      {currency}{prediction.value.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={change >= 0 ? "default" : "destructive"} className="text-xs glass-panel border-0">
                        {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                            style={{ width: `${Math.max(0, Math.min(100, (metaResult.metrics.r2 || 0) * 100))}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {Math.max(0, Math.min(100, (metaResult.metrics.r2 || 0) * 100)).toFixed(0)}%
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