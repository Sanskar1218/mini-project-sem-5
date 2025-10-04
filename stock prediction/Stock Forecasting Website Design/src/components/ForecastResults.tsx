import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import { Download, TrendingUp, TrendingDown, Activity } from "lucide-react";

interface ForecastResultsProps {
  symbol: string;
  days: number;
  results: ForecastData;
}

interface ForecastData {
  arima: ModelResult;
  lstm: ModelResult;
  ensemble: ModelResult;
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
  };
  historical: Array<{
    date: string;
    value: number;
  }>;
}

export function ForecastResults({ symbol, days, results }: ForecastResultsProps) {
  const [activeTab, setActiveTab] = useState("arima");

  const getCurrentResult = (): ModelResult => {
    return results[activeTab as keyof ForecastData];
  };

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

  const currentResult = getCurrentResult();
  const chartData = formatChartData(currentResult.historical, currentResult.predictions);
  const lastPrice = currentResult.historical[currentResult.historical.length - 1]?.value || 0;
  const firstForecast = currentResult.predictions[0]?.value || 0;
  const lastForecast = currentResult.predictions[currentResult.predictions.length - 1]?.value || 0;
  const totalChange = ((lastForecast - lastPrice) / lastPrice) * 100;

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5" />
              <span>Forecast Results - {symbol}</span>
            </CardTitle>
            <CardDescription>
              {days}-day forecast using AI models
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant={totalChange >= 0 ? "default" : "destructive"}>
              {totalChange >= 0 ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
              {totalChange.toFixed(2)}%
            </Badge>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export CSV
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="arima">ARIMA</TabsTrigger>
            <TabsTrigger value="lstm">LSTM</TabsTrigger>
            <TabsTrigger value="ensemble">Ensemble</TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab} className="space-y-6">
            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="text-2xl font-bold">{currentResult.metrics.mse.toFixed(4)}</div>
                  <p className="text-xs text-muted-foreground">Mean Squared Error</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-2xl font-bold">{currentResult.metrics.mae.toFixed(4)}</div>
                  <p className="text-xs text-muted-foreground">Mean Absolute Error</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-2xl font-bold">{currentResult.metrics.r2.toFixed(4)}</div>
                  <p className="text-xs text-muted-foreground">R-Squared</p>
                </CardContent>
              </Card>
            </div>

            {/* Chart */}
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                    formatter={(value: any, name: string) => [
                      value ? `$${value.toFixed(2)}` : null,
                      name === 'historical' ? 'Historical' : 
                      name === 'forecast' ? 'Forecast' :
                      name === 'upper' ? 'Upper Bound' : 'Lower Bound'
                    ]}
                  />
                  
                  {/* Confidence Interval */}
                  {currentResult.predictions[0]?.confidence_upper && (
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stroke="none"
                      fill="hsl(var(--primary))"
                      fillOpacity={0.1}
                      connectNulls={false}
                    />
                  )}
                  {currentResult.predictions[0]?.confidence_lower && (
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stroke="none"
                      fill="hsl(var(--background))"
                      fillOpacity={1}
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
                  
                  {/* Forecast Data */}
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 4 }}
                    connectNulls={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Forecast Table */}
            <div>
              <h4 className="mb-4">Forecasted Values</h4>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Predicted Price</TableHead>
                      <TableHead>Change</TableHead>
                      {currentResult.predictions[0]?.confidence_upper && (
                        <>
                          <TableHead>Lower Bound</TableHead>
                          <TableHead>Upper Bound</TableHead>
                        </>
                      )}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {currentResult.predictions.map((prediction, index) => {
                      const prevPrice = index === 0 ? lastPrice : currentResult.predictions[index - 1].value;
                      const change = ((prediction.value - prevPrice) / prevPrice) * 100;
                      
                      return (
                        <TableRow key={prediction.date}>
                          <TableCell className="font-medium">
                            {new Date(prediction.date).toLocaleDateString()}
                          </TableCell>
                          <TableCell>${prediction.value.toFixed(2)}</TableCell>
                          <TableCell>
                            <Badge variant={change >= 0 ? "default" : "destructive"} className="text-xs">
                              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                            </Badge>
                          </TableCell>
                          {prediction.confidence_lower && (
                            <TableCell>${prediction.confidence_lower.toFixed(2)}</TableCell>
                          )}
                          {prediction.confidence_upper && (
                            <TableCell>${prediction.confidence_upper.toFixed(2)}</TableCell>
                          )}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}