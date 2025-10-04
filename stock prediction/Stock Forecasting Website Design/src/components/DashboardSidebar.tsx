import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { TrendingUp, BookOpen, Settings, Star, Activity, History, Sparkles, Brain, BarChart3, Clock } from "lucide-react";

export function DashboardSidebar() {
  const indianStocks = [
    { symbol: "RELIANCE", name: "Reliance Industries", price: "₹2,487", change: "+3.2%", positive: true },
    { symbol: "TCS", name: "Tata Consultancy", price: "₹4,156", change: "+1.8%", positive: true },
    { symbol: "INFY", name: "Infosys Limited", price: "₹1,832", change: "-0.5%", positive: false },
    { symbol: "HDFC", name: "HDFC Bank", price: "₹1,654", change: "+2.1%", positive: true },
  ];

  const watchlist = [
    { symbol: "ICICI", name: "ICICI Bank", price: "₹1,278", change: "+1.4%", positive: true },
    { symbol: "SBI", name: "State Bank of India", price: "₹825", change: "-1.2%", positive: false },
    { symbol: "LT", name: "Larsen & Toubro", price: "₹3,456", change: "+2.8%", positive: true },
  ];

  return (
    <div className="w-80 glass-panel-subtle border-r border-glass-border flex flex-col backdrop-blur-xl">
      <div className="p-6 border-b border-glass-border">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-foreground">IndiaAI</span>
            <span className="block text-xs text-muted-foreground">Dashboard</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Navigation */}
        <div className="space-y-2">
          <div className="flex items-center space-x-3 px-4 py-3 rounded-xl glass-panel-elevated">
            <Activity className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">Forecasts</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-3 rounded-xl hover:glass-panel transition-all duration-200 cursor-pointer">
            <Star className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Watchlist</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-3 rounded-xl hover:glass-panel transition-all duration-200 cursor-pointer">
            <Brain className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Models</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-3 rounded-xl hover:glass-panel transition-all duration-200 cursor-pointer">
            <History className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">History</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-3 rounded-xl hover:glass-panel transition-all duration-200 cursor-pointer">
            <Settings className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Settings</span>
          </div>
        </div>

        {/* Popular Indian Stocks */}
        <div className="glass-panel rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Popular NSE Stocks</h3>
            <BarChart3 className="w-4 h-4 text-primary" />
          </div>
          <div className="space-y-4">
            {indianStocks.map((stock) => (
              <div key={stock.symbol} className="flex items-center justify-between group cursor-pointer">
                <div className="flex-1">
                  <div className="font-medium text-sm group-hover:text-primary transition-colors">{stock.symbol}</div>
                  <div className="text-xs text-muted-foreground">{stock.name}</div>
                  <div className="text-xs text-muted-foreground">{stock.price}</div>
                </div>
                <Badge 
                  variant={stock.positive ? "default" : "destructive"} 
                  className="text-xs glass-panel border-0"
                >
                  {stock.change}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        {/* Watchlist */}
        <div className="glass-panel rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">My Watchlist</h3>
            <Star className="w-4 h-4 text-accent" />
          </div>
          <div className="space-y-4">
            {watchlist.map((stock) => (
              <div key={stock.symbol} className="flex items-center justify-between group cursor-pointer">
                <div className="flex-1">
                  <div className="font-medium text-sm group-hover:text-primary transition-colors">{stock.symbol}</div>
                  <div className="text-xs text-muted-foreground">{stock.name}</div>
                  <div className="text-xs text-muted-foreground">{stock.price}</div>
                </div>
                <Badge 
                  variant={stock.positive ? "default" : "destructive"} 
                  className="text-xs glass-panel border-0"
                >
                  {stock.change}
                </Badge>
              </div>
            ))}
            <Button 
              variant="ghost" 
              size="sm" 
              className="w-full text-xs glass-panel-subtle hover:glass-panel transition-all duration-200"
            >
              + Add Stock
            </Button>
          </div>
        </div>

        {/* Model Status */}
        <div className="glass-panel rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">AI Models</h3>
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
                <span className="text-xs text-muted-foreground">Meta Ensemble</span>
              </div>
              <span className="text-xs font-medium text-accent">Active</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span className="text-xs text-muted-foreground">ARIMA</span>
              </div>
              <span className="text-xs font-medium">Ready</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span className="text-xs text-muted-foreground">LSTM</span>
              </div>
              <span className="text-xs font-medium">Ready</span>
            </div>
          </div>
        </div>

        {/* Market Status */}
        <div className="glass-panel rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Market Status</h3>
            <Clock className="w-4 h-4 text-chart-4" />
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">NSE</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-accent rounded-full"></div>
                <span className="text-xs font-medium">Open</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">BSE</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-accent rounded-full"></div>
                <span className="text-xs font-medium">Open</span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              Market closes at 3:30 PM IST
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}