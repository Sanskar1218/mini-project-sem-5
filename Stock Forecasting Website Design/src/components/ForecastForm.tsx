import { useState } from "react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Switch } from "./ui/switch";
import { Sparkles, Loader2, Brain, Zap, Shield } from "lucide-react";
import { useApiService } from "../services/api";

interface ForecastFormProps {
  onSubmit: (params: ForecastParams) => void;
  isLoading: boolean;
}

export interface ForecastParams {
  symbol: string;
  days: number;
  useAngelOne: boolean;
  totp?: string;
  useMock?: boolean;
}

export function ForecastForm({ onSubmit, isLoading }: ForecastFormProps) {
  const [symbol, setSymbol] = useState("RELIANCE-EQ");
  const [days, setDays] = useState(7);
  const [useAngelOne, setUseAngelOne] = useState(true);
  const [totp, setTotp] = useState("");
  const [useMock, setUseMock] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const apiService = useApiService();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsAnimating(true);

    // Reset animation after a delay
    setTimeout(() => setIsAnimating(false), 2000);

    onSubmit({
      symbol,
      days,
      useAngelOne,
      totp: totp || undefined,
      useMock
    });
  };

  const stockOptions = apiService.getStockOptions();

  return (
    <div className="glass-panel-elevated rounded-2xl p-8">
      <div className="flex items-center justify-between mb-8">
        <div className="space-y-2">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Meta Forecast Request</h2>
              <p className="text-sm text-muted-foreground">
                Configure parameters for ensemble AI prediction
              </p>
            </div>
          </div>
        </div>
        <Badge variant="secondary" className="glass-panel border-primary/20 text-primary">
          <Brain className="w-3 h-3 mr-1" />
          AI Ensemble
        </Badge>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <Label htmlFor="symbol" className="text-sm font-medium">Indian Stock Symbol</Label>
            <Select value={symbol} onValueChange={setSymbol}>
              <SelectTrigger className="glass-panel-subtle border-glass-border hover:glass-panel transition-all duration-200">
                <SelectValue placeholder="Select NSE/BSE symbol" />
              </SelectTrigger>
              <SelectContent className="glass-panel-elevated border-glass-border backdrop-blur-xl">
                {stockOptions.map((stock) => (
                  <SelectItem key={stock.value} value={stock.value}>
                    <div className="flex justify-between items-center w-full">
                      <span>{stock.label}</span>
                      <span className="text-xs text-muted-foreground ml-2">{stock.price}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label htmlFor="days" className="text-sm font-medium">Forecast Horizon</Label>
            <Select value={days.toString()} onValueChange={(value) => setDays(parseInt(value))}>
              <SelectTrigger className="glass-panel-subtle border-glass-border hover:glass-panel transition-all duration-200">
                <SelectValue placeholder="Select forecast days" />
              </SelectTrigger>
              <SelectContent className="glass-panel-elevated border-glass-border backdrop-blur-xl">
                <SelectItem value="1">1 Day</SelectItem>
                <SelectItem value="3">3 Days</SelectItem>
                <SelectItem value="7">7 Days</SelectItem>
                <SelectItem value="14">14 Days</SelectItem>
                <SelectItem value="30">30 Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-6">
          <div className="flex items-center justify-between p-4 glass-panel-subtle rounded-lg">
            <div className="flex items-center space-x-3">
              <Shield className="w-5 h-5 text-primary" />
              <div>
                <Label htmlFor="use-angel" className="text-sm font-medium">Use Angel One Live Data</Label>
                <p className="text-xs text-muted-foreground">Get real-time market data from Angel One broker</p>
              </div>
            </div>
            <Switch
              id="use-angel"
              checked={useAngelOne}
              onCheckedChange={setUseAngelOne}
            />
          </div>

          {useAngelOne && (
            <div className="space-y-3 p-4 glass-panel-subtle rounded-lg border border-primary/20">
              <Label htmlFor="totp" className="text-sm font-medium flex items-center">
                <Shield className="w-4 h-4 mr-2 text-primary" />
                TOTP Code (6-digit)
              </Label>
              <Input
                id="totp"
                type="text"
                placeholder="Enter 6-digit TOTP from authenticator app"
                value={totp}
                onChange={(e) => setTotp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                maxLength={6}
                className="glass-panel-subtle border-glass-border"
              />
              <p className="text-xs text-muted-foreground">
                Required for Angel One authentication. Get this from your authenticator app.
              </p>
            </div>
          )}

          <div className="flex items-center justify-between p-4 glass-panel-subtle rounded-lg">
            <div className="flex items-center space-x-3">
              <Zap className="w-5 h-5 text-accent" />
              <div>
                <Label htmlFor="use-mock" className="text-sm font-medium">Use Mock Data (Fast)</Label>
                <p className="text-xs text-muted-foreground">Use simulated data for quick testing</p>
              </div>
            </div>
            <Switch
              id="use-mock"
              checked={useMock}
              onCheckedChange={setUseMock}
            />
          </div>
        </div>

        <div className="relative overflow-hidden">
          {isAnimating && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/30 to-accent/30 rounded-xl animate-pulse"></div>
              <div className="absolute top-0 -left-full w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
            </div>
          )}
          <Button
            type="submit"
            className={`w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 transition-all duration-300 shadow-xl text-lg py-6 relative ${
              isAnimating ? 'scale-105 shadow-2xl' : ''
            }`}
            size="lg"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                <span className="animate-pulse">Generating Meta Forecast...</span>
              </>
            ) : (
              <>
                <Zap className={`w-5 h-5 mr-2 ${isAnimating ? 'animate-bounce' : ''}`} />
                Generate Meta Forecast
              </>
            )}
          </Button>
        </div>
      </form>
    </div>
  );
}