import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { TrendingUp, Brain, Database, Shield, Download, BarChart3, Clock, Zap, Activity, Sparkles } from "lucide-react";

interface LandingPageProps {
  onNavigate: (page: 'dashboard') => void;
}

export function LandingPage({ onNavigate }: LandingPageProps) {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 lg:py-32">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-10">
            <div className="space-y-6">
              <Badge variant="secondary" className="glass-panel px-4 py-2 text-primary border-primary/20">
                <Sparkles className="w-4 h-4 mr-2" />
                Ensemble AI Forecasting
              </Badge>
              <h1 className="text-5xl lg:text-7xl tracking-tight leading-tight">
                Glass‑feel AI forecasts for{" "}
                <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Indian stocks
                </span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-lg leading-relaxed">
                Ensemble Meta predictions powered by ARIMA + LSTM and live market data.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                size="lg" 
                onClick={() => onNavigate('dashboard')}
                className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 transition-all duration-300 shadow-2xl text-lg px-8 py-6"
              >
                <TrendingUp className="w-5 h-5 mr-2" />
                Try Demo
              </Button>
              <Button 
                variant="outline" 
                size="lg"
                className="glass-panel-subtle hover:glass-panel transition-all duration-300 border-glass-border text-lg px-8 py-6"
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                Connect API
              </Button>
            </div>
          </div>
          
          <div className="relative">
            <div className="glass-panel-elevated rounded-2xl p-8 shadow-2xl">
              <ImageWithFallback 
                src="https://images.unsplash.com/photo-1743712578499-7266d3d89dc3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmRpYW4lMjBzdG9jayUyMG1hcmtldCUyMG5zZSUyMGJvbWJheSUyMHRyYWRpbmd8ZW58MXx8fHwxNzU5NTgzODcyfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="Indian stock market trading"
                className="w-full h-72 object-cover rounded-xl opacity-80"
              />
              <div className="mt-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm text-muted-foreground">RELIANCE.NS</div>
                    <div className="font-semibold text-foreground">₹2,487.50</div>
                  </div>
                  <div className="flex items-center text-accent">
                    <TrendingUp className="w-4 h-4 mr-1" />
                    <span className="text-sm font-medium">+3.2%</span>
                  </div>
                </div>
                <div className="h-1 bg-muted rounded-full overflow-hidden">
                  <div className="h-full w-3/4 bg-gradient-to-r from-primary to-accent rounded-full"></div>
                </div>
                <div className="text-xs text-muted-foreground">7-day Meta prediction: ₹2,650 (+6.5%)</div>
              </div>
            </div>
            
            {/* Floating elements */}
            <div className="absolute -top-4 -right-4 glass-panel rounded-xl p-4 animate-pulse">
              <Activity className="w-6 h-6 text-primary" />
            </div>
            <div className="absolute -bottom-4 -left-4 glass-panel rounded-xl p-4 animate-pulse delay-1000">
              <Brain className="w-6 h-6 text-accent" />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-6 py-20">
        <div className="text-center space-y-6 mb-20">
          <h2 className="text-4xl lg:text-5xl tracking-tight">Ensemble AI features</h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Advanced Meta predictions combining multiple AI models with real-time Indian market data
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="glass-panel-elevated rounded-2xl p-8 text-center group hover:glass-panel transition-all duration-300">
            <div className="w-16 h-16 bg-gradient-to-br from-primary/20 to-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-4">Meta Ensemble</h3>
            <p className="text-muted-foreground leading-relaxed">
              Combines ARIMA + LSTM predictions into a single powerful Meta forecast with confidence bands
            </p>
          </div>
          
          <div className="glass-panel-elevated rounded-2xl p-8 text-center group hover:glass-panel transition-all duration-300">
            <div className="w-16 h-16 bg-gradient-to-br from-accent/20 to-accent/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
              <Shield className="w-8 h-8 text-accent" />
            </div>
            <h3 className="text-xl font-semibold mb-4">Auto-TOTP</h3>
            <p className="text-muted-foreground leading-relaxed">
              Secure automated time-based one-time password generation for Angel One API
            </p>
          </div>
          
          <div className="glass-panel-elevated rounded-2xl p-8 text-center group hover:glass-panel transition-all duration-300">
            <div className="w-16 h-16 bg-gradient-to-br from-chart-4/20 to-chart-4/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
              <Database className="w-8 h-8 text-chart-4" />
            </div>
            <h3 className="text-xl font-semibold mb-4">Live Trading</h3>
            <p className="text-muted-foreground leading-relaxed">
              Real-time NSE/BSE data integration with Indian stock symbols and market hours
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative py-20">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent"></div>
        <div className="container mx-auto px-6 relative">
          <div className="text-center space-y-6 mb-20">
            <h2 className="text-4xl lg:text-5xl tracking-tight">How it works</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Three seamless steps to get accurate Indian stock forecasts
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-12">
            <div className="text-center space-y-6 group">
              <div className="glass-panel-elevated w-20 h-20 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-all duration-300">
                <Database className="w-10 h-10 text-primary" />
              </div>
              <div className="space-y-3">
                <div className="inline-block px-3 py-1 glass-panel rounded-full text-sm text-primary border border-primary/20">
                  Step 1
                </div>
                <h3 className="text-2xl font-semibold">Fetch</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Connect to NSE/BSE live data and historical Indian stock information
                </p>
              </div>
            </div>
            
            <div className="text-center space-y-6 group">
              <div className="glass-panel-elevated w-20 h-20 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-all duration-300">
                <Brain className="w-10 h-10 text-accent" />
              </div>
              <div className="space-y-3">
                <div className="inline-block px-3 py-1 glass-panel rounded-full text-sm text-accent border border-accent/20">
                  Step 2
                </div>
                <h3 className="text-2xl font-semibold">Train</h3>
                <p className="text-muted-foreground leading-relaxed">
                  ARIMA and LSTM models analyze patterns and market behavior
                </p>
              </div>
            </div>
            
            <div className="text-center space-y-6 group">
              <div className="glass-panel-elevated w-20 h-20 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-all duration-300">
                <TrendingUp className="w-10 h-10 text-chart-4" />
              </div>
              <div className="space-y-3">
                <div className="inline-block px-3 py-1 glass-panel rounded-full text-sm text-chart-4 border border-chart-4/20">
                  Step 3
                </div>
                <h3 className="text-2xl font-semibold">Predict</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Generate Meta ensemble forecasts with confidence intervals
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-6 py-20">
        <div className="glass-panel-elevated rounded-3xl p-12 text-center">
          <div className="space-y-8">
            <div className="space-y-6">
              <h2 className="text-4xl lg:text-5xl tracking-tight">
                Ready to forecast{" "}
                <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Indian markets?
                </span>
              </h2>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                Get started with our demo using RELIANCE, TCS, INFY and other top Indian stocks
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <Button 
                size="lg" 
                onClick={() => onNavigate('dashboard')}
                className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 transition-all duration-300 shadow-2xl text-lg px-10 py-6"
              >
                <Zap className="w-5 h-5 mr-2" />
                Try Demo Now
              </Button>
              <Button 
                variant="outline" 
                size="lg"
                className="glass-panel-subtle hover:glass-panel transition-all duration-300 border-glass-border text-lg px-10 py-6"
              >
                <Download className="w-5 h-5 mr-2" />
                View Documentation
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="glass-panel-subtle border-t border-glass-border">
        <div className="container mx-auto px-6 py-16">
          <div className="grid md:grid-cols-4 gap-12">
            <div className="space-y-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <span className="font-bold text-foreground">IndiaAI</span>
                  <span className="block text-xs text-muted-foreground">Stock Forecasts</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Glass-feel AI forecasting for Indian equities powered by ensemble Meta predictions
              </p>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold">Product</h4>
              <div className="space-y-3 text-sm text-muted-foreground">
                <div className="hover:text-foreground transition-colors cursor-pointer">Meta Ensemble</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Angel One API</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Auto-TOTP</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Demo</div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold">Markets</h4>
              <div className="space-y-3 text-sm text-muted-foreground">
                <div className="hover:text-foreground transition-colors cursor-pointer">NSE Stocks</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">BSE Stocks</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Indian Indices</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Market Hours</div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold">Legal</h4>
              <div className="space-y-3 text-sm text-muted-foreground">
                <div className="hover:text-foreground transition-colors cursor-pointer">Privacy Policy</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Terms of Service</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">SEBI Compliance</div>
                <div className="hover:text-foreground transition-colors cursor-pointer">Security</div>
              </div>
            </div>
          </div>
          
          <div className="border-t border-glass-border mt-12 pt-8 text-center text-sm text-muted-foreground">
            © 2024 IndiaAI Stock Forecasts. All rights reserved. Not investment advice.
          </div>
        </div>
      </footer>
    </div>
  );
}