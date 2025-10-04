import { Button } from "./ui/button";
import { TrendingUp } from "lucide-react";

interface HeaderProps {
  onNavigate: (page: 'landing' | 'dashboard') => void;
  currentPage: 'landing' | 'dashboard';
}

export function Header({ onNavigate, currentPage }: HeaderProps) {
  return (
    <header className="glass-panel-subtle border-b border-glass-border backdrop-blur-xl sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center shadow-lg">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-foreground">IndiaAI</span>
              <span className="block text-xs text-muted-foreground">Stock Forecasts</span>
            </div>
          </div>
          
          {currentPage === 'landing' && (
            <nav className="hidden md:flex items-center space-x-8">
              <a href="#product" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Product</a>
              <a href="#docs" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Docs</a>
              <a href="#pricing" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Pricing</a>
              <a href="#faq" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">FAQ</a>
            </nav>
          )}
        </div>

        <div className="flex items-center space-x-4">
          {currentPage === 'landing' ? (
            <>
              <Button 
                variant="ghost" 
                onClick={() => onNavigate('dashboard')}
                className="glass-panel-subtle hover:glass-panel transition-all duration-200"
              >
                Connect API
              </Button>
              <Button 
                onClick={() => onNavigate('dashboard')}
                className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 transition-all duration-200 shadow-lg"
              >
                Try Demo
              </Button>
            </>
          ) : (
            <div className="flex items-center space-x-4">
              <div className="w-9 h-9 glass-panel rounded-xl flex items-center justify-center">
                <span className="text-muted-foreground text-sm font-medium">AI</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}