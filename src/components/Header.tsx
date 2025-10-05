import { Button } from "./ui/button";
import { TrendingUp, LogIn, User } from "lucide-react";

interface HeaderProps {
  onNavigate: (page: 'landing' | 'dashboard') => void;
  currentPage: 'landing' | 'dashboard';
}

export function Header({ onNavigate, currentPage }: HeaderProps) {
  return (
    <header className="fixed top-4 left-4 right-4 glass-panel-elevated rounded-2xl border border-glass-border backdrop-blur-xl z-50 shadow-2xl">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onNavigate('landing')}>
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center shadow-lg">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-foreground">IndiaAI</span>
              <span className="block text-xs text-muted-foreground">Stock Forecasts</span>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center space-x-8">
            <button 
              onClick={() => onNavigate('landing')}
              className={`text-sm transition-all duration-200 ${
                currentPage === 'landing' ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Home
            </button>
            <button 
              onClick={() => onNavigate('dashboard')}
              className={`text-sm transition-all duration-200 ${
                currentPage === 'dashboard' ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Dashboard
            </button>
            <a href="#product" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Product</a>
            <a href="#docs" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Docs</a>
            <a href="#pricing" className="text-muted-foreground hover:text-foreground transition-all duration-200 text-sm">Pricing</a>
          </nav>
        </div>

        <div className="flex items-center space-x-4">
          {currentPage === 'landing' ? (
            <>
              <Button 
                variant="ghost" 
                className="glass-panel-subtle hover:glass-panel transition-all duration-200"
              >
                <LogIn className="w-4 h-4 mr-2" />
                Login
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
              <Button variant="ghost" size="sm" className="glass-panel-subtle hover:glass-panel transition-all duration-200">
                <User className="w-4 h-4 mr-2" />
                Profile
              </Button>
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