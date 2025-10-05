import { useState } from "react";
import { Header } from "./components/Header";
import { LandingPage } from "./components/LandingPage";
import { Dashboard } from "./components/Dashboard";
import { Toaster } from "./components/ui/sonner";

export default function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'dashboard'>('landing');

  const handleNavigate = (page: 'landing' | 'dashboard') => {
    setCurrentPage(page);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header onNavigate={handleNavigate} currentPage={currentPage} />
      
      <div className="pt-20">
        {currentPage === 'landing' ? (
          <LandingPage onNavigate={handleNavigate} />
        ) : (
          <Dashboard />
        )}
      </div>

      <Toaster 
        toastOptions={{
          style: {
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: 'hsl(var(--foreground))'
          }
        }}
      />
    </div>
  );
}