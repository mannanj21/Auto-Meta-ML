import { useState } from 'react';

export default function VisualizationDisplay({ visualizationUrl }) {
  const [loading, setLoading] = useState(true);
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
      <h3 className="text-2xl font-bold text-gray-800 mb-6 text-center">
        📊 Recommendation Analysis
      </h3>
      
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 rounded-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-b-4 border-blue-500"></div>
          </div>
        )}
        
        <img
          src={`${API_URL}${visualizationUrl}`}
          alt="Algorithm Recommendations Visualization"
          className="w-full rounded-lg shadow-inner"
          onLoad={() => setLoading(false)}
          onError={() => setLoading(false)}
        />
      </div>

      <p className="text-sm text-gray-500 text-center mt-4">
        Visual representation of algorithm recommendations and confidence levels
      </p>
    </div>
  );
}
