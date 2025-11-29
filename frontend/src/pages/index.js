import { useState } from 'react';
import Head from 'next/head';
import FileUpload from '../components/FileUpload';
import RecommendationCard from '../components/RecommendationCard';
import VisualizationDisplay from '../components/VisualizationDisplay';

export default function Home() {
  const [recommendations, setRecommendations] = useState(null);
  const [visualization, setVisualization] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:10000';
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setRecommendations(data.recommendations);
      setVisualization(data.visualization_url);
      setDatasetInfo(data.dataset_info);
    } catch (error) {
      console.error('Upload failed:', error);
      setError('Failed to process your dataset. Please check the file format and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
    setVisualization(null);
    setDatasetInfo(null);
    setError(null);
  };

  return (
    <>
      <Head>
        <title>Meta-Learning Algorithm Recommender</title>
        <meta name="description" content="AI-powered ML algorithm recommendations" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-800">
                  🤖 Meta-Learning Algorithm Recommender
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Powered by AI and Meta-Learning
                </p>
              </div>
              <nav className="hidden md:flex space-x-6">
                <a href="#features" className="text-gray-600 hover:text-blue-600 transition">
                  Features
                </a>
                <a href="#how-it-works" className="text-gray-600 hover:text-blue-600 transition">
                  How It Works
                </a>
                <a href="https://github.com/D13garg/Auto-Meta-ML.git" 
                   className="text-gray-600 hover:text-blue-600 transition"
                   target="_blank" rel="noopener noreferrer">
                  GitHub
                </a>
              </nav>
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 py-12">
          {/* Hero Section */}
          {!recommendations && !loading && (
            <div className="text-center mb-12">
              <h2 className="text-5xl font-bold text-gray-800 mb-4">
                Find the Perfect Algorithm
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
                Upload your dataset and get personalized machine learning algorithm 
                recommendations based on your data characteristics
              </p>
              
              {/* Stats */}
              <div className="grid grid-cols-3 gap-6 max-w-2xl mx-auto mb-12">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-3xl font-bold text-blue-600">51</div>
                  <div className="text-sm text-gray-600 mt-1">Datasets Analyzed</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-3xl font-bold text-green-600">8</div>
                  <div className="text-sm text-gray-600 mt-1">ML Algorithms</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-3xl font-bold text-purple-600">98%</div>
                  <div className="text-sm text-gray-600 mt-1">Accuracy</div>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <svg className="w-6 h-6 text-red-600 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-red-800">{error}</p>
              </div>
            </div>
          )}

          {/* Upload Section */}
          {!recommendations && (
            <div className="max-w-3xl mx-auto">
              <FileUpload onUpload={handleUpload} loading={loading} />
              
              {/* How It Works */}
              <div id="how-it-works" className="mt-16">
                <h3 className="text-2xl font-bold text-gray-800 mb-6 text-center">
                  How It Works
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <span className="text-2xl">📤</span>
                    </div>
                    <h4 className="font-semibold mb-2">1. Upload</h4>
                    <p className="text-sm text-gray-600">Upload your CSV dataset</p>
                  </div>
                  <div className="text-center">
                    <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <span className="text-2xl">🔍</span>
                    </div>
                    <h4 className="font-semibold mb-2">2. Analyze</h4>
                    <p className="text-sm text-gray-600">AI extracts meta-features</p>
                  </div>
                  <div className="text-center">
                    <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <span className="text-2xl">🧠</span>
                    </div>
                    <h4 className="font-semibold mb-2">3. Predict</h4>
                    <p className="text-sm text-gray-600">Meta-learning predicts best algorithms</p>
                  </div>
                  <div className="text-center">
                    <div className="bg-orange-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <span className="text-2xl">✨</span>
                    </div>
                    <h4 className="font-semibold mb-2">4. Results</h4>
                    <p className="text-sm text-gray-600">Get personalized recommendations</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {recommendations && (
            <div className="max-w-6xl mx-auto">
              {/* Dataset Info */}
              {datasetInfo && (
                <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
                  <h3 className="text-xl font-semibold mb-4">Dataset Information</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <span className="text-gray-600 text-sm">Filename</span>
                      <p className="font-semibold">{datasetInfo.filename}</p>
                    </div>
                    <div>
                      <span className="text-gray-600 text-sm">Rows</span>
                      <p className="font-semibold">{datasetInfo.rows.toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="text-gray-600 text-sm">Columns</span>
                      <p className="font-semibold">{datasetInfo.columns}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations Header */}
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-2">
                  Your Algorithm Recommendations
                </h2>
                <p className="text-gray-600">
                  Based on your dataset characteristics, here are the top algorithms
                </p>
              </div>

              {/* Recommendation Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {recommendations.map((rec, idx) => (
                  <RecommendationCard key={idx} recommendation={rec} />
                ))}
              </div>

              {/* Visualization */}
              {visualization && (
                <VisualizationDisplay visualizationUrl={visualization} />
              )}

              {/* Action Buttons */}
              <div className="flex justify-center space-x-4 mt-8">
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-semibold shadow-lg"
                >
                  Try Another Dataset
                </button>
                <button
                  onClick={() => window.print()}
                  className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition font-semibold shadow-lg"
                >
                  Print Results
                </button>
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="bg-gray-800 text-white py-8 mt-16">
          <div className="container mx-auto px-4 text-center">
            <p className="text-sm">
              © 2024 Meta-Learning Algorithm Recommender. Built with ❤️ using Next.js and FastAPI.
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Powered by MLkNN on OpenML-CC18 benchmark
            </p>
          </div>
        </footer>
      </div>
    </>
  );
}
