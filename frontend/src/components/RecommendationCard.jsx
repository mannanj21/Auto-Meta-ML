import { motion } from 'framer-motion';

export default function RecommendationCard({ recommendation }) {
  const { algorithm, confidence, rank, explanation, characteristics } = recommendation;

  const getBadgeColor = (conf) => {
    if (conf > 0.7) return 'bg-green-100 text-green-800 border-green-300';
    if (conf > 0.5) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-orange-100 text-orange-800 border-orange-300';
  };

  const getRankBadgeColor = (r) => {
    if (r === 1) return 'bg-yellow-400 text-yellow-900';
    if (r === 2) return 'bg-gray-300 text-gray-800';
    return 'bg-orange-300 text-orange-900';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: rank * 0.15, duration: 0.5 }}
      whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.15)" }}
      className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all border-2 border-gray-100"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`${getRankBadgeColor(rank)} rounded-full w-10 h-10 flex items-center justify-center font-bold text-lg shadow-sm`}>
            {rank}
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-800">{algorithm}</h3>
            <p className="text-xs text-gray-500">Rank #{rank} Recommendation</p>
          </div>
        </div>
        
        <span className={`px-3 py-1.5 rounded-full text-sm font-semibold border-2 ${getBadgeColor(confidence)}`}>
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>

      {/* Explanation */}
      <p className="text-gray-600 text-sm leading-relaxed mb-4 min-h-[60px]">
        {explanation}
      </p>

      {/* Characteristics */}
      {characteristics && (
        <div className="pt-4 border-t border-gray-200">
          <p className="text-xs font-semibold text-gray-500 mb-2">CHARACTERISTICS</p>
          <div className="flex flex-wrap gap-2">
            {characteristics.speed && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium">
                ⚡ {characteristics.speed}
              </span>
            )}
            {characteristics.interpretability && (
              <span className="px-2 py-1 bg-purple-50 text-purple-700 rounded text-xs font-medium">
                📊 {characteristics.interpretability}
              </span>
            )}
            {characteristics.handles_non_linear && (
              <span className="px-2 py-1 bg-green-50 text-green-700 rounded text-xs font-medium">
                🔄 Non-linear
              </span>
            )}
            {characteristics.ensemble && (
              <span className="px-2 py-1 bg-orange-50 text-orange-700 rounded text-xs font-medium">
                🎯 Ensemble
              </span>
            )}
          </div>
        </div>
      )}

      {/* Confidence Indicator */}
      <div className="mt-4">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>Confidence Level</span>
          <span className="font-semibold">{confidence > 0.7 ? 'High' : confidence > 0.5 ? 'Medium' : 'Low'}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-1000 ${
              confidence > 0.7 ? 'bg-green-500' : confidence > 0.5 ? 'bg-yellow-500' : 'bg-orange-500'
            }`}
            style={{ width: `${confidence * 100}%` }}
          ></div>
        </div>
      </div>
    </motion.div>
  );
}
