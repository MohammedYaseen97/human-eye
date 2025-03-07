import React from 'react';

interface ComponentScore {
  position: number;
  task: number;
  visual: number;
}

export interface AttentionPoint {
  element_id: string;
  position: [number, number];
  candidate_type: 'ui_element' | 'platform_hotspot';
  score: number;
  component_scores: ComponentScore;
  reasoning: string;
  color?: string;
  scan_pattern?: string;
}

interface AttentionPointsProps {
  points: AttentionPoint[];
  onPointHover?: (point: AttentionPoint | null) => void;
  onPointClick?: (point: AttentionPoint) => void;
  selectedPoint?: AttentionPoint | null;
}

const AttentionPoints: React.FC<AttentionPointsProps> = ({
  points,
  onPointHover,
  onPointClick,
  selectedPoint
}) => {
  // Helper function to format score as percentage
  const formatScore = (score: number) => `${(score * 100).toFixed(1)}%`;

  // Helper function to get score color based on value
  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-400';
    if (score >= 0.4) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className="h-[calc(100vh-8rem)] p-4 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xl font-semibold text-white">Attention Analysis</h2>
        <div className="text-sm text-gray-400">
          {points.length} points detected
        </div>
      </div>

      {/* Points List */}
      <div className="overflow-y-auto flex-grow scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800 pr-2">
        {points.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400 text-center p-4">
            <div>
              <svg className="w-12 h-12 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <p className="text-lg font-medium">No attention points yet</p>
              <p className="text-sm mt-2">Upload an image and start prediction to see attention analysis</p>
            </div>
          </div>
        ) : (
          points.map((point, index) => (
            <div
              key={point.element_id || index}
              className={`mb-2 p-3 rounded-lg transition-all cursor-pointer ${
                selectedPoint?.element_id === point.element_id
                  ? 'bg-blue-500/20 border border-blue-400'
                  : 'bg-gray-700/50 hover:bg-gray-700 border border-gray-600'
              }`}
              onMouseEnter={() => onPointHover?.(point)}
              onMouseLeave={() => onPointHover?.(null)}
              onClick={() => onPointClick?.(point)}
            >
              {/* Point Header */}
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center">
                  {/* Color indicator matching the visualization */}
                  <div 
                    className="w-4 h-4 rounded mr-2 flex-shrink-0"
                    style={{ 
                      backgroundColor: point.color ? `rgba(${point.color})` : 'transparent',
                      border: '1px solid rgba(255,255,255,0.2)'
                    }}
                  />
                  <span className="font-medium text-white text-sm">
                    {point.candidate_type === 'ui_element' ? 'UI Element' : 'Platform Hotspot'}
                  </span>
                </div>
                <span className={`font-semibold text-sm ${getScoreColor(point.score)}`}>
                  {formatScore(point.score)}
                </span>
              </div>

              {point.scan_pattern && (
                <div className="text-xs text-gray-300 mb-1.5 italic">
                  Scan Pattern: {point.scan_pattern}
                </div>
              )}

              {/* Position and Scores in one row */}
              <div className="grid grid-cols-4 gap-2 mb-1.5 text-xs">
                <div className="text-gray-300">
                  ({point.position[0].toFixed(2)}, {point.position[1].toFixed(2)})
                </div>
                <div className="text-center">
                  <span className="text-gray-400 mr-1">P:</span>
                  <span className={getScoreColor(point.component_scores.position)}>
                    {formatScore(point.component_scores.position)}
                  </span>
                </div>
                <div className="text-center">
                  <span className="text-gray-400 mr-1">T:</span>
                  <span className={getScoreColor(point.component_scores.task)}>
                    {formatScore(point.component_scores.task)}
                  </span>
                </div>
                <div className="text-center">
                  <span className="text-gray-400 mr-1">V:</span>
                  <span className={getScoreColor(point.component_scores.visual)}>
                    {formatScore(point.component_scores.visual)}
                  </span>
                </div>
              </div>

              {/* Reasoning */}
              <div className="text-xs text-gray-300 bg-gray-800/50 p-2 rounded border border-gray-600">
                {point.reasoning}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AttentionPoints; 