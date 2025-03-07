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
    <div className="bg-gray-800 rounded-xl shadow-xl p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white">Attention Analysis</h2>
        <div className="text-sm text-gray-400">
          {points.length} points detected
        </div>
      </div>

      {/* Points List */}
      <div className="overflow-y-auto flex-grow">
        {points.map((point, index) => (
          <div
            key={point.element_id || index}
            className={`mb-3 p-4 rounded-lg transition-all cursor-pointer ${
              selectedPoint?.element_id === point.element_id
                ? 'bg-blue-500/20 border border-blue-400'
                : 'bg-gray-700/50 hover:bg-gray-700 border border-gray-600'
            }`}
            onMouseEnter={() => onPointHover?.(point)}
            onMouseLeave={() => onPointHover?.(null)}
            onClick={() => onPointClick?.(point)}
          >
            {/* Point Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className={`w-2 h-2 rounded-full mr-2 ${
                  point.candidate_type === 'ui_element' 
                    ? 'bg-purple-400' 
                    : 'bg-emerald-400'
                }`} />
                <span className="font-medium text-white">
                  {point.candidate_type === 'ui_element' ? 'UI Element' : 'Platform Hotspot'}
                </span>
              </div>
              <span className={`font-semibold ${getScoreColor(point.score)}`}>
                {formatScore(point.score)}
              </span>
            </div>

            {/* Position */}
            <div className="text-sm text-gray-300 mb-2">
              Position: ({point.position[0].toFixed(2)}, {point.position[1].toFixed(2)})
            </div>

            {/* Component Scores */}
            <div className="grid grid-cols-3 gap-2 mb-2">
              <div className="text-center">
                <div className="text-xs text-gray-400">Position</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.position)}`}>
                  {formatScore(point.component_scores.position)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">Task</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.task)}`}>
                  {formatScore(point.component_scores.task)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">Visual</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.visual)}`}>
                  {formatScore(point.component_scores.visual)}
                </div>
              </div>
            </div>

            {/* Reasoning */}
            <div className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded border border-gray-600">
              {point.reasoning}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttentionPoints; 