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
    if (score >= 0.7) return 'text-green-500';
    if (score >= 0.4) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 max-h-[600px] overflow-hidden flex flex-col">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Attention Points Analysis</h2>
      
      {/* Stats Summary */}
      <div className="mb-4 grid grid-cols-2 gap-4">
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-sm text-gray-600">UI Elements</p>
          <p className="text-lg font-semibold">
            {points.filter(p => p.candidate_type === 'ui_element').length}
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-sm text-gray-600">Platform Hotspots</p>
          <p className="text-lg font-semibold">
            {points.filter(p => p.candidate_type === 'platform_hotspot').length}
          </p>
        </div>
      </div>

      {/* Points List */}
      <div className="overflow-y-auto flex-grow">
        {points.map((point, index) => (
          <div
            key={point.element_id || index}
            className={`mb-3 p-4 rounded-lg transition-all cursor-pointer ${
              selectedPoint?.element_id === point.element_id
                ? 'bg-blue-50 border-2 border-blue-200'
                : 'bg-gray-50 hover:bg-gray-100'
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
                    ? 'bg-purple-500' 
                    : 'bg-green-500'
                }`} />
                <span className="font-medium text-gray-800">
                  {point.candidate_type === 'ui_element' ? 'UI Element' : 'Platform Hotspot'}
                </span>
              </div>
              <span className={`font-semibold ${getScoreColor(point.score)}`}>
                {formatScore(point.score)}
              </span>
            </div>

            {/* Position */}
            <div className="text-sm text-gray-600 mb-2">
              Position: ({point.position[0].toFixed(2)}, {point.position[1].toFixed(2)})
            </div>

            {/* Component Scores */}
            <div className="grid grid-cols-3 gap-2 mb-2">
              <div className="text-center">
                <div className="text-xs text-gray-500">Position</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.position)}`}>
                  {formatScore(point.component_scores.position)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">Task</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.task)}`}>
                  {formatScore(point.component_scores.task)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">Visual</div>
                <div className={`text-sm font-medium ${getScoreColor(point.component_scores.visual)}`}>
                  {formatScore(point.component_scores.visual)}
                </div>
              </div>
            </div>

            {/* Reasoning */}
            <div className="text-sm text-gray-600 bg-white p-2 rounded">
              {point.reasoning}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttentionPoints; 