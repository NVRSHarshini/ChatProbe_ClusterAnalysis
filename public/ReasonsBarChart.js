import React from 'react';
import { Bar } from 'react-chartjs-2';

const ReasonsBarChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return <div>No valid data available for the chart</div>;
  }

  const agentNames = Object.keys(data);
  const resolvedTypes = Object.keys(data[agentNames[0]]); // Assuming all agents have the same resolved types

  const chartData = {
    labels: agentNames,
    datasets: resolvedTypes.map((resolvedType, index) => ({
      label: resolvedType,
      backgroundColor: `rgba(${index * 70},${index * 50},${index * 30},0.6)`,
      borderColor: `rgba(${index * 70},${index * 50},${index * 30},1)`,
      borderWidth: 1,
      data: agentNames.map((agent) => data[agent][resolvedType] || 0),
    })),
  };

  const options = {
    scales: {
      yAxes: [{ stacked: true }],
      xAxes: [{ stacked: true }],
    },
  };

  return <Bar data={chartData} options={options} />;
};

export default ReasonsBarChart;
