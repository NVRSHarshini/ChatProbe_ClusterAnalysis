import React, { useState, useEffect } from 'react';
import Chart from 'chart.js/auto';
import * as XLSX from 'xlsx';
import { useNavigate } from 'react-router-dom';
const ChatbotWithExcelAndChart = () => {
  const [reasonsData, setReasonsData] = useState({});
  const navigate=useNavigate();
  useEffect(() => {
    const readExcelFile = async () => {
      try {
        const file = '/React_ChatProbe.xlsx'; // Update the file path accordingly
        const response = await fetch(file);
        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const excelData = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);

        const categorizedData = excelData.reduce((acc, item) => {
          const category = item.Category; // Assuming 'Category' is the column name

          if (category) {
            if (!acc[category]) {
              acc[category] = 1;
            } else {
              acc[category]++;
            }
          }
          return acc;
        }, {});

        setReasonsData(categorizedData);
      } catch (error) {
        console.error('Error reading file:', error);
      }
    };

    readExcelFile();
  }, []);

  useEffect(() => {
    const ctx = document.getElementById('chart'); // Assuming you have a canvas element with id='chart'
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Object.keys(reasonsData),
        datasets: [
          {
            label: 'Category Frequency',
            data: Object.values(reasonsData),
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: 'y', // To create a horizontal bar chart
        scales: {
          x: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Frequency',
            },
          },
          y: {
            title: {
              display: true,
              text: 'Categories',
            },
          },
        },
      },
    });
   

    return () => {
      // Clean up or destroy chart when component unmounts
      chart.destroy();
    };
  }, [reasonsData]);
  const handleChat = () => {
    navigate('/bot');
  };
  return (
    <div>
      <button onClick={handleChat}>Chat</button>
      <canvas id="chart" width={800} height={400}></canvas>
    </div>
  );
};

export default ChatbotWithExcelAndChart;
