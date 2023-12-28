import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

import ChatbotWithExcelAndChart from '../src/pages/chatBot_graph.jsx';
import ChatbotWithExcel from '../src/pages/chat_page.jsx';


const App = () => {
  return (
     <Router>
  <Routes>
    <Route exact path="/bot" element={<ChatbotWithExcel/>} />
    <Route exact path="/graph" element={<ChatbotWithExcelAndChart/>} />

    </Routes>

 </Router>
);















  //   <div style={{backgroundImage:img}}className="App">
  //     <ChatbotWithExcelReader/>
  //     {/* <h1>Excel File Reader</h1> */}
  //     {/* <ExcelReader /> */}
  //     {/* <h1>Chatbotwith excelreader</h1> */}
  //     <h1></h1>
  //     <h1></h1>
  //     <center><h1 style={{color:'white'}}> Hi there!, Welcome to ChatProbe</h1></center>
  //     {/* <ChatbotWithExcelReader/> */}
  //   </div>
  // );
//   const [userText, setUserText] = useState('');
//   const [messages, setMessages] = useState([]);
//   const [botReply, setBotReply] = useState({});

//   useEffect(() => {
//     readExcelFile(); // Fetch bot responses on component mount
//   }, []);

//   const readExcelFile = () => {
//     const file = 'chatprobe/public/ChatProbe.xlsx'; // Update the file path accordingly

//     fetch(file)
    
//       .then((res) => res.arrayBuffer())
//       .then((buffer) => {
//         const data = new Uint8Array(buffer);
//         const workbook = XLSX.read(data, { type: 'array' });
//         const sheetName = workbook.SheetNames[0];
//         const excelData = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);

//         const responses = {};
//         if (excelData && excelData.length > 0) {
//           excelData.forEach((row) => {
//             if (row.userText && row.BotReply) {
//               responses[row.userText.toLowerCase()] = row.BotReply;
//             }
//           });
//         }
//         setBotReply(responses);
//         console.log("bot:",botReply);
//       })
//       .catch((error) => {
//         console.error('Error reading file:', error);
//       });
//   };

//   const handleUserText = (e) => {
//     console.log("user",userText);
//     if (e.key === 'Enter') {
//       //const userMessage = userText.toLowerCase();
//       const userMessage=userText;
//       const botMessage = botReply[userMessage] || "I'm sorry, I couldn't understand that.";

//       setMessages([
//         ...messages,
//         { text: userMessage, type: 'user' },
//         { text: botMessage, type: 'bot' },
//       ]);
//       setUserText('');
//     }
//   };

//   return (
//     <div className="chatbot-container">
//       <div className="chat-window">
//         {messages.map((message, index) => (
//           <div key={index} className={`message ${message.type}`}>
//             {message.text}
//           </div>
//         ))}
//       </div>
//       <input
//         type="text"
//         value={userText}
//         onChange={(e) => setUserText(e.target.value)}
//         onKeyDown={handleUserText}
//         placeholder="Type your message..."
//       />
//     </div>
//   );
// };
};
export default App;
