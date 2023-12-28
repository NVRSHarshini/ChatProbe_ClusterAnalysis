import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';
import stringSimilarity from 'string-similarity';
import { useNavigate } from 'react-router-dom';
const ChatbotWithExcel = () => {
  const [data, setData] = useState({});
  const [userInput, setUserInput] = useState('');
  const [messages, setMessages] = useState([]);
  const navigate=useNavigate();
  useEffect(() => {
    readExcelFile(); // Fetch bot responses on component mount
  
    // Add the initial bot message to the chat
    const initialBotMessage = "Hi, we are here to help you. Go ahead, select your issue:\n1. Network-related issue\n2. Billing and payment issue\n3. Balance, Recharge and validity\n4. Plan/pack related issues\n5. Others";
    setMessages([{ text: initialBotMessage, type: 'bot' }]);
  }, []);

  const readExcelFile = () => {
    const file = '/React_ChatProbe.xlsx'; // Update the file path accordingly
    const reader = new FileReader();

    reader.onload = (event) => {
      const binaryString = event.target.result;
      const workbook = XLSX.read(binaryString, { type: 'binary' });
      const sheetName = workbook.SheetNames[0];
      const excelData = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);

      const chatData = excelData.reduce((acc, item) => {
        const userText = item.userText;
        const botReply = item.BotReply;
        if (userText && botReply) {
          acc[userText] = botReply;
        }
        return acc;
      }, {});

      setData(chatData);
    };

    fetch(file)
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        const data = new Uint8Array(buffer);
        reader.readAsBinaryString(new Blob([data]));
      })
      .catch((error) => {
        console.error('Error reading file:', error);
      });
  };

  const handleMessageSubmit = () => {
    if (userInput.trim() !== '') {
      const userInputLowerCase = userInput.toLowerCase();
      let botReply = '';

      // Check for partial matches using string similarity
      const matchingUserTexts = Object.keys(data).filter((key) => {
        const similarity = stringSimilarity.compareTwoStrings(userInputLowerCase, key.toLowerCase());
        return similarity > 0.5; 
      });

      
      if (matchingUserTexts.length > 0) {
        botReply = data[matchingUserTexts[0]];
      } else {
        botReply = "I'm sorry, I couldn't understand that.";
      }

      setMessages([
        ...messages,
        { text: userInput, type: 'user' },
        { text: botReply, type: 'bot' }
      ]);
      setUserInput('');
    }
  };
  const handleGraph = () => {
    navigate('/graph');
  };
  return (
    // <div className='wrapper'>
    <div>
      <button className='' onClick={handleGraph}>graph</button>
    <center>
       
    <div>
      <div className="chat-window">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            {message.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        placeholder="Type your message..."
      /></div>
      <button class="buttonfx curtaindown" onClick={handleMessageSubmit}>Send</button>
    </center>
    </div>
     
  );
};

export default ChatbotWithExcel;
