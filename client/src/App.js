import React, { useState, useRef, useEffect } from 'react';
import { VegaLite } from 'react-vega';
import * as d3 from 'd3-dsv';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Loader2, AlertCircle } from 'lucide-react';

// Update API URL construction to use correct endpoint
const BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://assignment3-f8gl.onrender.com'  // Your Render.com URL
  : 'http://0.0.0.0:10000';  // Match the backend port

const userImage = 'https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg';
const systemImage = 'https://img.freepik.com/free-vector/floating-robot_78370-3669.jpg';

export default function DataVizAssistant() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [fileData, setFileData] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [tableVisible, setTableVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleFileUpload = (file) => {
    if (file.type === "text/csv") {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsedData = d3.csvParse(reader.result, d3.autoType);
          if (parsedData && parsedData.length > 0) {
            setFileData(parsedData);
            setError(null);
            setMessages([{
              sender: 'System',
              text: `Dataset loaded successfully! Found ${parsedData.length} rows and ${Object.keys(parsedData[0]).length} columns.`,
              userImg: systemImage
            }]);
          } else {
            throw new Error('No data found in CSV file');
          }
        } catch (err) {
          setError('Failed to parse CSV file. Please check the file format.');
        }
      };
      reader.onerror = () => {
        setError('Failed to read the file. Please try again.');
      };
      reader.readAsText(file);
    } else {
      setError('Please upload a valid CSV file.');
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) handleFileUpload(file);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    if (!fileData) {
      setError('Please upload a dataset first.');
      return;
    }

    setMessages(prev => [...prev, { sender: 'You', text: message, userImg: userImage }]);
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: message,
          data: fileData
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Create array of messages to add
      const newMessages = [];
      
      // Add chart if present
      if (result.chart) {
        newMessages.push({
          sender: 'System',
          userImg: systemImage,
          vegaSpec: result.chart
        });
      }
      
      // Add text response
      newMessages.push({
        sender: 'System',
        text: result.text,
        userImg: systemImage
      });

      setMessages(prev => [...prev, ...newMessages]);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
      setMessage("");
    }
  };

  const clearMessages = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-900">
          Data Visualization Assistant
        </h1>

        {error && (
          <div className="text-red-500 mb-4">
            <AlertCircle className="inline h-4 w-4" />
            {error}
          </div>
        )}

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors
              ${dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
              id="file-input"
            />
            <label 
              htmlFor="file-input"
              className="cursor-pointer text-blue-600 hover:text-blue-800"
            >
              {dragging ? 'Drop your file here!' : 'Drop a CSV file here or click to upload'}
            </label>
          </div>

          {fileData && (
            <div className="mt-4">
              <button
                onClick={() => setTableVisible(!tableVisible)}
                className="w-full bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 transition-colors"
              >
                {tableVisible ? 'Hide Data Preview' : 'Show Data Preview'}
              </button>

              {tableVisible && (
                <div className="mt-4 overflow-auto max-h-60 rounded-lg border border-gray-200">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {Object.keys(fileData[0]).map(key => (
                          <th key={key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {fileData.slice(0, 5).map((row, i) => (
                        <tr key={i}>
                          {Object.values(row).map((val, j) => (
                            <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          <div className="mt-6 bg-gray-50 rounded-lg p-4">
            <div className="h-96 overflow-auto mb-4 space-y-4">
             {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.sender === 'You' ? 'justify-end' : 'justify-start'}`}>
                <div className="flex items-start space-x-2 max-w-2xl">
                  <img src={msg.userImg} alt={`${msg.sender} avatar`} className="w-8 h-8 rounded-full" />
                  <div className={`rounded-lg p-3 ${msg.sender === 'You' ? 'bg-blue-600 text-white' : 'bg-white shadow-sm'}`}>
                    <div className="text-sm font-semibold mb-1">{msg.sender}</div>
                    {msg.vegaLiteSpec && (
                      <div className="mt-2">
                        <VegaLite spec={msg.vegaLiteSpec} data={{ values: fileData }} />
                      </div>
                    )}
                    <ReactMarkdown remarkPlugins={[remarkGfm]} className="text-sm">
                      {msg.content}
                    </ReactMarkdown>
                    </div>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-center">
                  <Loader2 className="animate-spin text-blue-600" />
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="flex space-x-2">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                placeholder="Ask about your data..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={!fileData || loading}
              />
              <button
                onClick={sendMessage}
                disabled={!fileData || loading}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
              <button
                onClick={clearMessages}
                className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}