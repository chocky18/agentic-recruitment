import React, { useState, useEffect, useRef } from "react";

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const ws = useRef(null);

  // Detect environment (local vs production)
  const isProd = window.location.hostname !== "localhost";
  const apiUrl = isProd
    ? "https://agentic-recruitment.onrender.com"
    : "http://127.0.0.1:8000";
  const wsUrl = isProd
    ? "wss://agentic-recruitment.onrender.com/ws"
    : "ws://127.0.0.1:8000/ws";

  // Initialize WebSocket with auto-reconnect
  const initWebSocket = () => {
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log("WebSocket connected!");
    };

    ws.current.onmessage = (event) => {
      let msgData;
      try {
        msgData = JSON.parse(event.data);
      } catch {
        msgData = { type: "text", data: event.data };
      }

      setMessages((prev) => [
        ...prev,
        { from: "agent", text: msgData.data, type: msgData.type || "text" },
      ]);
    };

    ws.current.onclose = () => {
      console.log("WebSocket closed, retrying in 2s...");
      setTimeout(initWebSocket, 2000);
    };

    ws.current.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.current.close();
    };
  };

  useEffect(() => {
    initWebSocket();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  // Send user message
  const sendMessage = () => {
    if (!input.trim()) return;

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(input);
      setMessages((prev) => [...prev, { from: "user", text: input }]);
      setInput("");
    } else {
      console.warn("WebSocket not ready.");
      setMessages((prev) => [
        ...prev,
        { from: "agent", text: "⚠️ WebSocket not ready yet." },
      ]);
    }
  };

  // File selection
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Upload resume
  const uploadFile = async () => {
    if (!file) return alert("Please select a file first!");
    const formData = new FormData();
    formData.append("files", file);

    try {
      const res = await fetch(`${apiUrl}/upload_resumes`, {
        method: "POST",
        body: formData,
      });
      const result = await res.json();
      console.log("Upload response:", result);

      if (res.ok && result.uploaded_files?.length > 0) {
        setMessages((prev) => [
          ...prev,
          {
            from: "agent",
            text: `✅ Resume uploaded: ${result.uploaded_files[0].filename}`,
          },
        ]);
      } else {
        setMessages((prev) => [...prev, { from: "agent", text: "❌ Upload failed!" }]);
      }
    } catch (error) {
      console.error("Upload error:", error);
      setMessages((prev) => [
        ...prev,
        { from: "agent", text: "⚠️ Error uploading file." },
      ]);
    }
  };

  // Render message
  const renderMessage = (msg) => {
    if (!msg.text) return null;

    if (msg.type === "jd") {
      return (
        <div style={{ paddingLeft: "10px" }}>
          <div><b>Role:</b> {msg.text.role || "N/A"}</div>
          <div><b>Experience:</b> {msg.text.years_experience || 0} yrs</div>
          <div><b>Location:</b> {msg.text.location || "Any"}</div>
          <div><b>Skills:</b> {(msg.text.skills || []).join(", ")}</div>
          <div>
            <b>Responsibilities:</b>
            <ul>
              {(msg.text.responsibilities || []).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        </div>
      );
    }

    if (msg.type === "candidates") {
      return (
        <div style={{ paddingLeft: "10px" }}>
          {(msg.text || []).map((c, i) => (
            <div key={i} style={{ marginBottom: "10px", borderBottom: "1px dashed #ccc" }}>
              <b>{c.candidate.name}</b> - {c.candidate.designation}<br />
              <b>Exp:</b> {c.candidate.experience_years} yrs | <b>Location:</b> {c.candidate.location}<br />
              <b>Skills:</b> {(c.candidate.skills || []).join(", ")}<br />
              <b>Score:</b> {c.score.final_score}%
            </div>
          ))}
        </div>
      );
    }

    return <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{JSON.stringify(msg.text, null, 2)}</pre>;
  };

  return (
    <div className="chatbot-container" style={{ maxWidth: "600px", margin: "0 auto" }}>
      <div
        className="messages"
        style={{ minHeight: "300px", border: "1px solid #ccc", padding: "10px", overflowY: "auto" }}
      >
        {messages.map((msg, i) => (
          <div key={i} className={msg.from} style={{ marginBottom: "10px" }}>
            <b>{msg.from}: </b>
            {renderMessage(msg)}
          </div>
        ))}
      </div>

      <div className="input-bar" style={{ display: "flex", marginTop: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask agent..."
          style={{ flex: 1, padding: "8px" }}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} style={{ marginLeft: "5px" }}>Send</button>
      </div>

      <div className="upload-bar" style={{ marginTop: "10px" }}>
        <input type="file" onChange={handleFileChange} />
        <button onClick={uploadFile}>Upload Resume</button>
      </div>
    </div>
  );
};

export default ChatBot;
