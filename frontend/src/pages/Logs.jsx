import React, { useEffect, useState } from "react";

// Set this to your Render backend URL, or use relative if proxying
const LOG_API = process.env.REACT_APP_BACKEND_URL
  ? process.env.REACT_APP_BACKEND_URL + "/api/logs"
  : "/api/logs";

export default function Logs() {
  const [logs, setLogs] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(LOG_API)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch logs");
        return res.text();
      })
      .then((data) => {
        setLogs(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Backend Logs</h1>
      {loading && <div>Loading logs...</div>}
      {error && <div className="text-red-600">{error}</div>}
      <pre className="bg-gray-900 text-green-200 p-4 rounded overflow-x-auto text-xs max-h-[70vh] whitespace-pre-wrap">
        {logs}
      </pre>
    </div>
  );
}
