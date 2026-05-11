import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [location, setLocation] = useState("");
  const [experience, setExperience] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError("Please upload a resume.");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations([]);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `http://127.0.0.1:8000/recommend?preferred_location=${encodeURIComponent(
          location
        )}&experience_level=${encodeURIComponent(experience)}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch recommendations");
      }

      const data = await response.json();

      console.log(data);

      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error(err);
      setError("Something went wrong.");
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        maxWidth: "900px",
        margin: "40px auto",
        padding: "20px",
        fontFamily: "Arial",
      }}
    >
      <h1>AI Job Recommendation Engine</h1>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "15px" }}>
          <label>Upload Resume:</label>
          <br />
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>

        <div style={{ marginBottom: "15px" }}>
          <label>Preferred Location:</label>
          <br />
          <input
            type="text"
            placeholder="Atlanta"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            style={{
              width: "100%",
              padding: "10px",
            }}
          />
        </div>

        <div style={{ marginBottom: "15px" }}>
          <label>Desired Experience Level:</label>
          <br />
          <select
            value={experience}
            onChange={(e) => setExperience(e.target.value)}
            style={{
              width: "100%",
              padding: "10px",
            }}
          >
            <option value="">Select Experience Level</option>
            <option value="Internship">Internship</option>
            <option value="Entry level">Entry level</option>
            <option value="Associate">Associate</option>
            <option value="Mid-Senior level">Mid-Senior level</option>
            <option value="Director">Director</option>
            <option value="Executive">Executive</option>
          </select>
        </div>

        <button
          type="submit"
          style={{
            padding: "12px 20px",
            cursor: "pointer",
          }}
        >
          Get Recommendations
        </button>
      </form>

      {loading && (
        <p style={{ marginTop: "20px" }}>
          Loading recommendations...
        </p>
      )}

      {error && (
        <p style={{ color: "red", marginTop: "20px" }}>
          {error}
        </p>
      )}

      {recommendations.length > 0 && (
        <div style={{ marginTop: "40px" }}>
          <h2>Recommended Jobs</h2>

          {recommendations.map((job, index) => (
            <div
              key={index}
              style={{
                border: "1px solid #ccc",
                borderRadius: "10px",
                padding: "15px",
                marginBottom: "15px",
              }}
            >
              <h3>{job.title}</h3>

              <p>
                <strong>Location:</strong> {job.location}
              </p>

              <p>
                <strong>Similarity Score:</strong>{" "}
                {Number(job.similarity).toFixed(3)}
              </p>

              {job.experience_level && (
                <p>
                  <strong>Experience:</strong>{" "}
                  {job.experience_level}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;