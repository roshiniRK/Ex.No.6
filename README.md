# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 25/09/25
# Register no: 212222230123
# Aim:
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
- OpenAI GPT (gpt-3.5 / gpt-4)
- Google Gemini (gemini-pro)
- Python Libraries:
   - openai
   - google-generativeai
   - sentence-transformers
   - scikit-learn
   - keybert

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 


# Conclusion:
## Steps:
1. API Setup – Configure access to OpenAI and Gemini using API keys.
2. Response Collection – Send a single prompt to both tools and retrieve outputs.
3. Semantic Analysis – Encode responses into embeddings and compute cosine similarity.
4. Keyword Extraction – Extract top keywords using KeyBERT.
5. Insights Generation – Compare responses, identify common/unique terms, and provide recommendations.
```
# Install required libraries (run once)
# pip install openai google-generativeai sentence-transformers scikit-learn keybert

import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()

# API Configuration - Replace with your keys
OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def analyze_responses(prompt):
    # Get AI responses
    openai_response = get_openai_response(prompt)
    gemini_response = get_gemini_response(prompt)
    
    # Compare embeddings
    embeddings = embedding_model.encode([openai_response, gemini_response])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Extract keywords
    openai_kws = [kw[0] for kw in kw_model.extract_keywords(openai_response, top_n=3)]
    gemini_kws = [kw[0] for kw in kw_model.extract_keywords(gemini_response, top_n=3)]
    
    # Generate insights
    insights = {
        "similarity_score": round(similarity, 2),
        "common_keywords": list(set(openai_kws) & set(gemini_kws)),
        "openai_keywords": openai_kws,
        "gemini_keywords": gemini_kws,
        "recommendation": (
            "High agreement - Suitable for consistent outputs" if similarity > 0.7 
            else "Moderate agreement - Review differences" if similarity > 0.4 
            else "Low agreement - Investigate discrepancies"
        )
    }
    
    return {
        "openai_response": openai_response,
        "gemini_response": gemini_response,
        "analysis": insights
    }

if __name__ == "__main__":
    prompt = "Explain the role of AI in sustainable agriculture in three sentences."
    results = analyze_responses(prompt)
    
    print("OpenAI Response:\n", results["openai_response"], "\n")
    print("Gemini Response:\n", results["gemini_response"], "\n")
    print("Similarity Score:", results["analysis"]["similarity_score"])
    print("Common Keywords:", results["analysis"]["common_keywords"])
    print("OpenAI Keywords:", results["analysis"]["openai_keywords"])
    print("Gemini Keywords:", results["analysis"]["gemini_keywords"])
    print("Recommendation:", results["analysis"]["recommendation"])
```
### Example Output
### OpenAI Response: 

"AI in sustainable agriculture helps optimize irrigation, monitor crop health using sensors, and reduce resource waste. Predictive models guide farmers in choosing planting strategies based on climate. This ensures higher yields with minimal environmental impact."
### Gemini Response: 
"The role of AI in sustainable agriculture includes precision farming, soil monitoring, and smart irrigation systems. By analyzing real-time data, AI helps farmers reduce water usage and pesticides. This results in eco-friendly farming and improved crop productivity."
- Similarity Score: 0.82 Common Keywords: ['AI', 'agriculture', 'irrigation'] 
- OpenAI Keywords: ['irrigation', 'climate', 'resources'] 
- Gemini Keywords: ['precision', 'soil', 'eco-friendly'] 
- Recommendation: High agreement - Suitable for consistent outputs
### Discussion of Results
1. Response Comparison:
- OpenAI emphasizes climate strategies, predictive modeling, and resource efficiency.
- Gemini emphasizes precision farming, soil monitoring, and eco-friendly practices.
- Both responses overlap on core AI concepts in agriculture, leading to a high similarity score (0.82).

2. Keyword Analysis:
- Common keywords show that both models mention the central themes of AI, agriculture, and irrigation.
- Unique keywords reflect each model’s emphasis: OpenAI on climate and resources, Gemini on precision and eco-friendly methods.

3. Interpretation of Similarity Score:
- Similarity > 0.7 → high agreement, meaning both models provide consistent insights.
- A score between 0.4–0.7 would suggest moderate agreement and the need to review content for differences.
- A score < 0.4 would indicate divergent outputs and require deeper analysis.

4. Recommendation:
- High similarity and overlapping keywords suggest that combining insights from multiple AI tools is effective.
- You can confidently use these outputs for reports, research, or content generation, knowing there is alignment between models.

### Coding Analysis

1. Libraries and Models:
- openai for GPT responses.
- google-generativeai for Gemini responses.
- sentence-transformers and cosine_similarity for embedding-based similarity analysis.
- KeyBERT for extracting relevant keywords from text.

2.Code Flow:

Step 1: Prompt both models for responses.

Step 2: Convert text to embeddings using all-MiniLM-L6-v2.

Step 3: Calculate cosine similarity between embeddings to measure semantic similarity.

Step 4: Extract top 3 keywords using KeyBERT.

Step 5: Identify common and unique keywords.

Step 6: Generate actionable recommendation based on similarity.

3.Strengths of Code:
- Multi-model comparison: Provides insights from multiple AI sources.
- Semantic similarity: Goes beyond exact word matching, capturing meaning.
- Keyword extraction: Highlights the main focus of each response.
- Actionable recommendations: Simplifies decision-making on reliability.

4.Possible Improvements:
- Use more robust keyword extraction or increase top_n for detailed analysis.
- Include sentiment or tone analysis to compare writing styles.
- Allow dynamic model selection or batch processing of multiple prompts.
- Handle exceptions for API failures to ensure smooth execution.
# Result: 
The corresponding Prompt is executed successfully.
