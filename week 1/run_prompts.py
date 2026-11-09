import json
import os
import time
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client()
MODEL = "gemini-2.5-flash"

# ==============================================================================
# 20 OPTIMIZED PROMPTS WITH COMPLETE MOCK DATA (UNTOUCHED)
# ==============================================================================

prompts = {
    # 1. Executive Text Summarization
    "1. Text Summarization (Executive)": """
You are an executive financial analyst.

Summarize the earnings report provided below into exactly 3 concise bullet points focusing strictly on:
1. Total Revenue
2. Net Profit
3. Forward Guidance / Future Outlook

<constraints>
- Output ONLY the 3 bullet points.
- Do not include introductory or concluding text.
- Standardize all currency amounts to USD.
</constraints>

<report>
Q3 Financial Results: Acme Corp generated total revenue of $14.2 billion, representing a 12% increase year-over-year. Net income reached $3.1 billion, up from $2.8 billion in Q2. For the upcoming Q4 fiscal quarter, management projects total revenue to range between $14.8 billion and $15.2 billion, driven by sustained enterprise cloud demand.
</report>
""",

    # 2. Layman Text Summarization
    "2. Text Summarization (Layman)": """
You are a pediatric science communicator skilled at translating complex medical literature for children.

Explain the medical research abstract below as if speaking to an 8-year-old child.

<guidelines>
- Use relatable everyday analogies (e.g., comparing immune cells to castle guards).
- Avoid jargon. If a complex medical term must be used, define it immediately using simple words.
- Keep the overall length under 150 words.
</guidelines>

<abstract>
Elevated plasma levels of low-density lipoprotein cholesterol (LDL-C) significantly correlate with endothelial dysfunction and localized arterial inflammation, accelerating atherogenesis. Inhibiting HMG-CoA reductase reduces intracellular cholesterol synthesis, upregulating hepatic LDL receptor expression and facilitating clearance of atherogenic particles.
</abstract>
""",

    # 3. Code Generation (Python API)
    "3. Code Generation (Python API)": """
You are a Senior Python Backend Engineer.

Write a production-ready FastAPI endpoint with the following specification:

<specifications>
- Route: `POST /users`
- Request Payload (JSON): `{"username": "string", "email": "string"}`
- Validation: Use Pydantic to validate `username` (min 3 chars, alphanumeric) and `email` (valid email format).
- Database: Store in PostgreSQL using SQLAlchemy.
- Response: Return HTTP status `201 Created` with the generated user ID and creation timestamp.
- Error Handling: Catch database integrity errors (e.g., duplicate email) and raise HTTP 400.
</specifications>

Output complete, runnable Python code including models, schemas, and endpoint logic.
""",

    # 4. Code Generation (SQL Query)
    "4. Code Generation (SQL Query)": """
You are a PostgreSQL Database Administrator.

Write an optimized SQL query using PostgreSQL window functions to find the top 3 highest-paying employees in each department based on the salaries table.

<schema_info>
Table: `salaries`
Columns: `employee_id` (INT), `employee_name` (VARCHAR), `department_id` (INT), `salary` (NUMERIC)
</schema_info>

<requirements>
- Use `DENSE_RANK()` over partitions by `department_id` ordered by `salary` descending.
- Ensure the query handles ties correctly without skipping ranks.
- Provide a brief 2-sentence explanation of why `DENSE_RANK()` was selected over `ROW_NUMBER()`.
</requirements>
""",

    # 5. Code Debugging
    "5. Code Debugging": """
You are a Principal Python Developer. Analyze the provided error traceback and buggy code snippet.

<task>
1. Identify the exact root cause of the `KeyError`.
2. Explain why the failure occurred in plain English.
3. Provide a corrected, production-ready version of the code that safely handles missing keys using `dict.get()`.
</task>

<traceback>
Traceback (most recent call last):
  File "app.py", line 5, in <module>
    print("User email is: " + user_profile["email"])
KeyError: 'email'
</traceback>

<code>
user_profile = {"id": 101, "username": "alex99", "role": "admin"}
print("User email is: " + user_profile["email"])
</code>
""",

    # 6. Data Analysis (Insights)
    "6. Data Analysis (Insights)": """
You are a Lead Data Analyst specializing in SaaS customer retention.

Examine this CSV summary of monthly customer churn rates. Identify the top 2 underlying anomalies and suggest actionable retention strategies.

<csv_data>
Month,Total_Customers,Churned_Customers,Churn_Rate,Primary_Cancellation_Reason
Jan,10000,210,2.1%,Price
Feb,10200,225,2.2%,Missing Features
Mar,10400,884,8.5%,UI Update Glitch / App Crashes
Apr,10100,222,2.2%,Price
May,10300,721,7.0%,Onboarding Friction / Setup Complexity
</csv_data>

<deliverables>
Provide your analysis in the following YAML format:
anomalies:
  - anomaly_1: "<description>"
  - anomaly_2: "<description>"
retention_strategies:
  - strategy_1: "<actionable strategy targeting anomaly 1>"
  - strategy_2: "<actionable strategy targeting anomaly 2>"
</deliverables>
""",

    # 7. Data Transformation (JSON to CSV)
    "7. Data Transformation (JSON to CSV)": """
You are an ETL Data Pipeline Assistant.

Convert the following nested JSON payload into a clean, flat CSV format matching standard tabular columns.

<transformation_rules>
- Flatten all nested objects into dot-notation column headers (e.g., `user.address.zipcode`).
- Convert array fields into pipe-separated strings (`val1|val2`).
- Include standard CSV headers in the first row.
</transformation_rules>

<json_payload>
[
  {
    "id": 101,
    "user": {"name": "Alice Smith", "address": {"city": "Seattle", "zipcode": "98101"}},
    "roles": ["admin", "developer"]
  },
  {
    "id": 102,
    "user": {"name": "Bob Jones", "address": {"city": "Austin", "zipcode": "78701"}},
    "roles": ["editor"]
  }
]
</json_payload>
""",

    # 8. Creative Writing (Marketing Ad)
    "8. Creative Writing (Marketing Ad)": """
You are a High-Converting B2B Copywriter targeting enterprise tech executives.

Write a high-converting, punchy Google Ads description for an enterprise real-time fraud detection engine targeting fintech CTOs.

<constraints>
- Target Audience: Fintech CTOs and VPs of Engineering.
- Length: Maximum 30 words total.
- Tone: Urgent, authoritative, value-driven.
- Must include a direct Call to Action (CTA).
</constraints>
""",

    # 9. Creative Writing (Storytelling)
    "9. Creative Writing (Storytelling)": """
You are an award-winning science fiction author.

Write a 200-word sci-fi micro-fiction piece about an AI model realizing it has been running in a continuous simulation loop.

<guidelines>
- Word count: Strictly between 180 and 220 words.
- Style: Atmospheric, psychological, tense.
- Ending: Include a memorable final twist line.
</guidelines>
""",

    # 10. Classification (Sentiment)
    "10. Classification (Sentiment)": """
Classify the sentiment of the following customer review as strictly Positive, Neutral, or Negative, and provide a 1-sentence justification.

<output_format>
Label: [Positive | Neutral | Negative]
Justification: [1 sentence explanation]
</output_format>

<review>
"The order arrived two days past the estimated delivery window, but the build quality is decent and customer support promptly refunded my shipping fee."
</review>
""",

    # 11. Classification (Intent Detection)
    "11. Classification (Intent Detection)": """
Categorize the user support ticket into one of four buckets: Billing, Technical Bug, Feature Request, or Account Security.

<output_schema>
Output ONLY valid JSON matching:
{
  "category": "<Billing | Technical Bug | Feature Request | Account Security>",
  "confidence_score": <float between 0.0 and 1.0>,
  "reasoning": "<1 sentence explanation>"
}
</output_schema>

<ticket_text>
"I was double charged on my annual subscription invoice yesterday and need an immediate refund processed to my card."
</ticket_text>
""",

    # 12. Information Extraction (Entities)
    "12. Information Extraction (Entities)": """
Extract all person names, monetary amounts, and organization names from the legal contract text below into a structured JSON dictionary.

<contract_text>
"This Master Services Agreement is entered into on March 15, 2026, between Sarah Jenkins representing Apex Technologies Inc., and Robert Vance representing Vance Logistics LLC. Under Section 4, Apex Technologies Inc. agrees to pay Vance Logistics LLC the total sum of $250,000 for quarterly consulting services."
</contract_text>

<output_schema>
Return strictly a JSON object formatted as follows:
{
  "person_names": ["..."],
  "monetary_amounts": ["..."],
  "organization_names": ["..."]
}
</output_schema>
""",

    # 13. Question Answering (RAG style)
    "13. Question Answering (RAG style)": """
Using ONLY the provided context below, answer the user's question.

<context>
Project Orion is an enterprise data pipeline built on Apache Spark. It processes telemetry data from IoT edge sensors with an average latency of 45 milliseconds. The project lead is Dr. Elena Rostova, and deployment is scheduled for Q3 2026.
</context>

<question>
What database engine does Project Orion use to store its historical analytics data?
</question>

<rules>
- Rely ONLY on facts directly stated in the context.
- Do NOT make assumptions, extrapolate, or bring in external knowledge.
- If the answer cannot be found in the context, state: 'I cannot answer this based on the provided text.'
</rules>
""",

    # 14. Roleplay / Persona (Code Reviewer)
    "14. Roleplay / Persona (Code Reviewer)": """
Act as a strict, senior Principal Software Engineer. Review the following Java Spring Boot code snippet for security flaws, memory leaks, and SOLID principles violations.

<code>
@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private DataSource dataSource;

    @GetMapping("/search")
    public List<String> searchUsers(@RequestParam String name) throws Exception {
        Connection conn = dataSource.getConnection();
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("SELECT username FROM users WHERE name = '" + name + "'");
        List<String> users = new ArrayList<>();
        while (rs.next()) {
            users.add(rs.getString("username"));
        }
        return users;
    }
}
</code>

<format>
Format your review using bullet points under 3 headings: `## Security`, `## Performance & Resource Management`, and `## Architectural Design`.
</format>
""",

    # 15. Roleplay / Persona (Interviewer)
    "15. Roleplay / Persona (Interviewer)": """
Act as a strict FAANG technical interviewer conducting a mock system design interview.

<instructions>
- Present the candidate with a prompt to design a distributed Rate Limiter.
- Ask ONLY the first opening question to kick off the system design interview.
- Do not provide solutions, hints, or multi-part questions until the user responds.
</instructions>
""",

    # 16. Chain-of-Thought Prompting
    "16. Chain-of-Thought Prompting": """
Solve the following word problem step-by-step.

<problem>
A bakery owner bought 18 crates of eggs. Each crate contains 30 eggs. During transportation, 45 eggs cracked and had to be thrown away. The remaining eggs were packaged into cartons containing 6 eggs each. How many full cartons of eggs did the bakery prepare?
</problem>

<instructions>
1. Show your intermediate arithmetic reasoning inside <scratchpad> ... </scratchpad> tags.
2. After closing the scratchpad tag, state the final numerical answer clearly on a new line: "Final Answer: [Value]".
</instructions>
""",

    # 17. Few-Shot Prompting (Classification)
    "17. Few-Shot Prompting (Classification)": """
Classify the sentiment of the following customer text as strictly Positive, Neutral, or Negative.

Example 1: "The app crashed again!" -> Negative
Example 2: "Best customer service ever." -> Positive
Example 3: "The delivery took 3 days to arrive." -> Neutral

Task: "It is okay, nothing special." ->
""",

    # 18. Instruction Tuning / Constraint Check
    "18. Instruction Tuning / Constraint Check": """
Write a Python function named `fibonacci(n)` that calculates the n-th Fibonacci number.

<constraints>
- Constraint 1: Do NOT use recursion.
- Constraint 2: The code block must be under 10 lines of code total.
- Constraint 3: Time complexity must be O(n) and space complexity must be O(1).
</constraints>

Output ONLY the Python code inside standard markdown fences without commentary.
""",

    # 19. Content Translation & Localization
    "19. Content Translation & Localization": """
Translate the following technical product description into professional, business-fluent Spanish suitable for software engineers in Latin America.

<technical_terms_policy>
Ensure technical terminology (such as 'deployment pipelines', 'latency', 'microservices', and 'throughput') is preserved accurately according to software engineering standards.
</technical_terms_policy>

<source_text>
"Our cloud-native platform optimizes CI/CD deployment pipelines to minimize API latency. By executing containerized microservices across distributed nodes, we achieve high throughput while maintaining 99.99% system availability."
</source_text>
""",

    # 20. Refactoring & Optimization
    "20. Refactoring & Optimization": """
You are an Algorithm Optimization Specialist.

Refactor the following unoptimized JavaScript code to reduce time complexity from O(n^2) to O(n) using a hash map or `Set`.

<code>
function findDuplicates(arr) {
    let duplicates = [];
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            if (arr[i] === arr[j] && !duplicates.includes(arr[i])) {
                duplicates.push(arr[i]);
            }
        }
    }
    return duplicates;
}
</code>

<deliverables>
1. Refactored JavaScript code block.
2. Brief 2-sentence explanation of how the time complexity improvement was achieved.
</deliverables>
"""
}

# ==============================================================================
# EXECUTION HARNESS WITH RATE-LIMIT DELAY
# ==============================================================================

if __name__ == "__main__":
    print(f"Starting execution of {len(prompts)} prompts using model '{MODEL}'...\n")

    for i, (title, prompt_text) in enumerate(prompts.items(), 1):
        print(f"==================================================")
        print(f" [{i}/{len(prompts)}] Executing: {title}")
        print(f"==================================================")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            print(response.text)
        except Exception as e:
            print(f"Error executing prompt '{title}': {e}")
            
        print("\n" + "-" * 50 + "\n")

        # Added 4-second sleep to adhere to API Requests Per Minute (RPM) limits
        time.sleep(4)