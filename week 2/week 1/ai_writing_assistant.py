import os
import time
from google import genai
from google.genai import types

# Initialize Gemini Client (reads GEMINI_API_KEY environment variable)
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

# Safeguard flag for API quota limits
FORCE_OFFLINE_MODE = True


class AIWritingAssistant:
    """A modular AI writing assistant supporting multiple templates, temperatures, and output formatting."""

    TEMPLATES = {
        "blog": {
            "system": "You are an expert technical content writer. Write structured, engaging, and in-depth blog posts using Markdown headers.",
            "default_temp": 0.7,  # Higher creativity for writing
        },
        "email": {
            "system": "You are an executive communications manager. Write polished, professional, clear, and concise emails with clear subject lines.",
            "default_temp": 0.3,  # Lower creativity for professional precision
        },
        "code_explanation": {
            "system": "You are a senior software engineering instructor. Explain code snippets clearly, breaking down architecture, complexity, and logic step-by-step.",
            "default_temp": 0.2,  # Highly deterministic for technical facts
        },
        "social": {
            "system": "You are a growth-focused social media manager. Write punchy, high-engagement posts optimized with hooks and relevant hashtags.",
            "default_temp": 0.8,  # High creativity and engagement tone
        },
    }

    @classmethod
    def generate(cls, task_type: str, user_prompt: str, temperature: float = None) -> str:
        """Generates content based on the selected template task type and temperature."""
        task_config = cls.TEMPLATES.get(task_type.lower())
        if not task_config:
            return f"Error: Unknown task type '{task_type}'. Choose from {list(cls.TEMPLATES.keys())}"

        system_instruction = task_config["system"]
        temp = temperature if temperature is not None else task_config["default_temp"]

        # Simulated fallback if API quota is reached
        if FORCE_OFFLINE_MODE:
            time.sleep(0.4)
            return cls._get_mock_response(task_type, user_prompt)

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temp,
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f"API Error: {e}"

    @staticmethod
    def _get_mock_response(task_type: str, prompt: str) -> str:
        """Returns structured mock outputs for offline testing during quota limits."""
        mock_data = {
            "blog": f"### Understanding {prompt}\n\nIn modern software engineering, optimizing pipelines is critical. This blog explores key performance indicators, architectural bottlenecks, and scalable mitigation strategies.",
            "email": f"Subject: Update regarding {prompt}\n\nHi Team,\n\nI wanted to share a brief update on our progress. Everything is on track for milestone delivery. Let me know if you have any questions.\n\nBest regards,\nNitin",
            "code_explanation": f"### Code Breakdown: {prompt}\n\n1. **Initialization:** Sets up client dependencies and configurations.\n2. **Execution Flow:** Iterates through inputs and manages state.\n3. **Complexity:** Operates at $O(N)$ time complexity.",
            "social": f"🚀 Exciting breakthrough working on {prompt}! Clean architecture and automated pipelines make all the difference. Check it out below! 👇\n\n#AI #SoftwareEngineering #Python",
        }
        return mock_data.get(task_type, "Content successfully generated.")


# ==============================================================================
# DEMO EXECUTION SUITE ACROSS ALL 4 TEMPLATES
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" INITIALIZING AI WRITING ASSISTANT (WEEK 1 MINI PROJECT)")
    print("==================================================\n")

    test_requests = [
        ("blog", "Microservices vs Monoliths in Enterprise Software", 0.7),
        ("email", "Project deadline extension request to client", 0.3),
        ("code_explanation", "Binary Search Tree insertion algorithm in Python", 0.2),
        ("social", "Launching a new AI-powered code review assistant", 0.8),
    ]

    for task, prompt, temp in test_requests:
        print(f"[{task.upper()} TASK] (Temperature: {temp})")
        print(f"Prompt: {prompt}\n")
        
        output = AIWritingAssistant.generate(task_type=task, user_prompt=prompt, temperature=temp)
        
        print(f"Generated Output:\n{output}")
        print("=" * 60 + "\n")
        time.sleep(1)