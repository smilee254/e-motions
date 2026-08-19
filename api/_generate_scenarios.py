import os
import csv
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CSV_PATH = "data/kenyan_context.csv"

# Topics to ensure variety across the 100 scenarios
topics = [
    "University life, exam stress, and HELB loan delays",
    "Unemployment, the job hunt in Nairobi, and feeling left behind",
    "Workplace toxicity, underpayment, and the daily hustle",
    "Black tax, pressure from upcountry relatives, and being the firstborn",
    "Social media comparison, FOMO in Kilimani/Westlands, and feeling inadequate",
    "Navigating traditional African parents who don't believe in mental health",
    "Dating in Nairobi, relationship trauma, and loneliness",
    "Grief, loss of a loved one, and cultural expectations around mourning",
    "Financial anxiety, rising cost of living, and debts/Fuliza",
    "Identity, navigating modern life vs traditional expectations, and imposter syndrome"
]

def generate_batch(topic, batch_num):
    prompt = f"""
    You are an expert Kenyan psychologist and cultural expert.
    I am building a mental health AI for young Kenyans.
    Generate EXACTLY 10 deep, highly empathetic, and culturally nuanced Question/Answer pairs focusing on this specific topic:
    TOPIC: {topic}
    
    The 'Question' should be a realistic first-person message from a Kenyan user (use mild Sheng/Swahili where natural, but mostly English).
    The 'Answer' should be profound, empathetic, validating, and culturally relevant advice (1-2 paragraphs).
    
    Output STRICTLY in the following CSV format (do not include headers, do not include markdown, ONLY valid CSV rows):
    "Question string","Answer string"
    "Question string","Answer string"
    
    Ensure you properly escape quotes inside the strings by using double quotes if necessary. No introductory text.
    """
    
    print(f"Generating batch {batch_num}/10 for topic: {topic}...")
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_completion_tokens=4000
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating batch: {e}")
        return ""

def main():
    # Ensure file exists and has headers
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', encoding='utf-8') as f:
            f.write("Question,Answer\n")
            
    # Open in append mode
    with open(CSV_PATH, 'a', encoding='utf-8') as f:
        for i, topic in enumerate(topics, 1):
            csv_data = generate_batch(topic, i)
            if csv_data:
                # Clean up potential markdown formatting if the model disobeys
                csv_data = csv_data.replace("```csv", "").replace("```", "").strip()
                f.write(csv_data + "\n")
            time.sleep(2) # Brief pause to respect rate limits
            
    print(f"\n✅ Finished generating 100 scenarios! They have been appended to {CSV_PATH}")

if __name__ == "__main__":
    main()
