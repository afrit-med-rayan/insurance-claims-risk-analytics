"""
Natural Language Query Assistant for Insurance KPIs.

Uses Ollama (local LLM) by default to answer business questions 
based on pre-aggregated KPI tables.
"""

import json
import os
import textwrap

import pandas as pd
import requests

from dotenv import load_dotenv

load_dotenv()


def build_kpi_tables(features_path: str = None) -> dict:
    """Pre-aggregate features.csv into small KPI tables for the LLM."""
    if features_path is None:
        features_path = os.path.join("data", "processed", "features.csv")
        
    df = pd.read_csv(features_path)
    
    tables = {}
    
    # 1. Overall
    overall = pd.DataFrame({
        "Total_Policies": [len(df)],
        "Total_Exposure": [df["Exposure"].sum()],
        "Total_Claims": [df["ClaimNb"].sum()],
        "Total_Claim_Amount": [df["ClaimAmount_total"].sum()],
        "Total_Premium": [df["Premium"].sum()]
    })
    overall["Avg_Freq"] = overall["Total_Claims"] / overall["Total_Exposure"]
    overall["Avg_Severity"] = overall["Total_Claim_Amount"] / overall["Total_Claims"]
    overall["Loss_Ratio"] = overall["Total_Claim_Amount"] / overall["Total_Premium"]
    tables["overall"] = overall
    
    # Helper for groupby
    def agg_kpi(group_col):
        g = df.groupby(group_col).agg(
            Policies=("IDpol", "count"),
            Exposure=("Exposure", "sum"),
            Claims=("ClaimNb", "sum"),
            ClaimAmount=("ClaimAmount_total", "sum"),
            Premium=("Premium", "sum")
        )
        g["Freq"] = g["Claims"] / g["Exposure"]
        g["Severity"] = g["ClaimAmount"] / g["Claims"]
        g["LossRatio"] = g["ClaimAmount"] / g["Premium"]
        return g.reset_index()

    tables["region"] = agg_kpi("Region")
    tables["drivage"] = agg_kpi("DrivAgeBand")
    tables["vehpower"] = agg_kpi("VehPower")
    
    return tables


def route_question(question: str, tables: dict) -> tuple:
    """Naive router to pick the most relevant table."""
    q = question.lower()
    
    if "region" in q or "area" in q:
        return "region", tables["region"]
    elif "age" in q or "driver" in q or "young" in q or "old" in q:
        return "drivage", tables["drivage"]
    elif "power" in q or "engine" in q or "vehpower" in q:
        return "vehpower", tables["vehpower"]
    else:
        return "overall", tables["overall"]


def ask_llm(prompt: str, provider: str = None) -> str:
    """Send prompt to the specified LLM provider."""
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")
        
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"[Error connecting to Ollama: {str(e)}. Make sure Ollama is running and the model is pulled.]"
            
    elif provider == "openai":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            return "[Error: OPENAI_API_KEY not found in environment]"
            
        try:
            # Using new client syntax for latest openai package
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error connecting to OpenAI: {str(e)}]"
            
    else:
        return f"[Error: Unknown provider {provider}]"


def answer_question(question: str, tables: dict) -> str:
    """Format prompt and ask LLM."""
    table_name, df_context = route_question(question, tables)
    
    # Format table as markdown string
    context_str = df_context.to_markdown(index=False, floatfmt=".3f")
    
    prompt = textwrap.dedent(f"""
    You are a data analyst assistant at an insurance company. 
    Answer the user's question concisely based ONLY on the provided KPI table.
    
    Context Table: {table_name}
    {context_str}
    
    Question: {question}
    
    Answer:
    """)
    
    return ask_llm(prompt)


def generate_examples(tables: dict) -> None:
    """Generate sample Q&A pairs for documentation."""
    questions = [
        "Which driver age band has the highest claim frequency, and what is the rate?",
        "What is the overall portfolio loss ratio?",
        "Are higher vehicle powers associated with higher severity? Compare power 4 and 15."
    ]
    
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "genai_examples.md")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# GenAI Assistant Examples\n\n")
        
        for q in questions:
            print(f"  generating answer for: '{q}'")
            ans = answer_question(q, tables)
            
            f.write(f"**Q: {q}**\n\n")
            f.write(f"**A**: {ans}\n\n---\n\n")


def run_cli() -> None:
    """Interactive CLI."""
    print("  building KPI tables ...")
    tables = build_kpi_tables()
    
    print("  generating example outputs for documentation ...")
    generate_examples(tables)
    
    print("\n" + "="*50)
    print("Insurance Risk KPI Assistant (type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            q = input("\n> ")
            if q.lower() in ['quit', 'exit', 'q']:
                break
            if not q.strip():
                continue
                
            print("Thinking...")
            ans = answer_question(q, tables)
            print(f"\n{ans}")
            
        except KeyboardInterrupt:
            break
            

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--examples-only":
        tables = build_kpi_tables()
        generate_examples(tables)
    else:
        run_cli()
