"""Activity F: RAG evaluation and comparison."""

import importlib.util
import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_vector_database(db_file: str) -> pd.DataFrame:
    """Load vector database from pickle file."""
    with open(db_file, "rb") as f:
        df = pickle.load(f)
    return df


# Import RAG functions from previous scripts
spec = importlib.util.spec_from_file_location(
    "rag_system_basic", "examples/rag/04_rag_system_basic.py"
)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
retrieve_chunks = rag_module.retrieve_chunks
build_rag_prompt = rag_module.build_rag_prompt
generate_answer = rag_module.generate_answer


def direct_llm_query(query: str) -> str:
    """Query LLM directly without RAG."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


def rag_query(query: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Query using RAG system."""
    chunks = retrieve_chunks(query, df, top_k=5)
    prompt = build_rag_prompt(query, chunks)
    answer = generate_answer(prompt)
    return answer, chunks


def evaluate_answer(answer: str, expected: str) -> dict:
    """Evaluate answer quality against expected content.

    Returns dict with:
    - contains_fact: bool
    - has_citation: bool
    - hallucination_check: bool (inverse - does it avoid making things up)
    """
    return {
        "contains_expected": expected.lower() in answer.lower(),
        "has_citation": any(
            word in answer.lower()
            for word in ["project", "section", "document", "report"]
        ),
        "length": len(answer),
    }


def create_test_set() -> list[dict]:
    """Create evaluation test set.

    Each test has:
    - query: The question
    - expected_content: What should be in the answer
    - category: Type of question
    """
    return [
        {
            "query": "What were the response rates?",
            "expected_content": "response rate",
            "category": "factual_lookup",
        },
        {
            "query": "What challenges were encountered in the field?",
            "expected_content": "challenge",
            "category": "synthesis",
        },
        {
            "query": "What are best practices for community engagement?",
            "expected_content": "community",
            "category": "synthesis",
        },
        {
            "query": "What was the GDP of Kenya?",
            "expected_content": "not available",
            "category": "out_of_scope",
        },
    ]


def run_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Run evaluation comparing Direct LLM vs RAG."""
    test_set = create_test_set()
    results = []

    print("Running Evaluation")
    print("=" * 60)

    for test in test_set:
        print(f"\nTest: {test['query']}")
        print(f"Category: {test['category']}")
        print("-" * 60)

        # Direct LLM
        print("Querying direct LLM...")
        direct_answer = direct_llm_query(test["query"])
        direct_eval = evaluate_answer(direct_answer, test["expected_content"])

        # RAG
        print("Querying RAG system...")
        rag_answer, retrieved_chunks = rag_query(test["query"], df)
        rag_eval = evaluate_answer(rag_answer, test["expected_content"])

        # Store results
        results.append(
            {
                "query": test["query"],
                "category": test["category"],
                "direct_answer": direct_answer,
                "direct_contains_expected": direct_eval["contains_expected"],
                "direct_has_citation": direct_eval["has_citation"],
                "rag_answer": rag_answer,
                "rag_contains_expected": rag_eval["contains_expected"],
                "rag_has_citation": rag_eval["has_citation"],
                "rag_avg_similarity": retrieved_chunks["similarity"].mean(),
            }
        )

        print(
            f"  Direct - Contains expected: {direct_eval['contains_expected']}, Has citation: {direct_eval['has_citation']}"
        )
        print(
            f"  RAG - Contains expected: {rag_eval['contains_expected']}, Has citation: {rag_eval['has_citation']}"
        )

    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame, output_file: str) -> None:
    """Generate HTML evaluation report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #e7f3e7; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .method {{ flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Report</h1>

    <h2>Summary Metrics</h2>
    <div class="comparison">
        <div class="method">
            <h3>Direct LLM</h3>
            <p><strong>Factual Accuracy:</strong> {results["direct_contains_expected"].mean():.1%}</p>
            <p><strong>Citation Rate:</strong> {results["direct_has_citation"].mean():.1%}</p>
        </div>
        <div class="method">
            <h3>RAG System</h3>
            <p><strong>Factual Accuracy:</strong> {results["rag_contains_expected"].mean():.1%}</p>
            <p><strong>Citation Rate:</strong> {results["rag_has_citation"].mean():.1%}</p>
            <p><strong>Avg Retrieval Similarity:</strong> {results["rag_avg_similarity"].mean():.3f}</p>
        </div>
    </div>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Query</th>
            <th>Category</th>
            <th>Direct Answer</th>
            <th>RAG Answer</th>
            <th>Winner</th>
        </tr>
"""

    for _, row in results.iterrows():
        # Determine winner
        direct_score = int(row["direct_contains_expected"]) + int(
            row["direct_has_citation"]
        )
        rag_score = int(row["rag_contains_expected"]) + int(row["rag_has_citation"])

        if rag_score > direct_score:
            winner = "RAG ✓"
        elif direct_score > rag_score:
            winner = "Direct"
        else:
            winner = "Tie"

        html += f"""
        <tr>
            <td>{row["query"]}</td>
            <td>{row["category"]}</td>
            <td>{row["direct_answer"][:150]}...</td>
            <td>{row["rag_answer"][:150]}...</td>
            <td><strong>{winner}</strong></td>
        </tr>
"""

    html += """
    </table>

    <h2>Interpretation</h2>
    <div class="metric">
        <h3>Key Findings:</h3>
        <ul>
            <li><strong>Factual Accuracy:</strong> RAG should show higher accuracy for project-specific questions</li>
            <li><strong>Citation Rate:</strong> RAG should consistently provide citations</li>
            <li><strong>Out-of-scope handling:</strong> Both should say "not available" for questions outside document scope</li>
        </ul>
    </div>

</body>
</html>
"""

    Path(output_file).write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {output_file}")


def main() -> None:
    """Run RAG evaluation comparing direct LLM vs RAG system."""
    db_file = "outputs/02_vector_database.pkl"
    output_file = "outputs/04_rag_evaluation_report.html"

    print("RAG Evaluation")
    print("=" * 60)

    # Load database
    print(f"Loading vector database from {db_file}...")
    df = load_vector_database(db_file)
    print(f"Loaded {len(df)} chunks\n")

    # Run evaluation
    results = run_evaluation(df)

    # Show summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print("\nDirect LLM:")
    print(f"  Factual accuracy: {results['direct_contains_expected'].mean():.1%}")
    print(f"  Citation rate: {results['direct_has_citation'].mean():.1%}")

    print("\nRAG System:")
    print(f"  Factual accuracy: {results['rag_contains_expected'].mean():.1%}")
    print(f"  Citation rate: {results['rag_has_citation'].mean():.1%}")
    print(f"  Avg retrieval similarity: {results['rag_avg_similarity'].mean():.3f}")

    # Generate report
    generate_report(results, output_file)

    print("\n" + "=" * 60)
    print(f"Full report available at: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
