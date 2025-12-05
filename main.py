#!/usr/bin/env python3
"""
AI Tutor Multi-Agent System - Main Entry Point

This system uses three LangChain agents:
1. Summarizer - Summarizes PDFs using map-reduce
2. Examiner - Creates quiz questions with structured output
3. Supervisor - Provides Socratic tutoring feedback
"""
import os
from dotenv import load_dotenv

load_dotenv()

def check_api_key():
    """Check if GROQ_API_KEY is set."""
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Please set your Groq API key to use this system.")
        return False
    return True


def run_demo():
    """Run a demo of the AI Tutor system."""
    from agents.workflow import AITutor
    
    pdf_path = "documents/Lecture3.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return
    
    print("=" * 60)
    print("AI TUTOR MULTI-AGENT SYSTEM")
    print("=" * 60)
    
    tutor = AITutor(pdf_path)
    
    print("\n[Step 1] Ingesting and summarizing document...")
    summary = tutor.ingest_and_summarize()
    print("\n--- Document Summary ---")
    print(summary)
    
    print("\n[Step 2] Generating quiz questions...")
    quiz = tutor.generate_quiz(num_questions=5)
    print("\n--- Generated Quiz ---")
    for task in quiz.tasks:
        print(f"\nQ{task.task_id} ({task.task_type}): {task.task}")
        if task.answer_options:
            for i, opt in enumerate(task.answer_options):
                print(f"   {chr(65+i)}) {opt}")
    
    print("\n[Step 3] Simulating quiz answers...")
    sample_answers = [task.correct_answer for task in quiz.tasks[:3]]
    sample_answers.append("wrong answer")
    if len(quiz.tasks) > 4:
        sample_answers.append(quiz.tasks[4].correct_answer)
    
    while len(sample_answers) < len(quiz.tasks):
        sample_answers.append("I don't know")
    
    print("Submitted answers:", sample_answers)
    
    print("\n[Step 4] Getting Socratic feedback...")
    feedback = tutor.submit_answers(sample_answers)
    print("\n--- Supervisor Feedback ---")
    print(feedback)
    
    print("\n[Step 5] Follow-up chat...")
    response = tutor.chat("Can you explain more about the main concept from the document?")
    print("\n--- Tutor Response ---")
    print(response)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def interactive_mode():
    """Run in interactive mode with user input."""
    from agents.workflow import AITutor
    
    pdf_path = input("Enter path to PDF (default: documents/Lecture3.pdf): ").strip()
    if not pdf_path:
        pdf_path = "documents/Lecture3.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return
    
    tutor = AITutor(pdf_path)
    
    print("\nLoading and summarizing document...")
    summary = tutor.ingest_and_summarize()
    print("\n--- Summary ---")
    print(summary)
    
    num_q = input("\nHow many quiz questions? (default: 5): ").strip()
    num_questions = int(num_q) if num_q.isdigit() else 5
    
    print(f"\nGenerating {num_questions} questions...")
    quiz = tutor.generate_quiz(num_questions)
    
    print("\n--- Quiz ---")
    answers = []
    for task in quiz.tasks:
        print(f"\nQ{task.task_id} ({task.task_type}): {task.task}")
        if task.answer_options:
            for i, opt in enumerate(task.answer_options):
                print(f"   {chr(65+i)}) {opt}")
        answer = input("Your answer: ").strip()
        answers.append(answer)
    
    print("\nEvaluating your answers...")
    feedback = tutor.submit_answers(answers)
    print("\n--- Feedback ---")
    print(feedback)
    
    while True:
        user_input = input("\nAsk a follow-up question (or 'quit' to exit): ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        response = tutor.chat(user_input)
        print("\n--- Tutor ---")
        print(response)
    
    print("\nGoodbye!")


def main():
    print("AI Tutor Multi-Agent System")
    print("-" * 40)
    
    if not check_api_key():
        print("\nTo get a Groq API key, visit: https://console.groq.com/")
        return
    
    print("\nSelect mode:")
    print("1. Demo mode (automated test)")
    print("2. Interactive mode (user input)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_mode()
    else:
        run_demo()


if __name__ == "__main__":
    main()
