"""Main entry point for the tabular ML agent framework"""

import asyncio
import os
import sys
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.agents import ReActMLAgent, ReActAgentWithReflection
from datetime import datetime
import json


async def main():
    """Main function to run the ReAct ML Agent"""
    
    print("Tabular ML Agent Framework (ReAct Agent)")
    print("=" * 70)
    
    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: API key required!")
        return
    
    # Setup
    os.environ["OPENAI_API_KEY"] = api_key
    config = Config.from_env()
    config.llm.model = "gpt-4o-mini"  # or "gpt-4o" for better reasoning
    
    print(f"LLM Model: {config.llm.model}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Output Directory: {config.output_dir}")
    
    # Get dataset path
    default_dataset = "../oncologyml/task1_clinical.csv"
    dataset_path = input(f"\nEnter dataset path (default: {default_dataset}): ").strip()
    if not dataset_path:
        dataset_path = default_dataset
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: File not found: {dataset_path}")
        return
    
    # Get objective
    default_objective = "Build the best machine learning model for this dataset"
    objective = input(f"Enter your ML objective (default: {default_objective}): ").strip()
    if not objective:
        objective = default_objective
    
    # Optional: test set path
    testset_path = input("Enter test set path (optional, press Enter to skip): ").strip()
    if testset_path and not os.path.exists(testset_path):
        print(f"Warning: Test set file not found: {testset_path}, will auto-split instead")
        testset_path = None
    elif not testset_path:
        testset_path = None
    
    # Choose agent type
    print("\nChoose agent type:")
    print("1. Standard ReAct Agent")
    print("2. ReAct with Reflection (adds reflection after evaluations)")
    choice = input("Enter choice (1 or 2, default 1): ").strip() or "1"
    
    if choice == "2":
        print("\nUsing ReAct Agent with Reflection")
        agent = ReActAgentWithReflection(config)
    else:
        print("\nUsing Standard ReAct Agent")
        agent = ReActMLAgent(config)
    
    print(f"\nDataset: {dataset_path}")
    print(f"Objective: {objective}")
    
    try:
        # Run the agent
        result = await agent.run(
            dataset_path=dataset_path,
            testset_path=testset_path,
            objective=objective,
            max_iterations=20
        )
        
        if result["success"]:
            print("\n" + "=" * 70)
            print("AGENT COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            print(f"\nFINAL RESULTS:")
            print(f"   Iterations used: {result['iterations']}/{agent.max_iterations}")
            print(f"   Best model: {result['best_model']}")
            print(f"   Best score: {result['best_score']:.3f}")
            print(f"   Models trained: {', '.join(result['trained_models'])}")
            
            # Save results
            output_dir = Path("outputs/react_conversations")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_file = output_dir / f"react_conversation_{timestamp}.json"
            
            agent.save_conversation(str(conversation_file))
            
            print(f"\nFull conversation saved to: {conversation_file}")
            
            # Save evaluation metrics
            eval_dir = Path("outputs/evaluations")
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            eval_file = eval_dir / f"evaluation_metrics_{timestamp}.json"
            
            final_state = result["final_state"]
            
            evaluation_data = {
                "timestamp": timestamp,
                "dataset": dataset_path,
                "objective": objective,
                "task_type": final_state.get("feature_result", {}).get("task_type", "unknown"),
                "best_model": result["best_model"],
                "best_score": result["best_score"],
                "iterations": result["iterations"],
                "all_models": {}
            }
            
            # Add detailed metrics for each model
            if final_state.get("evaluation_results"):
                for model_name, eval_result in final_state["evaluation_results"].items():
                    metrics = eval_result.get("metrics", {})
                    model_info = final_state.get("trained_models", {}).get(model_name, {})
                    evaluation_data["all_models"][model_name] = {
                        "test_metrics": metrics,
                        "cv_metrics": model_info.get("cv_metrics", {}),
                        "training_time": model_info.get("training_time", 0),
                        "cv_score": model_info.get("cv_score", 0)
                    }
            
            with open(eval_file, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            
            print(f"Evaluation metrics saved to: {eval_file}")
            print("\n" + "=" * 70)
            print("Execution completed successfully!")
            print("=" * 70)
            
        else:
            print(f"\nAgent failed: {result.get('error')}")
            print(f"   Completed {result['iterations']} iterations before failure")
        
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
