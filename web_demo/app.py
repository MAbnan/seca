import sys
import os
import json
import threading

from flask import Flask, render_template, jsonify

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seca_core.seca_engine import SECAEngine
from data.dataset_loader import load_dataset

app = Flask(__name__)

LOG_FILE = "logs/live_results.json"

RUNNING = False
RESULTS = []

PROGRESS_STATE = {
    "generation": 0,
    "individual": 0,
    "total_individuals": 0,
    "estimated_remaining_seconds": 0,
    "results": []
}


# -------------------------------
# Dashboard
# -------------------------------

@app.route("/")
def dashboard():

    if not RESULTS:
        return render_template(
            "index.html",
            generations=[],
            accuracy=[],
            fitness=[],
            params=[],
            summary={
                "generations": 0,
                "best_accuracy": 0,
                "best_fitness": 0,
                "min_parameters": 0
            },
            best_model={"genome": "Run SECA to start evolution"}
        )

    generations = [r["generation"] for r in RESULTS]
    accuracy = [r["accuracy"] for r in RESULTS]
    fitness = [r["fitness"] for r in RESULTS]
    params = [r["params"] for r in RESULTS]

    summary = {
        "generations": len(RESULTS),
        "best_accuracy": max(accuracy),
        "best_fitness": max(fitness),
        "min_parameters": min(params)
    }

    best_model = RESULTS[-1]

    return render_template(
        "index.html",
        generations=generations,
        accuracy=accuracy,
        fitness=fitness,
        params=params,
        summary=summary,
        best_model=best_model,
        progress_state=PROGRESS_STATE
    )


# -------------------------------
# Start Evolution
# -------------------------------

@app.route("/run_seca")
def run_seca():

    global RUNNING

    if RUNNING:
        return jsonify({"status": "already running"})

    RUNNING = True

    thread = threading.Thread(target=start_evolution)
    thread.start()

    return jsonify({"status": "started"})


# -------------------------------
# Evolution Process
# -------------------------------

def start_evolution():

    global RUNNING, RESULTS, PROGRESS_STATE

    x_train, x_test, y_train, y_test = load_dataset()

    seca = SECAEngine(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    RESULTS = []
    
    import time
    start_time = time.time()
    
    total_individuals = seca.generations * seca.population_size
    
    PROGRESS_STATE = {
        "generation": 0,
        "individual": 0,
        "total_individuals": total_individuals,
        "estimated_remaining_seconds": 0,
        "results": []
    }
    
    individuals_processed = 0

    def on_individual(gen, idx, result):
        nonlocal individuals_processed, start_time
        individuals_processed += 1
        
        elapsed = time.time() - start_time
        avg_time_per_ind = elapsed / individuals_processed
        remaining_inds = total_individuals - individuals_processed
        
        PROGRESS_STATE["generation"] = gen + 1
        PROGRESS_STATE["individual"] = idx + 1
        PROGRESS_STATE["estimated_remaining_seconds"] = int(avg_time_per_ind * remaining_inds)

    def on_generation(gen, best_stats, best_genome):
        entry = {
            "generation": gen,
            "accuracy": best_stats["accuracy"],
            "fitness": best_stats["fitness"],
            "params": best_stats["params"],
            "genome": best_genome
        }
        RESULTS.append(entry)
        PROGRESS_STATE["results"] = RESULTS
        
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(PROGRESS_STATE, f, indent=4)

    best_genome, best_stats, history = seca.run_evolution(
        on_individual=on_individual,
        on_generation=on_generation
    )

    PROGRESS_STATE["estimated_remaining_seconds"] = 0
    PROGRESS_STATE["generation"] = seca.generations
    PROGRESS_STATE["individual"] = seca.population_size

    with open(LOG_FILE, "w") as f:
        json.dump(PROGRESS_STATE, f, indent=4)

    RUNNING = False


# -------------------------------
# Live Results API
# -------------------------------

@app.route("/get_progress")
def get_progress():

    if RUNNING:
        return jsonify(PROGRESS_STATE)

    if not os.path.exists(LOG_FILE):
        return jsonify({
            "generation": 0,
            "individual": 0,
            "total_individuals": 0,
            "estimated_remaining_seconds": 0,
            "results": []
        })

    with open(LOG_FILE) as f:
        data = json.load(f)

    # Convert old format list to new format dictionary for backward compatibility
    if isinstance(data, list):
        data = {
            "generation": len(data),
            "individual": 0,
            "total_individuals": len(data) * 6, # Assuming default pop size 6
            "estimated_remaining_seconds": 0,
            "results": data
        }

    return jsonify(data)


# -------------------------------
# Status API
# -------------------------------

@app.route("/status")
def status():
    return jsonify({"running": RUNNING})


# -------------------------------
# Run Server
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)