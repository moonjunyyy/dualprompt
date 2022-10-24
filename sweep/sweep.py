from calendar import EPOCH

sweep_config = {
    "method" : "bayes",
    "name"   : "Sweep",
    "metric" : {"name": "Average_Accuracy", "goal": "maximize"},
    "parameters" : {
        "model_args" : {
            "parameters" : {
                "lambd" : {'max': 1e8, 'min': 1e-8},
                "zetta" : {'max': 1e8, 'min': 1e-8}
            }
        }
    }
}