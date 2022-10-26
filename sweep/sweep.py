from calendar import EPOCH

sweep_config = {
    "method" : "bayes",
    "name"   : "Sweep",
    "metric" : {"name": "Average_Accuracy", "goal": "maximize"},
    "parameters" : {
        "model_args" : {
            "parameters" : {
                "lambd" : {'max': 1e4, 'min': 1e-4},
                "zetta" : {'max': 1e4, 'min': 1e-4},
                "xi"    : {'max': 10,  'min': 1e-4},
            }
        }
    }
}