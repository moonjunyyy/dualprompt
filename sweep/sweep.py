from calendar import EPOCH

sweep_config = {
    "method" : "random",
    "name"   : "Sweep",
    "metric" : {"name": "Average_Accuracy", "goal": "maximize"},
    "parameters" : {
        "model_args" : {
            "parameters" : {
                "lambd" : {'max': 1, 'min': 1e-3},
                "zetta" : {'max': 1, 'min': 1e-3},
                "xi"    : {'max': 1, 'min': 1e-3},
            }
        }
    }
}