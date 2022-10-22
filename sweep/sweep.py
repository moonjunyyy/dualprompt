from calendar import EPOCH

sweep_config = {
    "method" : "random",
    "name"   : "Sweep",
    "metric" : {"name": "Average_Accuracy", "goal": "maximize"},
    "parameters" : {
        "model_args" : {
            "parameters" : {
                "lambd" : {'max': 2.0, 'min': 1e-4},
                "zetta" : {'max': 2.0, 'min': 1e-4}
            }
        }
    }
}