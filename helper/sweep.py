from calendar import EPOCH


sweep_config = {

    "metric" : {"name": "accuracy", "goal": "maximize"},
    "method" : "random",
    "parameters" : {
        "epochs" : { "values" : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] },
        "model_args" : {
            "parameters" : {
                "lambd" : {'max': 1.0, 'min': 0.001},
                "zetta" : {'max': 1.0, 'min': 0.001}
            }
        }
    }
}