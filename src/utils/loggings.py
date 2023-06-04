from typing import Dict, List, Union

import mlflow


def get_log_text(
    values: Union[List, Dict, float],
    total_values: Dict[str, List],
    name="loss",
):
    """A helper function to log the values.

    Args:
        values (_type_): The values to log.
        total_values (_type_): All the values will be added to this dictionary.
        name (str, optional): The prefix of the values. Defaults to "loss".

    Returns:
        str: The log message.
    """
    log_text = ""
    if type(values) == list:
        for i, val in enumerate(values):
            log_text += "{}_{}: {:.4f} ".format(name, i, val)
            if name not in total_values.keys():
                total_values[name] = []
            total = total_values[name]
            total.append(val)
            mlflow.log_metric(f"{name}_{i}", val)
    elif type(values) == dict:
        for k, v in values.items():
            log_text += "{}: {:.4f} ".format(k, v)
            if k not in total_values.keys():
                total_values[k] = []
            total = total_values[k]
            total.append(v)
            mlflow.log_metric(k, v)
    else:
        log_text += "{}: {:.4f} ".format(name, values)
        if name not in total_values.keys():
            total_values[name] = []
        total = total_values[name]
        total.append(values)
        mlflow.log_metric(name, values)
    return log_text
